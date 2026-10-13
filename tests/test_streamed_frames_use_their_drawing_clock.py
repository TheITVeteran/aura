"""Queued frame delivery preserves source cadence and finite capture bounds."""
from __future__ import annotations

import asyncio
import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from core.perception import frames_as_they_are_drawn as frames
from core.perception.a_clock_for_drawn_frames import (
    DrawingClock,
    MeasuredFrameTime,
    _drawing_clock_invariant,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
CLIP = {"x": 0., "y": 0., "width": 16., "height": 16.}


def _encoded(colour=(40, 80, 120)):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), colour).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _stream(monkeypatch):
    clock = SimpleNamespace(at=11.)
    monkeypatch.setattr(frames, "time", SimpleNamespace(monotonic=lambda: clock.at))
    monkeypatch.setattr(frames, "STILL_AFTER_S", .001)
    stream = frames.PageFrames(SimpleNamespace())
    stream._session = SimpleNamespace(send=AsyncMock(), detach=AsyncMock())
    assert stream._clock.correlate("monotonic", 100., 10., 10.002)
    assert stream._clock.correlate("wall", 1000., 10., 10.002)
    return stream, clock


async def test_delayed_frames_keep_the_source_velocity_and_real_picture_age(monkeypatch):
    stream, clock = _stream(monkeypatch)
    stream._frame({"data": _encoded(), "metadata": {"monotonicTimestamp": 100.1}, "sessionId": 1})
    _, first = await stream.look(CLIP)
    clock.at = 11.001
    stream._frame({"data": _encoded((120, 80, 40)), "metadata": {"monotonicTimestamp": 100.2}, "sessionId": 2})
    _, second = await stream.look(CLIP)
    assert second - first == pytest.approx(.1)
    assert 3. / (second - first) == pytest.approx(30.)
    assert clock.at - second == pytest.approx(.801)
    assert first.lower == pytest.approx(10.1) and first.upper == pytest.approx(10.102)
    assert second.source == "browser-monotonic" and second.arrived == 11.001
    assert first.arrived == 11. and first.source_at == 100.1
    await stream.close()


async def test_monotonic_swap_time_takes_precedence_over_a_changed_wall_clock(monkeypatch):
    stream, _ = _stream(monkeypatch)
    stream._frame({"data": _encoded(), "metadata": {"monotonicTimestamp": 100.1, "timestamp": 5000.}})
    _, at = await stream.look(CLIP)
    assert at.source == "browser-monotonic" and at == pytest.approx(10.1)
    await stream.close()


@pytest.mark.parametrize("bad", [100.1, 100.05, float("nan"), float("inf"), True, "100.2", 110.])
async def test_repeated_regressed_or_future_source_times_do_not_renew_a_frame(monkeypatch, bad):
    stream, clock = _stream(monkeypatch)
    stream._frame({"data": _encoded(), "metadata": {"monotonicTimestamp": 100.1}, "sessionId": 1})
    old = await stream.look(CLIP)
    clock.at = 11.1
    stream._frame({"data": _encoded((120, 80, 40)), "metadata": {"monotonicTimestamp": bad}, "sessionId": 2})
    assert await stream.look(CLIP) is None
    assert stream._latest[2] is old[1]
    await asyncio.sleep(0)
    assert stream._session.send.await_count == 2  # both protocol messages still acknowledged
    await stream.close()


async def test_missing_metadata_retains_explicit_receipt_semantics(monkeypatch):
    stream, _ = _stream(monkeypatch)
    stream._frame({"data": _encoded(), "metadata": {}, "sessionId": 1})
    _, at = await stream.look(CLIP)
    assert isinstance(at, float) and isinstance(at, MeasuredFrameTime)
    assert at == 11. and at.source == "stream-receipt"
    assert at.lower is None and at.upper == at.arrived == 11.
    await stream.close()


async def test_a_missing_source_clock_cannot_switch_a_measured_stream_to_arrival_time(monkeypatch):
    stream, clock = _stream(monkeypatch)
    stream._frame({"data": _encoded(), "metadata": {"timestamp": 1000.1}})
    _, old = await stream.look(CLIP)
    clock.at = 12.
    stream._frame({"data": _encoded((80, 40, 120)), "metadata": {}})
    assert await stream.look(CLIP) is None
    assert stream._latest[2] is old
    await stream.close()


async def test_decode_delay_does_not_move_the_capture_interval(monkeypatch):
    stream, clock = _stream(monkeypatch)

    async def decode(function, *args):
        clock.at += .4
        return function(*args)

    monkeypatch.setattr(frames, "run_interactive_cpu", decode)
    stream._frame({"data": _encoded(), "metadata": {"timestamp": 1000.1}})
    _, at = await stream.look(CLIP)
    assert at == pytest.approx(10.1) and at.upper == pytest.approx(10.102)
    assert at.arrived == 11. and clock.at == 11.4
    await stream.close()


class _Session:
    def __init__(self, clock, *, performance=True, wall=True):
        self.clock, self.performance, self.wall = clock, performance, wall
        self.sent = []
        self.callback = None

    def on(self, event, callback):
        self.callback = callback

    async def send(self, method, params=None):
        self.sent.append(method)
        if method.startswith("Performance.") and not self.performance:
            raise RuntimeError("unsupported Performance domain")
        if method == "Performance.getMetrics":
            self.clock.at += .004
            return {"metrics": [{"name": "Timestamp", "value": 100.}]}
        if method == "Runtime.evaluate":
            if not self.wall:
                raise RuntimeError("unsupported Runtime clock read")
            assert params["expression"] == "Date.now() / 1000"
            self.clock.at += .006
            return {"result": {"value": 1000.}}
        return {}

    async def detach(self):
        return None


def _page(session):
    async def cdp(page):
        return session

    return SimpleNamespace(context=SimpleNamespace(new_cdp_session=cdp), evaluate=AsyncMock(return_value=[16, 16]))


async def test_startup_correlates_both_browser_clocks_inside_real_requests(monkeypatch):
    clock = SimpleNamespace(at=10.)
    session = _Session(clock)
    monkeypatch.setattr(frames, "time", SimpleNamespace(monotonic=lambda: clock.at))
    stream = frames.PageFrames(_page(session))
    assert await stream._start()
    clock.at = 11.
    stream._frame({"data": _encoded(), "metadata": {"monotonicTimestamp": 100.1}})
    _, at = await stream.look(CLIP)
    assert at == pytest.approx(10.1) and at.upper == pytest.approx(10.104)
    assert "Performance.getMetrics" in session.sent and "Runtime.evaluate" in session.sent
    await stream.close()


async def test_wall_correlation_works_when_monotonic_cdp_metrics_are_unsupported(monkeypatch):
    clock = SimpleNamespace(at=10.)
    session = _Session(clock, performance=False)
    monkeypatch.setattr(frames, "time", SimpleNamespace(monotonic=lambda: clock.at))
    stream = frames.PageFrames(_page(session))
    assert await stream._start() and not stream.unavailable
    clock.at = 11.
    stream._frame({"data": _encoded(), "metadata": {"timestamp": 1000.1}})
    _, at = await stream.look(CLIP)
    assert at == pytest.approx(10.099) and at.source == "browser-wall"
    assert at.upper == pytest.approx(10.106)
    await stream.close()


async def test_unsupported_cdp_clock_queries_do_not_kill_a_legacy_receipt_stream(monkeypatch):
    clock = SimpleNamespace(at=10.)
    session = _Session(clock, performance=False, wall=False)
    monkeypatch.setattr(frames, "time", SimpleNamespace(monotonic=lambda: clock.at))
    stream = frames.PageFrames(_page(session))
    assert await stream._start() and not stream.unavailable
    stream._frame({"data": _encoded(), "metadata": {}})
    _, at = await stream.look(CLIP)
    assert at == 10. and at.lower is None and at.source == "stream-receipt"
    await stream.close()


async def test_provided_source_time_without_a_correlation_remains_unknown():
    clock = DrawingClock()
    assert clock.frame({"timestamp": 1000.}, 10.) is None
    assert clock.frame({"monotonicTimestamp": 100.}, 10.) is None


async def test_a_wall_clock_jump_does_not_reanchor_old_frames_to_the_latest_arrival():
    clock = DrawingClock()
    assert clock.correlate("wall", 1000., 10., 10.01)
    original = clock.frame({"timestamp": 1000.1}, 11.)
    assert clock.frame({"timestamp": 999.9}, 11.1) is None
    assert clock.frame({"timestamp": 1200.}, 11.1) is None
    assert not clock.correlate("wall", 1200., 11.1, 11.11)
    assert original == pytest.approx(10.1) and original.epoch == 0


async def test_only_an_explicit_new_stream_epoch_can_bind_a_new_clock_origin():
    clock = DrawingClock()
    assert clock.correlate("monotonic", 100., 10., 10.01)
    old = clock.frame({"monotonicTimestamp": 100.1}, 11.)
    clock.reset()
    assert clock.frame({"monotonicTimestamp": 1.}, 11.1) is None
    assert clock.correlate("monotonic", 1., 11.1, 11.11)
    new = clock.frame({"monotonicTimestamp": 1.1}, 11.3)
    assert new.epoch == old.epoch + 1 and old == pytest.approx(10.1)
    assert new == pytest.approx(11.2)


async def test_arrival_supplies_a_capture_upper_bound_without_changing_source_cadence():
    clock = DrawingClock()
    assert clock.correlate("monotonic", 100., 10., 10.02)
    at = clock.frame({"monotonicTimestamp": 100.1}, 10.11)
    assert at.lower == pytest.approx(10.1) and at.upper == pytest.approx(10.11)
    with pytest.raises(AttributeError):
        at.lower = 11.
    with pytest.raises(AttributeError):
        del at.arrived
    assert _drawing_clock_invariant() == ()


async def test_one_stream_cannot_change_between_two_calibrated_clock_domains():
    clock = DrawingClock()
    assert clock.correlate('wall', 1000., 10., 10.01)
    assert clock.correlate('monotonic', 100., 10.004, 10.008)
    first = clock.frame({'timestamp': 1000.1}, 11.)
    # Even a better source appearing later cannot add its offset error to the
    # displacement interval already being measured on the first clock.
    second = clock.frame({'timestamp': 1000.2, 'monotonicTimestamp': 100.2}, 11.01)
    assert second.source == first.source == 'browser-wall'
    assert second - first == pytest.approx(.1)
    assert clock.frame({'monotonicTimestamp': 100.3}, 11.02) is None


async def test_capture_bounds_must_belong_wholly_to_one_delivered_control():
    from core.agency.which_one_answers_to_her import WhichIsHers

    hers = WhichIsHers()
    hers.holding('opaque-a', 10., trying=True)
    hers.holding('opaque-b', 10.15, trying=False)
    clock = DrawingClock()
    assert clock.correlate('monotonic', 100., 10., 10.02)
    first = clock.frame({'monotonicTimestamp': 100.1}, 11.)
    crossing = clock.frame({'monotonicTimestamp': 100.14}, 11.01)
    last = clock.frame({'monotonicTimestamp': 100.2}, 11.02)
    assert hers._held_at(first) == (10., 'opaque-a', True)
    assert hers._held_at(crossing) is None
    assert hers._held_at(last) == (10.15, 'opaque-b', False)
    unknown = MeasuredFrameTime(11.1, lower=None, upper=11.1, arrived=11.1,
                               source='stream-receipt', source_at=None, epoch=0)
    assert hers._held_at(unknown) is None


async def test_integer_millisecond_clock_reads_keep_the_discarded_fraction_in_the_bounds():
    clock = DrawingClock()
    # The browser read's true time can be .0009 after its integer value.
    assert clock.correlate('wall', 1000., 10., 10., resolution=.001)
    frame = clock.frame({'timestamp': 1000.1}, 11.)
    assert frame.lower <= 10.0991 <= frame.upper
    assert frame.upper - frame.lower == pytest.approx(.001)


async def test_the_drawing_clock_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'queued_frames_keep_the_drawing_clock')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())


async def test_capture_receipts_keep_their_bounds_through_private_worker_serialization():
    import copy
    import pickle

    clock = DrawingClock()
    assert clock.correlate('monotonic', 100., 10., 10.02)
    original = clock.frame({'monotonicTimestamp': 100.1}, 11.)
    for restored in (copy.deepcopy(original), pickle.loads(pickle.dumps(original))):
        assert isinstance(restored, MeasuredFrameTime) and restored == original
        assert vars(restored) == vars(original)
        with pytest.raises(AttributeError):
            restored.upper = 12.
