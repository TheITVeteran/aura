"""A slow compositor is sampled by time; source custody survives resampling."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime
from core.perception.frames_as_they_are_drawn import PageFrames
from core.perception.sampling_the_drawings_pace import DrawingCadence, _drawing_cadence_invariant


def captured(at, *, arrived=10., uncertainty=.001, epoch=1):
    return MeasuredFrameTime(at, lower=at, upper=at + uncertainty, arrived=arrived,
                             source='browser-wall', source_at=at + 1000., epoch=epoch)


def test_queued_arrivals_keep_the_source_sampling_pace():
    cadence = DrawingCadence(4, .125)
    for at in (1., 1.225, 1.45):
        cadence.observe(captured(at))
        assert cadence.next_stride() == 4
    cadence.observe(captured(1.675))
    assert cadence.next_stride() == 2
    for at in (2., 2.25, 2.5, 2.75):
        cadence.observe(captured(at))
    assert cadence.next_stride() == 1
    assert _drawing_cadence_invariant() == ()


def test_a_fast_source_and_one_slow_gap_do_not_resample():
    cadence = DrawingCadence(4, .125)
    for at in (1., 1.06, 1.12, 1.8, 1.86):
        cadence.observe(captured(at))
        assert cadence.next_stride() == 4


@pytest.mark.parametrize('missing', [None, 2., float('nan'), captured(2., epoch=2)])
def test_missing_receipts_and_new_clocks_break_the_interval_series(missing):
    cadence = DrawingCadence(4, .125)
    for at in (1., 1.225, 1.45):
        cadence.observe(captured(at))
    cadence.observe(missing)
    assert cadence.next_stride() == 4


def test_capture_uncertainty_cannot_overstate_a_slow_source():
    cadence = DrawingCadence(4, .125)
    for at in (1., 1.2, 1.4, 1.6):
        cadence.observe(captured(at, uncertainty=.1))
    assert cadence.next_stride() == 4


def test_repeated_or_reversed_captures_do_not_accumulate_intervals():
    cadence = DrawingCadence(4, .125)
    for at in (1., 1.2, 1.4, 1.4, 1.6, 1.5, 1.8):
        cadence.observe(captured(at))
        assert cadence.next_stride() == 4


class Session:
    def __init__(self):
        self.sent = []
        self.block_stop = None
        self.fail_start = False
        self.detached = False

    async def send(self, method, options=None):
        self.sent.append((method, options))
        if method == 'Page.stopScreencast' and self.block_stop is not None:
            await self.block_stop.wait()
        if method == 'Page.startScreencast' and self.fail_start:
            raise RuntimeError('the renderer closed')

    async def detach(self):
        self.detached = True


def slow_stream():
    stream = PageFrames(None, every_nth_frame=4)
    session = Session()
    stream._session, stream._generation = session, 1
    stream._options = {'format': 'jpeg', 'maxWidth': 640, 'maxHeight': 480, 'everyNthFrame': 4}
    stream._clock.correlate('wall', 1000., 1., 1.001)
    for at in (1., 1.225, 1.45, 1.675):
        stream._cadence.observe(captured(at))
    return stream, session


@pytest.mark.asyncio
async def test_resampling_preserves_the_producing_session_and_original_frame():
    stream, session = slow_stream()
    clock = stream._clock
    receipt = captured(1.675)
    stream._latest = ('pixels', {}, receipt)
    stream._taken_at = receipt
    assert await stream._sample_at_the_drawings_pace()
    assert stream.every_nth_frame == 2
    assert stream._clock is clock and stream._session is session
    assert stream._latest[2] is receipt and stream._taken_at is receipt
    assert [method for method, _ in session.sent] == ['Page.stopScreencast', 'Page.startScreencast']
    assert session.sent[-1][1] == {**stream._options, 'everyNthFrame': 2}
    await stream.close()


@pytest.mark.asyncio
async def test_close_during_resampling_cannot_start_an_orphan_stream():
    stream, session = slow_stream()
    session.block_stop = asyncio.Event()
    changing = asyncio.create_task(stream._sample_at_the_drawings_pace())
    await asyncio.sleep(0)
    closing = asyncio.create_task(stream.close())
    await asyncio.sleep(0)
    session.block_stop.set()
    assert not await changing
    await closing
    assert session.detached and stream._session is None
    assert all(method != 'Page.startScreencast' for method, _ in session.sent)


@pytest.mark.asyncio
async def test_failed_resampling_retires_the_stream_for_fresh_capture():
    stream, session = slow_stream()
    session.fail_start = True
    assert not await stream._sample_at_the_drawings_pace()
    assert stream.unavailable and stream._session is None and session.detached


def test_sampling_claim_is_registered():
    from core.organism.claims_realtime_control import install_realtime_control_claims

    suite = SimpleNamespace(tests=[], claims=[])
    suite.add_test = suite.tests.append
    suite.add_claim = suite.claims.append
    install_realtime_control_claims(suite)
    claim = next(c for c in suite.claims if c.test == 'drawing_sampling_uses_capture_cadence')
    test = next(t for t in suite.tests if t.name == claim.test)
    assert test.predict(None)
