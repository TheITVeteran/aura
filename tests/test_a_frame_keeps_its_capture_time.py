"""Slow delivery and stream silence cannot turn old pixels into new pixels."""
from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from PIL import Image

from core.perception import frames_as_they_are_drawn as frames
from core.skills import screen_pursuit_as_it_happens as pursuit

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
CLIP = {"x": 0., "y": 0., "width": 16., "height": 16.}


def _encoded(colour=(40, 80, 120)):
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), colour).save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _stream(monkeypatch):
    stream = frames.PageFrames(SimpleNamespace())
    stream._session = SimpleNamespace(send=AsyncMock(), detach=AsyncMock())
    stream._latest = (_encoded(), {"deviceWidth": 16}, 10.)
    monkeypatch.setattr(frames, "STILL_AFTER_S", .001)
    return stream


async def test_consuming_a_stream_frame_does_not_allow_it_to_be_reissued(monkeypatch):
    stream = _stream(monkeypatch)
    first = await stream.look(CLIP)
    assert first[1] == 10. and first[0].shape == (16, 16, 3)
    assert await stream.look(CLIP) is None
    assert await stream.look(CLIP) is None
    assert not stream.unavailable
    await stream.close()


async def test_an_identical_but_newly_delivered_frame_is_a_new_observation(monkeypatch):
    stream = _stream(monkeypatch)
    old = await stream.look(CLIP)
    stream._latest = (_encoded(), {"deviceWidth": 16}, 10.1)
    new = await stream.look(CLIP)
    assert np.array_equal(old[0], new[0])
    assert new[1] == 10.1
    await stream.close()


async def test_a_new_frame_arriving_during_the_wait_is_used(monkeypatch):
    import asyncio

    stream = _stream(monkeypatch)
    await stream.look(CLIP)
    monkeypatch.setattr(frames, "STILL_AFTER_S", 1.)
    monkeypatch.setattr(frames, "time", SimpleNamespace(monotonic=lambda: 10.2))
    waiting = asyncio.create_task(stream.look(CLIP))
    await asyncio.sleep(0)
    stream._frame({"data": _encoded((80, 40, 10)), "metadata": {"deviceWidth": 16}, "sessionId": 7})
    new = await waiting
    assert new[1] == 10.2
    await stream.close()


async def test_decode_time_does_not_replace_the_frame_receipt_time(monkeypatch):
    stream = _stream(monkeypatch)
    clock = SimpleNamespace(at=11.)
    monkeypatch.setattr(frames, "time", SimpleNamespace(monotonic=lambda: clock.at))

    async def decode(function, *args):
        clock.at += .3
        return function(*args)

    monkeypatch.setattr(frames, "run_interactive_cpu", decode)
    assert (await stream.look(CLIP))[1] == 10.
    assert clock.at == pytest.approx(11.3)
    await stream.close()


async def test_a_stalled_stream_uses_the_real_screenshot_fallback(monkeypatch):
    stream = _stream(monkeypatch)
    await stream.look(CLIP)
    clock = SimpleNamespace(at=11.)
    monkeypatch.setattr(pursuit, "time", SimpleNamespace(monotonic=lambda: clock.at))
    fresh = np.full((16, 16, 3), 180, dtype=np.uint8)

    async def screenshot(page, clip, **kwargs):
        assert clip == CLIP
        clock.at += .2
        return fresh

    capture = AsyncMock(side_effect=screenshot)
    monkeypatch.setattr("core.perception.a_picture_of_her_page.picture_of", capture)
    player = pursuit.PlayingAsItHappens(SimpleNamespace(), (0., 0., 1., 1.), "observe", 20.)
    player._clip, player._frames = CLIP, stream
    player._canvas = SimpleNamespace(unavailable=True)
    read = await player.look()
    capture.assert_awaited_once()
    assert read[0] is fresh
    assert read[1] == 11. and clock.at == pytest.approx(11.2)
    await stream.close()


async def test_a_canvas_request_retains_its_start_through_transport_and_decode(monkeypatch):
    clock = SimpleNamespace(at=10.)
    monkeypatch.setattr(frames, "time", SimpleNamespace(monotonic=lambda: clock.at))

    async def evaluate(script, clip):
        assert clip == CLIP
        clock.at += .1
        return "data:image/jpeg;base64," + _encoded()

    async def decode(function, *args):
        clock.at += .2
        return function(*args)

    monkeypatch.setattr(frames, "run_interactive_cpu", decode)
    canvas = frames.CanvasFrames(SimpleNamespace(evaluate=evaluate))
    canvas._observing, canvas._checked = True, True
    read = await canvas.look(CLIP)
    assert read[0].shape == (16, 16, 3)
    assert read[1] == 10. and clock.at == pytest.approx(10.3)


async def test_a_failed_fresh_capture_does_not_return_the_old_stream_frame(monkeypatch):
    stream = _stream(monkeypatch)
    await stream.look(CLIP)
    monkeypatch.setattr("core.perception.a_picture_of_her_page.picture_of", AsyncMock(return_value=None))
    player = pursuit.PlayingAsItHappens(SimpleNamespace(), (0., 0., 1., 1.), "observe", 20.)
    player._clip, player._frames = CLIP, stream
    player._canvas = SimpleNamespace(unavailable=True)
    assert await player.look() is None
    await stream.close()


async def test_registered_frame_claim_runs_the_adjacent_invariant():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert frames._frame_time_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == "repeated_stream_frames_keep_their_original_time")
    assert check.predict(None) is True
    assert any(c.test == check.name for c in suite.claims())
