"""An old browser callback or decode cannot publish into a replacement stream."""
import asyncio
import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from core.perception import frames_as_they_are_drawn as frames

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
CLIP = {'x':0., 'y':0., 'width':16., 'height':16.}


def encoded():
    image = io.BytesIO()
    Image.new('RGB', (16,16), 'green').save(image, format='JPEG')
    return base64.b64encode(image.getvalue()).decode('ascii')


class Session:
    def __init__(self):
        self.callback, self.calls = None, []
        self.pause, self.resume = None, None

    def on(self, _event, callback):
        self.callback = callback

    async def send(self, method, params=None):
        self.calls.append((method,params))
        if method == 'Performance.getMetrics':
            if self.pause is not None:
                self.pause.set()
                await self.resume.wait()
            return {'metrics':[{'name':'Timestamp','value':100.}]}
        if method == 'Runtime.evaluate':
            return {'result':{'value':1000.}}
        return {}

    async def detach(self):
        self.calls.append(('detach', None))


def page(*sessions):
    pending = iter(sessions)

    async def new(_page):
        return next(pending)

    async def evaluate(_script):
        return [16,16]

    return SimpleNamespace(context=SimpleNamespace(new_cdp_session=new), evaluate=evaluate)


async def test_an_old_callback_is_acknowledged_only_on_its_originating_session(monkeypatch):
    clock = SimpleNamespace(at=10.)
    monkeypatch.setattr(frames, 'time', SimpleNamespace(monotonic=lambda:clock.at))
    first, second = Session(), Session()
    stream = frames.PageFrames(page(first,second))
    assert await stream._start()
    old_callback = first.callback
    await stream.close()
    assert await stream._start()
    clock.at = 11.
    second.callback({'data':encoded(), 'metadata':{'monotonicTimestamp':100.1}, 'sessionId':22})
    latest = stream._latest
    old_callback({'data':encoded(), 'metadata':{'monotonicTimestamp':100.2}, 'sessionId':11})
    await asyncio.sleep(0)
    assert stream._latest is latest
    assert ('Page.screencastFrameAck', {'sessionId':11}) in first.calls
    assert ('Page.screencastFrameAck', {'sessionId':11}) not in second.calls
    await stream.close()


async def test_a_late_decode_cannot_return_pixels_from_the_closed_generation(monkeypatch):
    clock = SimpleNamespace(at=10.)
    monkeypatch.setattr(frames, 'time', SimpleNamespace(monotonic=lambda:clock.at))
    first, second = Session(), Session()
    stream = frames.PageFrames(page(first,second))
    assert await stream._start()
    clock.at = 11.
    first.callback({'data':encoded(), 'metadata':{'monotonicTimestamp':100.1}, 'sessionId':1})
    entered, finished = asyncio.Event(), asyncio.Event()

    async def decode(fn,*args):
        entered.set()
        await finished.wait()
        return fn(*args)

    monkeypatch.setattr(frames, 'run_interactive_cpu', decode)
    reading = asyncio.create_task(stream.look(CLIP))
    await entered.wait()
    await stream.close()
    assert await stream._start()
    finished.set()
    assert await reading is None
    await stream.close()


async def test_an_old_calibration_cannot_set_the_replacement_clock(monkeypatch):
    clock = SimpleNamespace(at=10.)
    monkeypatch.setattr(frames, 'time', SimpleNamespace(monotonic=lambda:clock.at))
    first, second = Session(), Session()
    first.pause, first.resume = asyncio.Event(), asyncio.Event()
    stream = frames.PageFrames(page(first,second))
    starting = asyncio.create_task(stream._start())
    await first.pause.wait()
    old_clock = stream._clock
    await stream.close()
    clock.at = 20.
    assert await stream._start()
    current_clock = stream._clock
    assert current_clock is not old_clock
    first.resume.set()
    assert not await starting
    clock.at = 21.
    second.callback({'data':encoded(), 'metadata':{'monotonicTimestamp':100.1}, 'sessionId':2})
    assert stream._latest[2] == pytest.approx(20.1)
    assert not any(method == 'Page.startScreencast' for method,_ in first.calls)
    await stream.close()


async def test_an_old_waiter_cannot_consume_the_replacement_streams_picture(monkeypatch):
    clock = SimpleNamespace(at=10.)
    monkeypatch.setattr(frames, 'time', SimpleNamespace(monotonic=lambda:clock.at))
    first, second = Session(), Session()
    stream = frames.PageFrames(page(first,second))
    assert await stream._start()
    reading = asyncio.create_task(stream.look(CLIP))
    await asyncio.sleep(0)
    await stream.close()
    assert await stream._start()
    clock.at = 11.
    second.callback({'data':encoded(), 'metadata':{'monotonicTimestamp':100.1}, 'sessionId':2})
    assert await reading is None
    assert stream._taken_at == -1.
    current = await stream.look(CLIP)
    assert current is not None and current[1] == pytest.approx(10.1)
    await stream.close()


async def test_the_stream_session_boundary_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert frames._frame_session_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'a_stream_frame_keeps_its_session_owner')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
