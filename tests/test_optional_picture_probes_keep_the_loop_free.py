"""Optional visual guesses cannot block input deadlines or outlive their owner."""
from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import Future

import numpy as np
import pytest

from core.perception import a_picture_examined_without_holding_the_loop as probing
from core.runtime.lockdep import checked_thread_semaphore

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_the_detached_probe_claim_is_registered():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert probing._picture_probe_inputs_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'picture_probes_own_detached_inputs')
    assert check.predict(None) and any(c.test == check.name for c in suite.claims())


@pytest.fixture
def slots(monkeypatch):
    capacity = checked_thread_semaphore("test.optional_picture_probe_operations", 1)
    monkeypatch.setattr(probing, "_SLOTS", capacity)
    return capacity


async def entered(event):
    for _ in range(200):
        if event.is_set():
            return
        await asyncio.sleep(.001)
    raise AssertionError("the invented detector did not start")


async def completed(raw):
    await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(raw)), 2.)


async def test_an_expensive_picture_check_keeps_the_event_loop_running(slots):
    started = threading.Event()
    worker_threads, beats = [], []

    def detector(picture):
        worker_threads.append(threading.get_ident())
        started.set()
        time.sleep(.06)
        return int(picture[2, 2, 0])

    probe = probing.PictureProbe(wait_s=.5)
    task = asyncio.create_task(probe.examine(detector, np.ones((6, 6, 3), np.uint8)))
    await entered(started)
    while not task.done():
        beats.append(time.monotonic())
        await asyncio.sleep(.002)
    assert await task == 1
    assert len(beats) >= 3 and len(worker_threads) == 1
    assert worker_threads[0] != threading.get_ident()
    assert probe.status == "examined" and not probe.busy
    await probe.close()


async def test_a_timeout_retains_its_worker_without_queuing_more_pictures(slots):
    started, release = threading.Event(), threading.Event()
    calls = []

    def detector(value):
        calls.append(value)
        started.set()
        assert release.wait(2.)
        return value

    probe = probing.PictureProbe(wait_s=.01)
    try:
        assert await probe.examine(detector, "old picture") is probing.UNMEASURED
        await entered(started)
        operation = probe._operation
        assert probe.busy and probe.status == "probe unmeasured"
        for _ in range(10):
            assert await probe.examine(detector, "new picture") is probing.UNMEASURED
        assert calls == ["old picture"] and not slots.acquire(blocking=False)
        release.set()
        await completed(operation.raw)
        assert await probe.examine(lambda: "fresh answer") == "fresh answer"
        assert calls == ["old picture"]
    finally:
        release.set()
        await probe.close()


async def test_cancellation_keeps_capacity_until_the_actual_worker_finishes(slots):
    started, release = threading.Event(), threading.Event()

    def detector():
        started.set()
        assert release.wait(2.)
        return "late answer"

    probe = probing.PictureProbe(wait_s=1.)
    task = asyncio.create_task(probe.examine(detector))
    try:
        await entered(started)
        operation = probe._operation
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert probe.status == "closed" and probe.busy
        assert await probe.examine(lambda: "wrong answer") is probing.UNMEASURED
        assert not slots.acquire(blocking=False)
        other = probing.PictureProbe()
        assert await other.examine(lambda: "queued answer") is probing.UNMEASURED
        assert other.status == "probe capacity unavailable" and other._operation is None
        await other.close()
        release.set()
        await completed(operation.raw)
        assert not probe.busy and probe.status == "closed"
        assert slots.acquire(blocking=False)
        slots.release()
    finally:
        release.set()
        await probe.close()


async def test_the_worker_owns_copied_pixels_and_interpretation_inputs(slots):
    started, release = threading.Event(), threading.Event()
    picture = np.ones((6, 6, 3), np.uint8)
    words, options = ["first"], {"rule": ["original"]}

    def detector(image, said, *, rules):
        started.set()
        assert release.wait(2.)
        return int(image[0, 0, 0]), said, rules

    probe = probing.PictureProbe(wait_s=.5)
    task = asyncio.create_task(probe.examine(detector, picture, words, rules=options))
    try:
        await entered(started)
        picture[:] = 99
        words.append("later")
        options["rule"].append("changed")
        release.set()
        assert await task == (1, ["first"], {"rule": ["original"]})
    finally:
        release.set()
        await probe.close()


async def test_a_returned_answer_is_detached_from_the_detectors_objects(slots):
    answer = {"places": [[1, 2]]}
    probe = probing.PictureProbe()
    measured = await probe.examine(lambda: answer)
    answer["places"][0][0] = 99
    assert measured == {"places": [[1, 2]]}
    await probe.close()


async def test_closing_the_owner_cannot_publish_a_late_answer(slots):
    started, release = threading.Event(), threading.Event()

    def detector():
        started.set()
        assert release.wait(2.)
        return {"board": True}

    probe = probing.PictureProbe(wait_s=1.)
    task = asyncio.create_task(probe.examine(detector))
    try:
        await entered(started)
        operation = probe._operation
        await probe.close()
        assert probe.status == "closed" and not slots.acquire(blocking=False)
        release.set()
        assert await task is probing.UNMEASURED
        await completed(operation.raw)
        assert probe.status == "closed"
        assert await probe.examine(lambda: None) is probing.UNMEASURED
    finally:
        release.set()
        await probe.close()


async def test_measured_none_and_unavailable_work_are_different_answers(slots):
    probe = probing.PictureProbe()
    assert await probe.examine(lambda: None) is None
    await probe.close()
    assert await probe.examine(lambda: None) is probing.UNMEASURED


async def test_a_second_caller_cannot_steal_the_first_callers_finished_answer(monkeypatch, slots):
    raw = Future()
    assert raw.set_running_or_notify_cancel()
    submitted = []

    def submit(*args, **kwargs):
        submitted.append((args, kwargs))
        return raw

    monkeypatch.setattr(probing, "submit_blocking_io", submit)
    probe = probing.PictureProbe()
    first = asyncio.create_task(probe.examine(lambda: "first answer"))
    await asyncio.sleep(0)
    raw.set_result("first answer")
    assert await probe.examine(lambda: "second answer") is probing.UNMEASURED
    assert await first == "first answer" and len(submitted) == 1
    await probe.close()


async def test_a_worker_error_is_visible_and_returns_actual_capacity(slots):
    def detector():
        raise ValueError("invented visual failure")

    probe = probing.PictureProbe()
    with pytest.raises(ValueError, match="invented visual failure"):
        await probe.examine(detector)
    assert probe.status == "closed" and slots.acquire(blocking=False)
    slots.release()


async def test_a_late_worker_error_is_received_after_the_owner_leaves(monkeypatch, slots):
    raw = Future()
    assert raw.set_running_or_notify_cancel()
    monkeypatch.setattr(probing, "submit_blocking_io", lambda *_args, **_kwargs: raw)
    loop, errors = asyncio.get_running_loop(), []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    try:
        probe = probing.PictureProbe(wait_s=.001)
        assert await probe.examine(lambda: "unused") is probing.UNMEASURED
        await probe.close()
        raw.set_exception(ValueError("the late visual check failed"))
        for _ in range(4):
            await asyncio.sleep(0)
        assert not errors and probe.status == "closed"
        assert slots.acquire(blocking=False)
        slots.release()
    finally:
        loop.set_exception_handler(previous_handler)


async def test_closing_a_queued_call_cancels_work_and_returns_unmeasured(monkeypatch, slots):
    raw = Future()
    monkeypatch.setattr(probing, "submit_blocking_io", lambda *_args, **_kwargs: raw)
    probe = probing.PictureProbe()
    task = asyncio.create_task(probe.examine(lambda: "queued answer"))
    await asyncio.sleep(0)
    await probe.close()
    assert await task is probing.UNMEASURED and raw.cancelled()
    assert probe.status == "closed" and slots.acquire(blocking=False)
    slots.release()


@pytest.mark.parametrize("wait_s", [False, True, 0., -1., float("nan"), float("inf"), None, "fast"])
async def test_the_caller_wait_is_finite_and_positive(wait_s):
    with pytest.raises(ValueError, match="finite and positive"):
        probing.PictureProbe(wait_s=wait_s)
