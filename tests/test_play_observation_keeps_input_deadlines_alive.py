"""The observation worker cannot hold an input or publish after its owner leaves."""
from __future__ import annotations

import asyncio
import gc
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.a_trial_has_a_physical_end import lease_control_trial
from core.agency.giving_a_trial_time import ControlTrial
from core.agency.inputs_while_measuring_control import hold_control
from core.agency.playing_as_it_happens import _Run
from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception import play_observation_without_holding_the_loop as observing
from core.perception.what_moves_in_the_picture import WhatMoves
from core.runtime.lockdep import checked_thread_semaphore

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
PICTURE = np.zeros((12, 16, 3), np.uint8)


class Eyes:
    __slots__ = ("_delay", "_entered", "_finished", "__dict__")

    def __init__(self, kinds=None, *, delay=0., entered=None, finished=None):
        self.things = {1: SimpleNamespace(number=1, kind=0, born=0., seen=0., x=0., y=0.,
                                         w=4., h=4., size=16., vx=0., vy=0., moved=True)}
        self.pending = {}
        self.kinds = list(kinds or [])
        self.shape = (12, 16)
        self.calls = []
        self._delay, self._entered, self._finished = delay, entered, finished

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, seconds):
        self._delay = seconds

    def see(self, picture, at):
        self.calls.append((at, int(picture[0, 0, 0]), threading.get_ident()))
        if self._entered is not None:
            self._entered.set()
        if self.delay:
            time.sleep(self.delay)
        body = self.things[1]
        body.x += 1.
        body.seen = at
        if self._finished is not None:
            self._finished.set()
        return [{"what": "measured", "at": at}]


async def finished(raw):
    await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(raw)), 2.)


@pytest.fixture
def slots(monkeypatch):
    capacity = checked_thread_semaphore("test.play_observation_operations", 1)
    monkeypatch.setattr(observing, "_SLOTS", capacity)
    return capacity


async def test_a_real_input_lease_expires_while_a_picture_is_still_being_measured(slots):
    entered, done = threading.Event(), threading.Event()
    eyes = Eyes(delay=.08, entered=entered, finished=done)
    observation = await observing.open_play_observation(factory=lambda **_kwargs: eyes)
    run, hers, actions = _Run(keys=["opaque"], began=time.monotonic()), WhichIsHers(), []

    class Hands:
        async def down(self, key):
            actions.append(("down", key, time.monotonic()))

        async def up(self, key):
            actions.append(("up", key, time.monotonic()))

    hands = Hands()
    await hold_control(hands, run, hers, "opaque", time.monotonic(), trying=True)
    run.things = observation.things
    run.body_trial = ControlTrial("opaque", run.held_since, run.held_since + .01, 0)
    assert lease_control_trial(hands, run, hers, run.body_trial, field="body_trial")
    task = asyncio.create_task(observation.see(PICTURE, time.monotonic()))
    await asyncio.wait_for(run._trial_input_deadline, 1.)
    assert entered.is_set() and not done.is_set() and not run.held
    assert actions[-1][:2] == ("up", "opaque")
    assert observation.things[1].x == 0.  # receipt readers retain the preceding detached body
    assert await task and done.is_set()
    assert observation.things[1].x == 1.
    await observation.close()


async def test_a_timed_out_operation_retains_its_slot_and_never_queues_another_picture(slots):
    eyes = Eyes(delay=.08)
    observation = await observing.open_play_observation(factory=lambda **_kwargs: eyes, frame_wait_s=.005)
    assert await observation.see(PICTURE, 1.) is None
    operation = observation._operation
    assert operation is not None and not operation.raw.done()
    assert not slots.acquire(blocking=False)
    for at in (2., 3., 4.):
        assert await observation.see(PICTURE, at) is None
    assert len(eyes.calls) == 1 and observation.things[1].x == 0.
    await finished(operation.raw)
    eyes.delay = 0.
    assert await observation.see(PICTURE, 5.) == [{"what": "measured", "at": 5.}]
    assert [call[0] for call in eyes.calls] == [1., 5.]
    assert observation.things[1].x == 2.  # private state continued without publishing the late frame
    await observation.close()


async def test_closing_a_running_operation_does_not_publish_or_return_its_permit_early(slots):
    eyes = Eyes(delay=.08)
    observation = await observing.open_play_observation(factory=lambda **_kwargs: eyes, frame_wait_s=.005)
    assert await observation.see(PICTURE, 1.) is None
    operation = observation._operation
    await observation.close()
    assert not slots.acquire(blocking=False)
    assert await observation.see(PICTURE, 2.) is None
    await finished(operation.raw)
    assert observation.things[1].x == 0. and observation.status == "closed"
    assert slots.acquire(blocking=False)
    slots.release()


async def test_cancelling_the_waiter_closes_its_owner_but_not_the_running_worker(slots):
    entered = threading.Event()
    eyes = Eyes(delay=.08, entered=entered)
    observation = await observing.open_play_observation(factory=lambda **_kwargs: eyes)
    task = asyncio.create_task(observation.see(PICTURE, 1.))
    for _ in range(100):
        if entered.is_set():
            break
        await asyncio.sleep(.001)
    assert entered.is_set()
    operation = observation._operation
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert observation.status == "closed" and not slots.acquire(blocking=False)
    await finished(operation.raw)
    assert observation.things[1].x == 0.


async def test_a_cancelled_queued_operation_does_not_construct_the_eyes(monkeypatch, slots):
    raw, called = Future(), []
    monkeypatch.setattr(observing, "submit_blocking_io", lambda *_args, **_kwargs: raw)
    observation = await observing.open_play_observation(factory=lambda **_kwargs: called.append(True), startup_wait_s=.005)
    assert not raw.done() and not slots.acquire(blocking=False)
    await observation.close()
    assert raw.cancelled() and not called
    assert slots.acquire(blocking=False)
    slots.release()


async def test_startup_is_off_loop_and_a_late_start_can_be_used_for_a_fresh_picture(slots):
    main_thread = threading.get_ident()
    created = []

    def factory(**kwargs):
        created.append(threading.get_ident())
        time.sleep(.08)
        return Eyes(**kwargs)

    observation = await observing.open_play_observation(factory=factory, startup_wait_s=.005)
    operation = observation._operation
    assert operation is not None and not operation.raw.done()
    assert await observation.see(PICTURE, 1.) is None
    await finished(operation.raw)
    assert await observation.see(PICTURE, 2.)
    assert created == [created[0]] and created[0] != main_thread
    assert observation.things[1].seen == 2.
    await observation.close()


async def test_public_things_keep_their_identity_and_only_change_when_the_loop_commits(slots):
    eyes = Eyes()
    observation = await observing.open_play_observation(factory=lambda **_kwargs: eyes)
    body = observation.things[1]
    assert body is not eyes.things[1]
    assert await observation.see(PICTURE, 1.)
    assert observation.things[1] is body and body.x == 1.
    assert await observation.see(PICTURE, 2.)
    assert observation.things[1] is body and body.x == 2.
    eyes.things[1].x = 100.
    assert body.x == 2.
    await observation.close()


async def test_a_nested_delegate_keeps_its_alias_cache_out_of_the_public_identity_ledger(slots):
    class Delegate(observing.SeenElsewhere):
        def __init__(self, **kwargs):
            object.__setattr__(self, "_here", None)
            holder = Eyes(**kwargs)
            holder._known = dict(holder.things)
            object.__setattr__(self, "_likeness", holder)

        def see(self, picture, at):
            return self._likeness.see(picture, at)

    observation = await observing.open_play_observation(factory=Delegate)
    assert observation.activity is None
    body = observation.things[1]
    assert observation.elsewhere and body is not observation._eyes._likeness.things[1]
    assert await observation.see(PICTURE, 1.)
    assert await observation.see(PICTURE, 2.)
    assert observation.things[1] is body and observation._mirror._known[1] is body
    assert body.x == 2.
    await observation.close()


async def test_concurrent_calls_share_one_inflight_operation(slots):
    eyes = Eyes(delay=.04)
    observation = await observing.open_play_observation(factory=lambda **_kwargs: eyes)
    first = asyncio.create_task(observation.see(PICTURE, 1.))
    await asyncio.sleep(.005)
    assert await observation.see(PICTURE, 2.) is None
    assert await first and len(eyes.calls) == 1
    await observation.close()


async def test_a_busy_global_lane_refuses_a_second_owner_without_queuing_work(slots):
    first = await observing.open_play_observation(factory=lambda **_kwargs: Eyes(delay=.08), frame_wait_s=.005)
    assert await first.see(PICTURE, 1.) is None
    called = []
    second = await observing.open_play_observation(factory=lambda **_kwargs: called.append(True))
    assert second.status == "observation capacity unavailable" and not called
    await second.close()
    operation = first._operation
    await first.close()
    await finished(operation.raw)


async def test_a_queued_frame_owns_its_pixels_and_source_clock(monkeypatch, slots):
    from core.perception.a_clock_for_drawn_frames import MeasuredFrameTime

    eyes = Eyes()
    observation = await observing.open_play_observation(factory=lambda **_kwargs: eyes)
    entered, release = threading.Event(), threading.Event()
    pool = ThreadPoolExecutor(max_workers=1)
    busy = pool.submit(lambda: (entered.set(), release.wait(1.)))
    assert entered.wait(.1)
    monkeypatch.setattr(observing, "submit_blocking_io", lambda fn, *args, **_kwargs: pool.submit(fn, *args))
    image = PICTURE.copy()
    at = MeasuredFrameTime(1., lower=1., upper=1.001, arrived=2., source="browser-monotonic", source_at=10., epoch=3)
    task = asyncio.create_task(observation.see(image, at))
    await asyncio.sleep(.005)
    image[0, 0, 0] = 99
    release.set()
    assert await task
    assert eyes.calls[0][1] == 0
    assert observation.things[1].seen is not at and vars(observation.things[1].seen) == vars(at)
    await observation.close()
    await finished(busy)
    pool.shutdown(wait=False)


async def test_a_cancelled_start_cannot_publish_when_its_factory_finishes(slots):
    entered = threading.Event()

    def factory(**kwargs):
        entered.set()
        time.sleep(.08)
        return Eyes(**kwargs)

    observation = observing.PlayObservation(factory=factory)
    task = asyncio.create_task(observation.start())
    for _ in range(100):
        if entered.is_set():
            break
        await asyncio.sleep(.001)
    operation = observation._operation
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert observation.status == "closed" and not observation.things
    await finished(operation.raw)
    assert not observation.things and observation._eyes is None


async def test_a_worker_error_is_visible_and_the_actual_operation_returns_its_slot(slots):
    def factory(**_kwargs):
        raise ValueError("cannot construct observer")

    with pytest.raises(ValueError, match="cannot construct observer"):
        await observing.open_play_observation(factory=factory)
    assert slots.acquire(blocking=False)
    slots.release()


async def test_a_late_worker_error_is_consumed_after_its_waiter_and_owner_leave(monkeypatch, slots):
    observation = await observing.open_play_observation(factory=lambda **_kwargs: Eyes(), frame_wait_s=.005)
    raw = Future()
    assert raw.set_running_or_notify_cancel()
    monkeypatch.setattr(observing, "submit_blocking_io", lambda *_args, **_kwargs: raw)
    loop, errors = asyncio.get_running_loop(), []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    try:
        assert await observation.see(PICTURE, 1.) is None
        await observation.close()
        raw.set_exception(ValueError("the late picture failed"))
        for _ in range(3):
            await asyncio.sleep(0)
        assert observation.things[1].x == 0.
        del observation
        gc.collect()
        await asyncio.sleep(0)
        assert not errors
        assert slots.acquire(blocking=False)
        slots.release()
    finally:
        loop.set_exception_handler(previous_handler)


async def test_the_actual_play_loop_keeps_a_physical_input_deadline_while_its_real_eyes_work(monkeypatch, slots):
    from core.agency import playing_as_it_happens as playing

    entered, done = threading.Event(), threading.Event()
    observation_threads, actions, created = [], [], []

    class DelayedEyes(WhatMoves):
        def see(self, picture, at):
            observation_threads.append(threading.get_ident())
            if self.pictures:
                entered.set()
                time.sleep(.3)
            happened = super().see(picture, at)
            if entered.is_set():
                done.set()
            return happened

    def factory(**kwargs):
        created.append(threading.get_ident())
        return DelayedEyes(**kwargs)

    class Hands:
        async def down(self, key):
            actions.append(("down", key, time.monotonic(), done.is_set()))

        async def up(self, key):
            actions.append(("up", key, time.monotonic(), done.is_set(), entered.is_set()))

    pressed = False

    async def trial(hands, run, _moves, hers, *_args, **_kwargs):
        nonlocal pressed
        if pressed:
            return
        pressed = True
        await hold_control(hands, run, hers, "opaque", time.monotonic(), trying=True)
        run.body_trial = ControlTrial("opaque", run.held_since, run.held_since + .2, hers.identification.epoch)
        assert lease_control_trial(hands, run, hers, run.body_trial, field="body_trial")

    pictures = 0

    async def look():
        nonlocal pictures
        pictures += 1
        return (PICTURE, time.monotonic()) if pictures <= 2 else None

    monkeypatch.setattr(playing, "eyes_for_play", factory)
    monkeypatch.setattr(playing, "_act", trial)
    result = await asyncio.wait_for(playing._play_as_it_happens(look, Hands(), keys=["opaque"], seconds=3.), 5.)
    assert result["pictures"] == 2 and entered.is_set() and done.is_set()
    assert [action[:2] for action in actions] == [("down", "opaque"), ("up", "opaque")]
    assert actions[1][3] is False  # physical release preceded completion of the second actual measurement
    assert actions[1][4] is True
    assert created and all(thread != threading.get_ident() for thread in created + observation_threads)


async def test_public_activity_preserves_measured_quiet_and_change_without_worker_pixels(slots):
    from core.agency.playing_as_it_happens import _over

    eyes = WhatMoves()
    observation = await observing.open_play_observation(factory=lambda **_kwargs: eyes)
    run = _Run(keys=[], began=1., last_moving=1.)
    for index in range(14):
        at = 1. + .25 * index
        assert await observation.see(PICTURE, at) is not None
    assert observation.activity.at == at and observation.activity.changed is False
    assert observation.activity.quiet_for(at) == at - 1.
    assert observation.activity._picture is None and eyes.activity._picture is not None
    assert _over(run, observation, [], at) == "nothing on the screen has moved for a while"
    changed = PICTURE.copy()
    changed[2:6, 3:7] = 255
    at += .25
    happened = await observation.see(changed, at)
    assert happened is not None and observation.activity.changed is True
    assert _over(run, observation, happened, at) == "" and run.last_moving == at
    await observation.close()


async def test_detached_state_has_a_local_invariant():
    assert observing._owned_play_observation_invariant() == ()
