"""Prepared motion reaches the actor with owned keys and a physical completion window."""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.agency import acting_on_a_prepared_motion as prepared
from core.agency.following_a_prepared_path import FollowingPreparedPath
from core.agency.inputs_while_measuring_control import (
    HandsWithControlReceipts,
    hold_control,
    release_after_play,
)
from core.agency.playing_as_it_happens import _act, _Run
from core.agency.solving_for_a_required_state import RequiredStatePlan
from core.agency.what_a_press_does import WhatAPressDoes
from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
KEYS = frozenset(('opaque-a', 'opaque-b'))


class Hands:
    def __init__(self):
        self.events, self.held = [], set()
        self.down_delay, self.failed_down, self.failed_up = {}, set(), set()
        self.entered, self.pause, self.on_up = None, None, None

    async def down(self, key):
        self.held.add(key)
        self.events.append(('down', key, time.monotonic()))
        if key in self.down_delay:
            await asyncio.sleep(self.down_delay[key])
        if self.entered is not None and key == 'opaque-b':
            self.entered.set()
            await self.pause.wait()
        if key in self.failed_down:
            raise OSError('down acknowledgement unavailable')

    async def up(self, key):
        self.events.append(('up', key, time.monotonic()))
        if self.on_up is not None:
            self.on_up()
        if key in self.failed_up:
            raise OSError('up acknowledgement unavailable')
        self.held.discard(key)


def fixture(*, duration=.03, neutral=False):
    at = time.monotonic()
    body = SimpleNamespace(number=1, kind=0, born=0., seen=at, x=0., y=0., w=4., h=4.,
                           moved=False, vx=0., vy=0.)
    moves = SimpleNamespace(things={1:body}, seeing=True, shape=(80, 120), scale=1.)
    hers = WhichIsHers()
    hers.number, hers.kind = 1, 0
    run = _Run(keys=sorted(KEYS), began=at, last_moving=at)
    run.things, run.presses = moves.things, WhatAPressDoes()
    owner, basis, goal = (1, 0., hers.identification.epoch), 'working pixels', 'destination'
    identity = 'measured response', owner, basis, goal
    plan = RequiredStatePlan((0, 1), (0., 1., 2.), True, 2, True)
    follower = FollowingPreparedPath(plan, identity)

    def observe(world, owned, captured_at):
        return prepared.ObservedMotion(owned.thing(world).x, captured_at, owner, basis, goal)

    actions = {0:prepared.TimedKeys(frozenset() if neutral else KEYS, duration),
               1:prepared.TimedKeys(frozenset(('opaque-b',)), .02)}
    run.prepared_motion_binding = prepared.PreparedMotionBinding(follower, 'measured response', owner, basis, goal,
            observe, lambda state, _action: state + 1., lambda state: state == 2., actions)
    raw = Hands()
    return run, moves, hers, raw, HandsWithControlReceipts(raw, run, hers), at


async def complete(run):
    task = run._prepared_input_set.task
    await asyncio.wait_for(asyncio.shield(task), 2.)


async def test_the_actual_actor_dispatches_the_supplied_path_as_a_set_and_finishes_without_more_frames():
    run, moves, hers, raw, hands, at = fixture()
    follower = run.prepared_motion_binding.follower
    await _act(hands, run, moves, hers, None, SimpleNamespace(), at)
    assert raw.held == KEYS and run.held == '' and not run.held_trying
    assert follower.cursor == 0 and follower.status == 'input_started'
    await complete(run)
    assert not raw.held and follower.cursor == 1 and run._prepared_input_set is None
    assert [event[:2] for event in raw.events][-8:] == [('down', 'opaque-a'), ('down', 'opaque-b'),
                                                 ('up', 'opaque-a'), ('up', 'opaque-b')]
    assert run._prepared_input_receipts[-1][4] is True


async def test_duration_begins_after_the_final_down_acknowledgement():
    run, moves, hers, raw, hands, at = fixture(duration=.04)
    raw.down_delay['opaque-b'] = .04
    assert await prepared.act_on_prepared_motion(hands, run, moves, hers, at)
    active = run._prepared_input_set
    assert active.began >= raw.events[1][2] + .04
    await complete(run)
    first_up = next(event[2] for event in raw.events if event[0] == 'up')
    assert first_up >= active.began + .04 and active.until == active.began + .04


async def test_only_a_newer_post_completion_observation_can_offer_the_next_action():
    run, moves, hers, raw, hands, at = fixture()
    assert await prepared.act_on_prepared_motion(hands, run, moves, hers, at)
    await complete(run)
    before = list(raw.events)
    assert await prepared.act_on_prepared_motion(hands, run, moves, hers, at)
    assert raw.events == before
    moves.things[1].x, moves.things[1].seen = 1., time.monotonic()
    assert await prepared.act_on_prepared_motion(hands, run, moves, hers, moves.things[1].seen)
    assert raw.held == {'opaque-b'}
    await complete(run)
    assert run.prepared_motion_binding.follower.cursor == 2


async def test_neutral_model_actions_have_a_full_physical_window_too():
    run, moves, hers, raw, hands, at = fixture(neutral=True)
    assert await prepared.act_on_prepared_motion(hands, run, moves, hers, at)
    active = run._prepared_input_set
    assert not raw.events and active.until > active.began
    await complete(run)
    assert not raw.events and run.prepared_motion_binding.follower.cursor == 1


async def test_partial_down_acknowledgement_releases_every_attempted_key_without_advancing():
    run, moves, hers, raw, hands, at = fixture()
    raw.failed_down.add('opaque-b')
    with pytest.raises(OSError, match='down acknowledgement'):
        await prepared.act_on_prepared_motion(hands, run, moves, hers, at)
    assert not raw.held and run._prepared_input_set is None
    follower = run.prepared_motion_binding.follower
    assert follower.cursor == 0 and follower.plan is None
    assert {event[1] for event in raw.events if event[0] == 'up'} == KEYS


async def test_cancellation_during_a_down_acknowledgement_still_cleans_up_attempted_keys():
    run, moves, hers, raw, hands, at = fixture()
    raw.entered, raw.pause = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(prepared.act_on_prepared_motion(hands, run, moves, hers, at))
    await asyncio.wait_for(raw.entered.wait(), 1.)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not raw.held and run._prepared_input_set is None
    assert run.prepared_motion_binding.follower.cursor == 0


async def test_a_foreign_same_key_input_releases_the_old_set_before_its_new_press():
    run, moves, hers, raw, hands, at = fixture(duration=.15)
    assert await prepared.act_on_prepared_motion(hands, run, moves, hers, at)
    await hands.down('opaque-a')
    assert raw.held == {'opaque-a'} and run._prepared_input_set is None
    before = list(raw.events)
    await asyncio.sleep(.18)
    assert raw.events == before and raw.held == {'opaque-a'}
    assert run.prepared_motion_binding.follower.cursor == 0
    await hands.up('opaque-a')


async def test_a_stale_set_cannot_release_a_new_set_with_the_same_keys():
    run, moves, hers, raw, hands, at = fixture(duration=.2)
    assert await prepared.act_on_prepared_motion(hands, run, moves, hers, at)
    old = run._prepared_input_set
    assert await prepared.abort_prepared_motion(hands, run, hers)
    fresh_run, _fresh_moves, _fresh_hers, _raw, _hands, _at = fixture(duration=.2)
    run.prepared_motion_binding = fresh_run.prepared_motion_binding
    moves.things[1].seen = time.monotonic()
    assert await prepared.act_on_prepared_motion(hands, run, moves, hers, moves.things[1].seen)
    fresh, before = run._prepared_input_set, list(raw.events)
    assert await prepared._release(hands, run, hers, old, completed=True)
    assert run._prepared_input_set is fresh and raw.held == KEYS and raw.events == before
    assert await prepared.abort_prepared_motion(hands, run, hers)


@pytest.mark.parametrize('change', ['body', 'basis', 'goal', 'model'])
async def test_changed_context_retires_and_releases_the_exact_input_set(change):
    run, moves, hers, raw, hands, at = fixture(duration=.2)
    binding = run.prepared_motion_binding
    assert await prepared.act_on_prepared_motion(hands, run, moves, hers, at)
    if change == 'body':
        hers.identification.epoch += 1
    elif change == 'basis':
        moves.shape = (160, 240)
    elif change == 'goal':
        run.prepared_motion_binding = replace(binding, observe=lambda world, owned, captured:
            prepared.ObservedMotion(owned.thing(world).x, captured, binding.owner, binding.basis_id, 'moved destination'))
    else:
        fresh, *_rest = fixture(duration=.2)
        run.prepared_motion_binding = fresh.prepared_motion_binding
    moves.things[1].seen = time.monotonic()
    assert await prepared.act_on_prepared_motion(hands, run, moves, hers, moves.things[1].seen)
    assert not raw.held and run._prepared_input_set is None
    assert binding.follower.cursor == 0 and binding.follower.plan is None


async def test_an_identity_change_during_up_acknowledgements_cannot_finish_the_old_plan():
    run, moves, hers, raw, hands, at = fixture()
    raw.on_up = lambda: setattr(hers.identification, 'epoch', hers.identification.epoch + 1)
    assert await prepared.act_on_prepared_motion(hands, run, moves, hers, at)
    await complete(run)
    assert not raw.held and run.prepared_motion_binding.follower.cursor == 0
    assert run._prepared_input_receipts[-1][4] is False


async def test_unconfirmed_up_prevents_another_input_until_cleanup_is_acknowledged():
    run, moves, hers, raw, hands, at = fixture()
    raw.failed_up.add('opaque-a')
    assert await prepared.act_on_prepared_motion(hands, run, moves, hers, at)
    await complete(run)
    assert raw.held == {'opaque-a'} and run._prepared_input_set.status == 'release unconfirmed'
    with pytest.raises(RuntimeError, match='release remains unconfirmed'):
        await hold_control(hands, run, hers, 'opaque-b', time.monotonic())
    assert not any(event[:2] == ('down', 'opaque-b') for event in raw.events[2:])
    raw.failed_up.clear()
    await release_after_play(hands, run, hers)
    assert not raw.held and run._prepared_input_set is None


async def test_chords_close_single_input_experiments_without_granting_new_independent_evidence():
    run, moves, hers, raw, hands, at = fixture()
    await hold_control(hands, run, hers, 'opaque-a', at, trying=True)
    began = run.held_since
    for when in (began + .001, began + .002):
        hers._hers.add('opaque-a', 10., 0., press=began, when=when)
    assert await prepared.act_on_prepared_motion(hands, run, moves, hers, at)
    assert ('opaque-a', began) in hers._hers.ended
    assert math.isinf(hers._beside_until) and not run.held_trying and not run.presses.watching
    await complete(run)
    assert math.isfinite(hers._beside_until)


async def test_absent_or_unqualified_bindings_leave_ordinary_policy_available():
    run, moves, hers, raw, hands, at = fixture()
    binding = run.prepared_motion_binding
    run.prepared_motion_binding = None
    assert not await prepared.act_on_prepared_motion(hands, run, moves, hers, at)
    run.prepared_motion_binding = binding
    hers.number = None
    assert not await prepared.act_on_prepared_motion(hands, run, moves, hers, at)
    assert not raw.events and binding.follower.cursor == 0


async def test_the_full_play_loop_keeps_a_provider_for_its_current_observed_owner(monkeypatch):
    import numpy as np

    from core.agency import playing_as_it_happens as playing
    from core.perception.what_moves_in_the_picture import Kind, Thing, WhatMoves

    run, _moves, owned, raw, _hands, _at = fixture(duration=.03)
    supplied, next_state, ended = [], [0.], asyncio.Event()

    class DirectEyes(WhatMoves):
        def see(self, picture, at):
            super().see(picture, at)
            pixels = np.full((4, 4, 3), 180, np.uint8)
            self.kinds = [Kind(0, pixels, 16., (180, 180, 180))]
            self.things = {1:Thing(number=1, kind=0, born=0., seen=float(at),
                x=next_state[0], y=0., vx=0., vy=0., w=4., h=4.,
                look=pixels, patch=pixels, colour=(180, 180, 180), motion_evidence='direct')}
            self.pending = {}
            return []

    def provide(moves, hers, at):
        # This fixture's public world names its controlled entity. Production
        # providers still own visual identity and state estimation separately.
        if 1 not in moves.things:
            return None
        hers.number, hers.kind = 1, 0
        if not supplied:
            owner = 1, 0., hers.identification.epoch
            identity = 'direct fixture', owner, 'working pixels', 'destination'
            follower = FollowingPreparedPath(RequiredStatePlan((0, 1), (0., 1., 2.), True, 2, True), identity)
            supplied.append(prepared.PreparedMotionBinding(follower, *identity,
                observe=lambda world, mine, captured: prepared.ObservedMotion(
                    mine.thing(world).x, captured, owner, identity[2], identity[3]),
                advance=lambda state, _action: state + 1., outcome=lambda state: state == 2.,
                actions={0:prepared.TimedKeys(KEYS, .03), 1:prepared.TimedKeys(KEYS, .03)}))
        return supplied[0]

    async def look():
        await asyncio.sleep(.01)
        completed = sum(event[0] == 'up' and event[1] == 'opaque-b' for event in raw.events)
        next_state[0] = float(completed)
        if supplied and supplied[0].follower.status == 'observed_goal':
            ended.set()
            return None
        return np.zeros((80, 120, 3), np.uint8), time.monotonic()

    monkeypatch.setattr(playing, 'eyes_for_play', lambda **kwargs: DirectEyes(**kwargs))
    keep = {'hers':owned, 'observed_model_binding':prepared.PreparedMotionProvider(provide)}
    await asyncio.wait_for(playing.play_as_it_happens(look, raw, keys=sorted(KEYS), keep=keep, seconds=1.), 3.)
    assert ended.is_set() and supplied[0].follower.status == 'observed_goal'
    assert next_state[0] == 2. and not raw.held
    assert [event[:2] for event in raw.events][-8:] == [('down', 'opaque-a'), ('down', 'opaque-b'),
        ('up', 'opaque-a'), ('up', 'opaque-b')] * 2


async def test_the_input_set_has_a_local_registered_invariant():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert prepared._prepared_input_set_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'prepared_motion_releases_only_its_owned_input_set')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
