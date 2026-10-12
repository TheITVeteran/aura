"""Independent control receipts stop when an additional input is attempted."""
import asyncio
from collections import Counter, deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from core.agency.inputs_while_measuring_control import (
    HandsWithControlReceipts,
    _single_input_interval_invariant,
    body_probe_is_active,
    hold_control,
)
from core.agency.what_a_press_does import WhatAPressDoes
from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = pytest.mark.unit


def _setup(monkeypatch):
    clock = SimpleNamespace(at=10.)
    monkeypatch.setattr('core.agency.inputs_while_measuring_control.time.monotonic', lambda: clock.at)
    hers = WhichIsHers()
    body = SimpleNamespace(number=7, kind=3, born=1., seen=10., x=50., y=100., vx=0., vy=0.,
                           w=12., h=12., path=[])
    run = SimpleNamespace(held='', held_since=None, held_trying=False, presses=WhatAPressDoes(),
                           things={7: body}, input_key_downs=Counter(), responses=deque())
    original = SimpleNamespace(**{name: AsyncMock() for name in ('tap', 'point', 'click', 'up', 'down', 'press')})
    return clock, hers, body, run, original, HandsWithControlReceipts(original, run, hers)


@pytest.mark.asyncio
@pytest.mark.parametrize('name,args', [('tap', ('b',)), ('press', ('b',)), ('down', ('b',)),
                                     ('up', ('b',)), ('click', (.3, .4)), ('point', (.3, .4))])
async def test_other_inputs_end_the_interval_without_revoking_its_measured_prefix(monkeypatch, name, args):
    clock, hers, body, run, original, hands = _setup(monkeypatch)
    await hold_control(hands, run, hers, 'a', clock.at, trying=True)
    for index in range(1, 5):
        clock.at = body.seen = 10. + index * .1
        body.y -= 5.
        run.presses.saw(run.things, clock.at)
        hers._by_thing[7].add('a', 0., -50., press=10., when=clock.at)
    clock.at = 10.41
    await getattr(hands, name)(*args)
    receipt = run.presses.entity_presses[0]
    assert receipt.began == 10. and receipt.ended == pytest.approx(10.4)
    assert not run.presses.watching
    before = list(hers._by_thing[7].by_press[('a', 10.)])
    clock.at = body.seen = 10.5
    body.y -= 50.
    run.presses.saw(run.things, clock.at)
    hers._by_thing[7].add('a', 0., -500., press=10., when=clock.at)
    assert hers._by_thing[7].by_press[('a', 10.)] == before
    assert list(run.presses.entity_presses) == [receipt]
    getattr(original, name).assert_awaited_with(*args)


@pytest.mark.asyncio
async def test_a_real_input_change_can_start_the_next_independent_trial(monkeypatch):
    clock, hers, body, run, _original, hands = _setup(monkeypatch)
    await hold_control(hands, run, hers, 'a', clock.at, trying=True)
    clock.at = body.seen = 10.3
    await hands.tap('b')
    clock.at = body.seen = 10.5
    await hold_control(hands, run, hers, '', clock.at, trying=True)
    assert run.held_trying and run.held_since == 10.5
    assert run.presses.watching and all(w.independent and w.began == 10.5 for w in run.presses.watching)


@pytest.mark.asyncio
async def test_a_secondary_input_in_another_task_cannot_borrow_primary_delivery_scope(monkeypatch):
    from core.agency.inputs_while_measuring_control import primary_delivery

    clock, hers, _body, run, _original, hands = _setup(monkeypatch)
    await hold_control(hands, run, hers, 'a', clock.at, trying=True)
    entered, resume = asyncio.Event(), asyncio.Event()

    async def primary():
        with primary_delivery(hands):
            entered.set()
            await resume.wait()

    task = asyncio.create_task(primary())
    await entered.wait()
    try:
        await hands.tap('b')
        assert ('a', 10.) in hers._hers.ended and not run.presses.watching
    finally:
        resume.set()
        await task


@pytest.mark.asyncio
async def test_a_child_task_cannot_inherit_permission_to_extend_a_primary_trial(monkeypatch):
    from core.agency.inputs_while_measuring_control import primary_delivery

    clock, hers, _body, run, _original, hands = _setup(monkeypatch)
    await hold_control(hands, run, hers, 'a', clock.at, trying=True)
    with primary_delivery(hands):
        await asyncio.create_task(hands.tap('b'))
    assert ('a', 10.) in hers._hers.ended and not run.presses.watching


def test_discovery_reserves_a_finite_pass_and_leaves_normal_actions_available_afterward():
    hers = SimpleNamespace(number=None, follows_pointer=False, keys_known=lambda _keys: False)
    run = SimpleNamespace(keys=['a', 'b'], trying=0, pointer_first=False)
    assert body_probe_is_active(run, hers, pointer_steps=6)
    run.trying = 7
    assert body_probe_is_active(run, hers, pointer_steps=6)
    run.trying = 8
    assert not body_probe_is_active(run, hers, pointer_steps=6)
    run.pointer_first, run.pointed = True, 5
    assert body_probe_is_active(run, hers, pointer_steps=6)
    run.pointed = 6
    assert not body_probe_is_active(run, hers, pointer_steps=6)


def test_registers_the_single_input_invariant():
    assert _single_input_interval_invariant() == ()


@pytest.mark.asyncio
async def test_a_secondary_trigger_waits_for_the_finite_probe_pass(monkeypatch):
    from core.agency import playing_as_it_happens as play

    clock, hers, _body, run, original, hands = _setup(monkeypatch)
    monkeypatch.setattr(play, '_no_body_to_find', lambda: False)
    run.keys, run.trying, run.pointer_first = ['a', 'b'], 0, False
    run.last_trigger_began, run.taps = -1., 0
    rules = SimpleNamespace(dodge_keys=[], fire_keys=['b'])
    moves = SimpleNamespace(things=run.things, shape=(200, 240))
    await play._fire_while_finding_herself(hands, run, hers, moves, rules, clock.at)
    original.tap.assert_not_awaited()
    assert run.taps == 0
    run.trying = 8
    await play._fire_while_finding_herself(hands, run, hers, moves, rules, clock.at)
    original.tap.assert_awaited_once_with('b')
    assert run.taps == 1
