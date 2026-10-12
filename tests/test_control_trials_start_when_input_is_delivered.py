"""A policy change cannot manufacture another physical control trial."""
from collections import Counter, deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.agency import playing_as_it_happens as play
from core.agency.controls_measured_on_one_thing import own_control_witness
from core.agency.which_one_answers_to_her import WhichIsHers

pytestmark = pytest.mark.unit


def test_registered_delivery_invariant():
    from core.agency.what_a_press_does import _unchanged_input_invariant

    assert _unchanged_input_invariant() == ()


class _Clock:
    at = 10.


class _Hands:
    def __init__(self, clock):
        self.clock = clock
        self.delivered = []

    async def down(self, key):
        self.delivered.append(("down", key, self.clock.at))
        self.clock.at += .025

    async def up(self, key):
        self.delivered.append(("up", key, self.clock.at))
        self.clock.at += .015


def _setup(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(play, "time", SimpleNamespace(monotonic=lambda: clock.at))
    hands = _Hands(clock)
    hers = WhichIsHers()
    presses = Mock()
    run = SimpleNamespace(held="", held_since=None, held_trying=False, presses=presses, things={},
                          input_key_downs=Counter(), responses=deque())
    return clock, hands, hers, presses, run


@pytest.mark.asyncio
@pytest.mark.parametrize("independent", [True, False])
async def test_rechoosing_a_held_key_keeps_its_delivered_start_and_trial_status(monkeypatch, independent):
    clock, hands, hers, presses, run = _setup(monkeypatch)
    await play._hold(hands, run, hers, "q", clock.at, trying=independent)
    delivered = run.held_since
    first_receipt = hers._held_at(delivered)
    for trying in (not independent, independent, not independent, independent):
        clock.at += .5
        await play._hold(hands, run, hers, "q", clock.at, trying=trying)

    assert hands.delivered == [("down", "q", 10.)]
    assert run.held_since == delivered
    assert run.held_trying is independent
    assert hers._held_at(clock.at) == first_receipt == (delivered, "q", independent)
    assert len(hers._held) == 1
    assert run.input_key_downs == {"q": 1} and len(run.responses) == 1
    presses.pressed.assert_called_once()
    assert presses.pressed.call_args.kwargs["independent"] is independent
    presses.released.assert_not_called()


@pytest.mark.asyncio
async def test_initial_neutral_policy_change_delivers_no_baseline_receipt(monkeypatch):
    clock, hands, hers, presses, run = _setup(monkeypatch)
    await play._hold(hands, run, hers, "", clock.at, trying=True)
    assert not hands.delivered and not hers._held
    assert run.held_since is None and run.held_trying is False
    presses.pressed.assert_not_called()


@pytest.mark.asyncio
async def test_only_a_real_release_and_repress_supply_a_new_independent_trial(monkeypatch):
    clock, hands, hers, presses, run = _setup(monkeypatch)
    await play._hold(hands, run, hers, "q", clock.at, trying=False)
    planned_start = run.held_since
    clock.at += .5
    await play._hold(hands, run, hers, "q", clock.at, trying=True)
    assert run.held_trying is False and run.held_since == planned_start

    clock.at += .5
    await play._hold(hands, run, hers, "", clock.at, trying=True)
    rest_start = run.held_since
    clock.at += .5
    await play._hold(hands, run, hers, "", clock.at, trying=False)
    assert run.held_since == rest_start and run.held_trying is True
    clock.at += .5
    await play._hold(hands, run, hers, "q", clock.at, trying=True)
    assert run.held_since > rest_start > planned_start
    assert run.held_trying is True
    assert [(kind, key) for kind, key, _at in hands.delivered] == [("down", "q"), ("up", "q"), ("down", "q")]
    assert [call.args[0] for call in presses.pressed.call_args_list] == ["q", "", "q"]
    assert [call.kwargs["independent"] for call in presses.pressed.call_args_list] == [False, True, True]
    assert [key for _at, key, _trying in hers._held] == ["q", "", "q"]


def _observe(hers, clock, body, *, speed):
    moves = SimpleNamespace(things={body.number: body})
    for _ in range(4):
        clock.at += .1
        body.seen = clock.at
        body.vx = speed
        body.x += speed * .1
        hers.saw(moves, [], clock.at)


@pytest.mark.asyncio
async def test_repeated_policy_flips_do_not_qualify_an_entity_without_repeated_inputs(monkeypatch):
    clock, hands, hers, _presses, run = _setup(monkeypatch)
    body = SimpleNamespace(number=7, kind=3, born=1., seen=clock.at, x=100., y=100.,
                           vx=0., vy=0., w=12., h=12., size=12., moved=True, path=[])
    run.things[body.number] = body
    for key, speed in (("q", 60.), ("", 0.)):
        await play._hold(hands, run, hers, key, clock.at, trying=True)
        _observe(hers, clock, body, speed=speed)
        for trying in (False, True, False, True):
            await play._hold(hands, run, hers, key, clock.at, trying=trying)
            _observe(hers, clock, body, speed=speed)

    evidence = hers._by_thing[body.number]
    assert Counter(key for key, _began in evidence.by_press) == {"q": 1, "": 1}
    assert own_control_witness(body.number, evidence, epoch=hers.identification.epoch) is None
    assert hers.number is None
    assert [(kind, key) for kind, key, _at in hands.delivered] == [("down", "q"), ("up", "q")]

    # Genuine repeated inputs still supply the distinct trials the witness needs.
    for key, speed in (("q", 60.), ("", 0.)):
        await play._hold(hands, run, hers, key, clock.at, trying=True)
        _observe(hers, clock, body, speed=speed)
    witness = own_control_witness(body.number, evidence, epoch=hers.identification.epoch)
    assert witness is not None
    assert dict(witness.independent_trials) == {"q": 2, "": 2}
