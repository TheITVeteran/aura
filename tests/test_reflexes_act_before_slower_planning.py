"""General consequence responses reach physical hands before a slower task."""
import asyncio
import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.choosing_where_to_be import _Choosing
from core.agency.inputs_while_measuring_control import (
    HandsWithControlReceipts,
    hold_control,
    release_after_play,
)
from core.agency.playing_as_it_happens import _Run
from core.agency.reflexes_from_measured_consequences import (
    ReflexCircuit,
    ReflexObservation,
    ReflexResponse,
    _reflex_consequence_invariant,
)
from core.agency.responding_before_slower_planning import _response_context, respond_before_planning
from core.agency.what_a_press_does import WhatAPressDoes
from core.agency.what_meeting_things_does import AVOID, WhatMeetingDoes
from core.agency.which_one_answers_to_her import WhichIsHers
from core.perception.what_moves_in_the_picture import Kind, Thing

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('action', [('ui', 'release drag'), ('body', 'step aside'), ('process', 'shed work')])
def test_the_same_circuit_responds_to_consequences_without_domain_vocabulary(action):
    circuit = ReflexCircuit()
    response = ReflexResponse(action, 0., .1)
    before = ReflexObservation(('model', action), 1., 8., (response,))
    assert circuit.inspect(before, 1.01) == response
    receipt = circuit.delivered(before, response, 1.01, 1.02)
    assert circuit.inspect(ReflexObservation(before.context, 1.05, 0., (response,)), 1.06) is None
    assert not circuit.outcomes  # an early picture cannot certify the held response
    circuit.ended(receipt, 1.12)
    assert circuit.inspect(ReflexObservation(before.context, 1.2, 0., (response,)), 1.21) is None
    assert circuit.outcomes[-1].observed_cost == 0.


def test_wrong_model_missing_feedback_and_ineffective_responses_do_not_pass():
    circuit = ReflexCircuit()
    response = ReflexResponse('opaque action', 0., .1)
    for start in (1., 2.):
        sample = ReflexObservation('same model', start, 8., (response,))
        assert circuit.inspect(sample, start + .01) == response
        receipt = circuit.delivered(sample, response, start + .01, start + .02)
        circuit.ended(receipt, start + .12)
        circuit.inspect(ReflexObservation(sample.context, start + .15, 8., (response,)), start + .16)
    assert circuit.inspect(ReflexObservation('same model', 3., 8., (response,)), 3.01) is None
    new = ReflexObservation('different checked model', 4., 8., (response,))
    assert circuit.inspect(new, 4.01) is response
    circuit.delivered(new, response, 4.01, 4.02)
    circuit.retire('no next observation')
    assert circuit.outcomes[-1].observed_cost is None


@pytest.mark.parametrize('stamp,now', [(1., 1.3), (2., 1.), (float('nan'), 2.)])
def test_stale_future_and_nonfinite_observations_cannot_act(stamp, now):
    if not np.isfinite(stamp):
        with pytest.raises(ValueError):
            ReflexObservation('context', stamp, 1., ())
    else:
        assert ReflexCircuit().inspect(ReflexObservation('context', stamp, 1.,
                       (ReflexResponse('a', 0., .1),)), now) is None


def test_receipts_cannot_be_minted_for_an_unselected_response():
    circuit = ReflexCircuit()
    sample = ReflexObservation('context', 1., 1., (ReflexResponse('a', 2., .1),))
    assert circuit.inspect(sample, 1.01) is None
    with pytest.raises(ValueError):
        circuit.delivered(sample, sample.responses[0], 1.01, 1.02)


@pytest.mark.parametrize('environment', ['cursor boundary', 'closing obstacle', 'backlog limit'])
def test_a_short_response_prevents_a_failure_before_a_slow_decision_can_arrive(environment):
    # Each plant has its own dynamics and independently checks a failure.
    # The circuit only receives costs from their numeric response models.
    if environment == 'cursor boundary':
        initial, normal_rate, response_rate, limit = .9, 1., 0., 1.
    elif environment == 'closing obstacle':
        initial, normal_rate, response_rate, limit = .1, -1., 1., 0.
    else:
        initial, normal_rate, response_rate, limit = 9., 10., -10., 10.

    def failed(value):
        return value <= limit if environment == 'closing obstacle' else value >= limit

    def cost(value, rate):
        future = value + rate * .15
        return max(0., limit - future) if environment == 'closing obstacle' else max(0., future - limit)

    circuit = ReflexCircuit()
    response = ReflexResponse('checked actuator', cost(initial, response_rate), .2)
    sample = ReflexObservation(environment, 1., cost(initial, normal_rate), (response,))
    assert circuit.inspect(sample, 1.01) is response
    receipt = circuit.delivered(sample, response, 1.01, 1.02)
    ordinary, reactive = initial, initial
    for _step in range(15):
        ordinary += normal_rate * .01  # the slower decision arrives after .4 seconds
        reactive += response_rate * .01
    assert failed(ordinary) and not failed(reactive)
    circuit.ended(receipt, 1.22)
    after = ReflexObservation(environment, 1.23, cost(reactive, response_rate), (response,))
    circuit.inspect(after, 1.24)
    assert circuit.outcomes[-1].observed_cost == 0.


class Hands:
    def __init__(self):
        self.events, self.held = [], set()

    async def down(self, key):
        self.events.append(('down', key, time.monotonic()))
        self.held.add(key)

    async def up(self, key):
        self.events.append(('up', key, time.monotonic()))
        self.held.discard(key)


def world(monkeypatch):
    at = time.monotonic()
    body = Thing(1, 40., 50., 8., 8., np.zeros(64), (80, 200, 20),
                 np.zeros((1, 1, 3), dtype=np.uint8), at - 1., at, kind=0)
    incoming = Thing(2, 60., 50., 8., 8., np.zeros(64), (200, 80, 20),
                     np.zeros((1, 1, 3), dtype=np.uint8), at - 1., at, kind=1)
    incoming.vx, incoming.moved = -100., True
    moves = SimpleNamespace(seeing=True, shape=(100, 100), things={1: body, 2: incoming},
                            kinds=[Kind(0, np.zeros(64), 64., body.colour),
                                   Kind(1, np.zeros(64), 64., incoming.colour)])
    hers = WhichIsHers()
    hers.number, hers.kind = 1, 0
    monkeypatch.setattr(hers, 'way_of', lambda key: (0., -120.) if key == 'a' else (0., 120.) if key == 'b' else None)
    run = _Run(keys=['a', 'b'], began=at, last_moving=at)
    run.things, run.presses = moves.things, WhatAPressDoes()
    meeting = WhatMeetingDoes()
    meeting.told[1] = AVOID
    choosing = _Choosing(moves, hers, meeting, run.keys)
    choosing.at = at
    raw = Hands()
    return run, moves, hers, choosing, raw, HandsWithControlReceipts(raw, run, hers), at


@pytest.mark.asyncio
async def test_real_hands_release_the_response_while_slower_planning_waits(monkeypatch):
    run, _moves, hers, choosing, raw, hands, at = world(monkeypatch)
    try:
        assert await respond_before_planning(hands, run, hers, choosing, at)
        assert raw.events[0][:2] == ('down', 'a')
        await asyncio.sleep(.2)  # a slower planner or word reader remains busy
        assert [e[:2] for e in raw.events] == [('down', 'a'), ('up', 'a')]
        assert raw.events[1][2] - raw.events[0][2] < .18
        assert not raw.held and run._reflex_input is None
        assert run._reflex_circuit.pending.ended_at is not None
    finally:
        await release_after_play(hands, run, hers)


@pytest.mark.asyncio
async def test_a_deadline_cannot_release_a_new_owners_input(monkeypatch):
    run, _moves, hers, choosing, raw, hands, at = world(monkeypatch)
    try:
        assert await respond_before_planning(hands, run, hers, choosing, at)
        await hold_control(hands, run, hers, 'b', time.monotonic())
        await asyncio.sleep(.18)
        assert raw.held == {'b'}
        assert all(e[:2] != ('up', 'b') for e in raw.events)
        assert run._reflex_circuit.outcomes[-1].observed_cost is None
    finally:
        await release_after_play(hands, run, hers)


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ['stale', 'lost', 'trial', 'unmeasured', 'carried'])
async def test_a_reflex_cannot_borrow_missing_control_evidence(monkeypatch, fault):
    run, _moves, hers, choosing, raw, hands, at = world(monkeypatch)
    if fault == 'stale':
        at -= 1.
    elif fault == 'lost':
        hers.number = None
    elif fault == 'trial':
        run.held_trying = True
    elif fault == 'unmeasured':
        monkeypatch.setattr(hers, 'way_of', lambda _key: None)
    else:
        hers.carried = True
    assert not await respond_before_planning(hands, run, hers, choosing, at)
    assert not raw.events


def test_small_measurement_changes_cannot_reset_failed_response_history(monkeypatch):
    run, _moves, hers, choosing, _raw, _hands, _at = world(monkeypatch)
    original = _response_context(run, choosing.mine, hers, choosing, {'a': (0., -120.), '': (0., 0.)})
    for speed in (119., 121., 115.):
        assert _response_context(run, choosing.mine, hers, choosing, {'a': (0., -speed), '': (0., 0.)}) == original
    assert _response_context(run, choosing.mine, hers, choosing, {'a': (0., 120.), '': (0., 0.)}) != original


def test_the_reflex_claim_has_a_registered_mechanical_check():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _reflex_consequence_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'fresh_consequences_can_interrupt_slower_planning')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
