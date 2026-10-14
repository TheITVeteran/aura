"""Source predictions can serve action without being labeled empirical responses."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency import playing_as_it_happens as play
from core.agency.a_source_motion_model_checked_in_view import (
    _source_motion_agreement_invariant,
    source_motion_checked,
)
from core.agency.giving_a_trial_time import ControlTrial
from core.agency.what_meeting_things_does import SHOOT, WhatMeetingDoes
from core.agency.which_one_answers_to_her import Makes, WhichIsHers
from core.cognition.a_guide_to_a_place import THE_GUIDE, Guide
from core.perception.a_world_as_its_code_makes_it import AWorldAsItsCodeMakesIt, HerBody
from core.perception.what_moves_in_the_picture import Kind, Thing

pytestmark = pytest.mark.unit


@pytest.fixture
def world(monkeypatch):
    body = Thing(1, 50., 100., 12., 12., np.zeros(64), (80, 200, 20),
                 np.zeros((1, 1, 3), dtype=np.uint8), 1., 10., kind=0)
    target = Thing(2, 50., 20., 12., 12., np.zeros(64), (200, 80, 20),
                   np.zeros((1, 1, 3), dtype=np.uint8), 1., 10., kind=1)
    moves = SimpleNamespace(seeing=True, shape=(150, 150), things={1: body, 2: target},
                            kinds=[Kind(0, np.zeros(64), 144., body.colour),
                                   Kind(1, np.zeros(64), 144., target.colour)])
    hers = WhichIsHers()
    hers.number, hers.kind = 1, 0
    monkeypatch.setattr(hers, 'way_of', lambda key: (-80., 0.) if key == 'a' else None)
    guide = Guide(world_its_code_makes=AWorldAsItsCodeMakesIt(stage=(150., 150.),
                  hers=HerBody(name='body', paces={'a': (-80., 0.), 'd': (80., 0.)}, drawn_in='#50c814')))
    run = play._Run(keys=['a', 'd'], began=1., trying=2, held='', held_since=8., held_trying=True,
                    body_trial=ControlTrial('', 8., 10., 0))
    meeting = WhatMeetingDoes()
    meeting.told[1] = SHOOT
    token = THE_GUIDE.set(guide)
    try:
        yield body, moves, hers, guide, run, meeting
    finally:
        THE_GUIDE.reset(token)


def test_only_agreeing_current_response_qualifies_a_complete_specification(world):
    _body, moves, hers, _guide, run, meeting = world
    choosing = play._Choosing(moves, hers, meeting, run.keys)
    assert source_motion_checked(run, hers, choosing, 10.)
    assert not hers.keys_known(run.keys) and hers.way_of('d') is None
    assert choosing.ways['d'] == (80., 0.)


@pytest.mark.parametrize('fault', ['unseen', 'unknown', 'opposite', 'faster', 'wrong-kind', 'carried', 'incomplete'])
def test_unqualified_state_retains_discovery(world, monkeypatch, fault):
    body, moves, hers, guide, run, meeting = world
    if fault == 'unseen':
        body.seen = 9.9
    elif fault == 'unknown':
        monkeypatch.setattr(hers, 'way_of', lambda _key: None)
    elif fault == 'opposite':
        monkeypatch.setattr(hers, 'way_of', lambda _key: (80., 0.))
    elif fault == 'faster':
        monkeypatch.setattr(hers, 'way_of', lambda _key: (-160., 0.))
    elif fault == 'wrong-kind':
        moves.kinds[0].colour = (200, 80, 20)
    elif fault == 'carried':
        hers.carried = True
    else:
        guide.world_its_code_makes.hers.paces.pop('d')
    assert not source_motion_checked(run, hers, play._Choosing(moves, hers, meeting, run.keys), 10.)


def test_source_agreement_cannot_skip_an_active_press_or_its_neutral_observation(world):
    _body, moves, hers, _guide, run, meeting = world
    choosing = play._Choosing(moves, hers, meeting, run.keys)
    run.body_trial = ControlTrial('', 8., 10.2, 0)
    assert not source_motion_checked(run, hers, choosing, 10.)
    run.held, run.body_trial = 'a', ControlTrial('a', 8., 9., 0)
    assert not source_motion_checked(run, hers, choosing, 10.)


@pytest.mark.asyncio
async def test_actual_actor_uses_a_checked_prediction_and_reaches_the_trigger(world, monkeypatch):
    _body, moves, hers, _guide, run, meeting = world
    hers.makes['j'] = Makes('j', 2, 0., -100., 0)
    monkeypatch.setattr(play, 'time', SimpleNamespace(monotonic=lambda: 10.))
    actions = []

    class Hands:
        async def tap(self, key):
            actions.append(('tap', key))

        async def down(self, key):
            actions.append(('down', key))

        async def up(self, key):
            actions.append(('up', key))

    await play._act(Hands(), run, moves, hers, meeting, play._Choosing(moves, hers, meeting, run.keys), 10.)
    assert ('tap', 'j') in actions and run.trying == 2
    assert hers.way_of('d') is None and not hers.keys_known(run.keys)


def test_the_source_agreement_claim_is_registered():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _source_motion_agreement_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'source_motion_needs_current_observed_agreement')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
