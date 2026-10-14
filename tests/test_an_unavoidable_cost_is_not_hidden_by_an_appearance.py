"""Appearance labels cannot hide a known cost that avoiding contact leaves unchanged."""
from types import SimpleNamespace

import numpy as np
import pytest

from core.agency.an_effect_contact_cannot_avoid import (
    _nonlocal_countermeasure_invariant,
    needs_a_countermeasure,
)
from core.agency.choosing_where_to_be import _Choosing
from core.agency.what_meeting_things_does import AVOID, MEET, SHOOT, WhatMeetingDoes
from core.agency.which_one_answers_to_her import Makes, WhichIsHers
from core.agency.who_is_who import PART_OF_THE_PLACE
from core.cognition.a_guide_to_a_place import THE_GUIDE
from core.perception.what_moves_in_the_picture import Kind, Thing

pytestmark = pytest.mark.unit


def decision(monkeypatch, *, countermeasure=True):
    body = Thing(1, 50., 100., 12., 12., np.zeros(64), (80, 200, 20),
                 np.zeros((1, 1, 3), dtype=np.uint8), 1., 10., kind=0)
    target = Thing(2, 50., 20., 12., 12., np.zeros(64), (200, 80, 20),
                   np.zeros((1, 1, 3), dtype=np.uint8), 1., 10., kind=1)
    moves = SimpleNamespace(seeing=True, shape=(150, 150), things={1: body, 2: target},
                            kinds=[Kind(0, np.zeros(64), 144., body.colour),
                                   Kind(1, np.zeros(64), 144., target.colour)])
    hers = WhichIsHers()
    hers.number, hers.kind = 1, 0
    monkeypatch.setattr(hers, 'way_of', lambda key: (-80., 0.) if key == 'a' else (80., 0.))
    if countermeasure:
        hers.makes['j'] = Makes('j', 2, 0., -100., 0)
    meeting = WhatMeetingDoes()
    meeting.told[1] = AVOID
    meeting.lands_anyway.add(1)
    token = THE_GUIDE.set(None)
    try:
        choose = _Choosing(moves, hers, meeting, ['a', 'd'])
    finally:
        THE_GUIDE.reset(token)
    choose.who = SimpleNamespace(of=lambda _thing: SimpleNamespace(name=PART_OF_THE_PLACE, sure=True))
    return choose, target


def test_the_actual_planner_keeps_a_known_cost_eligible_despite_a_scenery_label(monkeypatch):
    choose, target = decision(monkeypatch)
    assert choose.meeting.told[1] == AVOID
    assert choose.stance(target) == SHOOT
    _goal, why, aim = choose.target()
    assert why == 'shoot' and aim is target
    assert choose.fire(aim, why) == 'j'


def test_contact_harm_alone_cannot_override_the_appearance_prior(monkeypatch):
    choose, _target = decision(monkeypatch)
    choose.meeting.lands_anyway.clear()
    assert choose.target()[1] == 'wait'


def test_an_unknown_countermeasure_is_not_invented(monkeypatch):
    choose, target = decision(monkeypatch, countermeasure=False)
    assert choose.stance(target) == AVOID and choose.shot is None


def test_actual_beneficial_meetings_can_contradict_a_source_binding():
    meeting = SimpleNamespace(lands_anyway={1}, known=lambda _kind: True, stance=lambda _kind: MEET)
    assert not needs_a_countermeasure(meeting, 1, object())


def test_the_countermeasure_claim_is_registered():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert _nonlocal_countermeasure_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'nonlocal_cost_keeps_a_known_countermeasure_available')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
