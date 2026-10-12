"""A pooled appearance class never supplies the new entity's missing experiment."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.agency.causal_identification import CausalIdentification
from core.agency.controls_measured_on_one_thing import (
    _own_control_witness_invariant,
    own_control_witness,
)

pytestmark = pytest.mark.unit


def _evidence(*, left=2, right=2, statistic=80., effect=60.):
    by_press = {}
    for key, count, vx in (("left", left, -30.), ("right", right, 30.)):
        for index in range(count):
            by_press[(key, index + (0. if key == "left" else 10.))] = [(vx, 0.), (vx, 0.)]
    return SimpleNamespace(by_press=by_press, ratio=lambda: (statistic, effect))


def _choose(witness):
    ledger = CausalIdentification(statistic_over=20., effect_over=20.)
    return ledger.choose([] if witness is None else [witness], visible={"body"}, established=None, excluded=set())


def test_prior_kind_response_cannot_fill_a_new_entitys_missing_receipts():
    prior = _evidence(statistic=1000., effect=1000.)
    assert _choose(own_control_witness("body", prior, epoch=0)) == "body"
    for own in (None, _evidence(left=0, right=0), _evidence(left=1, right=1)):
        assert _choose(own_control_witness("body", own, epoch=0)) is None


def test_its_own_statistic_and_effect_cannot_be_replaced_by_a_kinds_stronger_score():
    own = _evidence(statistic=0., effect=0.)
    witness = own_control_witness("body", own, epoch=0)
    assert witness is not None and witness.statistic == 0. and witness.effect == 0.
    assert _choose(witness) is None


@pytest.mark.parametrize("exclusion", ["excluded", "a_gauge", "with_the_world"])
def test_other_measurements_exclude_it_from_every_kind_handoff(exclusion):
    own = _evidence()
    assert own_control_witness("body", own, epoch=0, **{exclusion: True}) is None


def test_a_current_continuing_entity_with_its_own_repeated_response_still_qualifies():
    witness = own_control_witness("body", _evidence(), epoch=0)
    assert witness is not None
    assert dict(witness.independent_trials) == {"left": 2, "right": 2}
    assert _choose(witness) == "body"


def test_a_nonfinite_receipt_cannot_supply_a_missing_repeat():
    own = _evidence(left=1)
    own.by_press[("left", 99.)] = [(float("nan"), 0.), (-30., 0.)]
    assert own_control_witness("body", own, epoch=0) is None


def test_the_entity_evidence_invariant():
    assert _own_control_witness_invariant() == ()


def _current_entity():
    from core.agency.which_one_answers_to_her import WhichIsHers

    hers = WhichIsHers()
    body = SimpleNamespace(number=1, kind=7, moved=True)
    moves = SimpleNamespace(things={1: body})
    hers._by_kind[7] = _evidence(statistic=1000., effect=1000.)
    return hers, body, moves


def test_the_live_kind_fallback_cannot_borrow_missing_entity_trials():
    hers, body, moves = _current_entity()
    hers._by_thing[body.number] = _evidence(left=0, right=0)
    assert hers._the_one_of_a_kind_that_answers(moves) is None


def test_the_live_kind_fallback_keeps_the_entitys_weak_result():
    hers, body, moves = _current_entity()
    hers._by_thing[body.number] = _evidence(statistic=0., effect=0.)
    assert hers._the_one_of_a_kind_that_answers(moves) is None


def test_the_live_kind_fallback_rejects_measured_scenery():
    hers, body, moves = _current_entity()
    for number in range(1, 6):
        hers._by_thing[number] = _evidence()
    assert hers._the_one_of_a_kind_that_answers(moves) is None


def test_the_live_kind_fallback_preserves_a_measured_gauge_exclusion():
    hers, body, moves = _current_entity()
    hers._by_thing[body.number] = _evidence()
    hers._outlines = SimpleNamespace(a_gauge=lambda _number: True)
    assert hers._the_one_of_a_kind_that_answers(moves) is None


def test_the_live_kind_fallback_can_confirm_an_entitys_own_response():
    hers, body, moves = _current_entity()
    hers._by_thing[body.number] = _evidence()
    assert hers._the_one_of_a_kind_that_answers(moves) == body.number


def test_a_kinds_repeated_tap_effects_cannot_install_an_unmeasured_entity():
    hers, body, _moves = _current_entity()
    hers.moved_by_her_presses(body, 10.)
    assert hers.number is None


def test_a_kinds_repeated_tap_effects_preserve_the_entitys_scene_exclusion():
    hers, body, _moves = _current_entity()
    for number in range(1, 6):
        hers._by_thing[number] = _evidence()
    hers.moved_by_her_presses(body, 10.)
    assert hers.number is None


def test_a_kinds_tap_effects_can_install_its_independently_measured_entity():
    hers, body, _moves = _current_entity()
    hers._by_thing[body.number] = _evidence()
    hers.moved_by_her_presses(body, 10.)
    assert hers.number == body.number and hers.kind == body.kind
