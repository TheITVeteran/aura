"""Ambiguous scene names cannot certify a procedure's physical result."""
from __future__ import annotations

from itertools import permutations

import pytest

from core.cognition.a_plan_to_an_end import Plan
from core.cognition.reading_the_rules import Frame, Rules
from core.perception.relations_between_things import (
    _relations_require_unique_finite_entities,
    holds,
    located,
    measured,
)
from core.runtime.skill_contract import PredicateState

INSIDE = {"text": "crate", "x": .15, "y": .15, "width": .1, "height": .1}
OUTSIDE = {"text": "crate", "x": .75, "y": .75, "width": .1, "height": .1}
ROOM = {"text": "room", "x": .1, "y": .1, "width": .3, "height": .3}
EXPECTED = "the crate is inside the room"


def _attempt(regions, expected=EXPECTED):
    frame = Frame("Inspect the arrangement", act="click things", thing="Inspect")
    rules = Plan(frames={frame.key: frame}, expects={frame.key: expected})
    before = {"text": "Arrangement", "surface_id": "editor", "regions": []}
    attempt = rules.begin('click "Inspect"', before)
    after = {"text": "Arrangement", "surface_id": "editor", "regions": list(regions)}
    receipt = rules.tried('click "Inspect"', True, after=after)
    return rules, frame, attempt, receipt


def test_registered_spatial_grounding_invariant():
    assert _relations_require_unique_finite_entities() == ()


@pytest.mark.parametrize("regions", list(permutations((INSIDE, OUTSIDE, ROOM))))
def test_every_order_of_ambiguous_named_entities_remains_unknown_in_the_real_procedure_caller(regions):
    assert located("the crate", regions) is None
    assert measured(EXPECTED, regions) is None
    rules, frame, attempt, receipt = _attempt(regions)
    assert receipt.attempt is attempt and receipt.state is PredicateState.UNKNOWN
    assert not rules.passed(frame) and frame.key not in rules.done
    assert rules.unknown_steps[frame.key] == 1 and not rules.failed_steps
    assert rules.pending_step is None


@pytest.mark.parametrize("candidate,state", [(INSIDE, PredicateState.SATISFIED),
                                          (OUTSIDE, PredicateState.UNSATISFIED)])
def test_one_uniquely_grounded_entity_keeps_its_measured_success_or_failure(candidate, state):
    rules, frame, attempt, receipt = _attempt((candidate, ROOM))
    assert receipt.attempt is attempt and receipt.state is state
    assert rules.passed(frame) is (state is PredicateState.SATISFIED)


def test_a_more_specific_name_resolves_one_entity_without_using_region_order():
    red = {**INSIDE, "text": "red crate"}
    blue = {**OUTSIDE, "text": "blue crate"}
    for regions in permutations((red, blue, ROOM)):
        assert located("the red crate", regions) == (.15, .15, .1, .1)
        rules, frame, _, receipt = _attempt(regions, "the red crate is inside the room")
        assert receipt.state is PredicateState.SATISFIED and rules.passed(frame)


@pytest.mark.parametrize("damage", [{"x": "nan"}, {"y": "inf"}, {"width": -1}, {"height": 0},
                                  {"width": None}, {"x": True}, {"height": "unmeasured"}])
def test_a_malformed_competing_entity_cannot_be_dropped_to_make_another_unique(damage):
    malformed = {**OUTSIDE, **damage}
    for regions in ((INSIDE, malformed, ROOM), (malformed, INSIDE, ROOM)):
        rules, frame, _, receipt = _attempt(regions)
        assert receipt.state is PredicateState.UNKNOWN and not rules.passed(frame)


@pytest.mark.parametrize("damage", [{"x": "nan"}, {"y": "inf"}, {"width": -1}, {"height": 0},
                                  {"width": None}, {"x": True}, {"height": "unmeasured"}])
def test_unmeasured_geometry_cannot_certify_success_or_failure_even_when_the_name_is_unique(damage):
    rules, frame, _, receipt = _attempt(({**INSIDE, **damage}, ROOM))
    assert receipt.state is PredicateState.UNKNOWN and not rules.passed(frame)


def test_two_equally_named_destinations_are_unknown_too():
    elsewhere = {**ROOM, "x": .6, "y": .6}
    for regions in permutations((INSIDE, ROOM, elsewhere)):
        rules, frame, _, receipt = _attempt(regions)
        assert receipt.state is PredicateState.UNKNOWN and not rules.passed(frame)


def test_an_invalid_more_specific_candidate_does_not_fall_back_to_a_generic_one():
    invalid = {**OUTSIDE, "text": "red crate", "width": None}
    assert located("the red crate", (INSIDE, invalid)) is None


def test_direct_relation_measurement_rejects_invalid_extents_and_unknown_relations():
    assert holds("inside", (.15, .15, float("nan"), .1), (.1, .1, .3, .3)) is None
    assert holds("inside", (.15, .15, .1, 0), (.1, .1, .3, .3)) is None
    assert holds("not a known relation", (.15, .15, .1, .1), (.1, .1, .3, .3)) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_raw_nonfinite_coordinates_are_unknown_to_perception_and_rejected_by_the_receipt_boundary(value):
    regions = [{**INSIDE, "x": value}, ROOM]
    assert measured(EXPECTED, regions) is None
    frame = Frame("Inspect the arrangement", act="click things", thing="Inspect")
    plan = Plan(frames={frame.key: frame}, expects={frame.key: EXPECTED})
    attempt = plan.begin('click "Inspect"', {"text": "Arrangement", "regions": []})
    with pytest.raises(ValueError, match="JSON compliant"):
        plan.tried('click "Inspect"', True, after={"regions": regions})
    assert plan.pending_step is attempt and not plan.passed(frame)


def test_bare_rules_navigation_receipts_still_work_without_a_spatial_expectation():
    frame = Frame("Open", act="click things", thing="Open")
    rules = Rules(frames={frame.key: frame})
    rules.begin('click "Open"', {"text": "Closed"})
    receipt = rules.tried('click "Open"', True, after={"text": "Open"})
    assert receipt.state is PredicateState.SATISFIED and rules.passed(frame)
