"""A step's end that is a relation between two things ("the roller sits at the end of the arrow") is measured on where
the two things are, not looked for in the screen's words, which never say it.

LIVE 2026-10-10 a plan step "Test the trap" was to show "the cage drops on Jerry"; held to the screen's words, a
physical outcome could only ever be judged not shown.
"""
from __future__ import annotations

import pytest

from core.perception.relations_between_things import a_relation_in, holds, measured

pytestmark = pytest.mark.unit

REGIONS = [{"text": "the roller", "x": 300, "y": 200, "width": 40, "height": 30},
           {"text": "the arrow", "x": 260, "y": 205, "width": 50, "height": 20},
           {"text": "the cage", "x": 600, "y": 100, "width": 80, "height": 60},
           {"text": "Jerry", "x": 610, "y": 160, "width": 30, "height": 30}]


def test_a_sentence_is_read_for_two_things_and_the_relation_between_them():
    assert a_relation_in("the roller sits at the end of the arrow") == ("the roller", "at", "the arrow")
    assert a_relation_in("The box is under the ledge.") == ("The box", "below", "the ledge")
    assert a_relation_in("A list of device types appears") is None


def test_the_relation_is_measured_on_where_the_things_are():
    assert measured("the roller sits at the end of the arrow", REGIONS) is True
    assert measured("the cage drops on Jerry", REGIONS) is True
    assert measured("the roller is on the cage", REGIONS) is False
    assert measured("the anvil is on Jerry", REGIONS) is None            # no anvil is found: nothing is measured
    assert holds("left of", (0, 0, 10, 10), (50, 0, 10, 10)) and not holds("right of", (0, 0, 10, 10), (50, 0, 10, 10))


def test_a_steps_end_that_is_a_relation_settles_the_step_by_measurement():
    from core.cognition.reading_the_rules import EvidenceSnapshot, StepAttempt, _step_state
    from core.runtime.skill_contract import PredicateState

    before = EvidenceSnapshot.of({"text": "A ROOM OF DEVICES", "regions": []})
    attempt = StepAttempt(1, "k", 'drag "a roller" to "the end of the arrow"', before,
                          "the roller sits at the end of the arrow")
    shown = EvidenceSnapshot.of({"text": "A ROOM OF DEVICES", "regions": REGIONS})
    assert _step_state(attempt, shown, True)[0] == PredicateState.SATISFIED
    elsewhere = [dict(r, x=r["x"] + 300) if r["text"] == "the roller" else r for r in REGIONS]
    assert _step_state(attempt, EvidenceSnapshot.of({"text": "x", "regions": elsewhere}), True)[0] == PredicateState.UNSATISFIED
    unseen = EvidenceSnapshot.of({"text": "A ROOM OF DEVICES", "regions": []})
    assert _step_state(attempt, unseen, True)[0] == PredicateState.UNKNOWN
