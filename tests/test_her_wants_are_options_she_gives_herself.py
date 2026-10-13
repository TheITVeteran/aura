"""Her wants reach the slate she chooses from, and are weighed on what she has found.

The record is core/conation/what_she_wants.py. These follow a want from that
record through the initiative synthesizer, the choice engine and the loop
that acts on the winner.
"""
from __future__ import annotations

import random

import pytest

from core.agency.subjective_choice import ChoiceOption, SubjectiveChoiceEngine
from core.conation import what_she_wants
from core.conation.engine import ConationEngine
from core.conation.what_she_wants import _conation
from core.initiative_synthesis import InitiativeSynthesizer
from core.runtime.overt_action_loop import OvertActionResult, _note_want

TIDE, LOGS = "topic:tide pools", "topic:log files"
LABELS = {TIDE: "tide pools", LOGS: "log files"}


@pytest.fixture
def record(tmp_path):
    held = what_she_wants.reset_for_test(tmp_path / "wants.json", engine=ConationEngine())
    rng = random.Random(7)
    for turn in range(24):
        key, paid = (TIDE, 0.6) if turn % 2 == 0 else (LOGS, -0.5)
        held.went(key, "web_search", label=LABELS[key])
        held.met(["self", key], paid + rng.uniform(-0.2, 0.2), reachable=[key], labels=LABELS)
    for satiation in _conation(held).dynamics._satiation.values():
        satiation.last_update -= 600.0
    yield held
    what_she_wants.reset_for_test(tmp_path / "after.json", engine=ConationEngine())


def test_her_wants_go_on_the_slate_with_the_thing_the_way_and_the_measurements(record):
    synthesizer = InitiativeSynthesizer()
    synthesizer._gather_want_impulses()
    hers = [impulse for impulse in synthesizer._impulse_queue if impulse.source == "want"]
    assert [impulse.content for impulse in hers] == ["tide pools"]
    want = hers[0].metadata["want"]
    assert (want["thing"], want["way"], want["known"], want["contacts"]) == (TIDE, "web_search", True, 12)
    assert hers[0].metadata["skill"] == "web_search"
    assert hers[0].metadata["params"] == {"query": "tide pools"}
    assert hers[0].urgency == pytest.approx(want["priority"], abs=1e-6)


def test_a_liking_she_has_measured_weighs_in_place_of_the_words_in_the_label(record, tmp_path):
    """The live scan matched one keyword and was chosen 379 times of 500."""
    want = record.wants()[0].as_dict()
    chooser = SubjectiveChoiceEngine(tmp_path / "choices.json", mirror_identity=False)
    options = [
        {"goal": "tide pools", "metadata": {"want": want}},
        {"goal": "Running a self-integrity scan", "metadata": {}},
    ]

    class Scored:
        def __init__(self, initiative):
            self.initiative, self.final_score = initiative, 0.6

    chosen, receipt = chooser.choose_from_scored_initiatives([Scored(o) for o in options], context="test")
    assert chosen.initiative["goal"] == "tide pools"
    assert receipt.preference_scores["0"] == pytest.approx(want["liking"])
    assert not any(receipt.option_features["0"].values())
    assert "measured from her own turns" in receipt.rationale


def test_an_outcome_teaches_no_value_through_an_option_weighed_on_her_record(tmp_path):
    chooser = SubjectiveChoiceEngine(tmp_path / "choices.json", mirror_identity=False)
    before = chooser.preferences()
    receipt = chooser.choose(
        [ChoiceOption(id="a", label="design a safe structure of her own", measured_preference=0.5)],
        context="test",
    )
    chooser.appraise_outcome(receipt.choice_id, outcome="went badly", satisfaction=-1.0)
    assert chooser.preferences() == before


def test_a_thing_she_cannot_yet_say_she_likes_is_weighed_like_any_other_option(tmp_path):
    chooser = SubjectiveChoiceEngine(tmp_path / "choices.json", mirror_identity=False)

    class Scored:
        final_score = 0.6
        initiative = {"goal": "lichen", "metadata": {"want": {"known": None, "liking": 0.3}}}

    _chosen, receipt = chooser.choose_from_scored_initiatives([Scored()], context="test")
    assert "measured from her own turns" not in receipt.rationale


def _acted(record, *, verified: bool, mode: str = "skill", unfinished: str = ""):
    want = {"thing": TIDE, "label": "tide pools", "way": "web_search", "unfinished": unfinished}
    result = OvertActionResult(
        action_id="act-9",
        status="verified" if verified else "failed",
        objective=unfinished or "tide pools",
        skill="web_search" if mode == "skill" else "autonomous_task_engine",
        verified=verified,
        execution_mode=mode,
    )
    _note_want({"metadata": {"want": want}}, result)
    return result


def test_acting_on_a_want_is_a_going_credited_when_the_turn_is_read(record):
    before = record.thing(TIDE).ways["web_search"].times
    _acted(record, verified=True)
    record.met(["self"], 0.7)
    assert record.thing(TIDE).ways["web_search"].times == before + 1
    assert record.thing(TIDE).contacts == 13


def test_a_plan_she_did_not_finish_stays_open_until_she_finishes_it(record):
    words = "Draw the zones of a tide pool from low water to the splash line"
    _acted(record, verified=False, mode="planned_goal", unfinished=words)
    assert list(record.thing(TIDE).open.values()) == [words]
    assert record.wants()[0].kind == "finish"
    _acted(record, verified=True, mode="planned_goal", unfinished=words)
    assert record.thing(TIDE).open == {}


def test_a_search_that_failed_leaves_nothing_open(record):
    _acted(record, verified=False)
    assert record.thing(TIDE).open == {}
