"""Calibration contracts for the accuracy harness (CP215).

This tool twice reported 0% exact-match at every depth. Neither was a
capability result; both were harness faults:

  1. RecurrenceTrainingTask has no ``.score()``, so every task raised
     AttributeError inside ``except Exception: return False`` and was
     recorded as a WRONG ANSWER -- a crash silently rendered as model
     failure.
  2. A 40-token budget could not reach FINAL_ANSWER at all, because the
     model writes hundreds of tokens of prose first.

Either would have been caught in seconds by handing the harness a perfect
answer and requiring 100%. These tests make that calibration mandatory and
prove the gate FAILS CLOSED on both historical bug shapes -- an instrument
that has never been checked against known truth is not evidence-grade.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.rlc_accuracy_ladder import (
    CalibrationError,
    HarnessError,
    _score,
    _tally,
    calibrate_scoring,
)

GOLD = 'FINAL_ANSWER: {"node":6}'


def _task(answer: str = GOLD, family: str = "khop"):
    return SimpleNamespace(answer=answer, family=family)


# ── The four outcomes must be distinguishable ───────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (GOLD, "correct"),
        ("reasoning first\n" + GOLD, "correct"),
        ('FINAL_ANSWER: {"node":9}', "incorrect"),
        ('FINAL_ANSWER: {"other":6}', "incorrect"),
        ("To solve this we follow the edges...", "unparseable"),
        ("", "unparseable"),
        ("FINAL_ANSWER: {malformed", "unparseable"),
    ],
)
def test_outcome_classification(text, expected):
    assert _score(_task(), text) == expected


def test_harness_fault_raises_and_is_never_an_incorrect_answer():
    """The exact bug: a fault inside scoring must not become 'wrong'."""
    with pytest.raises(HarnessError, match="gold answer"):
        _score(_task(answer="this gold cannot parse"), GOLD)


# ── Aggregation must not make 0% ambiguous ──────────────────────────────


def test_perfect_run_aggregates_to_exactly_one_hundred_percent():
    tally = _tally(["correct"] * 6)
    assert tally["accuracy"] == 1.0
    assert tally["contract_compliance"] == 1.0
    assert tally["correct"] == 6 and tally["unparseable"] == 0


def test_silent_model_is_distinguishable_from_a_bad_reasoner():
    """0% accuracy means opposite things in these two cases, so the
    compliance number must separate them."""
    silent = _tally(["unparseable"] * 6)
    bad_reasoner = _tally(["incorrect"] * 6)
    assert silent["accuracy"] == bad_reasoner["accuracy"] == 0.0
    assert silent["contract_compliance"] == 0.0
    assert bad_reasoner["contract_compliance"] == 1.0
    assert silent["unparseable"] == 6 and bad_reasoner["incorrect"] == 6


def test_mixed_outcomes_report_both_rates():
    tally = _tally(["correct", "incorrect", "unparseable", "correct"])
    assert tally["accuracy"] == 0.5
    assert tally["contract_compliance"] == 0.75
    assert tally["n"] == 4


def test_empty_tally_does_not_divide_by_zero():
    tally = _tally([])
    assert tally["accuracy"] == 0.0 and tally["n"] == 0


# ── The gate itself ─────────────────────────────────────────────────────


def test_healthy_harness_passes_calibration():
    receipt = calibrate_scoring()
    assert receipt["passed"] is True
    assert receipt["failures"] == []
    assert len(receipt["checks"]) >= 12
    assert all(row["passed"] for row in receipt["checks"])


def test_calibration_fails_closed_when_everything_scores_incorrect(monkeypatch):
    """Bug shape 1: scoring swallowed a crash and marked all tasks wrong."""
    import tools.rlc_accuracy_ladder as ladder

    monkeypatch.setattr(ladder, "_score", lambda task, text: "incorrect")
    with pytest.raises(CalibrationError, match="failed calibration"):
        ladder.calibrate_scoring()


def test_calibration_fails_closed_when_nothing_parses(monkeypatch):
    """Bug shape 2: the token budget never reached a FINAL_ANSWER."""
    import tools.rlc_accuracy_ladder as ladder

    monkeypatch.setattr(ladder, "_score", lambda task, text: "unparseable")
    with pytest.raises(CalibrationError, match="failed calibration"):
        ladder.calibrate_scoring()


def test_calibration_fails_closed_when_harness_faults_stop_raising(monkeypatch):
    """A scorer that returns instead of raising on a fault is the most
    dangerous shape: it manufactures quiet negative evidence."""
    import tools.rlc_accuracy_ladder as ladder

    monkeypatch.setattr(
        ladder, "_score", lambda task, text: "correct"
    )  # never raises, even on unparseable gold
    with pytest.raises(CalibrationError, match="failed calibration"):
        ladder.calibrate_scoring()


# ── Reasoning must be separable from instruction-following ──────────────


def test_correct_answer_in_the_wrong_format_is_not_a_reasoning_failure():
    """Observed on a real 1.5B: it traces all 8 hops correctly, then writes
    'JSON key node: {...}' instead of the literal FINAL_ANSWER marker.
    Scoring that as a reasoning failure conflates instruction-following
    with reasoning and reports the sum as reasoning."""
    prose = (
        "1. Start at node 11.\n2. Follow 11->2.\n3. From 2 -> 9.\n"
        'The final answer is: 6\nJSON key node: {"node": 6}'
    )
    assert _score(_task(), prose) == "correct_lenient"


def test_wrong_answer_in_the_wrong_format_is_still_wrong():
    prose = 'reasoning...\nJSON key node: {"node": 99}'
    assert _score(_task(), prose) == "incorrect_lenient"


def test_lenient_extraction_cannot_invent_an_answer():
    """It matches only objects carrying the EXPECTED keys, so unrelated
    JSON in the output can never be promoted into an answer."""
    assert _score(_task(), 'here is config {"unrelated": 6}') == "unparseable"
    assert _score(_task(), "no json at all here") == "unparseable"


def test_strict_and_reasoning_rates_are_reported_separately():
    tally = _tally(["correct", "correct_lenient", "incorrect_lenient", "unparseable"])
    assert tally["accuracy"] == 0.25            # strict: contract satisfied
    assert tally["reasoning_accuracy"] == 0.5   # right answer in any shape
    assert tally["contract_compliance"] == 0.25
    assert tally["answered_at_all"] == 0.75


def test_perfect_contract_run_scores_identically_on_both_metrics():
    tally = _tally(["correct"] * 4)
    assert tally["accuracy"] == tally["reasoning_accuracy"] == 1.0
