"""The 2026-08-10 composition failure, reproduced through the real seam.

The count was 9, in four verified receipts that survived a restart. Retrieval
found them. A correction stated the number. The person was told seventeen.

No stage was wrong. A shaping pass read the record as off-topic and removed
it; a repair pass replaced the reply with a denial that the read had happened.
Each was behaving correctly on what it could see, and what none of them could
see was that the turn had already established the answer.

These tests drive the same shape: hold a fact, run text through the mutation
seam every stage in the reply path actually uses, and assert that the loss is
detected, attributed to the stage that caused it, and repaired at the boundary.
"""

from __future__ import annotations

import logging

import pytest

from core.runtime.fact_custody import (
    BreakKind,
    ValueKind,
    current_custody,
    custody_report,
    hold_fact,
    inspect_mutation,
    reset_custody_for_test,
    restore_held_facts,
)
from core.runtime.turn_outcome import TurnOutcome, VerificationGrade, bind_turn

_RECORDED = (
    "From my own receipts, the count was 9 — that is the recorded value, "
    "not a recollection."
)


@pytest.fixture(autouse=True)
def _clean_custody():
    reset_custody_for_test()
    yield
    reset_custody_for_test()


def _hold_the_count(grade: VerificationGrade = VerificationGrade.OBSERVED):
    return hold_fact(
        subject="read_directory",
        predicate="count",
        value="9",
        subject_cues=("count", "files", "directory"),
        canonical_rendering=_RECORDED,
        established_by="test",
        grade=grade,
        kind=ValueKind.NUMBER,
        evidence=("count=9 sha256=…",),
    )


# ── holding ────────────────────────────────────────────────────────────────


def test_no_turn_bound_holds_nothing() -> None:
    """A background tick has no reply whose custody could break."""

    assert _hold_the_count() is None
    assert current_custody() is None


def test_a_bound_turn_holds_the_fact() -> None:
    with bind_turn(TurnOutcome("t1", origin="test")):
        fact = _hold_the_count()
        assert fact is not None
        assert fact.value == "9"
        assert fact.restorable
        custody = current_custody()
        assert custody is not None
        assert len(custody.facts()) == 1


def test_stronger_evidence_replaces_weaker_and_a_tie_keeps_the_incumbent() -> None:
    with bind_turn(TurnOutcome("t2", origin="test")):
        _hold_the_count(grade=VerificationGrade.ASSERTED)
        _hold_the_count(grade=VerificationGrade.OBSERVED)
        held = current_custody().facts()
        assert len(held) == 1
        assert held[0].grade is VerificationGrade.OBSERVED

        # A weaker re-derivation does not demote what a stronger one measured.
        hold_fact(
            subject="read_directory",
            predicate="count",
            value="17",
            subject_cues=("count",),
            canonical_rendering="the count was 17",
            established_by="test",
            grade=VerificationGrade.ASSERTED,
            kind=ValueKind.NUMBER,
        )
        assert current_custody().facts()[0].value == "9"


# ── detection ──────────────────────────────────────────────────────────────


def test_a_stage_that_strips_the_record_is_named() -> None:
    """The off-topic strip. It removed the answer and said nothing about it."""

    with bind_turn(TurnOutcome("t3", origin="test")):
        _hold_the_count()
        breaks = inspect_mutation(
            "chat.reply_shaping",
            f"Here is what you asked for. {_RECORDED}",
            "Here is what you asked for.",
        )
        assert len(breaks) == 1
        assert breaks[0].kind is BreakKind.DROPPED
        assert breaks[0].stage == "chat.reply_shaping"
        assert breaks[0].fact.value == "9"


def test_a_stage_that_states_a_different_number_is_a_contradiction() -> None:
    """The live sentence, verbatim: a number in words, and the wrong one."""

    with bind_turn(TurnOutcome("t4", origin="test")):
        _hold_the_count()
        breaks = inspect_mutation(
            "chat.response_repair",
            _RECORDED,
            "The count of files in the directory was seventeen, if I recall correctly.",
        )
        assert len(breaks) == 1
        assert breaks[0].kind is BreakKind.CONTRADICTED
        assert "seventeen" in breaks[0].detail


def test_an_unrelated_number_elsewhere_is_not_a_contradiction() -> None:
    """A sentence has to be about this subject before its number counts."""

    with bind_turn(TurnOutcome("t5", origin="test")):
        _hold_the_count()
        breaks = inspect_mutation(
            "chat.append",
            _RECORDED,
            _RECORDED + " It took 42 seconds and used 3 tools.",
        )
        assert breaks == ()


def test_a_stage_that_only_reformats_breaks_nothing() -> None:
    with bind_turn(TurnOutcome("t6", origin="test")):
        _hold_the_count()
        breaks = inspect_mutation(
            "chat.whitespace",
            f"  {_RECORDED}  ",
            _RECORDED,
        )
        assert breaks == ()


def test_a_fact_absent_before_the_stage_is_not_that_stage_s_fault() -> None:
    """Attribution has to survive a pipeline where damage is inherited."""

    with bind_turn(TurnOutcome("t7", origin="test")):
        _hold_the_count()
        first = inspect_mutation("chat.strip", _RECORDED, "Nothing to report.")
        second = inspect_mutation("chat.polish", "Nothing to report.", "Nothing to report!")
        assert len(first) == 1
        assert second == ()


# ── enforcement ────────────────────────────────────────────────────────────


def test_a_dropped_fact_is_restored_at_the_boundary() -> None:
    with bind_turn(TurnOutcome("t8", origin="test")):
        _hold_the_count()
        inspect_mutation("chat.reply_shaping", _RECORDED, "Here is what you asked for.")
        result = restore_held_facts("Here is what you asked for.")
        assert result.changed
        assert "9" in result.text
        assert result.restored[0].kind is BreakKind.DROPPED


def test_a_contradicting_sentence_is_replaced_not_argued_with() -> None:
    """Two sentences disagreeing about one count is not better than one."""

    wrong = "The count of files in the directory was seventeen, if I recall correctly."
    with bind_turn(TurnOutcome("t9", origin="test")):
        _hold_the_count()
        inspect_mutation("chat.response_repair", _RECORDED, wrong)
        result = restore_held_facts(wrong)
        assert result.changed
        assert "seventeen" not in result.text
        assert _RECORDED in result.text


def test_a_fact_a_later_stage_restated_is_not_restored_again() -> None:
    """Custody is a property of what gets delivered, not a grudge."""

    with bind_turn(TurnOutcome("t10", origin="test")):
        _hold_the_count()
        inspect_mutation("chat.strip", _RECORDED, "Gone.")
        inspect_mutation("chat.regenerate", "Gone.", _RECORDED)
        result = restore_held_facts(_RECORDED)
        assert not result.changed
        assert result.text.count("9") == 1


def test_a_merely_asserted_fact_is_reported_and_never_rewrites_text() -> None:
    """A component asserting its own success cannot overwrite a later stage."""

    with bind_turn(TurnOutcome("t11", origin="test")):
        _hold_the_count(grade=VerificationGrade.ASSERTED)
        inspect_mutation("chat.strip", _RECORDED, "Gone.")
        result = restore_held_facts("Gone.")
        assert not result.changed
        assert result.text == "Gone."
        assert len(result.unrestored) == 1


def test_restoration_is_safe_with_no_turn_bound() -> None:
    result = restore_held_facts("anything")
    assert result.text == "anything"
    assert not result.changed


# ── the seam ───────────────────────────────────────────────────────────────


def test_the_real_mutation_seam_checks_custody() -> None:
    """Not a parallel path: the ledger every reply-path gate already calls.

    This is the property that makes the mechanism general. A gate written next
    year is covered because it records its mutation here, not because someone
    remembered to add a custody call to it.
    """

    from core.conversation.turn_arbitration import ledger_for, reset_turn_ledgers_for_test

    reset_turn_ledgers_for_test()
    with bind_turn(TurnOutcome("t12", origin="test")):
        _hold_the_count()
        ledger_for("t12").record_suppression(
            "chat.off_topic_strip",
            "off_topic",
            before=_RECORDED,
            after="Here is what you asked for.",
        )
        breaks = current_custody().breaks()
        assert len(breaks) == 1
        assert breaks[0].stage == "chat.off_topic_strip"


def test_the_health_report_names_the_stages_that_broke_custody() -> None:
    with bind_turn(TurnOutcome("t13", origin="test")):
        _hold_the_count()
        inspect_mutation("chat.reply_shaping", _RECORDED, "Gone.")
    report = custody_report()
    assert report["turns_with_breaks"] == 1
    assert "chat.reply_shaping" in report["stages_that_broke_custody"]


def test_validation_can_exercise_custody_without_emitting_production_events(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with bind_turn(TurnOutcome("t-validation", origin="validation")):
        _hold_the_count()
        with caplog.at_level(logging.WARNING, logger="core.runtime.fact_custody"):
            breaks = inspect_mutation(
                "validation.strip",
                _RECORDED,
                "Gone.",
                emit_log=False,
            )
            result = restore_held_facts("Gone.", emit_log=False)

    assert len(breaks) == 1
    assert result.changed
    assert not any("[CUSTODY]" in record.message for record in caplog.records)


def test_the_producer_takes_custody_when_it_answers_from_receipts() -> None:
    """The value is held where the evidence is, not re-derived downstream."""

    import core.introspection.self_evidence as self_evidence

    with bind_turn(TurnOutcome("t14", origin="test")):
        self_evidence._take_custody_of_recorded_value(
            key="count",
            value="9",
            entry={"action": "read_directory", "evidence": "count=9 path=~/Documents"},
            asked={"count", "files", "directory"},
            rendering=_RECORDED,
        )
        held = current_custody().facts()
        assert len(held) == 1
        assert held[0].predicate == "count"
        assert held[0].value == "9"
        assert held[0].grade is VerificationGrade.OBSERVED
        assert "count" in held[0].subject_cues
