"""The turn is answered once, by the best answer, and suppression has a name.

Every mechanism in the response path is locally correct, and they interfere
anyway, because nothing observes the interference:

* an honesty gate suppresses a good answer and the reason lives in a log line
  nobody correlates with the apology the user received;
* two lanes both produce a reply and whichever finishes last wins, so quality
  depends on timing rather than merit;
* a deep lane returns after a quick lane already answered, and speaks into a
  conversation that has moved on.

None is fixable inside the mechanism that caused it — each behaves correctly on
what it can see. What was missing is a record of the turn as a whole, which
makes three things decidable instead of accidental: whether a late lane may
speak, which candidate wins, and which gate discarded what.
"""
from __future__ import annotations

import pytest

from core.conversation.turn_arbitration import (
    LanePrecedence,
    TurnLedger,
    ledger_for,
    reset_turn_ledgers_for_test,
)


@pytest.fixture(autouse=True)
def _clean_ledgers():
    reset_turn_ledgers_for_test()
    yield
    reset_turn_ledgers_for_test()


# ── A turn is answered once ────────────────────────────────────────────────

def test_the_first_lane_may_answer() -> None:
    ledger = ledger_for("t1")
    allowed, why = ledger.should_serve("quick", LanePrecedence.QUICK)
    assert allowed and "unanswered" in why


def test_a_second_lane_may_not_answer_again() -> None:
    ledger = ledger_for("t2")
    ledger.record_candidate("quick", LanePrecedence.QUICK, chars=120)
    ledger.record_served("quick")
    allowed, why = ledger.should_serve("full", LanePrecedence.FULL)
    assert not allowed
    # A higher-precedence lane gets the more specific refusal, which says both
    # that it was better and that it was late.
    assert "after the turn was answered" in why


def test_a_better_answer_arriving_late_is_still_late() -> None:
    """A deep reply that lands after the quick one is a second reply.

    Serving it would put two answers in front of the person, the second one
    addressing a question they have already moved past. Being better does not
    make it timely.
    """
    ledger = ledger_for("t3")
    ledger.record_candidate("quick", LanePrecedence.QUICK, chars=90)
    ledger.record_served("quick")
    allowed, why = ledger.should_serve("deep", LanePrecedence.DEEP)
    assert not allowed
    assert "outranks" in why and "after the turn was answered" in why


def test_the_same_lane_cannot_answer_twice() -> None:
    ledger = ledger_for("t4")
    ledger.record_candidate("quick", LanePrecedence.QUICK, chars=90)
    ledger.record_served("quick")
    allowed, why = ledger.should_serve("quick", LanePrecedence.QUICK)
    assert not allowed
    assert "already answered" in why


# ── The best answer wins, not the fastest ─────────────────────────────────

def test_merit_beats_arrival_order() -> None:
    ledger = ledger_for("t5")
    ledger.record_candidate("quick", LanePrecedence.QUICK, chars=200)
    ledger.record_candidate("deep", LanePrecedence.DEEP, chars=150)
    best = ledger.best_candidate()
    assert best is not None and best.lane == "deep"


def test_authentic_text_beats_repair_machinery() -> None:
    """Theatre must not win a tie-break on length."""
    ledger = ledger_for("t6")
    ledger.record_candidate("repair", LanePrecedence.REPAIR, chars=800, authentic=False)
    ledger.record_candidate("full", LanePrecedence.FULL, chars=120, authentic=True)
    best = ledger.best_candidate()
    assert best is not None and best.lane == "full"


def test_an_inauthentic_candidate_is_better_than_nothing() -> None:
    ledger = ledger_for("t7")
    ledger.record_candidate("fallback", LanePrecedence.FALLBACK, chars=60, authentic=False)
    best = ledger.best_candidate()
    assert best is not None and best.lane == "fallback"


def test_no_candidates_means_no_answer() -> None:
    assert ledger_for("t8").best_candidate() is None


def test_an_empty_candidate_never_wins() -> None:
    ledger = ledger_for("t9")
    ledger.record_candidate("deep", LanePrecedence.DEEP, chars=0)
    ledger.record_candidate("quick", LanePrecedence.QUICK, chars=40)
    best = ledger.best_candidate()
    assert best is not None and best.lane == "quick"


# ── Suppression has a name and a size ─────────────────────────────────────

def test_a_gate_that_discards_an_answer_is_named() -> None:
    ledger = ledger_for("t10")
    ledger.record_suppression(
        "chat.full_mind_contract_fail_closed",
        "confidence:degraded",
        before="x" * 900,
        after="I couldn't get my full attention onto that one.",
    )
    lost = ledger.lost_work()
    assert len(lost) == 1
    assert "full_mind_contract" in lost[0].gate
    assert lost[0].suppressed_chars == 900


def test_tidying_is_not_counted_as_lost_work() -> None:
    """A whitespace repair and a discarded answer must not look alike."""
    ledger = ledger_for("t11")
    ledger.record_suppression("format", "spacing", before="hello  there", after="hello there")
    assert ledger.lost_work() == []


def test_the_turn_reads_as_one_line() -> None:
    ledger = ledger_for("t12")
    ledger.record_candidate("quick", LanePrecedence.QUICK, chars=90)
    ledger.record_suppression("guard", "weather", before="a" * 200, after="")
    ledger.record_served("quick")
    line = ledger.narrative()
    assert "turn=t12" in line and "quick(QUICK/90c)" in line and "served=quick" in line
    assert "guard suppressed 200 chars" in line


def test_an_unanswered_turn_says_so() -> None:
    assert "served=nothing" in ledger_for("t13").narrative()


# ── The ledger must not become the leak it helps diagnose ─────────────────

def test_ledgers_are_bounded() -> None:
    for index in range(200):
        ledger_for(f"turn-{index}").record_candidate("quick", LanePrecedence.QUICK, chars=1)
    from core.conversation import turn_arbitration

    assert len(turn_arbitration._LEDGERS) <= 64


def test_a_turn_can_be_forgotten() -> None:
    from core.conversation import turn_arbitration

    ledger_for("t14")
    turn_arbitration.forget_turn("t14")
    assert "t14" not in turn_arbitration._LEDGERS


# ── It is wired to the path every gate takes ──────────────────────────────

def test_the_chat_mutation_funnel_records_suppression() -> None:
    from pathlib import Path

    src = Path("interface/routes/chat.py").read_text(encoding="utf-8")
    funnel = src[src.index("def _append_turn_text_mutation") :]
    funnel = funnel[: funnel.index("def _merge_turn_text_mutations")]
    assert "from core.conversation.turn_arbitration import ledger_for" in funnel
    assert "record_suppression(" in funnel


# ── A gate that takes an answer must be able to give it back ──────────────

def test_a_suppression_keeps_what_it_took() -> None:
    """Sizes let a loss be counted. Text lets it be undone.

    The refusal site says "the engine produced no acceptable reply" while
    standing next to the acceptable reply a gate removed, and without the text
    there is nothing to hand back.
    """
    ledger = TurnLedger(turn_id="recoverable")
    answer = "The very first thing you asked was what it's actually like in here right now."
    ledger.record_suppression("honesty_gate", "unverified_claim", before=answer, after="")
    assert ledger.recoverable_text() == answer


def test_a_trivial_edit_is_not_offered_as_a_recovery() -> None:
    ledger = TurnLedger(turn_id="tidy")
    ledger.record_suppression("whitespace", "trailing_space", before="Hello. ", after="Hello.")
    assert ledger.recoverable_text() == ""


def test_the_largest_loss_wins() -> None:
    ledger = TurnLedger(turn_id="several")
    ledger.record_suppression("a", "r", before="x" * 90, after="")
    ledger.record_suppression("b", "r", before="y" * 400, after="")
    assert ledger.recoverable_text() == "y" * 400


def test_retained_text_stays_bounded() -> None:
    """A turn in trouble must not become a memory leak."""
    ledger = TurnLedger(turn_id="flood")
    for index in range(120):
        ledger.record_suppression(f"gate{index}", "r", before="z" * 50_000, after="")
    assert len(ledger.suppressions) <= 24
    assert all(len(item.suppressed_text) <= 8000 for item in ledger.suppressions)
