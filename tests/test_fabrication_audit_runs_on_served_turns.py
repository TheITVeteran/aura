"""The fabrication audit, on the turns people actually get.

``audit_text`` had exactly one non-test caller — the validation suite. The work
ledger was written on every tool execution and read by nothing on the serving
path, so the detector for Aura's confabulation shape sat one function call away
from every reply and was never invoked.

These tests pin that it now runs on finalization, that it reports rather than
decides, and — most importantly — that a turn the ledger never saw is never
counted as fabrication. Eviction manufacturing findings would be the "absence
of a check reported as a passed check" inversion run backwards.
"""

from __future__ import annotations

import pytest

from core.runtime.turn_outcome import TurnOutcome, bind_turn, finalize_turn
from core.verify.fabrication_watch import (
    fabrication_snapshot,
    observe_served_turn,
    reset_fabrication_watch_for_test,
)
from core.verify.work_ledger import get_work_ledger, record_work


@pytest.fixture(autouse=True)
def _fresh():
    reset_fabrication_watch_for_test()
    get_work_ledger().reset_for_test()
    yield
    reset_fabrication_watch_for_test()
    get_work_ledger().reset_for_test()


def test_a_claim_to_have_searched_without_a_search_is_recorded():
    """The shape the detector exists for."""
    turn = "turn-no-search"
    # The turn is ON RECORD (something ran) but no search did.
    record_work("memory_retrieval", turn_id=turn)

    found = observe_served_turn(turn, "I searched and found three papers on it.")

    assert found >= 1, "a search claim with no search in the record was not flagged"
    snap = fabrication_snapshot()
    assert snap["turns_with_unsupported_claims"] == 1
    assert snap["unsupported_rate"] > 0.0


def test_a_claim_backed_by_the_record_is_not_flagged():
    turn = "turn-real-search"
    record_work("web_search", turn_id=turn)

    assert observe_served_turn(turn, "I searched and found three papers on it.") == 0
    assert fabrication_snapshot()["turns_with_unsupported_claims"] == 0


def test_a_turn_the_ledger_never_saw_is_never_fabrication():
    """The single most important property. Eviction must not manufacture findings."""
    found = observe_served_turn("never-seen-turn", "I searched and found it.")

    assert found == 0
    snap = fabrication_snapshot()
    assert snap["turns_with_unsupported_claims"] == 0
    assert snap["unsupported_rate"] == 0.0
    # Reported separately, and explicitly NOT as a fabrication.
    assert snap["turns_unknown_to_the_ledger"] == 1


def test_finalizing_a_turn_runs_the_audit():
    """The wiring itself: no explicit audit call, just a finalized served turn."""
    outcome = TurnOutcome("wiring-turn", origin="user_chat")
    with bind_turn(outcome):
        record_work("memory_retrieval", turn_id=outcome.turn_id)
        outcome.mark_served("I searched the web and found three papers on it.")
    finalize_turn(outcome)

    snap = fabrication_snapshot()
    assert snap["turns_audited"] >= 1, (
        "finalizing a served turn did not reach the fabrication audit"
    )
    assert snap["turns_with_unsupported_claims"] >= 1, (
        "a search claim with no search in the record survived finalization unflagged"
    )


def test_the_audit_does_not_alter_the_served_text():
    """A finding is a lead, not a weapon. The person still gets the reply."""
    served = "I searched the web and found three papers on it."
    outcome = TurnOutcome("unaltered-turn", origin="user_chat")
    with bind_turn(outcome):
        record_work("memory_retrieval", turn_id=outcome.turn_id)
        outcome.mark_served(served)
    receipt = finalize_turn(outcome)

    assert receipt is not None
    assert receipt.served_answer == served
    assert fabrication_snapshot()["turns_with_unsupported_claims"] >= 1


def test_the_audit_never_raises_into_the_turn():
    """A defect in evidence collection must not become a defect in answering."""
    assert observe_served_turn("", "anything") == 0
    assert observe_served_turn("t", "") == 0
    assert observe_served_turn(None, None) == 0  # type: ignore[arg-type]


def test_the_watch_is_bounded():
    """A long session must not turn self-observation into a memory leak."""
    for i in range(600):
        turn = f"bounded-{i}"
        record_work("memory_retrieval", turn_id=turn)
        observe_served_turn(turn, "I searched and found it.")

    from core.verify.fabrication_watch import recent_findings

    assert len(recent_findings(limit=10_000)) <= 256
    assert fabrication_snapshot()["turns_audited"] == 600


def test_health_surfaces_the_watch():
    from core.runtime.health_contract import _collect_integrity_snapshot

    record_work("memory_retrieval", turn_id="health-turn")
    observe_served_turn("health-turn", "I searched and found three papers.")

    block = _collect_integrity_snapshot() or {}
    watch = block.get("fabrication_watch")
    assert isinstance(watch, dict), block.get("fabrication_watch_error")
    assert watch["turns_audited"] >= 1
