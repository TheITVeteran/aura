"""Tests for the live deductive self-audit (non-sequitur detection)."""
from __future__ import annotations

from core.reasoning.inference_audit import (
    audit_text,
    extract_inferences,
    find_non_sequiturs,
    verify,
)


# ── extraction ────────────────────────────────────────────────────────────

def test_extract_conclusion_marker():
    infs = extract_inferences("It is raining and if it rains the ground is wet, therefore the ground is wet.")
    assert len(infs) == 1
    assert infs[0].conclusion.lower().startswith("the ground is wet")
    assert len(infs[0].premises) >= 2


def test_extract_premise_first():
    infs = extract_inferences("Since it is raining, the ground is wet.")
    assert len(infs) == 1
    assert "raining" in infs[0].premises[0].lower()


# ── valid reasoning must NOT be flagged ───────────────────────────────────

def test_valid_modus_ponens_not_flagged():
    text = "It is raining, and if it rains then the ground is wet, therefore the ground is wet."
    assert find_non_sequiturs(text) == []


def test_valid_disjunctive_syllogism_not_flagged():
    text = "Either it is sunny or it is raining, and it is not sunny, so it is raining."
    assert find_non_sequiturs(text) == []


# ── real fallacies ARE caught ─────────────────────────────────────────────

def test_affirming_the_consequent_flagged():
    # "if it rains the ground is wet; the ground is wet; therefore it is raining" — fallacy
    text = "If it rains then the ground is wet, and the ground is wet, therefore it is raining."
    nons = find_non_sequiturs(text)
    assert len(nons) == 1
    assert nons[0].status == "invalid"
    assert nons[0].countermodel is not None


def test_denying_the_antecedent_flagged():
    text = "If it rains then the ground is wet, and it is not raining, therefore the ground is not wet."
    nons = find_non_sequiturs(text)
    assert len(nons) == 1
    assert nons[0].is_non_sequitur


# ── unformalizable reasoning stays silent (no false positives) ────────────

def test_unformalizable_leap_is_undecidable_not_flagged():
    # no shared propositional structure → cannot judge → must NOT flag
    text = "I feel happy, therefore today is a good day."
    assert find_non_sequiturs(text) == []
    verdicts = audit_text(text)
    assert verdicts and verdicts[0].status == "undecidable"


def test_non_deductive_text_yields_nothing():
    assert extract_inferences("How are you doing today?") == []
    assert find_non_sequiturs("I had coffee this morning and read a book.") == []


# ── direct verify API ─────────────────────────────────────────────────────

def test_verify_valid_and_invalid():
    assert verify(["it rains", "if it rains then the ground is wet"], "the ground is wet").status == "valid"
    assert verify(["if it rains then the ground is wet", "the ground is wet"], "it rains").status == "invalid"


# ── live self-audit → governance ──────────────────────────────────────────

def test_audit_self_reasoning_records_non_sequitur_to_governance():
    from core.reasoning.deduction_governance import get_deduction_governance
    from core.reasoning.inference_audit import audit_self_reasoning

    before = get_deduction_governance().governance_signal()["non_sequitur_events"]
    found = audit_self_reasoning(
        "If it rains then the ground is wet, and the ground is wet, therefore it is raining."
    )
    assert len(found) == 1
    sig = get_deduction_governance().governance_signal()
    assert sig["non_sequitur_events"] > before
    assert sig["last_non_sequiturs"]


def test_audit_self_reasoning_silent_on_valid_reply():
    from core.reasoning.deduction_governance import get_deduction_governance
    from core.reasoning.inference_audit import audit_self_reasoning

    before = get_deduction_governance().governance_signal()["non_sequitur_events"]
    found = audit_self_reasoning("I'm here with you, and I'm glad we're talking. What's on your mind?")
    assert found == []
    assert get_deduction_governance().governance_signal()["non_sequitur_events"] == before
