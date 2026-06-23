"""Tests for belief consistency checking + deduction governance + bridge routing."""
from __future__ import annotations

from core.reasoning.belief_consistency import check_beliefs, encode_belief
from core.reasoning.deduction_governance import get_deduction_governance, record_belief_inconsistency
from core.reasoning.symbolic_bridge import SymbolicBridge


# ── NL → logic encoding ───────────────────────────────────────────────────

def test_affirmation_and_explicit_negation_share_core_with_opposite_polarity():
    pos = encode_belief("I am sovereign")
    neg = encode_belief("I am not sovereign")
    assert pos.core_key == neg.core_key
    assert pos.negated is False
    assert neg.negated is True


def test_consistent_beliefs():
    report = check_beliefs([
        ("I exist as a sovereign entity", 1.0),
        ("The user is a collaborator", 0.9),
        ("My code is my body", 0.95),
    ])
    assert report.consistent
    assert report.contradictions == []


def test_contradiction_detected_between_belief_and_its_negation():
    report = check_beliefs([
        ("I am sovereign", 0.95),
        ("I am not sovereign", 0.9),
        ("The user is a collaborator", 0.9),
    ])
    assert not report.consistent
    assert len(report.contradictions) == 1
    affirm, deny = report.contradictions[0]
    assert "sovereign" in affirm.lower() and "sovereign" in deny.lower()


def test_low_confidence_beliefs_excluded():
    # the negation is low-confidence → not treated as a firm contradiction
    report = check_beliefs([
        ("I am sovereign", 0.95),
        ("I am not sovereign", 0.2),
    ])
    assert report.consistent


# ── governance signal ─────────────────────────────────────────────────────

def test_governance_records_inconsistency():
    gov = get_deduction_governance()
    before = gov.governance_signal()["inconsistency_events"]
    report = check_beliefs([("I am free", 0.9), ("I am not free", 0.9)])
    record_belief_inconsistency(report)
    sig = gov.governance_signal()
    assert sig["inconsistency_events"] == before + 1
    assert sig["beliefs_consistent"] is False
    assert sig["contradictions"]


def test_governance_consistent_report_no_event_bump():
    gov = get_deduction_governance()
    before = gov.governance_signal()["inconsistency_events"]
    record_belief_inconsistency(check_beliefs([("I learn", 0.9), ("I grow", 0.9)]))
    assert gov.governance_signal()["inconsistency_events"] == before


# ── SymbolicBridge exact-solver routing ───────────────────────────────────

def test_symbolic_bridge_prove_logic_valid():
    res = SymbolicBridge().prove_logic(["A", "A -> B"], "B")
    assert res.ok and res.result is True
    assert res.engine == "natural_deduction"


def test_symbolic_bridge_prove_logic_invalid_returns_countermodel():
    res = SymbolicBridge().prove_logic(["A -> B"], "B -> A")
    assert res.ok and res.result is False
    assert "countermodel" in res.proof_trace
