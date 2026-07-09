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


def test_implication_belief_encodes_as_implies():
    from core.reasoning.belief_consistency import encode_belief
    from core.reasoning.natural_deduction import Implies

    enc = encode_belief("if it rains then the ground is wet")
    assert isinstance(enc.formula, Implies)


def test_chained_modus_ponens_contradiction_detected():
    # {rain, rain→wet, ¬wet} is inconsistent — the prover chains the implication.
    report = check_beliefs([
        ("it rains", 0.9),
        ("if it rains then the ground is wet", 0.9),
        ("the ground is not wet", 0.9),
    ])
    assert not report.consistent
    assert report.contradictions          # the conflicting source beliefs are surfaced
    assert len(report.minimal_core) >= 2


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

def test_belief_engine_demotes_weaker_side_of_conflict():
    from core.epistemics.belief_revision import Belief, BeliefRevisionEngine

    eng = BeliefRevisionEngine.__new__(BeliefRevisionEngine)
    eng.beliefs = [
        Belief(id="b1", content="I am sovereign", confidence=0.95, domain="self", source="axiom"),
        Belief(id="b2", content="I am not sovereign", confidence=0.7, domain="self", source="conversation"),
    ]
    eng.check_belief_consistency()
    weaker = next(b for b in eng.beliefs if b.content == "I am not sovereign")
    stronger = next(b for b in eng.beliefs if b.content == "I am sovereign")
    assert weaker.confidence < 0.7                       # demoted
    assert stronger.confidence == 0.95                   # stronger side untouched
    assert "logical_conflict" in weaker.supporting_evidence


def test_symbolic_bridge_prove_logic_valid():
    res = SymbolicBridge().prove_logic(["A", "A -> B"], "B")
    assert res.ok and res.result is True
    assert res.engine == "natural_deduction"


def test_symbolic_bridge_prove_logic_invalid_returns_countermodel():
    res = SymbolicBridge().prove_logic(["A -> B"], "B -> A")
    assert res.ok and res.result is False
    assert "countermodel" in res.proof_trace


def test_symbolic_bridge_catches_arithmetic_error():
    errs = SymbolicBridge().check_arithmetic_claims("The total is 2 + 2 = 5, obviously.")
    assert len(errs) == 1
    assert errs[0]["stated"] == 5.0 and errs[0]["correct"] == 4.0


def test_symbolic_bridge_accepts_correct_arithmetic():
    assert SymbolicBridge().check_arithmetic_claims("We have 12 * 3 = 36 widgets.") == []
    # algebra / non-numeric "=" must not be mis-flagged
    assert SymbolicBridge().check_arithmetic_claims("Let x = 5 and y = 7.") == []


def test_symbolic_bridge_audit_reasoning_gateway():
    audit = SymbolicBridge().audit_reasoning(
        "If it rains then the ground is wet, and the ground is wet, therefore it is raining. Also 3 + 4 = 8."
    )
    assert not audit["clean"]
    assert len(audit["non_sequiturs"]) == 1          # affirming the consequent
    assert len(audit["arithmetic_errors"]) == 1      # 3+4≠8
    assert SymbolicBridge().audit_reasoning("I'm glad we're talking. What's up?")["clean"]
