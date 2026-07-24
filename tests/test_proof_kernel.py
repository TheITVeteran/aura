"""Contract tests for the trusted proof kernel (de Bruijn criterion).

The kernel must (1) accept every certificate the search produces for a real
proof, (2) reject forged or tampered certificates, (3) report the true
premise footprint (axiom audit), and (4) keep honest books on admitted
(sorry) claims, taint, and discharge.
"""
from __future__ import annotations

import pytest

from core.reasoning.natural_deduction import (
    Atom,
    Bot,
    CertStep,
    Implies,
    Not,
    parse,
    prove,
)
from core.reasoning.proof_kernel import (
    TheoremLedger,
    check_proof,
    prove_certified,
    prove_certified_text,
    reset_theorem_ledger_for_test,
)


@pytest.fixture(autouse=True)
def _fresh_ledger():
    reset_theorem_ledger_for_test()
    yield
    reset_theorem_ledger_for_test()


# ── search emits certificates, kernel accepts them ────────────────────────

@pytest.mark.parametrize(
    "premises,goal",
    [
        (["A", "A -> B"], "B"),                          # modus ponens
        (["A -> B", "B -> C"], "A -> C"),                # hypothetical syllogism
        (["A | B", "~A"], "B"),                          # disjunctive syllogism
        (["A & B"], "A"),                                 # ∧-elimination
        (["~~A"], "A"),                                   # double negation
        (["A -> B", "~B"], "~A"),                         # modus tollens
        ([], "A | ~A"),                                   # excluded middle (no premises)
        (["A <-> B", "A"], "B"),                          # iff use
    ],
)
def test_kernel_accepts_search_certificates(premises, goal):
    cp = prove_certified_text(premises, goal)
    assert cp.proof.provable
    assert cp.proof.certificate is not None
    assert cp.verified, cp.verdict.reason if cp.verdict else "no verdict"
    assert cp.theorem is not None


def test_unprovable_goal_yields_countermodel_and_no_theorem():
    cp = prove_certified_text(["A -> B", "B"], "A")      # affirming the consequent
    assert not cp.proof.provable
    assert cp.proof.countermodel is not None
    assert cp.verdict is None
    assert cp.theorem is None


def test_prove_bot_from_inconsistent_set_is_certified():
    cp = prove_certified_text(["A", "~A"], "⊥" if False else "A & ~A")
    # A ∧ ¬A is provable from {A, ¬A}; also check direct Bot goal via formulas
    assert cp.verified
    direct = prove_certified([parse("A"), parse("~A")], Bot())
    assert direct.verified


# ── forged / tampered certificates are rejected ───────────────────────────

def test_kernel_rejects_forged_closure():
    # Certificate claims {A, ¬A} closes a branch that contains neither.
    forged = CertStep("close", Atom("Z"))
    verdict = check_proof([parse("A")], parse("B"), forged)
    assert not verdict.verified
    assert "closure" in verdict.reason


def test_kernel_rejects_expansion_of_absent_target():
    forged = CertStep("expand", parse("P & Q"), (CertStep("close", Atom("P")),))
    verdict = check_proof([parse("A")], parse("B"), forged)
    assert not verdict.verified
    assert "not in the branch" in verdict.reason


def test_kernel_rejects_wrong_branch_count():
    # A∨B mandates two branches; certificate supplies one.
    premises = [parse("A | B"), parse("~A"), parse("~B")]
    forged = CertStep(
        "expand", parse("A | B"), (CertStep("close", Atom("A")),)
    )
    verdict = check_proof(premises, Bot(), forged)
    assert not verdict.verified
    assert "mandates" in verdict.reason


def test_kernel_rejects_truncated_real_certificate():
    proof = prove([parse("A | B"), parse("~A"), parse("~B")], Bot())
    assert proof.provable and proof.certificate is not None
    cert = proof.certificate
    # Walk down to a node with two children and drop one.
    def truncate(node: CertStep) -> CertStep | None:
        if len(node.children) == 2:
            return CertStep(node.kind, node.target, node.children[:1])
        for i, c in enumerate(node.children):
            t = truncate(c)
            if t is not None:
                kids = list(node.children)
                kids[i] = t
                return CertStep(node.kind, node.target, tuple(kids))
        return None

    tampered = truncate(cert)
    assert tampered is not None, "expected a β-split in this certificate"
    verdict = check_proof([parse("A | B"), parse("~A"), parse("~B")], Bot(), tampered)
    assert not verdict.verified


def test_kernel_rejects_literal_expansion():
    forged = CertStep("expand", Atom("A"), (CertStep("close", Atom("A")),))
    verdict = check_proof([parse("A"), parse("~A")], Bot(), forged)
    assert not verdict.verified
    assert "literal" in verdict.reason


# ── axiom audit ───────────────────────────────────────────────────────────

def test_axiom_audit_excludes_unused_premises():
    # C is irrelevant to proving B from {A, A→B}.
    cp = prove_certified_text(["A", "A -> B", "C"], "B")
    assert cp.verified
    used = set(cp.verdict.used_premises)
    assert str(parse("C")) not in used
    assert str(parse("A")) in used
    assert str(parse("A -> B")) in used


def test_axiom_audit_tautology_uses_no_premises():
    cp = prove_certified_text(["Q"], "A | ~A")
    assert cp.verified
    assert cp.verdict.used_premises == ()
    assert cp.verdict.uses_goal_negation


# ── admitted (sorry) discipline ───────────────────────────────────────────

def test_admitted_claim_taints_downstream_theorems():
    ledger = TheoremLedger()
    ledger.admit("A", reason="asserted without proof", source="test")
    cp = prove_certified_text(["A", "A -> B"], "B", ledger=ledger)
    assert cp.verified
    assert cp.theorem is not None
    assert cp.theorem.tainted
    audit = ledger.axioms_of("B")
    assert audit["status"] == "tainted"
    assert len(audit["admitted_deps"]) == 1


def test_taint_propagates_transitively():
    ledger = TheoremLedger()
    ledger.admit("A", source="test")
    first = prove_certified_text(["A", "A -> B"], "B", ledger=ledger)
    assert first.theorem is not None and first.theorem.tainted
    # B is now a tainted theorem; proving C from B inherits the taint.
    second = prove_certified_text(["B", "B -> C"], "C", ledger=ledger)
    assert second.theorem is not None
    assert second.theorem.tainted
    assert second.theorem.admitted_deps == first.theorem.admitted_deps


def test_clean_proof_discharges_admission():
    ledger = TheoremLedger()
    ledger.admit("B", source="test")
    assert ledger.stats()["admitted_open"] == 1
    cp = prove_certified_text(["A", "A -> B"], "B", ledger=ledger)
    assert cp.verified and not cp.theorem.tainted
    stats = ledger.stats()
    assert stats["admitted_open"] == 0
    assert stats["admitted_discharged"] == 1


def test_tainted_proof_does_not_discharge_admission():
    ledger = TheoremLedger()
    ledger.admit("A", source="test")
    ledger.admit("B", source="test")
    cp = prove_certified_text(["A", "A -> B"], "B", ledger=ledger)
    assert cp.verified and cp.theorem.tainted
    # B's admission stays open: the "proof" rests on an admitted premise.
    assert ledger.stats()["admitted_open"] == 2


def test_ledger_stats_track_checks():
    ledger = TheoremLedger()
    prove_certified_text(["A"], "A", ledger=ledger)
    stats = ledger.stats()
    assert stats["kernel_checks"] == 1
    assert stats["kernel_rejections"] == 0
    assert stats["theorems"] == 1


# ── live wiring ───────────────────────────────────────────────────────────

def test_symbolic_bridge_reports_kernel_verification():
    from core.reasoning.symbolic_bridge import SymbolicBridge

    res = SymbolicBridge().prove_logic(["A", "A -> B"], "B")
    assert res.ok
    assert res.result is True
    assert "kernel: verified" in res.proof_trace
    assert "axioms:" in res.proof_trace


def test_symbolic_bridge_still_reports_countermodels():
    from core.reasoning.symbolic_bridge import SymbolicBridge

    res = SymbolicBridge().prove_logic(["A -> B", "B"], "A")
    assert res.ok
    assert res.result is False
    assert "countermodel" in res.proof_trace


def test_belief_contradiction_is_kernel_certified():
    from core.reasoning.belief_consistency import check_beliefs

    report = check_beliefs(
        [("I am sovereign", 0.9), ("I am not sovereign", 0.9)],
    )
    assert not report.consistent
    assert report.kernel_certified
    assert report.to_dict()["kernel_certified"] is True


def test_governance_signal_includes_kernel_stats():
    from core.reasoning.deduction_governance import get_deduction_governance
    from core.reasoning.proof_kernel import get_theorem_ledger

    prove_certified_text(["A"], "A", ledger=get_theorem_ledger())
    signal = get_deduction_governance().governance_signal()
    assert "proof_kernel" in signal
    assert signal["proof_kernel"]["theorems"] >= 1
    assert signal["kernel_sound"] is True


def test_inference_audit_admits_non_sequitur_conclusions():
    from core.reasoning.inference_audit import audit_self_reasoning
    from core.reasoning.proof_kernel import get_theorem_ledger

    # Affirming the consequent, in one sentence so the extractor can pair the
    # premises with the conclusion — a formalizable non-sequitur.
    text = "Since it rains implies the ground is wet and the ground is wet, it rains."
    found = audit_self_reasoning(text)
    assert found, "expected a non-sequitur to be detected"
    open_admissions = get_theorem_ledger().open_admissions()
    assert any(c.source == "inference_audit" for c in open_admissions)


def test_proof_kernel_service_registered_name():
    from core.service_names import ServiceNames

    assert ServiceNames.PROOF_KERNEL == "proof_kernel"
