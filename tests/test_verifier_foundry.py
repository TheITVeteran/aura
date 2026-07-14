"""Verifier Foundry: reliability measurement and the self-training gate.

The contract under test (frontier-general arc P1): trust in verifiers is
EARNED — pessimistic statistics over graded verdicts; admission to the
self-training loop requires evidence; seeds are revocable; the hard gate is
never softened by weighting; and the ledger is tamper-evident.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.verifiers.base import VerificationResult, combine_results
from core.brain.verifiers.foundry import (
    SEED_ADMITTED_DOMAINS,
    VerifierFoundry,
    wilson_lower_bound,
    wilson_upper_bound,
)

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self, t: float = 1_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def foundry(tmp_path):
    f = VerifierFoundry(root=tmp_path / "foundry", clock=FakeClock())
    yield f
    f.close()


def _feed(foundry, *, verifier="probe", domain="factual", n=60,
          correct_rate=1.0, verdict_pass=True):
    """Record n verdicts and grade them with the given correctness rate."""
    wrong = round(n * (1.0 - correct_rate))
    for i in range(n):
        vid = foundry.record_verdict(verifier=verifier, domain=domain,
                                     hard_pass=verdict_pass, score=0.9,
                                     checked=True)
        truth = verdict_pass if i >= wrong else (not verdict_pass)
        foundry.grade_verdict(vid, truth_pass=truth, source="test_audit")


# ─────────────────────────────────────────────────────────────────────────────
# Pessimistic statistics
# ─────────────────────────────────────────────────────────────────────────────

def test_wilson_bounds_are_pessimistic_and_sane():
    assert wilson_lower_bound(0, 0) == 0.0          # no evidence → no trust
    assert wilson_upper_bound(0, 0) == 1.0
    assert wilson_lower_bound(10, 10) < 1.0          # perfect-but-small < 1
    assert wilson_lower_bound(100, 100) > wilson_lower_bound(10, 10)
    assert 0.6 < wilson_lower_bound(90, 100) < 0.9   # 90% with margin
    assert wilson_upper_bound(0, 100) < 0.05         # 0/100 failures stays low


def test_unchecked_verdicts_are_not_evidence(foundry):
    vid = foundry.record_verdict(verifier="x", domain="factual",
                                 hard_pass=True, score=0.9, checked=False)
    assert vid == ""
    assert foundry.reliability("x", "factual").recorded == 0


def test_grading_updates_accuracy_brier_and_false_pass(foundry):
    _feed(foundry, n=100, correct_rate=0.9)
    cell = foundry.reliability("probe", "factual")
    assert cell.graded == 100
    assert cell.correct == 90
    assert cell.false_passes == 10           # said pass, truth was fail
    assert 0.8 < cell.accuracy_lb() < 0.9    # pessimistic but close
    assert cell.brier() < 0.2


def test_grade_unknown_verdict_is_rejected(foundry):
    assert foundry.grade_verdict("vd-nonexistent", truth_pass=True,
                                 source="test") is False


# ─────────────────────────────────────────────────────────────────────────────
# The admission gate
# ─────────────────────────────────────────────────────────────────────────────

def test_unknown_domain_needs_evidence(foundry):
    decision = foundry.domain_admitted("poetry_quality")
    assert not decision.admitted
    assert decision.reason == "insufficient_evidence"


def test_domain_earns_admission_with_clean_evidence(foundry):
    _feed(foundry, domain="factual", n=80, correct_rate=1.0)
    decision = foundry.domain_admitted("factual")
    assert decision.admitted
    assert decision.reason == "earned_by_evidence"
    assert decision.evidence["graded"] == 80


def test_accuracy_below_threshold_blocks_admission(foundry):
    _feed(foundry, domain="factual", n=80, correct_rate=0.7)
    decision = foundry.domain_admitted("factual")
    assert not decision.admitted
    assert decision.reason in ("accuracy_below_threshold",
                               "false_pass_rate_too_high")


def test_false_passes_specifically_block_admission(foundry):
    # 92% accurate overall but every error is a FALSE PASS — the poison type
    _feed(foundry, domain="factual", n=100, correct_rate=0.92,
          verdict_pass=True)
    decision = foundry.domain_admitted("factual")
    assert not decision.admitted


def test_weakest_verifier_bounds_the_domain(foundry):
    # one excellent engine and one leaky engine in the same domain: the
    # domain is only as trustworthy as its leakiest used checker
    _feed(foundry, verifier="good", domain="factual", n=80, correct_rate=1.0)
    _feed(foundry, verifier="leaky", domain="factual", n=80, correct_rate=0.75)
    decision = foundry.domain_admitted("factual")
    assert not decision.admitted


def test_seed_domains_are_admitted_from_birth(foundry):
    for domain in SEED_ADMITTED_DOMAINS:
        decision = foundry.domain_admitted(domain)
        assert decision.admitted
        assert decision.reason == "seed_admitted"


def test_seed_admission_is_revocable_by_evidence(foundry):
    # a "deterministic" checker turning out leaky loses its birthright
    _feed(foundry, verifier="code_engine", domain="code", n=80,
          correct_rate=0.6, verdict_pass=True)
    decision = foundry.domain_admitted("code")
    assert not decision.admitted
    assert decision.reason == "seed_revoked_by_evidence"
    # and the revocation is durable state, not a transient verdict
    assert not foundry.domain_admitted("code").admitted


# ─────────────────────────────────────────────────────────────────────────────
# Folding weights: measured reliability, hard gate untouched
# ─────────────────────────────────────────────────────────────────────────────

def test_unmeasured_verifiers_keep_neutral_weight(foundry):
    assert foundry.weight_for("brand_new", "factual") == 1.0


def test_measured_bad_verifier_is_downweighted_but_floored(foundry):
    _feed(foundry, verifier="leaky", domain="factual", n=100, correct_rate=0.5)
    w = foundry.weight_for("leaky", "factual")
    assert 0.25 <= w < 0.5


def test_weighted_fold_shifts_soft_score_only():
    good = VerificationResult(domain="factual", ok=True, checked=True,
                              score=0.9, engine="good")
    leaky = VerificationResult(domain="factual", ok=True, checked=True,
                               score=0.2, engine="leaky")
    plain = combine_results("factual", [good, leaky])
    weighted = combine_results("factual", [good, leaky],
                               weights={"good": 1.0, "leaky": 0.25})
    assert plain.score == pytest.approx(0.55, abs=0.01)
    assert weighted.score > plain.score      # leaky pessimism counts less
    assert weighted.ok is plain.ok is True


def test_hard_gate_is_never_weighted():
    passing = VerificationResult(domain="code", ok=True, checked=True,
                                 score=0.95, engine="good")
    failing = VerificationResult(domain="code", ok=False, checked=True,
                                 score=0.4, engine="leaky")
    # even a fully muted engine's PROVEN failure remains final
    weighted = combine_results("code", [passing, failing],
                               weights={"good": 1.0, "leaky": 0.0})
    assert weighted.ok is False


# ─────────────────────────────────────────────────────────────────────────────
# Ledger: durability and tamper evidence
# ─────────────────────────────────────────────────────────────────────────────

def test_state_survives_restart(tmp_path):
    clock = FakeClock()
    f1 = VerifierFoundry(root=tmp_path / "foundry", clock=clock)
    _feed(f1, domain="factual", n=60, correct_rate=1.0)
    f1.close()

    f2 = VerifierFoundry(root=tmp_path / "foundry", clock=clock)
    try:
        cell = f2.reliability("probe", "factual")
        assert cell.graded == 60
        assert f2.domain_admitted("factual").admitted
    finally:
        f2.close()


def test_ledger_verifies_clean_and_detects_tampering(foundry):
    _feed(foundry, n=20, correct_rate=1.0)
    ok, problems = foundry.verify_ledger()
    assert ok, problems

    lines = foundry.events_path.read_text().splitlines()
    doctored = [line.replace('"truth_pass": true', '"truth_pass": false', 1)
                if '"grade"' in line and '"truth_pass": true' in line else line
                for line in lines]
    foundry.events_path.write_text("\n".join(doctored) + "\n")
    ok, problems = foundry.verify_ledger()
    assert not ok


# ─────────────────────────────────────────────────────────────────────────────
# Registry integration: verdicts recorded, folding weighted
# ─────────────────────────────────────────────────────────────────────────────

class _StubEngine:
    def __init__(self, name, ok=True, score=0.8, checked=True):
        self.name = name
        self.domains = ("factual",)
        self._r = (ok, score, checked)

    def handles(self, task_type):
        return task_type in self.domains

    async def verify(self, candidate, *, context=None):
        ok, score, checked = self._r
        return VerificationResult(domain="factual", ok=ok, checked=checked,
                                  score=score, engine=self.name)


def test_registry_records_verdicts_into_foundry(tmp_path, monkeypatch):
    from core.brain.verifiers.registry import VerifierRegistry

    foundry = VerifierFoundry(root=tmp_path / "foundry", clock=FakeClock())
    try:
        monkeypatch.setattr(VerifierRegistry, "_foundry",
                            staticmethod(lambda: foundry))
        registry = VerifierRegistry(verifiers=[_StubEngine("alpha"),
                                               _StubEngine("beta", score=0.6)])
        result = asyncio.run(registry.verify("the sky is blue",
                                             task_type="factual"))
        assert result.checked
        verdicts = result.detail.get("foundry_verdicts", {})
        assert set(verdicts) == {"alpha", "beta"}
        assert foundry.reliability("alpha", "factual").recorded == 1
        # grading through the returned ids closes the loop
        assert foundry.grade_verdict(verdicts["alpha"], truth_pass=True,
                                     source="test")
        assert foundry.reliability("alpha", "factual").graded == 1
    finally:
        foundry.close()


def test_registry_without_foundry_behaves_as_before(monkeypatch):
    from core.brain.verifiers.registry import VerifierRegistry

    monkeypatch.setattr(VerifierRegistry, "_foundry",
                        staticmethod(lambda: None))
    registry = VerifierRegistry(verifiers=[_StubEngine("alpha")])
    result = asyncio.run(registry.verify("x", task_type="factual"))
    assert result.checked
    assert "foundry_verdicts" not in result.detail


# ─────────────────────────────────────────────────────────────────────────────
# The training gate (the causal consumer)
# ─────────────────────────────────────────────────────────────────────────────

def test_record_win_blocked_for_unadmitted_domain(tmp_path, monkeypatch):
    import core.brain.reasoning_self_improvement as rsi_mod

    foundry = VerifierFoundry(root=tmp_path / "foundry", clock=FakeClock())
    try:
        monkeypatch.setattr(
            "core.runtime.service_access.optional_service",
            lambda name, default=None: foundry if name == "verifier_foundry"
            else default,
        )
        monkeypatch.setenv("AURA_REASONING_SELF_IMPROVEMENT", "1")
        rsi = rsi_mod.ReasoningSelfImprovement(
            path=tmp_path / "traces.json",
            cacheable_task_types=frozenset({"math", "factual"}),
        )

        # math is seed-admitted → captured
        assert rsi.record_win("2+2?", "math", answer="4", confidence=0.9,
                              mode="deep", verified=True) is True
        # factual has no evidence yet → the gate refuses training capture
        assert rsi.record_win("capital of France?", "factual",
                              answer="Paris", confidence=0.9,
                              mode="deep", verified=True) is False
        assert rsi._stats["unadmitted"] == 1

        # after clean evidence accumulates, factual earns its way in
        _feed(foundry, domain="factual", n=80, correct_rate=1.0)
        assert rsi.record_win("capital of France?", "factual",
                              answer="Paris", confidence=0.9,
                              mode="deep", verified=True) is True
    finally:
        foundry.close()


def test_record_win_legacy_behavior_without_foundry(tmp_path, monkeypatch):
    import core.brain.reasoning_self_improvement as rsi_mod

    monkeypatch.setattr(
        "core.runtime.service_access.optional_service",
        lambda name, default=None: default,
    )
    monkeypatch.setenv("AURA_REASONING_SELF_IMPROVEMENT", "1")
    rsi = rsi_mod.ReasoningSelfImprovement(
        path=tmp_path / "traces.json",
        cacheable_task_types=frozenset({"factual"}),
    )
    assert rsi.record_win("capital of France?", "factual", answer="Paris",
                          confidence=0.9, mode="deep", verified=True) is True
