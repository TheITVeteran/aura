"""CP126 promotion_governor — the gate that writes into LIVE SOURCE.

Everything this accepts, the running system becomes, so each test pins one
finding from artifacts/closeout/semantic_review/cp126/.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.architect.config import ASAConfig
from core.architect.errors import PromotionError
from core.architect.models import (
    BehaviorDelta,
    RefactorStep,
    MutationTier,
    ProofReceipt,
    ProofResult,
    PromotionStatus,
    RefactorPlan,
    RollbackPacket,
)
from core.architect.promotion_governor import PromotionGovernor, _contained_target
from core.architect.shadow_workspace import ShadowRun


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "core").mkdir(parents=True)
    (root / "core" / "widget.py").write_text("original\n", encoding="utf-8")
    return root


@pytest.fixture
def governor(repo, tmp_path):
    # ASAConfig is a frozen dataclass: build it, don't mutate it.
    config = ASAConfig(
        repo_root=repo,
        artifact_root=tmp_path / "artifacts",
        max_tier=MutationTier.T3_BEHAVIORAL_IMPROVEMENT,
    )
    return PromotionGovernor(config)


def _plan(files=("core/widget.py",), tier=MutationTier.T1_CLEANUP, eligible=True):
    # changed_files is DERIVED from the steps' target paths, not from
    # affected_files — the gate reasons over the steps.
    steps = tuple(
        RefactorStep(
            id=f"s{index}",
            description="edit",
            operation="rewrite",
            target_path=rel,
        )
        for index, rel in enumerate(files)
    )
    return RefactorPlan(
        id="plan-1",
        objective="tidy",
        risk_tier=tier,
        affected_files=tuple(files),
        affected_symbols=(),
        semantic_surfaces=(),
        steps=steps,
        proof_obligations=(),
        expected_smell_reduction=(),
        expected_behavior_delta="none",
        promotion_eligible=eligible,
    )


def _proof(plan, *, equivalent=True, regressions=(), plan_id=None, tier=None):
    return ProofReceipt(
        run_id="run-1",
        plan_id=plan_id or plan.id,
        tier=tier or plan.risk_tier,
        results=(ProofResult(obligation_id="o1", passed=True, status="PASS"),),
        behavior_delta=BehaviorDelta(
            equivalent=equivalent, improved=False, regressions=tuple(regressions)
        ),
        rollback_packet_hash="rb",
        shadow_artifact_path="",
        decision_hash="dh",
    )


def _rollback(plan, candidate_hashes, *, run_id="run-1", files=None):
    changed = tuple(files if files is not None else plan.changed_files)
    return RollbackPacket(
        run_id=run_id,
        timestamp=0.0,
        repo_root_hash="rh",
        changed_files=changed,
        original_hashes={rel: "orig" for rel in changed},
        candidate_hashes=dict(candidate_hashes),
        packet_path="",
        dry_run_passed=True,
    )


class TestPathContainment:
    """89cfe03a: a plan target may not escape the repository."""

    def test_absolute_and_traversal_paths_are_refused(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        assert _contained_target(root, "/etc/passwd") is None
        assert _contained_target(root, "../outside.py") is None
        assert _contained_target(root, "core/../../escape.py") is None
        assert _contained_target(root, "") is None

    def test_ordinary_relative_paths_resolve(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        assert _contained_target(root, "core/widget.py") == (root / "core" / "widget.py").resolve()

    def test_decide_rejects_escaping_targets(self, governor):
        plan = _plan(files=("../escape.py",))
        candidate_hashes = {"../escape.py": _sha(b"x")}
        decision = governor.decide(plan, _proof(plan), _rollback(plan, candidate_hashes))
        assert decision.status is PromotionStatus.REJECTED
        assert "escape" in decision.reason


class TestTierReclassification:
    """6f938e23: protected paths cannot hide inside an under-tiered plan."""

    def test_ineligible_plan_is_proposal_only(self, governor):
        plan = _plan(eligible=False)
        decision = governor.decide(plan, _proof(plan), _rollback(plan, {"core/widget.py": _sha(b"c")}))
        assert decision.status is PromotionStatus.PROPOSAL_ONLY

    def test_understated_tier_is_rejected(self, governor, monkeypatch):
        plan = _plan(tier=MutationTier.T0_SYNTAX_STYLE)
        monkeypatch.setattr(
            governor.classifier, "classify_path",
            lambda path: MutationTier.T2_REFACTOR,
        )
        decision = governor.decide(plan, _proof(plan), _rollback(plan, {"core/widget.py": _sha(b"c")}))
        assert decision.status is PromotionStatus.REJECTED
        assert "understates" in decision.reason

    def test_protected_path_is_proposal_only(self, governor, monkeypatch):
        plan = _plan(tier=MutationTier.T3_BEHAVIORAL_IMPROVEMENT)
        monkeypatch.setattr(
            governor.classifier, "classify_path",
            lambda path: MutationTier.T4_GOVERNANCE_SENSITIVE,
        )
        decision = governor.decide(plan, _proof(plan), _rollback(plan, {"core/widget.py": _sha(b"c")}))
        assert decision.status is PromotionStatus.PROPOSAL_ONLY


class TestReceiptBindings:
    """f69faa30: a proof and rollback must describe THIS plan and run."""

    def test_proof_for_another_plan_is_rejected(self, governor):
        plan = _plan()
        proof = _proof(plan, plan_id="some-other-plan")
        decision = governor.decide(plan, proof, _rollback(plan, {"core/widget.py": _sha(b"c")}))
        assert decision.status is PromotionStatus.REJECTED
        assert "plan_id" in decision.reason

    def test_rollback_from_another_run_is_rejected(self, governor):
        plan = _plan()
        rollback = _rollback(plan, {"core/widget.py": _sha(b"c")}, run_id="run-999")
        decision = governor.decide(plan, _proof(plan), rollback)
        assert decision.status is PromotionStatus.REJECTED
        assert "run_id" in decision.reason

    def test_tier_mismatch_is_rejected(self, governor):
        plan = _plan(tier=MutationTier.T1_CLEANUP)
        proof = _proof(plan, tier=MutationTier.T3_BEHAVIORAL_IMPROVEMENT)
        decision = governor.decide(plan, proof, _rollback(plan, {"core/widget.py": _sha(b"c")}))
        assert decision.status is PromotionStatus.REJECTED
        assert "tier" in decision.reason

    def test_rollback_file_set_must_match(self, governor):
        plan = _plan()
        rollback = _rollback(plan, {"core/other.py": _sha(b"c")}, files=("core/other.py",))
        decision = governor.decide(plan, _proof(plan), rollback)
        assert decision.status is PromotionStatus.REJECTED
        assert "file set" in decision.reason

    def test_missing_candidate_hash_is_rejected(self, governor):
        plan = _plan()
        decision = governor.decide(plan, _proof(plan), _rollback(plan, {}))
        assert decision.status is PromotionStatus.REJECTED
        assert "candidate hash" in decision.reason


class TestBehaviorRegressionAtEveryTier:
    """3d88c6f7: a T3 plan cannot carry regressions to SHADOW_PASSED."""

    def test_t3_regression_is_rejected(self, governor):
        plan = _plan(tier=MutationTier.T3_BEHAVIORAL_IMPROVEMENT)
        proof = _proof(plan, equivalent=False, regressions=("latency",))
        decision = governor.decide(plan, proof, _rollback(plan, {"core/widget.py": _sha(b"c")}))
        assert decision.status is PromotionStatus.REJECTED
        assert "behavior regression" in decision.reason

    def test_clean_plan_passes(self, governor):
        plan = _plan()
        decision = governor.decide(plan, _proof(plan), _rollback(plan, {"core/widget.py": _sha(b"c")}))
        assert decision.status is PromotionStatus.SHADOW_PASSED


class TestCandidateIntegrityAndAtomicity:
    """cde68e74 + c54936d6: proved bytes only, and all-or-nothing."""

    def _shadow(self, tmp_path, mapping):
        # ShadowRun is frozen: construct it.
        return ShadowRun(
            run_id="run-1",
            shadow_root=str(tmp_path),
            artifact_dir=str(tmp_path),
            plan_id="plan-1",
            changed_files=tuple(mapping),
            candidate_files=dict(mapping),
        )

    def test_edited_candidate_is_refused(self, governor, tmp_path, repo):
        plan = _plan()
        cand = tmp_path / "cand.py"
        cand.write_text("proved\n", encoding="utf-8")
        rollback = _rollback(plan, {"core/widget.py": _sha(b"proved\n")})
        # The snapshot is mutated AFTER proof.
        cand.write_text("tampered\n", encoding="utf-8")
        shadow = self._shadow(tmp_path, {"core/widget.py": str(cand)})
        with pytest.raises(PromotionError, match="proved hash"):
            governor.promote(plan, shadow, _proof(plan), rollback)
        # Live source untouched.
        assert (repo / "core" / "widget.py").read_text(encoding="utf-8") == "original\n"

    def test_partial_failure_rolls_back_the_prefix(self, governor, tmp_path, repo):
        (repo / "core" / "second.py").write_text("second-original\n", encoding="utf-8")
        plan = _plan(files=("core/widget.py", "core/second.py"))
        first = tmp_path / "a.py"
        first.write_text("new-first\n", encoding="utf-8")
        rollback = _rollback(
            plan,
            {
                "core/widget.py": _sha(b"new-first\n"),
                "core/second.py": _sha(b"new-second\n"),
            },
        )
        # The SECOND candidate is missing: staging must fail before any write.
        shadow = self._shadow(tmp_path, {"core/widget.py": str(first)})
        with pytest.raises(PromotionError, match="candidate snapshot missing"):
            governor.promote(plan, shadow, _proof(plan), rollback)
        assert (repo / "core" / "widget.py").read_text(encoding="utf-8") == "original\n"
        assert (repo / "core" / "second.py").read_text(encoding="utf-8") == "second-original\n"

    def test_verified_promotion_writes_live_source(self, governor, tmp_path, repo):
        plan = _plan()
        cand = tmp_path / "cand.py"
        cand.write_text("promoted\n", encoding="utf-8")
        rollback = _rollback(plan, {"core/widget.py": _sha(b"promoted\n")})
        shadow = self._shadow(tmp_path, {"core/widget.py": str(cand)})
        decision = governor.promote(plan, shadow, _proof(plan), rollback)
        assert decision.status is PromotionStatus.PROMOTED
        assert (repo / "core" / "widget.py").read_text(encoding="utf-8") == "promoted\n"
