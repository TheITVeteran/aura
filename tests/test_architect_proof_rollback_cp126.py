"""CP126 architect — proofs must measure, rollback must be safe and atomic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.architect.config import ASAConfig
from core.architect.errors import RollbackError
from core.architect.models import (
    MutationTier,
    RefactorPlan,
    RefactorStep,
    RollbackPacket,
)
from core.architect.proof_obligations import ProofVerifier
from core.architect.rollback_manager import (
    ABSENT_SENTINEL,
    RollbackManager,
    _contained,
    compute_receipt_hash,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "a.py").write_text("original-a\n", encoding="utf-8")
    return root


@pytest.fixture
def manager(repo, tmp_path):
    return RollbackManager(
        ASAConfig(repo_root=repo, artifact_root=tmp_path / "artifacts")
    )


def _packet(manager, repo, *, original="original-a\n", candidate="candidate-a\n"):
    """A hand-built packet with a correct receipt, laid out on disk."""
    run_id = "run-1"
    packet_dir = manager.root / run_id
    (packet_dir / "original" / "pkg").mkdir(parents=True, exist_ok=True)
    if original is not None:
        (packet_dir / "original" / "pkg" / "a.py").write_text(original, encoding="utf-8")
    originals = {"pkg/a.py": _sha(original.encode()) if original is not None else ABSENT_SENTINEL}
    candidates = {"pkg/a.py": _sha(candidate.encode())}
    repo_hash = "repohash"
    return RollbackPacket(
        run_id=run_id,
        timestamp=0.0,
        repo_root_hash=repo_hash,
        changed_files=("pkg/a.py",),
        original_hashes=originals,
        candidate_hashes=candidates,
        packet_path=str(packet_dir),
        receipt_hash=compute_receipt_hash(run_id, repo_hash, originals, candidates),
        dry_run_passed=True,
    )


class TestPathContainment:
    """dd364ff8: plan paths and packet roots are contained."""

    def test_traversal_and_absolute_refused(self, tmp_path):
        for bad in ("../escape.py", "/etc/passwd", "pkg/../../out.py"):
            with pytest.raises(RollbackError, match="escapes its root"):
                _contained(tmp_path, bad)

    def test_empty_refused(self, tmp_path):
        with pytest.raises(RollbackError, match="empty"):
            _contained(tmp_path, "")

    def test_ordinary_path_resolves(self, tmp_path):
        assert _contained(tmp_path, "pkg/a.py") == (tmp_path / "pkg" / "a.py").resolve()


class TestPacketIdentityIsReverified:
    """f4bc655d: a loaded packet is untrusted until its receipt recomputes."""

    def test_tampered_hashes_are_rejected(self, manager, repo):
        packet = _packet(manager, repo)
        tampered = RollbackPacket(
            **{**packet.__dict__, "original_hashes": {"pkg/a.py": _sha(b"forged")}}
        )
        with pytest.raises(RollbackError, match="receipt does not match"):
            manager.restore(tampered, require_candidate_generation=False)

    def test_redirected_packet_path_is_rejected(self, manager, repo, tmp_path):
        packet = _packet(manager, repo)
        elsewhere = RollbackPacket(**{**packet.__dict__, "packet_path": str(tmp_path / "elsewhere")})
        with pytest.raises(RollbackError, match="packet_path does not match"):
            manager.restore(elsewhere, require_candidate_generation=False)


class TestInterveningWorkIsProtected:
    """2bcfa2c3: restore must not silently replace newer work."""

    def test_restore_refuses_when_live_is_not_the_promoted_generation(self, manager, repo):
        packet = _packet(manager, repo)
        (repo / "pkg" / "a.py").write_text("someone else's edit\n", encoding="utf-8")
        with pytest.raises(RollbackError, match="intervening work"):
            manager.restore(packet)
        assert (repo / "pkg" / "a.py").read_text(encoding="utf-8") == "someone else's edit\n"

    def test_force_revert_is_explicit(self, manager, repo):
        packet = _packet(manager, repo)
        (repo / "pkg" / "a.py").write_text("broken\n", encoding="utf-8")
        restored = manager.restore(packet, require_candidate_generation=False)
        assert restored.post_restore_verified is True
        assert (repo / "pkg" / "a.py").read_text(encoding="utf-8") == "original-a\n"

    def test_matching_promoted_generation_restores_without_force(self, manager, repo):
        packet = _packet(manager, repo)
        (repo / "pkg" / "a.py").write_text("candidate-a\n", encoding="utf-8")
        restored = manager.restore(packet)
        assert restored.post_restore_verified is True


class TestCreatedFilesAreRollbackable:
    """78c68165: a plan that CREATES a file gets truthful coverage."""

    def test_absent_original_is_deleted_on_restore(self, manager, repo):
        created = repo / "pkg" / "new.py"
        created.write_text("candidate-a\n", encoding="utf-8")
        run_id = "run-1"
        packet_dir = manager.root / run_id
        packet_dir.mkdir(parents=True, exist_ok=True)
        originals = {"pkg/new.py": ABSENT_SENTINEL}
        candidates = {"pkg/new.py": _sha(b"candidate-a\n")}
        packet = RollbackPacket(
            run_id=run_id,
            timestamp=0.0,
            repo_root_hash="rh",
            changed_files=("pkg/new.py",),
            original_hashes=originals,
            candidate_hashes=candidates,
            packet_path=str(packet_dir),
            receipt_hash=compute_receipt_hash(run_id, "rh", originals, candidates),
            dry_run_passed=True,
        )
        manager.restore(packet)
        assert not created.exists(), "a created file was not removed by rollback"


class TestProofsMeasureRatherThanAssert:
    """17be716a + a8434f1a: wording and key-presence are not proofs."""

    def _plan(self, *, tier, metadata=None, delta="improved throughput"):
        step = RefactorStep(
            id="s1",
            description="d",
            operation="rewrite",
            target_path="pkg/a.py",
            metadata=metadata or {},
        )
        return RefactorPlan(
            id="p1",
            objective="o",
            risk_tier=tier,
            affected_files=("pkg/a.py",),
            affected_symbols=(),
            semantic_surfaces=(),
            steps=(step,),
            proof_obligations=("unused_import_static_proof",),
            expected_smell_reduction=(),
            expected_behavior_delta=delta,
            promotion_eligible=True,
        )

    def test_t3_wording_alone_is_unproven(self):
        plan = self._plan(tier=MutationTier.T3_BEHAVIORAL_IMPROVEMENT)
        result = ProofVerifier._tier3_improvement_proof(plan)
        assert result.passed is False
        assert result.status.startswith("unproven")

    def test_t3_measured_target_passes(self):
        plan = self._plan(
            tier=MutationTier.T3_BEHAVIORAL_IMPROVEMENT,
            metadata={"improvement_target": {"metric": "p95_ms", "baseline": 120, "goal": 90}},
        )
        result = ProofVerifier._tier3_improvement_proof(plan)
        assert result.passed is True

    def test_t3_zero_effect_size_is_unproven(self):
        plan = self._plan(
            tier=MutationTier.T3_BEHAVIORAL_IMPROVEMENT,
            metadata={"improvement_target": {"metric": "p95_ms", "baseline": 90, "goal": 90}},
        )
        assert ProofVerifier._tier3_improvement_proof(plan).status == "unproven_no_effect_size"

    def test_t3_incomplete_target_is_unproven(self):
        plan = self._plan(
            tier=MutationTier.T3_BEHAVIORAL_IMPROVEMENT,
            metadata={"improvement_target": {"metric": "", "baseline": "fast", "goal": "faster"}},
        )
        assert ProofVerifier._tier3_improvement_proof(plan).passed is False


class TestFingerprintDerivedObligations:
    """da90a3bf: compatibility and bypass counts come from measurement."""

    class _FP:
        def __init__(self, changed=(), bypasses=0):
            self.changed_public_apis = changed
            self.protected_bypass_count = bypasses

    def test_changed_public_api_fails_compatibility(self):
        results = ProofVerifier._fingerprint_derived_obligations(
            self._FP(), self._FP(changed=("pkg.mod.gone",)), None
        )
        api = next(r for r in results if r.obligation_id == "t2_public_api_compatibility")
        assert api.passed is False
        assert api.evidence["changed_public_apis"] == ["pkg.mod.gone"]

    def test_new_bypass_fails(self):
        results = ProofVerifier._fingerprint_derived_obligations(
            self._FP(bypasses=1), self._FP(bypasses=3), None
        )
        bypass = next(r for r in results if r.obligation_id == "t2_no_new_bypasses")
        assert bypass.passed is False
        assert bypass.evidence == {"before": 1, "after": 3}

    def test_clean_fingerprints_pass(self):
        results = ProofVerifier._fingerprint_derived_obligations(
            self._FP(bypasses=2), self._FP(bypasses=2), None
        )
        assert all(r.passed for r in results)
