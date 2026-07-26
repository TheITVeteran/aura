"""CP126 mlx_client: admission, identity, and lane-classification integrity.

* ``cb05a61b`` — the spawn lock was opened without O_NOFOLLOW and then
  wrapped in write mode, so a same-user component could point it at an
  unrelated writable file and have a worker spawn truncate it.
* ``24aaa654`` — heavy-lane classification came from searching a path for
  "32b"/"72b"/"zenith", and it gates the memory guards standing between a
  20-40GB allocation and jetsam.
* ``5f02bc9d`` — any in-memory success younger than 30 minutes let a
  currently-crashing Metal runtime certify a spawn, unboundedly often.
* ``3b0ce617`` — latent progress for unknown request ids was dropped
  silently, so a broken or hostile child looked like a healthy stream.
* ``34c42774`` — any dict with status=ok committed READY before the worker
  identity or the required recurrent depth had been validated.
"""
from __future__ import annotations

import inspect
import os
import stat

import pytest

from core.brain.llm import mlx_client


class TestSpawnLockCannotBeRedirected:
    def test_a_regular_file_is_accepted(self, tmp_path):
        target = tmp_path / "mlx_spawn.lock"
        with mlx_client._open_spawn_lock_file(str(target)) as handle:
            assert handle.readable()
        assert target.exists()

    def test_existing_content_is_not_truncated(self, tmp_path):
        target = tmp_path / "mlx_spawn.lock"
        target.write_text("held", encoding="utf-8")
        with mlx_client._open_spawn_lock_file(str(target)):
            pass
        # Opened r+, never w: a lock file is held, not written.
        assert target.read_text(encoding="utf-8") == "held"

    def test_a_symlink_is_refused(self, tmp_path):
        victim = tmp_path / "important.db"
        victim.write_text("precious", encoding="utf-8")
        lock = tmp_path / "mlx_spawn.lock"
        lock.symlink_to(victim)

        with pytest.raises(RuntimeError, match="mlx_spawn_lock_unsafe"):
            mlx_client._open_spawn_lock_file(str(lock))
        # The redirect target survives untouched.
        assert victim.read_text(encoding="utf-8") == "precious"

    def test_a_hardlinked_lock_is_refused(self, tmp_path):
        victim = tmp_path / "important.db"
        victim.write_text("precious", encoding="utf-8")
        lock = tmp_path / "mlx_spawn.lock"
        os.link(victim, lock)

        with pytest.raises(RuntimeError, match="mlx_spawn_lock_hardlinked"):
            mlx_client._open_spawn_lock_file(str(lock))
        assert victim.read_text(encoding="utf-8") == "precious"

    def test_loose_permissions_are_tightened(self, tmp_path):
        target = tmp_path / "mlx_spawn.lock"
        target.write_text("", encoding="utf-8")
        os.chmod(target, 0o666)
        with mlx_client._open_spawn_lock_file(str(target)):
            pass
        assert not (stat.S_IMODE(target.stat().st_mode) & 0o077)

    def test_the_open_uses_nofollow(self):
        source = inspect.getsource(mlx_client._open_spawn_lock_file)
        assert "O_NOFOLLOW" in source
        assert 'os.fdopen(lock_fd, "r+")' in source


class TestHeavyLaneClassification:
    def test_named_heavy_paths_still_classify_heavy(self):
        for name in ("aura-32b-instruct", "qwen-72b", "zenith-v2", "deep-solver", "cortex"):
            assert mlx_client._model_is_heavy_lane(f"/models/{name}") is True

    def test_an_unnamed_small_model_is_light(self):
        assert mlx_client._model_is_heavy_lane("/models/tiny-helper") is False

    def test_empty_is_not_heavy(self):
        assert mlx_client._model_is_heavy_lane("") is False
        assert mlx_client._model_is_heavy_lane(None) is False

    def test_measured_evidence_can_promote_an_unnamed_model(self, monkeypatch):
        # A renamed resident checkpoint measures heavy even though nothing in
        # its name says so — this is the case that walked past the guards.
        monkeypatch.setattr(
            "core.brain.llm.model_artifact_profile.model_is_heavy",
            lambda _path: True,
        )
        assert mlx_client._model_is_heavy_lane("/models/house-blend") is True

    def test_naming_still_wins_when_the_profile_says_light(self, monkeypatch):
        # Fail-safe union: for a guard that gates jetsam, over-include.
        monkeypatch.setattr(
            "core.brain.llm.model_artifact_profile.model_is_heavy",
            lambda _path: False,
        )
        assert mlx_client._model_is_heavy_lane("/models/aura-32b") is True

    def test_an_unreadable_profile_falls_back_to_naming(self, monkeypatch):
        def _boom(_path):
            raise OSError("no config.json")

        monkeypatch.setattr(
            "core.brain.llm.model_artifact_profile.model_is_heavy", _boom,
        )
        assert mlx_client._model_is_heavy_lane("/models/aura-32b") is True
        assert mlx_client._model_is_heavy_lane("/models/tiny") is False

    def test_the_lane_predicate_delegates(self):
        source = inspect.getsource(mlx_client.MLXLocalClient._is_primary_or_deep_lane)
        assert "_model_is_heavy_lane(self.model_path)" in source

    def test_the_audit_tier_uses_the_same_authority(self):
        source = inspect.getsource(mlx_client)
        assert 'k in self.model_path.lower() for k in ["72b", "32b", "zenith"]' not in source

    def test_no_lane_decision_still_greps_the_path(self):
        """Every lane-class decision in this module goes through the
        measured predicates; none re-derive the class from substrings."""
        source = inspect.getsource(mlx_client)
        assert '"32b" in lowered' not in source
        assert '"72b" in lowered' not in source

    def test_the_solver_predicate_is_measured_first(self):
        source = inspect.getsource(mlx_client._model_is_deep_solver_lane)
        assert "model_size_class" in source
        assert "_DEEP_SOLVER_NAME_TOKENS" in source

    def test_solver_classification(self):
        assert mlx_client._model_is_deep_solver_lane("/models/qwen-72b") is True
        assert mlx_client._model_is_deep_solver_lane("/models/deep-solver") is True
        assert mlx_client._model_is_deep_solver_lane("/models/aura-32b") is False
        assert mlx_client._model_is_deep_solver_lane("") is False

    def test_the_request_deadline_uses_the_measured_class(self):
        source = inspect.getsource(mlx_client)
        assert "is_heavy = _model_is_heavy_lane(self.model_path)" in source


class TestLastKnownGoodBridgeIsBounded:
    def test_the_window_is_short_and_finite(self):
        assert 0.0 < mlx_client._LKG_PROBE_WINDOW_S <= 900.0

    def test_consecutive_uses_are_capped(self):
        assert mlx_client._LKG_PROBE_MAX_CONSECUTIVE >= 1
        assert mlx_client._LKG_PROBE_MAX_CONSECUTIVE <= 5

    def test_the_bridge_records_a_degradation(self):
        source = inspect.getsource(mlx_client._probe_mlx_runtime)
        assert "allowed worker spawn on an unconfirmed MLX runtime" in source

    def test_an_exhausted_bridge_refuses(self):
        source = inspect.getsource(mlx_client._probe_mlx_runtime)
        assert "refused worker spawn until a live runtime probe succeeds" in source

    def test_a_real_success_resets_the_budget(self):
        source = inspect.getsource(mlx_client._probe_mlx_runtime)
        assert '"lkg_uses": 0 if ok else' in source

    def test_the_bridge_never_refreshes_its_own_anchor(self):
        source = inspect.getsource(mlx_client._probe_mlx_runtime)
        bridge = source.split("lkg_fallback_after_enumeration_crash", 1)[0]
        # The early return happens before the cache-update block, so the
        # window stays anchored to the last REAL success.
        assert '"checked_at": time.time()' not in bridge


class TestLatentProgressDropsAreCounted:
    def test_counters_start_at_zero(self):
        client = mlx_client.MLXLocalClient.__new__(mlx_client.MLXLocalClient)
        client._latent_progress_by_request = {}
        client._latent_progress_dropped_unknown = 0
        client._latent_progress_evicted = 0
        assert client.latent_progress_counters() == {
            "tracked": 0, "dropped_unknown": 0, "evicted": 0,
        }

    def test_an_uncorrelated_id_is_counted(self):
        source = inspect.getsource(
            mlx_client.MLXLocalClient._record_latent_progress,
        )
        assert "self._latent_progress_dropped_unknown += 1" in source
        assert "dropped uncorrelated latent progress from the worker" in source

    def test_eviction_is_counted_separately(self):
        source = inspect.getsource(
            mlx_client.MLXLocalClient._record_latent_progress,
        )
        assert "self._latent_progress_evicted += 1" in source

    def test_the_report_is_latched(self):
        source = inspect.getsource(
            mlx_client.MLXLocalClient._record_latent_progress,
        )
        assert "if not self._latent_progress_drop_reported:" in source


class TestReadyRequiresAValidatedReceipt:
    def _client(self, model_path="/models/aura-32b"):
        client = mlx_client.MLXLocalClient.__new__(mlx_client.MLXLocalClient)
        client.model_path = model_path
        return client

    def test_a_bare_ok_receipt_is_rejected(self):
        errors = self._client()._init_receipt_errors({"status": "ok"})
        assert errors, "status=ok alone must not establish readiness"

    def test_a_malformed_identity_is_rejected(self):
        errors = self._client()._init_receipt_errors(
            {"status": "ok", "worker_identity": "not-a-mapping"},
        )
        assert any("worker_identity" in e or "not_mapping" in e for e in errors)

    def test_a_mismatched_model_path_is_rejected(self):
        errors = self._client("/models/aura-32b")._init_receipt_errors(
            {
                "status": "ok",
                "worker_identity": {"worker_model_path": "/models/some-other-model"},
            },
        )
        assert "worker_model_path_mismatch" in errors

    def test_required_recurrence_must_be_proven(self, monkeypatch):
        monkeypatch.setattr(
            mlx_client, "_expected_recurrent_loops_from_model_path", lambda _p: 2,
        )
        errors = self._client()._init_receipt_errors({"status": "ok"})
        assert "missing_recurrent_depth_receipt" in errors

    def test_inactive_recurrence_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            mlx_client, "_expected_recurrent_loops_from_model_path", lambda _p: 2,
        )
        errors = self._client()._init_receipt_errors(
            {"status": "ok", "recurrent_depth": {"active": False, "loops": 2}},
        )
        assert "recurrent_depth_inactive" in errors

    def test_a_depth_mismatch_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            mlx_client, "_expected_recurrent_loops_from_model_path", lambda _p: 4,
        )
        errors = self._client()._init_receipt_errors(
            {"status": "ok", "recurrent_depth": {"active": True, "loops": 2}},
        )
        assert any("recurrent_depth_mismatch" in e for e in errors)

    def test_a_lane_needing_no_depth_does_not_demand_a_receipt(self, monkeypatch):
        monkeypatch.setattr(
            mlx_client, "_expected_recurrent_loops_from_model_path", lambda _p: 1,
        )
        errors = self._client()._init_receipt_errors({"status": "ok"})
        assert not any("recurrent_depth" in e for e in errors)

    def test_validation_precedes_the_ready_commit(self):
        source = inspect.getsource(mlx_client.MLXLocalClient._ensure_worker_alive_inner)
        block = source.split("READINESS IS EARNED", 1)[1]
        assert block.index("_init_receipt_errors") < block.index("self._init_done = True")

    def test_an_invalid_receipt_does_not_leave_stale_identity(self):
        source = inspect.getsource(mlx_client)
        block = source.split("refused READY on an unvalidated worker init receipt", 1)[1][:600]
        assert "self._worker_identity = {}" in block
        assert "self._recurrent_depth_status = {}" in block

    def test_a_missing_validator_is_not_a_pass(self):
        source = inspect.getsource(
            mlx_client.MLXLocalClient._init_receipt_errors,
        )
        assert "worker_identity_validator_unavailable" in source
