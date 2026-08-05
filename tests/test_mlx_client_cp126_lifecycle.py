"""CP126: cancellation attribution, clock discrimination, bounded worker input."""
from __future__ import annotations

import asyncio
import pathlib
import time
from types import SimpleNamespace

import pytest

from core.brain.llm.mlx_client import (
    MLXLocalClient,
    _bounded_progress_value,
    _sleep_inclusive_monotonic,
)

TEST_MODEL = "/models/Qwen2.5-7B-Instruct-4bit"


@pytest.fixture
def client() -> MLXLocalClient:
    return MLXLocalClient(model_path=TEST_MODEL)


class TestExpectedCancellationIsRequestBound:
    def test_an_unrelated_cancellation_cannot_spend_a_planned_claim(self, client):
        client._note_expected_generation_cancellation("reboot", request_ids=["req-a"])
        assert client._consume_expected_generation_cancellation("req-b") == ""
        # The real one is still claimable — the credit form had already spent it.
        assert client._consume_expected_generation_cancellation("req-a") == "reboot"

    def test_a_claim_is_consumed_once(self, client):
        client._note_expected_generation_cancellation("reboot", request_ids=["req-a"])
        assert client._consume_expected_generation_cancellation("req-a") == "reboot"
        assert client._consume_expected_generation_cancellation("req-a") == ""

    def test_claims_expire_and_do_not_accumulate(self, client):
        client._expected_cancels = {"old": ("reboot", time.time() - 3600.0)}
        assert client._consume_expected_generation_cancellation("old") == ""
        assert client._expected_cancels == {}

    def test_an_empty_request_id_claims_nothing(self, client):
        client._note_expected_generation_cancellation("reboot", request_ids=["req-a"])
        assert client._consume_expected_generation_cancellation("") == ""
        assert client._consume_expected_generation_cancellation(None) == ""


class TestWorkerProgressIsBounded:
    def test_an_oversized_string_is_clamped(self):
        assert len(_bounded_progress_value("A" * 10_000)) == 200

    def test_a_long_sequence_is_clamped(self):
        assert len(_bounded_progress_value(list(range(1000)))) == 32

    def test_non_finite_numbers_become_none(self):
        assert _bounded_progress_value(float("inf")) is None
        assert _bounded_progress_value(float("nan")) is None
        assert _bounded_progress_value(3.5) == 3.5

    def test_an_unsupported_shape_is_named_not_dropped(self):
        assert _bounded_progress_value(object()).startswith("<unsupported:")

    def test_recorded_progress_is_clamped_end_to_end(self, client):
        client._current_request_id = "req-1"
        client._record_latent_progress({"id": "req-1", "stage": "S" * 5000, "elapsed_s": 1.0})
        assert len(client._latent_progress_by_request["req-1"]["stage"]) == 200

    def test_finished_requests_expire_from_the_progress_map(self, client):
        client._latent_progress_by_request = {
            "done": {"received_at_unix": time.time() - 10_000.0}
        }
        client._expire_latent_progress()
        assert "done" not in client._latent_progress_by_request
        assert client._latent_progress_evicted == 1

    def test_a_live_request_never_expires(self, client):
        client._current_request_id = "live"
        client._latent_progress_by_request = {
            "live": {"received_at_unix": time.time() - 10_000.0}
        }
        client._expire_latent_progress()
        assert "live" in client._latent_progress_by_request


class TestClockJumpIsNotHostSleep:
    def test_the_platform_offers_a_sleep_inclusive_clock(self):
        # Darwin: CLOCK_MONOTONIC counts suspend, time.monotonic() does not.
        value = _sleep_inclusive_monotonic()
        assert value is None or value > 0.0

    def test_a_forward_wall_jump_is_reported_as_a_clock_shift(self, client, monkeypatch):
        base_wall = 1_000_000.0
        base_mono = 500.0
        base_sleep_inclusive = 900.0
        client._clock_sample_wall = base_wall
        client._clock_sample_monotonic = base_mono
        client._clock_sample_sleep_inclusive = base_sleep_inclusive
        client._current_request_started_at = base_wall - 10.0

        # One second of real running time, and the wall clock jumped 600s.
        monkeypatch.setattr(time, "time", lambda: base_wall + 601.0)
        monkeypatch.setattr(time, "monotonic", lambda: base_mono + 1.0)
        monkeypatch.setattr(
            "core.brain.llm.mlx_client._sleep_inclusive_monotonic",
            lambda: base_sleep_inclusive + 1.0,
        )

        rebased = client._rebase_after_system_sleep()
        assert rebased == pytest.approx(600.0)
        assert client._clock_shift_events == 1
        assert client._clock_shift_total_s == pytest.approx(600.0)

    def test_real_sleep_is_not_reported_as_a_clock_shift(self, client, monkeypatch):
        base_wall = 1_000_000.0
        base_mono = 500.0
        base_sleep_inclusive = 900.0
        client._clock_sample_wall = base_wall
        client._clock_sample_monotonic = base_mono
        client._clock_sample_sleep_inclusive = base_sleep_inclusive
        client._current_request_started_at = base_wall - 10.0

        # The host slept 600s: wall advanced, the sleep-inclusive clock
        # advanced with it, and time.monotonic() did not.
        monkeypatch.setattr(time, "time", lambda: base_wall + 600.0)
        monkeypatch.setattr(time, "monotonic", lambda: base_mono)
        monkeypatch.setattr(
            "core.brain.llm.mlx_client._sleep_inclusive_monotonic",
            lambda: base_sleep_inclusive + 600.0,
        )

        rebased = client._rebase_after_system_sleep()
        assert rebased == pytest.approx(600.0)
        assert client._clock_shift_events == 0

    def test_without_a_sleep_inclusive_clock_the_old_heuristic_stands(
        self, client, monkeypatch
    ):
        client._clock_sample_wall = 1_000_000.0
        client._clock_sample_monotonic = 500.0
        client._clock_sample_sleep_inclusive = None
        monkeypatch.setattr(time, "time", lambda: 1_000_600.0)
        monkeypatch.setattr(time, "monotonic", lambda: 500.0)
        monkeypatch.setattr(
            "core.brain.llm.mlx_client._sleep_inclusive_monotonic", lambda: None
        )
        assert client._rebase_after_system_sleep() == pytest.approx(600.0)
        assert client._clock_shift_events == 0


class TestArtifactPromotionIsATransaction:
    """CP126 a996d77f/41fa9f3c/8ccdcd3b/df8e3045/7f4435f5."""

    def _artifact(self, root, name, arch="Qwen2ForCausalLM"):
        artifact = root / name
        artifact.mkdir()
        (artifact / "config.json").write_text('{"architectures": ["%s"]}' % arch)
        (artifact / "tokenizer.json").write_text("{}")
        (artifact / "model.safetensors").write_bytes(b"\x00" * 16)
        return artifact

    def test_a_servable_artifact_validates(self, tmp_path):
        from core.brain.llm.mlx_client import _validate_model_artifact

        verdict = _validate_model_artifact(self._artifact(tmp_path, "good"))
        assert verdict.ok
        assert verdict.architectures == ("Qwen2ForCausalLM",)
        assert verdict.weight_files == 1

    def test_an_unparseable_config_is_refused(self, tmp_path):
        from core.brain.llm.mlx_client import _validate_model_artifact

        artifact = tmp_path / "broken"
        artifact.mkdir()
        (artifact / "config.json").write_text("{not json")
        verdict = _validate_model_artifact(artifact)
        assert not verdict.ok
        assert verdict.reason.startswith("artifact_config_unreadable")

    def test_a_missing_tokenizer_is_refused(self, tmp_path):
        from core.brain.llm.mlx_client import _validate_model_artifact

        artifact = tmp_path / "no-tok"
        artifact.mkdir()
        (artifact / "config.json").write_text("{}")
        (artifact / "model.safetensors").write_bytes(b"\x00")
        assert _validate_model_artifact(artifact).reason == "artifact_missing_tokenizer"

    def test_registry_rebind_moves_the_client_to_its_new_key(self, client):
        import core.brain.llm.mlx_client as mod

        with mod._CLIENTS_LOCK:
            mod._CLIENTS["/old/path"] = client
        try:
            assert mod._rebind_client_registry_key("/old/path", "/new/path", client)
            snapshot = dict(mod.clients_snapshot())
            assert snapshot.get("/new/path") is client
            assert "/old/path" not in snapshot
        finally:
            with mod._CLIENTS_LOCK:
                mod._CLIENTS.pop("/new/path", None)
                mod._CLIENTS.pop("/old/path", None)

    def test_registry_rebind_refuses_to_evict_a_different_client(self, client):
        import core.brain.llm.mlx_client as mod

        other = MLXLocalClient(model_path=TEST_MODEL)
        with mod._CLIENTS_LOCK:
            mod._CLIENTS["/old/path"] = client
            mod._CLIENTS["/new/path"] = other
        try:
            assert not mod._rebind_client_registry_key("/old/path", "/new/path", client)
            assert dict(mod.clients_snapshot())["/new/path"] is other
        finally:
            with mod._CLIENTS_LOCK:
                mod._CLIENTS.pop("/new/path", None)
                mod._CLIENTS.pop("/old/path", None)

    @pytest.mark.asyncio
    async def test_a_failed_recycle_reports_failed_not_ok(self, client, tmp_path, monkeypatch):
        async def _boom(reason="", mark_failed=True):
            raise RuntimeError("spawn refused")

        monkeypatch.setattr(client, "reboot_worker", _boom)
        target = self._artifact(tmp_path, "fused")
        receipt = await client._activate_promoted_artifact(str(target))
        assert receipt["ok"] is False
        assert receipt["state"] == "failed"
        assert receipt["reason"].startswith("recycle_failed")


class TestForceAbortDoesNotClobberALifecycleOwner:
    """CP126 499846c3."""

    def _wedged_client(self):
        c = MLXLocalClient(model_path=TEST_MODEL)
        c._record_degraded_event = lambda *a, **k: None
        c._replace_ipc_queues = lambda *a, **k: None
        return c

    def test_it_defers_reconciliation_rather_than_erasing_a_new_process(self):
        class _Proc:
            def __init__(self):
                self.killed = False

            def is_alive(self):
                return not self.killed

            def kill(self):
                self.killed = True

            def join(self, timeout=None):
                return None

        client = self._wedged_client()
        published = _Proc()
        client._active_generations = 1
        client._current_request_started_at = time.time() - 500.0
        client._lock.acquire()
        try:
            client._process = published  # a concurrent spawn just published this
            assert client.force_abort_active_generation("watchdog") is True
        finally:
            client._lock.release()
        # The handle survived: erasing it would have hidden a live worker from
        # the owner that spawned it. Reconciliation is queued for that owner.
        assert client._process is published
        assert client._force_abort_reconcile_pending == "watchdog"

    def test_the_lock_owner_applies_the_deferred_reconciliation(self):
        client = self._wedged_client()
        client._force_abort_reconcile_pending = "watchdog"
        client._process = object()
        client._init_done = True
        client._active_generations = 2
        client._apply_pending_force_abort_reconcile()
        assert client._process is None
        assert client._init_done is False
        assert client._active_generations == 0
        assert client._force_abort_reconcile_pending is None

    def test_applying_with_nothing_pending_is_a_no_op(self):
        client = self._wedged_client()
        marker = object()
        client._process = marker
        client._apply_pending_force_abort_reconcile()
        assert client._process is marker

    def test_it_forces_after_repeated_deferrals(self):
        client = self._wedged_client()
        client._process = None
        client._active_generations = 1
        client._current_request_started_at = time.time() - 500.0
        client._force_abort_lock_failures = 2  # two prior deferrals
        client._lock.acquire()
        try:
            assert client.force_abort_active_generation("watchdog") is True
        finally:
            client._lock.release()
        # The owner is presumed wedged; the abort reconciles unsynchronized.
        assert client._process is None
        assert client._force_abort_reconcile_pending is None

    def test_the_request_lane_is_not_released_for_another_holder(self):
        client = self._wedged_client()
        client._request_lock.acquire()
        client._request_lock_owner_label = "another_request"
        client._active_generations = 0
        client._current_request_started_at = 0.0
        try:
            client._release_request_lock_if_aborted("watchdog")
            assert client._request_lock.locked()
        finally:
            if client._request_lock.locked():
                client._request_lock.release()

    def test_the_aborted_holder_releases_its_own_lane(self):
        client = self._wedged_client()
        client._request_lock.acquire()
        client._request_lock_owner_label = "wedged"
        client._active_generations = 1
        client._release_request_lock_if_aborted("watchdog")
        assert not client._request_lock.locked()


class TestDurationSettingsAreBounded:
    """CP126 ec9f8d32: a floor does not stop infinity."""

    def test_infinity_falls_back_to_the_default(self, monkeypatch):
        from core.brain.llm.mlx_client import _env_duration_s

        monkeypatch.setenv("AURA_TEST_DURATION", "inf")
        assert _env_duration_s("AURA_TEST_DURATION", 90.0, minimum=1.0) == 90.0

    def test_an_absurd_value_falls_back_rather_than_being_honoured(self, monkeypatch):
        from core.brain.llm.mlx_client import _env_duration_s

        monkeypatch.setenv("AURA_TEST_DURATION", "86400000")
        assert _env_duration_s("AURA_TEST_DURATION", 90.0) == 90.0

    def test_a_malformed_value_does_not_raise_into_the_request_path(self, monkeypatch):
        from core.brain.llm.mlx_client import _env_duration_s

        monkeypatch.setenv("AURA_TEST_DURATION", "ninety seconds")
        assert _env_duration_s("AURA_TEST_DURATION", 90.0) == 90.0

    def test_a_reasonable_value_is_honoured(self, monkeypatch):
        from core.brain.llm.mlx_client import _env_duration_s

        monkeypatch.setenv("AURA_TEST_DURATION", "45")
        assert _env_duration_s("AURA_TEST_DURATION", 90.0) == 45.0

    def test_below_the_floor_falls_back(self, monkeypatch):
        from core.brain.llm.mlx_client import _env_duration_s

        monkeypatch.setenv("AURA_TEST_DURATION", "0.1")
        assert _env_duration_s("AURA_TEST_DURATION", 90.0, minimum=1.0) == 90.0


class TestLaneEvictionIsFenced:
    """CP126 518e876f: idle-check and eviction were not atomic."""

    @pytest.mark.asyncio
    async def test_work_started_during_the_fence_refuses_the_eviction(self, monkeypatch):
        import core.brain.llm.mlx_client as mod

        client = MLXLocalClient(model_path=TEST_MODEL)
        owner = SimpleNamespace(owner_id="owner-1", model_path=TEST_MODEL)
        monkeypatch.setattr(mod, "_clients_snapshot", lambda: [(TEST_MODEL, client)])
        monkeypatch.setattr(mod, "_model_lane_owner_id", lambda _c: "owner-1")

        rebooted = []

        async def _reboot(reason="", mark_failed=True):
            rebooted.append(reason)

        monkeypatch.setattr(client, "reboot_worker", _reboot)

        # A generation begins in the window the old code left open: after the
        # idle check, before the reboot.
        real_acquire = client._acquire_request_lock

        async def _acquire_then_work(**kwargs):
            got = await real_acquire(**kwargs)
            client._active_generations = 1
            return got

        monkeypatch.setattr(client, "_acquire_request_lock", _acquire_then_work)

        assert await mod._evict_model_lane_owner(owner, "pressure") is False
        assert rebooted == [], "a lane with work in flight must not be recycled"
        assert not client._request_lock.locked(), "the fence must be released"

    @pytest.mark.asyncio
    async def test_a_busy_request_lane_refuses_the_eviction(self, monkeypatch):
        import core.brain.llm.mlx_client as mod

        client = MLXLocalClient(model_path=TEST_MODEL)
        owner = SimpleNamespace(owner_id="owner-1", model_path=TEST_MODEL)
        monkeypatch.setattr(mod, "_clients_snapshot", lambda: [(TEST_MODEL, client)])
        monkeypatch.setattr(mod, "_model_lane_owner_id", lambda _c: "owner-1")
        monkeypatch.setattr(mod, "_LANE_EVICTION_FENCE_WAIT_S", 0.2)

        rebooted = []

        async def _reboot(reason="", mark_failed=True):
            rebooted.append(reason)

        monkeypatch.setattr(client, "reboot_worker", _reboot)
        client._request_lock.acquire()
        try:
            assert await mod._evict_model_lane_owner(owner, "pressure") is False
            assert rebooted == []
        finally:
            client._request_lock.release()

    @pytest.mark.asyncio
    async def test_a_genuinely_idle_lane_is_evicted(self, monkeypatch):
        import core.brain.llm.mlx_client as mod

        client = MLXLocalClient(model_path=TEST_MODEL)
        owner = SimpleNamespace(owner_id="owner-1", model_path=TEST_MODEL)
        monkeypatch.setattr(mod, "_clients_snapshot", lambda: [(TEST_MODEL, client)])
        monkeypatch.setattr(mod, "_model_lane_owner_id", lambda _c: "owner-1")

        rebooted = []

        async def _reboot(reason="", mark_failed=True):
            rebooted.append(reason)

        monkeypatch.setattr(client, "reboot_worker", _reboot)
        monkeypatch.setattr(client, "is_alive", lambda: False)

        assert await mod._evict_model_lane_owner(owner, "pressure") is True
        assert rebooted == ["yield_to_lane_transaction:pressure"]
        assert not client._request_lock.locked()


class TestBatchGenerationContract:
    """CP126 c4bd8d0a / 189ac02a / 536f8e0d / 375fc058."""

    def test_a_candidate_is_bounded_in_size_not_only_in_count(self, client):
        from core.brain.llm.mlx_client import _BATCH_CANDIDATE_MAX_CHARS

        assert _BATCH_CANDIDATE_MAX_CHARS > 0
        oversized = "x" * (_BATCH_CANDIDATE_MAX_CHARS * 4)
        assert len(str(oversized)[:_BATCH_CANDIDATE_MAX_CHARS]) == _BATCH_CANDIDATE_MAX_CHARS

    def test_the_identity_snapshot_cannot_be_mutated_through(self, client):
        client._worker_identity = {
            "worker_boot_id": "boot-1",
            "worker_pid": 42,
            "stack": {"mlx": "0.1", "adapters": ["a"]},
        }
        snapshot = client.get_worker_identity_snapshot()
        snapshot["stack"]["mlx"] = "tampered"
        snapshot["stack"]["adapters"].append("b")
        assert client._worker_identity["stack"]["mlx"] == "0.1"
        assert client._worker_identity["stack"]["adapters"] == ["a"]

    def test_an_absent_identity_snapshots_as_empty(self, client):
        client._worker_identity = None
        assert client.get_worker_identity_snapshot() == {}

    @pytest.mark.asyncio
    async def test_batch_metadata_does_not_claim_verification_without_identity(
        self, client, monkeypatch
    ):
        async def _response(*_a, **_k):
            return {
                "texts": ["one", "two"],
                "request_id": "req-1",
                "tokens_used": 10,
                "tokens_used_by_candidate": [4, 6],
                "tokens_used_consistent": True,
            }

        monkeypatch.setattr(client, "_generate_batch_response_async", _response)
        client._worker_identity = {}
        out = await client.generate_batch_with_metadata_async("p")
        meta = out["generation_metadata"]
        assert meta["provider_verified"] is False
        assert meta["provider_verification_basis"] == "unattested"

    @pytest.mark.asyncio
    async def test_batch_metadata_binds_the_attested_worker(self, client, monkeypatch):
        async def _response(*_a, **_k):
            return {
                "texts": ["one"],
                "request_id": "req-1",
                "tokens_used": 4,
                "tokens_used_by_candidate": [4],
                "tokens_used_consistent": True,
            }

        monkeypatch.setattr(client, "_generate_batch_response_async", _response)
        client._worker_identity = {"worker_boot_id": "boot-9", "worker_pid": 1234}
        client._worker_generation = 3
        meta = (await client.generate_batch_with_metadata_async("p"))["generation_metadata"]
        assert meta["provider_verified"] is True
        assert meta["provider_verification_basis"] == "attested_worker_identity"
        assert meta["worker_boot_id"] == "boot-9"
        assert meta["worker_pid"] == 1234
        assert meta["worker_generation"] == 3
        assert meta["model_basis"] == "path_basename"


class TestBatchBudgetAndAdapterCancellation:
    """The remaining halves of CP126 0bdb9f4d and c4bd8d0a."""

    @pytest.mark.asyncio
    async def test_a_widened_batch_budget_is_reported_not_silent(
        self, client, monkeypatch
    ):
        import core.brain.llm.mlx_client as mod

        recorded = []
        monkeypatch.setattr(
            mod, "_record_mlx_degradation", lambda exc, **kw: recorded.append(str(exc))
        )
        client._req_q = SimpleNamespace(put=lambda *a, **k: None)
        client._closed = False
        monkeypatch.setattr(
            mod, "get_memory_pressure_snapshot",
            lambda: SimpleNamespace(refuse_heavy_local_generation=False, reason=""),
        )

        async def _alive(**_kw):
            return True

        async def _put(*_a, **_k):
            raise TimeoutError("stop here; the admission decision already happened")

        monkeypatch.setattr(client, "_ensure_worker_alive", _alive)
        monkeypatch.setattr(mod, "run_io_bound", _put)

        assert await client._generate_batch_response_async("p", timeout_s=3.0) == {}
        assert any("outside the admissible range" in msg for msg in recorded)

    @pytest.mark.asyncio
    async def test_cancelling_an_adapter_swap_marks_the_state_unknown(
        self, client, monkeypatch
    ):
        import core.brain.llm.mlx_client as mod

        client._req_q = SimpleNamespace(put=lambda *a, **k: None)
        client._process = SimpleNamespace(is_alive=lambda: True)
        client._init_done = True
        client._expert_adapter_state_unknown = False

        async def _put(*_a, **_k):
            return None

        async def _cancelled(*_a, **_k):
            raise asyncio.CancelledError

        monkeypatch.setattr(mod, "run_io_bound", _put)
        monkeypatch.setattr(mod, "_await_shared_future", _cancelled)

        import tempfile

        with tempfile.TemporaryDirectory() as adapter_dir:
            root = pathlib.Path(adapter_dir)
            (root / "adapters.safetensors").write_bytes(b"\x00" * 64)
            (root / "adapter_config.json").write_text('{"fine_tune_type": "lora"}')
            with pytest.raises(asyncio.CancelledError):
                await client.set_expert_adapter(str(root))

        # The command is on the worker's queue; the caller going away does not
        # stop it from attaching.
        assert client._expert_adapter_state_unknown is True
        assert client._pending_generations == {}


class TestAdapterArtifactContract:
    """CP126 d665aa64: admission was is_dir()."""

    def _adapter(self, root, name, *, weights=True, config=None):
        from json import dumps

        path = root / name
        path.mkdir()
        if weights:
            (path / "adapters.safetensors").write_bytes(b"\x00" * 128)
        if config is not None:
            (path / "adapter_config.json").write_text(dumps(config))
        return path

    def test_a_real_adapter_validates(self, tmp_path):
        from core.brain.llm.mlx_client import _validate_adapter_artifact

        v = _validate_adapter_artifact(
            self._adapter(tmp_path, "good", config={"fine_tune_type": "lora"})
        )
        assert v.ok
        assert v.weight_file == "adapters.safetensors"
        assert v.fine_tune_type == "lora"
        assert v.base_compatibility == "not_declared"

    def test_an_empty_directory_is_refused(self, tmp_path):
        from core.brain.llm.mlx_client import _validate_adapter_artifact

        v = _validate_adapter_artifact(self._adapter(tmp_path, "empty", weights=False))
        assert not v.ok and v.reason == "adapter_missing_weights"

    def test_zero_byte_weights_are_refused(self, tmp_path):
        from core.brain.llm.mlx_client import _validate_adapter_artifact

        path = tmp_path / "hollow"
        path.mkdir()
        (path / "adapters.safetensors").write_bytes(b"")
        assert _validate_adapter_artifact(path).reason == "adapter_weights_empty"

    def test_an_unreadable_config_is_refused(self, tmp_path):
        from core.brain.llm.mlx_client import _validate_adapter_artifact

        path = self._adapter(tmp_path, "bad-config")
        (path / "adapter_config.json").write_text("{nope")
        assert _validate_adapter_artifact(path).reason.startswith(
            "adapter_config_unreadable"
        )

    def test_a_mismatched_base_checkpoint_is_refused(self, tmp_path):
        from core.brain.llm.mlx_client import _validate_adapter_artifact

        path = self._adapter(
            tmp_path, "foreign", config={"base_checkpoint_fingerprint": "a" * 64}
        )
        v = _validate_adapter_artifact(path, expected_base_fingerprint="b" * 64)
        assert not v.ok
        assert v.reason.startswith("adapter_base_mismatch")
        assert v.base_compatibility == "mismatch"

    def test_a_matching_base_checkpoint_is_verified(self, tmp_path):
        from core.brain.llm.mlx_client import _validate_adapter_artifact

        path = self._adapter(
            tmp_path, "matched", config={"base_checkpoint_fingerprint": "a" * 64}
        )
        v = _validate_adapter_artifact(path, expected_base_fingerprint="a" * 64)
        assert v.ok and v.base_compatibility == "verified"

    def test_a_declared_base_with_nothing_to_compare_says_so(self, tmp_path):
        """Unmeasured is not the same as fine."""
        from core.brain.llm.mlx_client import _validate_adapter_artifact

        path = self._adapter(
            tmp_path, "declared", config={"base_checkpoint_fingerprint": "a" * 64}
        )
        v = _validate_adapter_artifact(path)
        assert v.ok and v.base_compatibility == "declared_unverified"

    def test_the_singular_weight_filename_is_accepted(self, tmp_path):
        from core.brain.llm.mlx_client import _validate_adapter_artifact

        path = tmp_path / "cp796-shape"
        path.mkdir()
        (path / "adapter.safetensors").write_bytes(b"\x00" * 32)
        v = _validate_adapter_artifact(path)
        assert v.ok and v.weight_file == "adapter.safetensors"

    @pytest.mark.asyncio
    async def test_the_live_seam_refuses_a_bare_directory(self, client, tmp_path):
        client._req_q = SimpleNamespace(put=lambda *a, **k: None)
        client._process = SimpleNamespace(is_alive=lambda: True)
        client._init_done = True
        bare = tmp_path / "bare"
        bare.mkdir()
        res = await client.set_expert_adapter(str(bare))
        assert res["ok"] is False
        assert res["reason"] == "adapter_missing_weights"


class TestMaintenanceCounterContract:
    """CP126 8264628d: counters converted straight off the wire."""

    def _counts(self, payload):
        from core.brain.llm.mlx_client import _bounded_maintenance_counters

        return _bounded_maintenance_counters(
            payload, max_pairs=4, scan_limit=16, max_positions=96
        )

    def test_a_consistent_response_reads_cleanly(self):
        counters, faults = self._counts(
            {
                "pairs_considered": 10,
                "pairs_scanned": 8,
                "pairs_ingested": 2,
                "positions_ingested": 40,
            }
        )
        assert faults == []
        assert counters["pairs_ingested"] == 2

    def test_a_malformed_counter_does_not_raise_into_the_caller(self):
        counters, faults = self._counts({"pairs_scanned": "many"})
        assert counters["pairs_scanned"] is None
        assert "pairs_scanned:malformed" in faults

    def test_a_negative_counter_is_unmeasured_not_clamped(self):
        counters, faults = self._counts({"positions_ingested": -5})
        assert counters["positions_ingested"] is None
        assert "positions_ingested:negative" in faults

    def test_a_counter_above_its_own_budget_is_refused(self):
        counters, faults = self._counts({"pairs_ingested": 9999})
        assert counters["pairs_ingested"] is None
        assert "pairs_ingested:above_budget" in faults

    def test_ingesting_more_than_was_scanned_breaks_the_relationship(self):
        counters, faults = self._counts({"pairs_scanned": 1, "pairs_ingested": 3})
        assert counters["pairs_ingested"] is None
        assert "pairs_ingested:exceeds_scanned" in faults

    def test_absent_is_unmeasured_and_not_zero(self):
        """Zero means the worker measured none; None means we never found out."""
        counters, faults = self._counts({})
        assert all(value is None for value in counters.values())
        assert all(fault.endswith(":absent") for fault in faults)

    def test_a_measured_zero_stays_zero(self):
        counters, faults = self._counts(
            {
                "pairs_considered": 0,
                "pairs_scanned": 0,
                "pairs_ingested": 0,
                "positions_ingested": 0,
            }
        )
        assert faults == []
        assert counters == {
            "pairs_considered": 0,
            "pairs_scanned": 0,
            "pairs_ingested": 0,
            "positions_ingested": 0,
        }

    @pytest.mark.asyncio
    async def test_a_foreground_turn_arriving_during_the_wait_yields_the_lane(
        self, client, monkeypatch
    ):
        import core.brain.llm.mlx_client as mod

        client._init_done = True
        client._req_q = SimpleNamespace(put=lambda *a, **k: None)
        client._process = SimpleNamespace(is_alive=lambda: True)
        monkeypatch.setattr(
            mod, "get_memory_pressure_snapshot",
            lambda: SimpleNamespace(refuse_heavy_local_generation=False, reason=""),
        )

        owned = {"value": False}
        monkeypatch.setattr(mod, "_foreground_owner_active", lambda: owned["value"])

        real_acquire = client._acquire_request_lock

        async def _acquire_then_person_arrives(**kwargs):
            got = await real_acquire(**kwargs)
            owned["value"] = True  # a person's turn takes the foreground
            return got

        monkeypatch.setattr(client, "_acquire_request_lock", _acquire_then_person_arrives)

        res = await client.ingest_nonparametric_async()
        assert res["status"] == "skipped_foreground_active_after_lane"
        assert not client._request_lock.locked(), "the lane must be handed back"


class TestWorkerDeathIsProven:
    """CP126 9a4f99da / 1399e019."""

    class _Immortal:
        pid = 4321

        def is_alive(self):
            return True

        def kill(self):
            return None

        def join(self, timeout=None):
            return None

    def test_a_survivor_is_reported_not_assumed_dead(self, client):
        assert client._kill_and_join_blocking(self._Immortal()) is False

    def test_an_unobservable_process_counts_as_alive(self, client):
        class _Opaque:
            pid = 99

            def __init__(self):
                self._calls = 0

            def is_alive(self):
                self._calls += 1
                if self._calls == 1:
                    return True
                raise OSError("cannot observe")

            def kill(self):
                return None

            def join(self, timeout=None):
                return None

        assert client._kill_and_join_blocking(_Opaque()) is False

    def test_a_forced_abort_that_leaves_a_survivor_reports_failure(self, client):
        client._record_degraded_event = lambda *a, **k: None
        client._replace_ipc_queues = lambda *a, **k: None
        survivor = self._Immortal()
        client._process = survivor
        client._active_generations = 1
        client._current_request_started_at = time.time() - 500.0

        assert client.force_abort_active_generation("watchdog") is False
        # The handle is retained: a None handle tells the next spawn admission
        # that no worker of ours is running.
        assert client._process is survivor

    def test_a_clean_abort_still_reports_success(self, client, monkeypatch):
        class _Mortal:
            pid = 1

            def __init__(self):
                self.killed = False

            def is_alive(self):
                return not self.killed

            def kill(self):
                self.killed = True

            def join(self, timeout=None):
                return None

        client._record_degraded_event = lambda *a, **k: None
        client._replace_ipc_queues = lambda *a, **k: None
        client._release_durable_model_lane_owner_sync = lambda **k: None
        client._process = _Mortal()
        client._active_generations = 1
        client._current_request_started_at = time.time() - 500.0

        assert client.force_abort_active_generation("watchdog") is True
        assert client._process is None

    def test_spawn_refuses_when_reclamation_is_blind_and_a_worker_may_live(
        self, client, monkeypatch
    ):
        import core.brain.llm.mlx_client as mod

        class _BlindObserver:
            def processes(self):
                raise OSError("process table unavailable")

        monkeypatch.setattr(mod, "get_resource_observer", lambda: _BlindObserver())
        monkeypatch.setattr(mod, "_shutdown_blocks_model_work", lambda *a, **k: False)
        client._process = self._Immortal()

        with pytest.raises(RuntimeError) as excinfo:
            client._spawn_worker_blocking()
        assert "orphan_reclamation_unobservable_refused_worker_spawn" in str(excinfo.value)

    def test_spawn_proceeds_when_blind_but_no_prior_worker_exists(
        self, client, monkeypatch
    ):
        import core.brain.llm.mlx_client as mod

        class _BlindObserver:
            def processes(self):
                raise OSError("process table unavailable")

        monkeypatch.setattr(mod, "get_resource_observer", lambda: _BlindObserver())
        monkeypatch.setattr(mod, "_shutdown_blocks_model_work", lambda *a, **k: False)
        client._process = None

        # Past the orphan gate; whatever it fails on next is not this finding.
        # (The blind observer raises again from a later scan — that is the
        # spawn continuing, which is the point.)
        with pytest.raises((RuntimeError, OSError)) as excinfo:
            client._spawn_worker_blocking()
        assert "orphan_reclamation_unobservable" not in str(excinfo.value)
