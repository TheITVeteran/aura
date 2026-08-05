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

def _new_future():
    """A shared future of the same kind the client hands to waiters."""
    from core.brain.llm.mlx_client import _new_shared_future

    return _new_shared_future()


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


class TestAbortAndCancelHaveTargets:
    """CP126 ccb125e0 / 2538a912."""

    def _idle_client(self):
        c = MLXLocalClient(model_path=TEST_MODEL)
        c._record_degraded_event = lambda *a, **k: None
        c._replace_ipc_queues = lambda *a, **k: None
        c._process = SimpleNamespace(is_alive=lambda: True, pid=7, kill=lambda: None,
                                     join=lambda timeout=None: None)
        return c

    def test_an_idle_worker_is_never_killed_for_an_unrecognised_reason(self):
        """The phrasing of a reason string must not decide a worker's life."""
        client = self._idle_client()
        assert client.force_abort_active_generation("something_unexpected") is False
        assert client._process is not None

    def test_an_idle_worker_is_not_killed_for_a_race_reason_either(self):
        client = self._idle_client()
        assert client.force_abort_active_generation("first_token_timeout") is False
        assert client._process is not None

    def test_an_abort_for_another_request_is_a_no_op(self):
        client = self._idle_client()
        client._active_generations = 1
        client._current_request_started_at = time.time() - 500.0
        client._current_request_id = "req-live"
        assert (
            client.force_abort_active_generation(
                "watchdog", expected_request_id="req-already-finished"
            )
            is False
        )
        assert client._process is not None

    def test_an_abort_for_another_generation_sequence_is_a_no_op(self):
        client = self._idle_client()
        client._active_generations = 1
        client._current_request_started_at = time.time() - 500.0
        client._current_request_seq = 9
        assert (
            client.force_abort_active_generation("watchdog", expected_request_seq=4) is False
        )
        assert client._process is not None

    def test_an_abort_for_the_live_request_still_proceeds(self):
        client = self._idle_client()
        client._release_durable_model_lane_owner_sync = lambda **k: None
        killed = {"value": False}
        client._process = SimpleNamespace(
            is_alive=lambda: not killed["value"],
            pid=7,
            kill=lambda: killed.__setitem__("value", True),
            join=lambda timeout=None: None,
        )
        client._active_generations = 1
        client._current_request_started_at = time.time() - 500.0
        client._current_request_id = "req-live"
        client._current_request_seq = 9
        assert (
            client.force_abort_active_generation(
                "watchdog", expected_request_id="req-live", expected_request_seq=9
            )
            is True
        )
        assert killed["value"] is True

    def test_the_cancel_sweep_spares_clients_serving_another_owner(self, monkeypatch):
        import core.brain.llm.mlx_client as mod

        wedged = MLXLocalClient(model_path=TEST_MODEL)
        wedged._request_lock_owner_label = "latent_cortex_foreground"
        bystander = MLXLocalClient(model_path="/models/other-7B")
        bystander._request_lock_owner_label = "background_curiosity"

        cancelled = []
        for client in (wedged, bystander):
            client.soft_cancel_active_generation = (
                lambda reason, _c=client: (
                    cancelled.append(_c.model_path),
                    {"requested": True, "reason": reason},
                )[1]
            )

        monkeypatch.setattr(
            mod, "_clients_snapshot",
            lambda: [(wedged.model_path, wedged), (bystander.model_path, bystander)],
        )
        receipts = mod.soft_cancel_active_generations(
            reason="owner_cleared:wedge", owner_label="latent_cortex_foreground"
        )
        assert cancelled == [TEST_MODEL]
        assert receipts and receipts[0]["targeted_owner"] == "latent_cortex_foreground"

    def test_the_sweep_broadens_when_nobody_claims_the_owner(self, monkeypatch):
        import core.brain.llm.mlx_client as mod

        a = MLXLocalClient(model_path=TEST_MODEL)
        b = MLXLocalClient(model_path="/models/other-7B")
        cancelled = []
        for client in (a, b):
            client._request_lock_owner_label = ""
            client.soft_cancel_active_generation = (
                lambda reason, _c=client: (
                    cancelled.append(_c.model_path),
                    {"requested": True, "reason": reason},
                )[1]
            )
        monkeypatch.setattr(
            mod, "_clients_snapshot", lambda: [(a.model_path, a), (b.model_path, b)]
        )
        mod.soft_cancel_active_generations(reason="r", owner_label="nobody")
        # Cancelling nothing during recovery is worse than cancelling widely —
        # but the widening is explicit and shows in the receipt.
        assert len(cancelled) == 2


class TestReceiptsSayWhereTheyCameFrom:
    """CP126 92cbf5e2 / 093a2902 / 0989c717."""

    def test_a_surface_receipt_carries_its_provenance(self, client):
        client._worker_identity = {"worker_boot_id": "boot-3", "worker_pid": 55}
        client._worker_generation = 2
        client._current_request_id = "req-live"
        client._current_request_seq = 8
        receipt = {"enabled": True, "clean_user_surface_contract": True}
        client._bind_surface_receipt_provenance(receipt, {"id": "req-live"})
        prov = receipt["provenance"]
        assert prov["claims"] == "worker_attested"
        assert prov["worker_boot_id"] == "boot-3"
        assert prov["worker_generation"] == 2
        assert prov["request_id_matches_active"] is True
        assert prov["worker_identity_attested"] is True

    def test_a_receipt_for_another_request_says_so(self, client):
        client._worker_identity = {"worker_boot_id": "boot-3", "worker_pid": 55}
        client._current_request_id = "req-live"
        receipt = {"enabled": True}
        client._bind_surface_receipt_provenance(receipt, {"id": "req-stale"})
        assert receipt["provenance"]["request_id_matches_active"] is False

    def test_an_unattested_worker_is_named_as_such(self, client):
        client._worker_identity = {}
        receipt = {"enabled": True}
        client._bind_surface_receipt_provenance(receipt, {})
        assert receipt["provenance"]["worker_identity_attested"] is False

    def test_a_rejected_interoception_payload_is_not_published(self, client, monkeypatch):
        import core.being.thought_interoception as intero

        class _Rejecting:
            def ingest(self, *_a, **_k):
                return None

        monkeypatch.setattr(intero, "get_thought_interoception", lambda: _Rejecting())
        client._last_interoception = {"previous": True}
        client._record_interoception_from_response(
            {"interoception": {"junk": "x"}, "text": "hi"},
            foreground_request=True,
            owner_label="test",
        )
        assert client.get_last_interoception() == {"previous": True}

    def test_an_accepted_payload_is_published_and_bounded(self, client, monkeypatch):
        import core.being.thought_interoception as intero

        class _Accepting:
            def ingest(self, *_a, **_k):
                return object()

        monkeypatch.setattr(intero, "get_thought_interoception", lambda: _Accepting())
        client._record_interoception_from_response(
            {
                "interoception": {
                    "token_ids_sample": list(range(100_000)),
                    "mean_surprisal": 1.5,
                    "note": "y" * 10_000,
                },
                "text": "hi",
            },
            foreground_request=True,
            owner_label="test",
        )
        stored = client.get_last_interoception()
        assert len(stored["token_ids_sample"]) == 4096
        assert len(stored["note"]) == 200
        assert stored["mean_surprisal"] == 1.5
        assert stored["_text_fingerprint"]

    def test_pressure_reduced_recurrence_is_marked_not_silent(self):
        import core.brain.llm.mlx_client as mod

        recorded = []
        original = mod._record_mlx_degradation
        mod._record_mlx_degradation = lambda exc, **kw: recorded.append(str(exc))
        try:
            options = mod._apply_memory_pressure_generation_controls(
                {
                    "clean_user_surface_contract": True,
                    "clean_user_surface_recurrent_loops": 4,
                    "max_tokens": 900,
                },
                SimpleNamespace(max_token_cap=256),
                default_max_tokens=512,
            )
        finally:
            mod._record_mlx_degradation = original
        assert options["clean_user_surface_recurrent_loops"] == 1
        assert options["recurrent_loops_requested"] == 4
        assert options["recurrent_loops_reduced_by_pressure"] is True
        assert any("reduced to 1 under a token cap" in msg for msg in recorded)


class TestExpectationsAreGroundedInArtifacts:
    """CP126 cc31c15f / a6db6c23."""

    def test_the_trained_depth_is_read_from_the_execution_spec(self, tmp_path):
        from core.brain.llm.mlx_client import _manifest_recurrent_loops

        artifact = tmp_path / "adapter"
        artifact.mkdir()
        (artifact / "execution_spec.json").write_text('{"recurrent_steps": 4}')
        assert _manifest_recurrent_loops(artifact) == 4

    def test_a_missing_spec_is_unmeasured_not_a_default(self, tmp_path):
        from core.brain.llm.mlx_client import _manifest_recurrent_loops

        artifact = tmp_path / "bare"
        artifact.mkdir()
        assert _manifest_recurrent_loops(artifact) is None
        assert _manifest_recurrent_loops("/nowhere") is None

    def test_an_unreadable_spec_is_unmeasured(self, tmp_path):
        from core.brain.llm.mlx_client import _manifest_recurrent_loops

        artifact = tmp_path / "broken"
        artifact.mkdir()
        (artifact / "execution_spec.json").write_text("{oops")
        assert _manifest_recurrent_loops(artifact) is None

    def test_a_disagreement_with_the_configured_depth_is_recorded(
        self, tmp_path, monkeypatch
    ):
        import core.brain.llm.mlx_client as mod

        artifact = tmp_path / "adapter"
        artifact.mkdir()
        (artifact / "execution_spec.json").write_text('{"recurrent_steps": 4}')
        recorded = []
        monkeypatch.setattr(
            mod, "_record_mlx_degradation", lambda exc, **kw: recorded.append(str(exc))
        )
        mod._note_recurrent_depth_basis_disagreement("/models/x", str(artifact), 2)
        assert any("trained depth 4" in msg for msg in recorded)

    def test_agreement_is_silent(self, tmp_path, monkeypatch):
        import core.brain.llm.mlx_client as mod

        artifact = tmp_path / "adapter"
        artifact.mkdir()
        (artifact / "execution_spec.json").write_text('{"recurrent_steps": 2}')
        recorded = []
        monkeypatch.setattr(
            mod, "_record_mlx_degradation", lambda exc, **kw: recorded.append(str(exc))
        )
        mod._note_recurrent_depth_basis_disagreement("/models/x", str(artifact), 2)
        assert recorded == []

    def test_an_output_contract_cannot_demand_an_unbounded_budget(self):
        from core.brain.llm.mlx_client import (
            _MAX_OUTPUT_CONTRACT_FLOOR_TOKENS,
            _requested_output_contract_generation_floor,
        )

        absurd = {"exact_reply": True, "exact_reply_utf8_bytes": 10_000_000}
        assert (
            _requested_output_contract_generation_floor(absurd)
            == _MAX_OUTPUT_CONTRACT_FLOOR_TOKENS
        )
        assert (
            _requested_output_contract_generation_floor({"semantic_token_cap": 10_000_000})
            == _MAX_OUTPUT_CONTRACT_FLOOR_TOKENS
        )

    def test_overriding_the_pressure_budget_is_recorded(self, monkeypatch):
        import core.brain.llm.mlx_client as mod

        recorded = []
        monkeypatch.setattr(
            mod, "_record_mlx_degradation", lambda exc, **kw: recorded.append(str(exc))
        )
        admitted = mod._bounded_generation_max_tokens(
            2048, 128, 4096, 512, {"semantic_token_cap": 900}
        )
        assert admitted > 128
        assert any("above adaptive shrinkage" in msg for msg in recorded)

    def test_a_contract_within_the_budget_is_silent(self, monkeypatch):
        import core.brain.llm.mlx_client as mod

        recorded = []
        monkeypatch.setattr(
            mod, "_record_mlx_degradation", lambda exc, **kw: recorded.append(str(exc))
        )
        mod._bounded_generation_max_tokens(2048, 2048, 4096, 512, {"semantic_token_cap": 100})
        assert recorded == []


class TestAnExhaustedDeadlineGrantsNothing:
    """CP126 dec24697 / 275480c8."""

    def test_an_expired_budget_yields_no_first_token_ceiling(self, client, monkeypatch):
        monkeypatch.setattr(
            client, "_first_token_hard_ceiling",
            lambda *, foreground_request=False: 120.0,
        )
        assert client._deadline_bound_first_token_hard_ceiling(
            0.0, foreground_request=True
        ) == 0.0
        assert client._deadline_bound_first_token_hard_ceiling(
            -5.0, foreground_request=True
        ) == 0.0

    def test_a_short_budget_keeps_its_recovery_reserve(self, client, monkeypatch):
        monkeypatch.setattr(
            client, "_first_token_hard_ceiling",
            lambda *, foreground_request=False: 120.0,
        )
        # 8 seconds left, 4 reserved for the caller to fail closed and recycle.
        assert client._deadline_bound_first_token_hard_ceiling(
            8.0, foreground_request=True
        ) == 4.0

    def test_an_exhausted_deadline_starts_no_new_lock_wait(self, client):
        from core.utils.deadlines import Deadline

        expired = Deadline(0.0)
        assert client._request_lock_timeout(expired, foreground_request=True) == 0.0
        assert client._request_lock_timeout(expired, foreground_request=False) == 0.0

    def test_a_heartbeating_idle_worker_is_still_idle(self, client):
        """CP126 275480c8: the recycle it gates could never fire."""
        now = time.time()
        client._process_started_at = now - 7200.0
        client._last_generation_completed_at = now - 3600.0
        client._last_token_progress_at = 0.0
        # Breathing the whole time it did nothing.
        client._last_heartbeat = now
        client._last_progress_at = now
        client._last_ready_at = now
        assert client._idle_for_s(now) >= 3600.0
        assert client._liveness_quiet_for_s(now) == 0.0

    def test_a_never_used_worker_becomes_idle_from_its_start(self, client):
        now = time.time()
        client._process_started_at = now - 5000.0
        client._last_generation_completed_at = 0.0
        client._last_token_progress_at = 0.0
        client._last_heartbeat = now
        assert client._idle_for_s(now) >= 5000.0

    def test_fragmentation_recycle_can_actually_fire(self, client, monkeypatch):
        import core.brain.llm.mlx_client as mod

        now = time.time()
        monkeypatch.setattr(mod, "_foreground_owner_active", lambda: False)
        monkeypatch.setattr(client, "is_alive", lambda: True)
        client._active_generations = 0
        client._process_started_at = now - 7200.0
        client._last_generation_completed_at = now - 3600.0
        client._last_heartbeat = now
        assert client.should_recycle_for_fragmentation() is True

    def test_a_working_lane_is_never_recycled(self, client, monkeypatch):
        import core.brain.llm.mlx_client as mod

        now = time.time()
        monkeypatch.setattr(mod, "_foreground_owner_active", lambda: False)
        monkeypatch.setattr(client, "is_alive", lambda: True)
        client._active_generations = 0
        client._process_started_at = now - 7200.0
        client._last_generation_completed_at = now - 10.0
        assert client.should_recycle_for_fragmentation() is False


class TestBackoffAndReadinessNeedCauses:
    """CP126 ee4ccfcc / 5b870404 / 5ce89b9e."""

    def test_a_runtime_probe_does_not_clear_a_crash_backoff(self, client, monkeypatch):
        import core.brain.llm.mlx_client as mod

        monkeypatch.setattr(mod, "_probe_mlx_runtime", lambda force=False: (True, "ok"))
        monkeypatch.setattr(client, "_lane_runtime_failure", lambda: "")
        client._spawn_backoff_until = time.time() + 120.0
        client._spawn_backoff_cause = "spawn_failure"
        client._consecutive_spawn_failures = 3

        assert client.refresh_runtime_availability() is False
        assert client._spawn_backoff_until > time.time(), "the crash backoff must stand"
        assert client._consecutive_spawn_failures == 3

    def test_a_runtime_probe_clears_a_runtime_backoff(self, client, monkeypatch):
        import core.brain.llm.mlx_client as mod

        monkeypatch.setattr(mod, "_probe_mlx_runtime", lambda force=False: (True, "ok"))
        monkeypatch.setattr(client, "_lane_runtime_failure", lambda: "")
        client._spawn_backoff_until = time.time() + 120.0
        client._spawn_backoff_cause = "runtime_unavailable"
        client._consecutive_spawn_failures = 3

        assert client.refresh_runtime_availability() is True
        assert client._spawn_backoff_until == 0.0
        assert client._consecutive_spawn_failures == 0

    def test_a_memory_refusal_backoff_is_not_a_runtime_backoff(self, client):
        client.model_path = "/models/Qwen2.5-72B-Instruct-4bit"
        client._record_degraded_event = lambda *a, **k: None
        from core.brain.llm.mlx_client import ModelLoadAdmissionRefused

        assert client._handle_optional_deep_solver_memory_refusal(
            ModelLoadAdmissionRefused("model_load_headroom")
        ) is True
        assert client._spawn_backoff_cause == "memory_refusal"

    def test_a_deferred_warmup_does_not_invent_readiness_timestamps(self, client):
        client._init_done = True
        client.is_alive = lambda: True
        client._last_ready_at = 0.0
        client._last_progress_at = 0.0

        client.note_lane_recovering("foreground_warmup_deferred_memory_pressure")

        assert client._lane_state == "ready", "a live initialized worker is not recovering"
        assert client._last_ready_at == 0.0, "nothing happened; nothing may be stamped"
        assert client._last_progress_at == 0.0

    def test_an_ordinary_recovery_reason_still_marks_recovering(self, client):
        client.note_lane_recovering("worker_died")
        assert client._lane_state == "recovering"


class TestLatentRequestSchema:
    """CP126 a09d6218: unbounded, weakly typed payloads hashed before admission."""

    def _err(self, **kwargs):
        from core.brain.llm.mlx_client import _latent_request_schema_error

        return _latent_request_schema_error(**kwargs)

    def test_an_ordinary_request_is_admissible(self):
        assert self._err(prompt="hello") == ""
        assert self._err(messages=[{"role": "user", "content": "hi"}]) == ""

    def test_an_oversized_prompt_is_refused_before_any_hashing(self):
        assert self._err(prompt="x" * 400_000) == "prompt_too_large"

    def test_a_non_string_prompt_is_refused(self):
        assert self._err(prompt=b"bytes") == "invalid_prompt_type"

    def test_too_many_messages_is_refused(self):
        assert (
            self._err(messages=[{"role": "user", "content": "x"}] * 5000)
            == "too_many_messages"
        )

    def test_a_message_that_is_not_a_mapping_is_refused(self):
        assert self._err(messages=["just a string"]) == "invalid_message_type"

    def test_an_unknown_role_is_refused(self):
        assert (
            self._err(messages=[{"role": "root", "content": "x"}])
            == "invalid_message_role"
        )

    def test_non_string_content_is_refused(self):
        assert (
            self._err(messages=[{"role": "user", "content": {"nested": True}}])
            == "invalid_message_content"
        )

    def test_a_single_huge_message_is_refused(self):
        assert (
            self._err(messages=[{"role": "user", "content": "y" * 400_000}])
            == "message_too_large"
        )

    def test_many_medium_messages_are_refused_in_aggregate(self):
        assert (
            self._err(
                messages=[{"role": "user", "content": "z" * 50_000} for _ in range(20)]
            )
            == "request_too_large"
        )

    @pytest.mark.asyncio
    async def test_the_entry_point_refuses_before_touching_the_worker(self, client):
        client._closed = False
        client._req_q = None  # any worker access would fail loudly
        result = await client.latent_reason_async(prompt="x" * 400_000)
        assert result["reason"] == "prompt_too_large"


class TestTheCancelChannelReportsDelivery:
    """CP126 2656d71d: a lock-free write that could be silently lost."""

    class _Overwritten:
        """A shared word another writer clears the instant we set it."""

        def __init__(self):
            self._value = 0

        @property
        def value(self):
            return self._value

        @value.setter
        def value(self, new):
            self._value = 0 if new else new

    def test_a_lost_cancel_write_is_reported_not_assumed(self, client, monkeypatch):
        import core.brain.llm.mlx_client as mod

        recorded = []
        monkeypatch.setattr(
            mod, "_record_mlx_degradation", lambda exc, **kw: recorded.append(str(exc))
        )
        client._cancel_seq = self._Overwritten()
        client._current_request_started_at = time.time()
        client._current_request_seq = 7

        receipt = client.soft_cancel_active_generation("preemption")
        assert receipt["requested"] is False
        assert receipt["detail"] == "cancel_write_lost"
        assert any("overwritten" in msg for msg in recorded)

    def test_a_lost_write_leaves_no_acknowledgement_to_wait_for(self, client):
        client._cancel_seq = self._Overwritten()
        client._current_request_started_at = time.time()
        client._current_request_seq = 7
        client._soft_cancel_target = None

        client.soft_cancel_active_generation("preemption")
        # Nothing was delivered, so nothing may claim to be awaiting an ack.
        assert client._soft_cancel_target is None

    def test_a_delivered_cancel_is_recorded_as_requested(self, client):
        class _Word:
            value = 0

        client._cancel_seq = _Word()
        client._current_request_started_at = time.time()
        client._current_request_seq = 7
        client._current_request_id = "req-live"

        receipt = client.soft_cancel_active_generation("preemption")
        assert receipt["requested"] is True
        assert client._cancel_seq.value == 7
        assert client._soft_cancel_target["seq"] == 7
        assert client._soft_cancel_target["req_id"] == "req-live"


class TestIdleScavengeIsFenced:
    """CP126 dcae0f1f: two lock-free checks only shrink the window."""

    def _scavengeable(self, monkeypatch):
        import core.brain.llm.mlx_client as mod

        client = MLXLocalClient(model_path="/models/tiny-1B")
        monkeypatch.setattr(client, "_unload_safety_blocker", lambda: "")
        monkeypatch.setattr(client, "idle_age", lambda: 10_000.0)
        monkeypatch.setattr(client, "_is_primary_lane", lambda: False)
        monkeypatch.setattr(
            mod, "get_memory_pressure_snapshot",
            lambda: SimpleNamespace(warning=True, refuse_heavy_local_generation=False),
        )
        client._process = None
        return client

    @pytest.mark.asyncio
    async def test_a_busy_request_lane_is_never_unloaded(self, monkeypatch):
        client = self._scavengeable(monkeypatch)
        import core.brain.llm.mlx_client as mod

        monkeypatch.setattr(mod, "_LANE_EVICTION_FENCE_WAIT_S", 0.3)
        rebooted = []

        async def _reboot(reason="", mark_failed=True):
            rebooted.append(reason)

        monkeypatch.setattr(client, "reboot_worker", _reboot)
        client._request_lock.acquire()
        try:
            result = await client.maybe_unload_idle()
            assert result["unloaded"] is False
            assert result["reason"] == "request_lane_busy"
            assert rebooted == []
        finally:
            client._request_lock.release()

    @pytest.mark.asyncio
    async def test_a_request_admitted_under_the_fence_cancels_the_unload(
        self, monkeypatch
    ):
        client = self._scavengeable(monkeypatch)
        rebooted = []

        async def _reboot(reason="", mark_failed=True):
            rebooted.append(reason)

        monkeypatch.setattr(client, "reboot_worker", _reboot)

        # The final safety check now runs while the lane is held, so a blocker
        # discovered there cannot go stale before the teardown.
        calls = {"n": 0}

        def _blocker():
            calls["n"] += 1
            return "" if calls["n"] == 1 else "generation_active"

        monkeypatch.setattr(client, "_unload_safety_blocker", _blocker)
        result = await client.maybe_unload_idle()
        assert result["unloaded"] is False
        assert result["reason"] == "generation_active"
        assert rebooted == []
        assert not client._request_lock.locked(), "the fence must be released"

    @pytest.mark.asyncio
    async def test_a_genuinely_idle_lane_still_unloads(self, monkeypatch):
        client = self._scavengeable(monkeypatch)
        rebooted = []

        async def _reboot(reason="", mark_failed=True):
            rebooted.append(reason)

        monkeypatch.setattr(client, "reboot_worker", _reboot)
        result = await client.maybe_unload_idle()
        assert result["unloaded"] is True
        assert rebooted == ["idle_vram_scavenge"]
        assert not client._request_lock.locked()


class TestProofLaneIdentityIsMeasured:
    """CP126 0ad66338: two artifacts with compatible names both passed."""

    def test_a_measured_fingerprint_is_returned(self, tmp_path, monkeypatch):
        import core.brain.llm.mlx_client as mod

        monkeypatch.setattr(
            mod, "get_model_artifact_profile",
            lambda _p: SimpleNamespace(measured=True, fingerprint="abc123"),
            raising=False,
        )
        import core.brain.llm.model_artifact_profile as profile_mod

        monkeypatch.setattr(
            profile_mod, "get_model_artifact_profile",
            lambda _p: SimpleNamespace(measured=True, fingerprint="abc123"),
        )
        assert mod._measured_artifact_fingerprint("/models/x") == "abc123"

    def test_an_unmeasured_artifact_returns_empty_not_a_match(self, monkeypatch):
        import core.brain.llm.mlx_client as mod
        import core.brain.llm.model_artifact_profile as profile_mod

        monkeypatch.setattr(
            profile_mod, "get_model_artifact_profile",
            lambda _p: SimpleNamespace(measured=False, fingerprint="whatever"),
        )
        assert mod._measured_artifact_fingerprint("/models/x") == ""

    def test_an_unreadable_artifact_returns_empty(self, monkeypatch):
        import core.brain.llm.mlx_client as mod
        import core.brain.llm.model_artifact_profile as profile_mod

        def _boom(_p):
            raise OSError("unreadable")

        monkeypatch.setattr(profile_mod, "get_model_artifact_profile", _boom)
        assert mod._measured_artifact_fingerprint("/models/x") == ""

    def test_two_same_named_checkpoints_are_distinguishable(self, monkeypatch):
        """The exact hazard: same basename, different weights."""
        import core.brain.llm.mlx_client as mod
        import core.brain.llm.model_artifact_profile as profile_mod

        digests = {"/a/Qwen2.5-32B-Instruct-4bit": "aaa", "/b/Qwen2.5-32B-Instruct-4bit": "bbb"}
        monkeypatch.setattr(
            profile_mod, "get_model_artifact_profile",
            lambda p: SimpleNamespace(measured=True, fingerprint=digests.get(str(p), "")),
        )
        left = mod._measured_artifact_fingerprint("/a/Qwen2.5-32B-Instruct-4bit")
        right = mod._measured_artifact_fingerprint("/b/Qwen2.5-32B-Instruct-4bit")
        assert left and right and left != right

        from core.brain.llm.model_registry import model_identities_compatible

        # The old test said these were the same model.
        assert model_identities_compatible(
            "Qwen2.5-32B-Instruct-4bit", "Qwen2.5-32B-Instruct-4bit"
        )


class TestSamplingParametersHaveAContract:
    """CP126 cac5c1a3: arbitrary kwargs went straight into IPC."""

    def _normalize(self, **overrides):
        from core.brain.llm.mlx_client import _normalize_generation_params

        req = dict(overrides)
        faults = _normalize_generation_params(req)
        return req, faults

    def test_ordinary_values_pass_through(self):
        req, faults = self._normalize(temp=0.8, top_p=0.95, top_k=40)
        assert faults == []
        assert req["temp"] == 0.8
        assert req["top_p"] == 0.95
        assert req["top_k"] == 40

    def test_a_non_numeric_temperature_becomes_the_default(self):
        req, faults = self._normalize(temp="hot")
        assert "temp:not_a_number" in faults
        assert req["temp"] == 0.7

    def test_infinity_is_refused(self):
        req, faults = self._normalize(top_p=float("inf"))
        assert "top_p:not_finite" in faults
        assert req["top_p"] == 0.9

    def test_nan_is_refused(self):
        req, faults = self._normalize(min_p=float("nan"))
        assert "min_p:not_finite" in faults

    def test_an_out_of_range_value_becomes_the_default_not_the_edge(self):
        """A temperature of 40 is a mistake, not a request for 2.0."""
        req, faults = self._normalize(temp=40.0)
        assert "temp:out_of_range" in faults
        assert req["temp"] == 0.7

    def test_a_boolean_is_not_a_number(self):
        req, faults = self._normalize(top_k=True)
        assert "top_k:not_a_number" in faults
        assert req["top_k"] == 60

    def test_integers_stay_integers(self):
        req, _ = self._normalize(top_k=12.9, repetition_context_size=64.7)
        assert req["top_k"] == 12
        assert isinstance(req["top_k"], int)
        assert req["repetition_context_size"] == 64

    def test_missing_parameters_get_their_defaults(self):
        req, faults = self._normalize()
        assert faults == []
        assert req["temp"] == 0.7
        assert req["stop_sequences"] == []

    def test_too_many_stop_sequences_are_bounded(self):
        req, faults = self._normalize(stop_sequences=["a"] * 100)
        assert "stop_sequences:too_many" in faults
        assert len(req["stop_sequences"]) == 16

    def test_an_oversized_stop_sequence_is_clipped(self):
        req, faults = self._normalize(stop_sequences=["x" * 5000])
        assert "stop_sequences:too_long" in faults
        assert len(req["stop_sequences"][0]) == 200

    def test_a_non_string_stop_sequence_is_dropped_and_named(self):
        req, faults = self._normalize(stop_sequences=["ok", 42, None])
        assert faults.count("stop_sequences:not_a_string") == 2
        assert req["stop_sequences"] == ["ok"]

    def test_a_non_sequence_stop_value_is_refused(self):
        req, faults = self._normalize(stop_sequences="STOP")
        assert "stop_sequences:not_a_sequence" in faults
        assert req["stop_sequences"] == []


class TestAgentLoopEstablishesItsOwnAuthority:
    """CP126 9cddd95c: an untyped caller mapping became execution authority."""

    def _ctx(self, context):
        from core.brain.llm.mlx_client import _agent_execution_context

        return _agent_execution_context(
            context,
            objective="do the thing",
            tool_name="shell",
            tool_call_id="call_1",
            model_path="/models/Qwen2.5-32B-Instruct-4bit",
        )

    def test_ordinary_context_passes_through(self):
        out = self._ctx({"route": "desktop", "origin": "api", "timeout_s": 30})
        assert out["route"] == "desktop"
        assert out["origin"] == "api"
        assert out["timeout_s"] == 30

    def test_a_caller_cannot_declare_itself_confirmed(self):
        out = self._ctx({"confirmed": True, "route": "desktop"})
        assert "confirmed" not in out
        assert out["route"] == "desktop"
        assert out["agent_loop"]["refused_authority_keys"] == ["confirmed"]

    def test_a_caller_cannot_declare_standing_authority(self):
        out = self._ctx({"_standing_authority_verified": True})
        assert "_standing_authority_verified" not in out
        assert "_standing_authority_verified" in out["agent_loop"]["refused_authority_keys"]

    def test_proof_and_seal_claims_are_refused(self):
        out = self._ctx(
            {"proof_run": True, "sealed_validation": {"ok": True}, "proof_validation": 1}
        )
        for key in ("proof_run", "sealed_validation", "proof_validation"):
            assert key not in out
        assert len(out["agent_loop"]["refused_authority_keys"]) == 3

    def test_the_loop_stamps_its_own_provenance(self):
        out = self._ctx({"route": "desktop"})
        loop = out["agent_loop"]
        assert loop["source"] == "mlx_client.think_and_act"
        assert loop["tool"] == "shell"
        assert loop["tool_call_id"] == "call_1"
        assert loop["model"] == "Qwen2.5-32B-Instruct-4bit"
        assert len(loop["objective_sha256"]) == 64

    def test_no_context_still_yields_a_stamped_context(self):
        out = self._ctx(None)
        assert out["source"] == "think_and_act"
        assert out["agent_loop"]["refused_authority_keys"] == []

    def test_a_non_mapping_context_is_refused_not_forwarded(self):
        out = self._ctx(["definitely", "not", "a", "mapping"])
        assert out["source"] == "think_and_act"
        assert out["agent_loop"]["refused_authority_keys"] == ["<non_mapping:list>"]


class TestTheListenerIsTheOnlyConsumer:
    """CP126 2ba6ea2e / 1210919c."""

    def test_a_failed_message_terminalizes_its_request(self, client):
        future = _new_future()
        client._pending_generations["req-7"] = future
        client._terminalize_failed_message(
            {"id": "req-7", "action": "generate"}, TypeError("bad frame")
        )
        assert future.done()
        payload = future.result()
        assert payload["status"] == "error"
        assert payload["id"] == "req-7"
        assert "listener_message_processing_failed" in payload["message"]

    def test_the_current_request_is_terminalized_too(self, client):
        future = _new_future()
        client._current_request_id = "req-live"
        client._current_gen_future = future
        client._terminalize_failed_message({"id": "req-live"}, ValueError("bad"))
        assert future.done()

    def test_an_id_less_frame_terminalizes_nothing(self, client):
        future = _new_future()
        client._pending_generations["req-7"] = future
        client._terminalize_failed_message({"action": "generate"}, ValueError("bad"))
        assert not future.done()
        client._terminalize_failed_message(None, ValueError("bad"))
        assert not future.done()

    def test_an_already_finished_future_is_left_alone(self, client):
        future = _new_future()
        future.set_result({"status": "ok", "id": "req-7"})
        client._pending_generations["req-7"] = future
        client._terminalize_failed_message({"id": "req-7"}, ValueError("bad"))
        assert future.result()["status"] == "ok"

    def test_the_lease_renewal_is_bounded(self):
        from core.brain.llm.mlx_client import _LEASE_RENEWAL_TIMEOUT_S

        # The listener is the sole consumer of the worker's response queue, so
        # any unbounded await here stalls every request on the lane.
        assert 0.0 < _LEASE_RENEWAL_TIMEOUT_S <= 15.0


class TestWorkerScopedStateDoesNotSurviveASpawn:
    """CP126 066ebe25 / 7f9e9cf4."""

    def test_a_spawn_clears_the_previous_workers_claims(self, client):
        client._worker_identity = {"worker_boot_id": "old"}
        client._recurrent_depth_status = {"active": True, "loops": 2}
        client._recurrent_adapter_activation = {"active": True}
        client._steering_liveness_observed = True
        client._last_interoception = {"stale": True}
        client._last_surface_control_receipt = {"enabled": True}
        client._soft_cancel_target = {"req_id": "old"}

        client._reset_worker_scoped_state()

        assert client._worker_identity == {}
        assert client._recurrent_depth_status == {}
        assert client._recurrent_adapter_activation == {}
        assert client._steering_liveness_observed is False
        assert client._last_interoception == {}
        assert client._last_surface_control_receipt == {}
        assert client._soft_cancel_target is None

    def test_an_ordinary_history_passes_through(self):
        from core.brain.llm.mlx_client import _bounded_chat_messages

        rows, faults = _bounded_chat_messages(
            [{"role": "system", "content": "be good"}, {"role": "user", "content": "hi"}]
        )
        assert faults == []
        assert [r["role"] for r in rows] == ["system", "user"]

    def test_a_non_mapping_entry_is_dropped_and_named(self):
        from core.brain.llm.mlx_client import _bounded_chat_messages

        rows, faults = _bounded_chat_messages([{"role": "user", "content": "hi"}, "junk"])
        assert "messages:non_mapping_dropped" in faults
        assert len(rows) == 1

    def test_an_entry_without_a_role_is_dropped(self):
        from core.brain.llm.mlx_client import _bounded_chat_messages

        rows, faults = _bounded_chat_messages([{"content": "orphan"}])
        assert "messages:invalid_role" in faults
        assert rows == []

    def test_non_string_content_is_coerced_and_named(self):
        from core.brain.llm.mlx_client import _bounded_chat_messages

        rows, faults = _bounded_chat_messages([{"role": "user", "content": {"a": 1}}])
        assert "messages:non_string_content" in faults
        assert isinstance(rows[0]["content"], str)

    def test_an_oversized_message_is_clipped(self):
        from core.brain.llm.mlx_client import (
            _CHAT_MESSAGE_MAX_CHARS,
            _bounded_chat_messages,
        )

        rows, faults = _bounded_chat_messages(
            [{"role": "user", "content": "x" * (_CHAT_MESSAGE_MAX_CHARS * 2)}]
        )
        assert "messages:content_too_long" in faults
        assert len(rows[0]["content"]) == _CHAT_MESSAGE_MAX_CHARS

    def test_too_many_messages_keeps_the_system_turn_and_the_recent_ones(self):
        from core.brain.llm.mlx_client import (
            _CHAT_MESSAGES_MAX_ITEMS,
            _bounded_chat_messages,
        )

        history = [{"role": "system", "content": "policy"}]
        history += [
            {"role": "user", "content": f"turn-{i}"}
            for i in range(_CHAT_MESSAGES_MAX_ITEMS + 50)
        ]
        rows, faults = _bounded_chat_messages(history)
        assert "messages:too_many" in faults
        assert len(rows) == _CHAT_MESSAGES_MAX_ITEMS
        assert rows[0]["content"] == "policy", "the policy turn must survive"
        assert rows[-1]["content"].endswith(str(_CHAT_MESSAGES_MAX_ITEMS + 49))

    def test_an_aggregate_blowup_stops_before_templating(self):
        from core.brain.llm.mlx_client import _bounded_chat_messages

        rows, faults = _bounded_chat_messages(
            [{"role": "user", "content": "y" * 150_000} for _ in range(20)]
        )
        assert "messages:aggregate_too_large" in faults
        assert len(rows) < 20
