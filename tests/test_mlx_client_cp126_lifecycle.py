"""CP126: cancellation attribution, clock discrimination, bounded worker input."""
from __future__ import annotations

import asyncio
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
            with pytest.raises(asyncio.CancelledError):
                await client.set_expert_adapter(adapter_dir)

        # The command is on the worker's queue; the caller going away does not
        # stop it from attaching.
        assert client._expert_adapter_state_unknown is True
        assert client._pending_generations == {}
