from __future__ import annotations

import time
from types import SimpleNamespace


def test_external_memory_sentinel_uses_current_footprint_not_lifetime_peak():
    from tools.memory_sentinel import _RUsageV4, current_phys_footprint_bytes

    usage = _RUsageV4()
    usage.ri_resident_size = 2 * 1024**3
    usage.ri_phys_footprint = 7 * 1024**3
    usage.ri_lifetime_max_phys_footprint = 105 * 1024**3

    assert current_phys_footprint_bytes(usage) == 7 * 1024**3


def test_memory_pressure_generation_controls_clamp_tokens_and_recurrent_depth():
    from core.brain.llm.mlx_client import _apply_memory_pressure_generation_controls

    options = {
        "max_tokens": 2048,
        "clean_user_surface_contract": True,
        "clean_user_surface_recurrent_loops": 2,
    }
    snapshot = SimpleNamespace(max_token_cap=192)

    controlled = _apply_memory_pressure_generation_controls(options, snapshot)

    assert controlled["max_tokens"] == 192
    assert controlled["clean_user_surface_recurrent_loops"] == 1


def test_memory_pressure_generation_controls_preserve_depth_without_cap():
    from core.brain.llm.mlx_client import _apply_memory_pressure_generation_controls

    options = {
        "max_tokens": 2048,
        "clean_user_surface_contract": True,
        "clean_user_surface_recurrent_loops": 2,
    }
    snapshot = SimpleNamespace(max_token_cap=None)

    controlled = _apply_memory_pressure_generation_controls(options, snapshot)

    assert controlled["max_tokens"] == 2048
    assert controlled["clean_user_surface_recurrent_loops"] == 2


def test_memory_pressure_generation_controls_use_model_default_when_unspecified():
    from core.brain.llm.mlx_client import _apply_memory_pressure_generation_controls

    options = {
        "clean_user_surface_contract": True,
        "clean_user_surface_recurrent_loops": 2,
    }
    snapshot = SimpleNamespace(max_token_cap=192)

    controlled = _apply_memory_pressure_generation_controls(
        options,
        snapshot,
        default_max_tokens=4096,
    )

    assert controlled["max_tokens"] == 192
    assert controlled["clean_user_surface_recurrent_loops"] == 1


def test_mlx_client_retains_surface_control_receipt_from_worker_response():
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient("Aura-32B-20260510-151144")

    client._record_surface_control_receipt_from_response(
        {
            "status": "ok",
            "surface_control_receipt": {
                "enabled": True,
                "live_mind_controls_bound": True,
                "clean_user_surface_contract": True,
                "surface_alpha_applied": 0.22,
                "surface_alpha_applied_ok": True,
                "recurrent_runtime_loops_applied": 2,
                "recurrent_runtime_loops_applied_ok": True,
                "surface_quality_gate_enabled": True,
                "surface_quality_gate_passed": True,
                "surface_quality_gate_attempts": 1,
                "surface_quality_gate_reasons": ["retryable_draft"],
                "applied": True,
                "untrusted_extra": "drop-me",
            },
        }
    )

    receipt = client.get_last_surface_control_receipt()
    assert receipt["enabled"] is True
    assert receipt["live_mind_controls_bound"] is True
    assert receipt["clean_user_surface_contract"] is True
    assert receipt["surface_alpha_applied"] == 0.22
    assert receipt["recurrent_runtime_loops_applied"] == 2
    assert receipt["surface_quality_gate_enabled"] is True
    assert receipt["surface_quality_gate_passed"] is True
    assert receipt["surface_quality_gate_attempts"] == 1
    assert receipt["surface_quality_gate_reasons"] == ["retryable_draft"]
    assert receipt["applied"] is True
    assert "untrusted_extra" not in receipt


def test_mlx_foreground_first_token_watchdog_aborts_tokenless_wall_clock_stall(monkeypatch):
    from core.brain.llm import mlx_client

    timers = []
    degraded = []
    aborted = []

    class FakeTimer:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            self.daemon = False
            self.name = ""
            self.cancelled = False

        def start(self):
            timers.append(self)

        def cancel(self):
            self.cancelled = True

    client = mlx_client.MLXLocalClient("Aura-32B-20260510-151144")
    monkeypatch.setattr(mlx_client._threading, "Timer", FakeTimer)
    monkeypatch.setattr(mlx_client, "_runtime_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        client,
        "_first_token_hard_ceiling",
        lambda *, foreground_request=False: 0.01,
    )
    monkeypatch.setattr(
        client,
        "_record_degraded_event",
        lambda *args, **kwargs: degraded.append((args, kwargs)),
    )
    monkeypatch.setattr(
        client,
        "force_abort_active_generation",
        lambda reason: aborted.append(reason) or True,
    )
    client._mark_generation_started("req-live", prompt_chars=32, requested_max_tokens=16)
    client._current_request_started_at = time.time() - 1.0

    timer = client._start_foreground_first_token_watchdog(
        "req-live",
        foreground_request=True,
    )

    assert timer is timers[0]
    assert timer.delay >= 10.0
    timer.callback()
    assert aborted == ["first_token_wall_clock_watchdog"]
    assert degraded
    assert degraded[0][0][0] == "first_token_wall_clock_watchdog"


def test_mlx_first_token_ceiling_is_bounded_by_request_deadline(monkeypatch):
    from core.brain.llm import mlx_client

    client = mlx_client.MLXLocalClient("Aura-32B-20260510-151144")
    monkeypatch.setattr(
        client,
        "_first_token_hard_ceiling",
        lambda *, foreground_request=False: 120.0 if foreground_request else 90.0,
    )

    assert client._deadline_bound_first_token_hard_ceiling(
        None,
        foreground_request=True,
    ) == 120.0
    assert client._deadline_bound_first_token_hard_ceiling(
        45.0,
        foreground_request=True,
    ) == 41.0
    assert client._deadline_bound_first_token_hard_ceiling(
        8.0,
        foreground_request=True,
    ) == 10.0
    assert client._deadline_bound_first_token_hard_ceiling(
        0.0,
        foreground_request=True,
    ) == 10.0


def test_mlx_generation_tracking_carries_deadline_bound_first_token_ceiling():
    from core.brain.llm import mlx_client

    client = mlx_client.MLXLocalClient("Aura-32B-20260510-151144")

    client._mark_generation_started(
        "req-live",
        prompt_chars=32,
        requested_max_tokens=16,
        first_token_hard_ceiling_s=41.0,
    )

    assert client._current_first_token_hard_ceiling_s == 41.0

    client._clear_active_generation_tracking()

    assert client._current_first_token_hard_ceiling_s == 0.0


def test_mlx_force_abort_kills_worker_before_lifecycle_lock_cleanup(monkeypatch):
    from core.brain.llm import mlx_client

    class FakeProcess:
        def __init__(self):
            self.killed = False
            self.joined = False

        def is_alive(self):
            return not self.killed

        def kill(self):
            self.killed = True

        def join(self, timeout=None):
            self.joined = True

    process = FakeProcess()
    client = mlx_client.MLXLocalClient("Aura-32B-20260510-151144")
    client._process = process
    client._active_generations = 1
    client._current_request_started_at = time.time() - 500.0
    monkeypatch.setattr(client, "_replace_ipc_queues", lambda *args, **kwargs: None)
    monkeypatch.setattr(client, "_record_degraded_event", lambda *args, **kwargs: None)

    client._lock.acquire()
    started = time.time()
    try:
        aborted = client.force_abort_active_generation("test_lock_unavailable_abort")
    finally:
        client._lock.release()

    assert aborted is True
    assert process.killed is True
    assert process.joined is True
    assert time.time() - started < 1.0


def test_mlx_worker_spawn_blocks_32b_when_headroom_is_too_low(monkeypatch):
    from core.brain.llm import mlx_client

    gib = 1024**3
    monkeypatch.setattr(
        mlx_client.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=int(64.0 * gib)),
    )
    snapshot = SimpleNamespace(
        refuse_heavy_local_generation=False,
        available_gb=12.0,
        reason="",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot)
    monkeypatch.delenv("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", raising=False)

    reason = mlx_client._memory_pressure_blocks_worker_spawn(
        "/models/Qwen2.5-32B-Instruct-8bit",
    )

    assert reason is not None
    assert "model_load_headroom" in reason
    assert "required 24.0GB" in reason


def test_mlx_worker_spawn_allows_32b_with_sufficient_headroom(monkeypatch):
    from core.brain.llm import mlx_client

    gib = 1024**3
    monkeypatch.setattr(
        mlx_client.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=int(64.0 * gib)),
    )
    snapshot = SimpleNamespace(
        refuse_heavy_local_generation=False,
        available_gb=24.0,
        reason="",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot)

    assert mlx_client._memory_pressure_blocks_worker_spawn(
        "/models/Qwen2.5-32B-Instruct-8bit",
    ) is None


def test_mlx_worker_spawn_blocks_projected_32b_overcommit(monkeypatch):
    from core.brain.llm import mlx_client

    gib = 1024**3
    monkeypatch.setattr(
        mlx_client.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=int(64.0 * gib)),
    )
    snapshot = SimpleNamespace(
        refuse_heavy_local_generation=False,
        available_gb=28.0,
        process_rss_gb=8.0,
        process_rss_limit_gb=38.0,
        reason="",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot)
    monkeypatch.delenv("AURA_MLX_32B_PROJECTED_FOOTPRINT_GB", raising=False)

    reason = mlx_client._memory_pressure_blocks_worker_spawn(
        "/models/Qwen2.5-32B-Instruct-8bit",
    )

    assert reason is not None
    assert "projected_process_tree_rss:8.0GB+35.0GB=43.0GB" in reason


def test_mlx_worker_spawn_blocks_when_unified_guard_refuses(monkeypatch):
    from core.brain.llm import mlx_client

    snapshot = SimpleNamespace(
        refuse_heavy_local_generation=True,
        available_gb=40.0,
        reason="process_tree_rss:54GB/48GB",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot)

    reason = mlx_client._memory_pressure_blocks_worker_spawn(
        "/models/Qwen2.5-72B-Instruct-4bit",
    )

    assert reason == "process_tree_rss:54GB/48GB"


def test_mlx_worker_spawn_blocks_72b_on_64gb_without_large_free_headroom(monkeypatch):
    from core.brain.llm import mlx_client

    gib = 1024**3
    monkeypatch.setattr(
        mlx_client.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=int(64.0 * gib)),
    )
    snapshot = SimpleNamespace(
        refuse_heavy_local_generation=False,
        available_gb=48.0,
        reason="",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot)
    monkeypatch.delenv("AURA_MLX_72B_LOAD_MIN_AVAILABLE_GB", raising=False)

    reason = mlx_client._memory_pressure_blocks_worker_spawn(
        "/models/Qwen2.5-72B-Instruct-4bit",
    )

    assert reason is not None
    assert "model_load_headroom" in reason
    assert "required 52.0GB" in reason


def test_mlx_worker_spawn_blocks_72b_projected_process_overcommit(monkeypatch):
    from core.brain.llm import mlx_client

    gib = 1024**3
    monkeypatch.setattr(
        mlx_client.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=int(128.0 * gib)),
    )
    snapshot = SimpleNamespace(
        refuse_heavy_local_generation=False,
        available_gb=72.0,
        process_rss_gb=12.0,
        process_rss_limit_gb=48.0,
        reason="",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot)
    monkeypatch.delenv("AURA_MLX_72B_PROJECTED_FOOTPRINT_GB", raising=False)

    reason = mlx_client._memory_pressure_blocks_worker_spawn(
        "/models/Qwen2.5-72B-Instruct-4bit",
    )

    assert reason is not None
    assert "projected_process_tree_rss:12.0GB+41.0GB=53.0GB" in reason


def test_worker_memory_sentinel_uses_bounded_heavy_lane_limits(monkeypatch):
    from core.brain.llm.mlx_worker import WorkerMemorySentinel

    writer = SimpleNamespace(put=lambda _payload: None)
    sentinel_32b = WorkerMemorySentinel(writer, "/models/Qwen2.5-32B-Instruct-8bit")
    sentinel_72b = WorkerMemorySentinel(writer, "/models/Qwen2.5-72B-Instruct-4bit")

    assert sentinel_32b._worker_rss_limit_gb(64.0) <= 36.0
    assert sentinel_72b._worker_rss_limit_gb(64.0) <= 40.0

    monkeypatch.setenv("AURA_MLX_WORKER_RSS_LIMIT_GB", "44")
    assert sentinel_32b._worker_rss_limit_gb(64.0) == 44.0


def test_desktop_safe_boot_disables_primary_prompt_cache_retention(monkeypatch):
    from core.brain.llm.mlx_worker import _prompt_cache_entry_budget_for_model

    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")

    assert _prompt_cache_entry_budget_for_model("/models/Qwen2.5-32B-Instruct-8bit") == 0


def test_clean_user_surface_bypasses_worker_prompt_cache():
    from core.brain.llm.mlx_worker import _job_requires_prompt_cache_bypass

    assert _job_requires_prompt_cache_bypass({"clean_user_surface_contract": True}) is True
    assert _job_requires_prompt_cache_bypass({"proof_evaluation_contract": True}) is True
    assert _job_requires_prompt_cache_bypass({"action": "generate"}) is False
