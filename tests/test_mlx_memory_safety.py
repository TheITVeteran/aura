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
    assert "projected_process_tree_rss:8.0GB+35.0GB+reserve3.0GB=46.0GB" in reason


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
    assert "projected_process_tree_rss:12.0GB+41.0GB+reserve5.0GB=58.0GB" in reason


def test_mlx_worker_spawn_uses_auto_projection_from_local_artifact(monkeypatch, tmp_path):
    from core.brain.llm import mlx_client

    gib = 1024**3
    model_dir = tmp_path / "Aura-32B-20260510-151144"
    model_dir.mkdir()
    weights = model_dir / "weights.safetensors"
    with weights.open("wb") as handle:
        handle.truncate(int(17.0 * gib))

    snapshot = SimpleNamespace(
        refuse_heavy_local_generation=False,
        available_gb=30.0,
        process_rss_gb=8.0,
        process_rss_limit_gb=40.0,
        reason="",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot)
    monkeypatch.setenv("AURA_MLX_32B_PROJECTED_FOOTPRINT_GB", "auto")

    assert mlx_client._memory_pressure_blocks_worker_spawn(str(model_dir)) is None

    snapshot_tight = SimpleNamespace(
        refuse_heavy_local_generation=False,
        available_gb=30.0,
        process_rss_gb=15.0,
        process_rss_limit_gb=40.0,
        reason="",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot_tight)

    reason = mlx_client._memory_pressure_blocks_worker_spawn(str(model_dir))

    assert reason is not None
    assert "projected_process_tree_rss" in reason


def test_worker_memory_sentinel_uses_bounded_heavy_lane_limits(monkeypatch):
    from core.brain.llm.mlx_worker import WorkerMemorySentinel

    writer = SimpleNamespace(put=lambda _payload: None)
    sentinel_32b = WorkerMemorySentinel(writer, "/models/Qwen2.5-32B-Instruct-8bit")
    sentinel_72b = WorkerMemorySentinel(writer, "/models/Qwen2.5-72B-Instruct-4bit")

    assert sentinel_32b._worker_rss_limit_gb(64.0) <= 36.0
    assert sentinel_72b._worker_rss_limit_gb(64.0) <= 40.0

    monkeypatch.setenv("AURA_MLX_WORKER_RSS_LIMIT_GB", "44")
    assert sentinel_32b._worker_rss_limit_gb(64.0) == 44.0


def test_worker_memory_sentinel_clamps_override_in_desktop_safe_boot(monkeypatch):
    from core.brain.llm.mlx_worker import WorkerMemorySentinel

    writer = SimpleNamespace(put=lambda _payload: None)
    sentinel_32b = WorkerMemorySentinel(writer, "/models/Qwen2.5-32B-Instruct-8bit")

    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)
    monkeypatch.setenv("AURA_MLX_WORKER_RSS_LIMIT_GB", "44")

    assert sentinel_32b._worker_rss_limit_gb(64.0) <= 36.0


def test_desktop_safe_boot_disables_primary_prompt_cache_retention(monkeypatch):
    from core.brain.llm.mlx_worker import _prompt_cache_entry_budget_for_model

    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "1")

    assert _prompt_cache_entry_budget_for_model("/models/Qwen2.5-32B-Instruct-8bit") == 0


def test_clean_user_surface_bypasses_worker_prompt_cache():
    from core.brain.llm.mlx_worker import _job_requires_prompt_cache_bypass

    assert _job_requires_prompt_cache_bypass({"clean_user_surface_contract": True}) is True
    assert _job_requires_prompt_cache_bypass({"proof_evaluation_contract": True}) is True
    assert _job_requires_prompt_cache_bypass({"action": "generate"}) is False


def test_optional_deep_solver_memory_refusal_stays_noncritical(monkeypatch):
    from core.brain.llm import mlx_client

    events = []
    client = mlx_client.MLXLocalClient("/models/Qwen2.5-72B-Instruct-4bit")
    monkeypatch.setattr(
        client,
        "_record_degraded_event",
        lambda reason, **kwargs: events.append((reason, kwargs)),
    )

    handled = client._handle_optional_deep_solver_memory_refusal(
        "memory_pressure_refused_worker_spawn:model_load_headroom:25.5GB < required 52.0GB"
    )

    assert handled is True
    assert client._lane_state == "cold"
    assert client._init_future is None
    assert client._consecutive_spawn_failures == 0
    assert client._spawn_backoff_until > time.time()
    assert events == [
        (
            "optional_deep_solver_memory_refusal",
            {
                "detail": (
                    "Qwen2.5-72B-Instruct-4bit:"
                    "memory_pressure_refused_worker_spawn:model_load_headroom:25.5GB < required 52.0GB"
                ),
                "severity": "warning",
                "foreground_request": False,
                "classification": "non_critical_fallback",
            },
        )
    ]
    assert client._classify_failure(
        reason="memory_pressure_refused_worker_spawn:model_load_headroom",
    ) == "non_critical_fallback"


def test_optional_deep_solver_handler_ignores_primary_32b_memory_refusal(monkeypatch):
    from core.brain.llm import mlx_client

    events = []
    client = mlx_client.MLXLocalClient("/models/Qwen2.5-32B-Instruct-8bit")
    monkeypatch.setattr(
        client,
        "_record_degraded_event",
        lambda reason, **kwargs: events.append((reason, kwargs)),
    )

    handled = client._handle_optional_deep_solver_memory_refusal(
        "memory_pressure_refused_worker_spawn:model_load_headroom:12.0GB < required 24.0GB"
    )

    assert handled is False
    assert events == []


def test_mlx_worker_accepts_zeroed_shared_substrate_for_affective_sync():
    source = open("core/brain/llm/mlx_worker.py", encoding="utf-8").read()

    assert "if substrate_mem is not None:" in source
    assert "engine.start_substrate_sync(shared_state=substrate_mem)" in source
    assert '"steering_active": bool(_steering_active)' in source


def test_mlx_client_records_worker_steering_liveness_receipt():
    source = open("core/brain/llm/mlx_client.py", encoding="utf-8").read()

    assert 'raw_steering = res.get("steering_active")' in source
    # Strict receipt typing: only an actual bool may activate the shared
    # steering channels (bool("false") is True — CP126 finding).
    assert "if isinstance(raw_steering, bool):" in source
    assert "self._steering_active.value = steering_active" in source
    assert "self._substrate_mem[-1] = 1.0 if steering_active else 0.0" in source
    assert "self._steering_liveness_observed = True" in source


# ── pressure-adaptive token-progress budgets ─────────────────────────────────
# Under unified-memory contention a resident heavy model's first token slows
# because prompt eval competes for bandwidth; killing it pays a ~20GB reload
# that deepens the contention (the Jul 7 soak doom loop). These tests pin the
# bounded stretch: heavy lanes only, emergency excluded, caller deadlines
# still dominate, and the whole feature is env-pinned OFF for the suite.

def _fake_snapshot(level: str):
    tiers = ["warning", "high", "critical", "emergency"]
    idx = tiers.index(level) if level in tiers else -1
    return SimpleNamespace(
        level=level,
        warning=idx >= 0,
        high=idx >= 1,
        critical=idx >= 2,
        emergency=idx >= 3,
    )


def _budget_client(model_path: str = "Qwen2.5-32B-cortex"):
    from core.brain.llm.mlx_client import MLXLocalClient

    return MLXLocalClient(model_path=model_path)


def test_pressure_stretch_is_pinned_off_for_the_suite(monkeypatch):
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("critical"),
    )
    client = _budget_client()
    factor, reason = client._pressure_adaptive_stretch()
    assert factor == 1.0 and reason == ""


def test_pressure_stretch_scales_token_budgets_by_tier(monkeypatch):
    monkeypatch.setenv("AURA_FIRST_TOKEN_PRESSURE_ADAPT", "1")
    client = _budget_client()
    for level, expected in (("warning", 1.2), ("high", 1.35), ("critical", 1.5)):
        monkeypatch.setattr(
            "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
            lambda _level=level, **_kw: _fake_snapshot(_level),
        )
        factor, reason = client._pressure_adaptive_stretch()
        assert factor == expected
        assert reason == f"memory_pressure_{level}"
        assert client._token_stall_after(foreground_request=True) == 40.0 * expected


def test_pressure_stretch_excluded_at_emergency(monkeypatch):
    monkeypatch.setenv("AURA_FIRST_TOKEN_PRESSURE_ADAPT", "1")
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("emergency"),
    )
    client = _budget_client()
    factor, _ = client._pressure_adaptive_stretch()
    assert factor == 1.0  # the refuse-generation path owns emergencies


def test_pressure_stretch_ignores_light_lanes(monkeypatch):
    monkeypatch.setenv("AURA_FIRST_TOKEN_PRESSURE_ADAPT", "1")
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("critical"),
    )
    client = _budget_client(model_path="Qwen2.5-1.5B-reflex")
    factor, _ = client._pressure_adaptive_stretch()
    assert factor == 1.0
    assert client._token_stall_after() == 8.0


def test_hard_ceiling_stretches_bounded_under_pressure(monkeypatch):
    monkeypatch.setenv("AURA_FIRST_TOKEN_PRESSURE_ADAPT", "1")
    monkeypatch.delenv("AURA_FIRST_TOKEN_ABSOLUTE_CEILING_S", raising=False)
    client = _budget_client()
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("critical"),
    )
    stretched = client._first_token_hard_ceiling(foreground_request=True)
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("nominal"),
    )
    baseline = client._first_token_hard_ceiling(foreground_request=True)
    assert baseline < stretched <= baseline * 1.5


def test_caller_deadline_still_dominates_stretch(monkeypatch):
    monkeypatch.setenv("AURA_FIRST_TOKEN_PRESSURE_ADAPT", "1")
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("critical"),
    )
    client = _budget_client()
    bounded = client._deadline_bound_first_token_hard_ceiling(
        20.0, foreground_request=True
    )
    assert bounded <= 16.0  # remaining - reserve, stretch cannot exceed it


def test_stall_receipts_name_the_pressure_tier(monkeypatch):
    client = _budget_client()
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("high"),
    )
    assert client._pressure_receipt_suffix() == ":memory=high"
    monkeypatch.setattr(
        "core.brain.llm.mlx_client.get_memory_pressure_snapshot",
        lambda **_kw: _fake_snapshot("nominal"),
    )
    assert client._pressure_receipt_suffix() == ""
