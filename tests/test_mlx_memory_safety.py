from __future__ import annotations

from types import SimpleNamespace


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
