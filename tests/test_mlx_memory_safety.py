from __future__ import annotations

from types import SimpleNamespace


def test_mlx_worker_spawn_blocks_32b_when_headroom_is_too_low(monkeypatch):
    from core.brain.llm import mlx_client

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
    assert "required 16.0GB" in reason


def test_mlx_worker_spawn_allows_32b_with_sufficient_headroom(monkeypatch):
    from core.brain.llm import mlx_client

    snapshot = SimpleNamespace(
        refuse_heavy_local_generation=False,
        available_gb=20.0,
        reason="",
    )
    monkeypatch.setattr(mlx_client, "get_memory_pressure_snapshot", lambda: snapshot)

    assert mlx_client._memory_pressure_blocks_worker_spawn(
        "/models/Qwen2.5-32B-Instruct-8bit",
    ) is None


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


def test_worker_memory_sentinel_uses_bounded_heavy_lane_limits(monkeypatch):
    from core.brain.llm.mlx_worker import WorkerMemorySentinel

    writer = SimpleNamespace(put=lambda _payload: None)
    sentinel_32b = WorkerMemorySentinel(writer, "/models/Qwen2.5-32B-Instruct-8bit")
    sentinel_72b = WorkerMemorySentinel(writer, "/models/Qwen2.5-72B-Instruct-4bit")

    assert sentinel_32b._worker_rss_limit_gb(64.0) <= 52.0
    assert sentinel_72b._worker_rss_limit_gb(64.0) <= 56.0

    monkeypatch.setenv("AURA_MLX_WORKER_RSS_LIMIT_GB", "44")
    assert sentinel_32b._worker_rss_limit_gb(64.0) == 44.0
