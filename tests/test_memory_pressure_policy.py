from __future__ import annotations

from types import SimpleNamespace

import pytest


def _vm(*, total_gb: float, available_gb: float, percent: float) -> SimpleNamespace:
    gib = 1024**3
    return SimpleNamespace(
        total=int(total_gb * gib),
        available=int(available_gb * gib),
        percent=percent,
    )


def test_memory_pressure_snapshot_classifies_64gb_emergency(monkeypatch):
    import core.utils.memory_monitor as memory_monitor

    monkeypatch.setattr(
        memory_monitor.psutil,
        "virtual_memory",
        lambda: _vm(total_gb=64.0, available_gb=2.5, percent=96.0),
    )

    snapshot = memory_monitor.get_memory_pressure_snapshot()

    assert snapshot.level == "emergency"
    assert snapshot.emergency is True
    assert snapshot.refuse_heavy_local_generation is True
    assert snapshot.max_token_cap == 32
    assert "memory_pressure:96.0%" in snapshot.reason


def test_memory_pressure_snapshot_caps_but_does_not_refuse_high_pressure(monkeypatch):
    import core.utils.memory_monitor as memory_monitor

    monkeypatch.setattr(
        memory_monitor.psutil,
        "virtual_memory",
        lambda: _vm(total_gb=64.0, available_gb=9.5, percent=86.0),
    )

    snapshot = memory_monitor.get_memory_pressure_snapshot()

    assert snapshot.level == "high"
    assert snapshot.refuse_heavy_local_generation is False
    assert snapshot.max_token_cap == 192


@pytest.mark.asyncio
async def test_mlx_client_refuses_heavy_generation_under_emergency_memory(monkeypatch):
    import core.utils.memory_monitor as memory_monitor
    from core.brain.llm.mlx_client import MLXLocalClient

    monkeypatch.setattr(
        memory_monitor.psutil,
        "virtual_memory",
        lambda: _vm(total_gb=64.0, available_gb=2.0, percent=96.0),
    )

    client = MLXLocalClient("/models/Aura-Cortex-32B-MLX")
    request_lock_calls = 0

    async def record_unexpected_request_lock(*_args, **_kwargs):
        nonlocal request_lock_calls
        request_lock_calls += 1
        return False

    monkeypatch.setattr(client, "_acquire_request_lock", record_unexpected_request_lock)

    result = await client.generate("hello", foreground_request=True, origin="desktop")

    assert result is None
    assert request_lock_calls == 0
    assert client._lane_state == "cold"
