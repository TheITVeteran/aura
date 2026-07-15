"""Regression: the foreground-protection shed must NOT unload the small
fallback models when memory is abundant.

Lived 2026-07-15 soak: with the router routing AROUND a not-ready cortex,
shedding the 7B/1.5B fallbacks left nothing resident to answer — every
warming-cortex turn cascaded to a no-reply death spiral (7B >56s, 1.5B
>14.7s thrashing to reload) despite 42GB free. The small models are the
guaranteed fast-answer path; only genuine memory pressure may shed them.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.brain.inference_gate as ig

pytestmark = pytest.mark.unit


class _FakeClient:
    def __init__(self):
        self.rebooted = False

    def is_alive(self):
        return True

    async def reboot_worker(self, reason="", mark_failed=False):
        self.rebooted = True


@pytest.mark.asyncio
async def test_shed_skips_when_memory_is_abundant(monkeypatch):
    gate = ig.InferenceGate.__new__(ig.InferenceGate)
    gate._last_background_memory_shed_at = 0.0
    gate._mlx_client = object()

    fake = _FakeClient()
    monkeypatch.setattr("core.brain.llm.mlx_client._CLIENTS", {"/m/qwen-7b": fake}, raising=False)
    # 50GB free — far above the 24GB cortex + 8GB fallback reserve.
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda *a, **k: SimpleNamespace(available_gb=50.0),
    )

    await gate._shed_background_workers_for_memory_pressure(force=True)
    assert fake.rebooted is False, "fallback worker must stay resident with 50GB free"


@pytest.mark.asyncio
async def test_shed_proceeds_under_real_pressure(monkeypatch):
    gate = ig.InferenceGate.__new__(ig.InferenceGate)
    gate._last_background_memory_shed_at = 0.0
    gate._mlx_client = object()

    fake = _FakeClient()
    monkeypatch.setattr("core.brain.llm.mlx_client._CLIENTS", {"/m/qwen-7b": fake}, raising=False)
    # 10GB free — below the cortex requirement; shedding is correct.
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda *a, **k: SimpleNamespace(available_gb=10.0),
    )

    await gate._shed_background_workers_for_memory_pressure(force=True)
    assert fake.rebooted is True, "fallback worker must be shed under real pressure"
