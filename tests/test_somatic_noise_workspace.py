from __future__ import annotations

import random

import pytest

from core.consciousness.global_workspace import GlobalWorkspace


@pytest.mark.asyncio
async def test_somatic_noise_injects_bounded_workspace_candidate(monkeypatch):
    monkeypatch.setenv("AURA_SOMATIC_NOISE", "1")
    monkeypatch.setenv("AURA_SOMATIC_NOISE_FORCE", "1")
    workspace = GlobalWorkspace()
    workspace._somatic_noise.rng = random.Random(7)

    winner = await workspace.run_competition()

    assert winner is not None
    assert winner.source == "somatic_noise"
    assert 0.18 <= winner.priority <= workspace._somatic_noise.max_priority
    snapshot = workspace.get_snapshot()
    assert snapshot["somatic_noise"]["injected_count"] == 1
    assert snapshot["somatic_noise"]["last_reason"]


@pytest.mark.asyncio
async def test_somatic_noise_respects_minimum_tick_interval(monkeypatch):
    monkeypatch.setenv("AURA_SOMATIC_NOISE", "1")
    monkeypatch.delenv("AURA_SOMATIC_NOISE_FORCE", raising=False)
    workspace = GlobalWorkspace()
    workspace._somatic_noise.rate = 1.0
    workspace._somatic_noise.rng = random.Random(3)

    first_tick = await workspace.run_competition()

    assert first_tick is None
    winner = None
    for _ in range(workspace._somatic_noise.min_ticks_between - 1):
        winner = await workspace.run_competition()
        if winner is not None:
            break
    assert winner is not None
    assert winner.source == "somatic_noise"
