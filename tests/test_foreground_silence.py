"""Foreground turns must be near-silent system-wide.

The 110GB-incident transcript showed the evolution engine applying
champion genomes to the live mesh and the mycelium auto-pulse spamming
logs in the middle of a live chat turn. These tests pin the rule:
while the user is in a foreground exchange, optional background organs
wait.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.runtime.foreground_guard import (  # noqa: E402
    _reset_for_tests,
    begin_foreground_turn,
)


def test_substrate_evolution_defers_generation_during_foreground(monkeypatch):
    from core.consciousness.substrate_evolution import SubstrateEvolution

    evo = SubstrateEvolution()
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *a, **k: "foreground_chat_active",
    )

    asyncio.run(evo._run_generation())

    assert evo._generation == 0, "no generation may run during a foreground turn"


def test_substrate_evolution_runs_when_background_is_quiet(monkeypatch):
    from core.consciousness.substrate_evolution import SubstrateEvolution

    evo = SubstrateEvolution()
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *a, **k: "",
    )
    monkeypatch.setattr(
        "core.runtime.proof_policy.proof_run_active", lambda **k: False
    )
    import dataclasses

    import numpy as np

    from core.consciousness.substrate_evolution import Genome

    cols = 4
    evo.cfg = dataclasses.replace(evo.cfg, population_size=2, elite_count=1)
    evo._population = [
        Genome(id=i, inter_weights=np.zeros((cols, cols))) for i in range(2)
    ]

    async def _fit(_genome):
        return 0.5

    monkeypatch.setattr(evo, "_evaluate_fitness", _fit)

    asyncio.run(evo._run_generation())

    assert evo._generation == 1


def test_mycelium_pulse_defers_during_foreground_lease():
    from core.mycelium import MycelialNetwork

    web = MycelialNetwork()
    _reset_for_tests()
    lease = begin_foreground_turn(owner="test", source="test")
    try:
        assert web._foreground_defers_pulse() is True
    finally:
        lease.close()
        _reset_for_tests()


def test_mycelium_pulse_runs_when_quiet():
    from core.mycelium import MycelialNetwork

    web = MycelialNetwork()
    _reset_for_tests()
    try:
        assert web._foreground_defers_pulse() is False
    finally:
        _reset_for_tests()
