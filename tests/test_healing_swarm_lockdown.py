"""The immune lane must outrank the failure lockdown it exists to clear.

Live incident (July 2, 2026): mind_tick died; the healer dispatched its
repair 66 times and the background policy deferred every attempt with
``failure_lockdown_1.00`` — a lockdown caused by the very subsystem the
repair targets. Runtime-contract subsystems' repairs must proceed through
failure lockdown; luxury repairs must still defer.
"""
from __future__ import annotations

import asyncio

import pytest

from core.resilience.healing_swarm import HealingSwarmService


class _RecordingSovereignSwarm:
    def __init__(self):
        self.spawned: list[str] = []

    async def spawn_shard(self, goal, context):
        self.spawned.append(goal)
        return True


@pytest.fixture()
def swarm():
    instance = HealingSwarmService.__new__(HealingSwarmService)
    sovereign = _RecordingSovereignSwarm()

    class _Orchestrator:
        sovereign_swarm = sovereign

    instance.orchestrator = _Orchestrator()
    instance._repair_history = {}
    instance._test_repairs = sovereign.spawned
    return instance


def _lockdown_policy(monkeypatch):
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "failure_lockdown_1.00",
    )


def _clear_policy(monkeypatch):
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "",
    )


def test_contract_subsystem_repair_proceeds_through_lockdown(swarm, monkeypatch):
    _lockdown_policy(monkeypatch)
    asyncio.run(swarm.attempt_repair("mind_tick", {"status": "STALE"}))
    assert len(swarm._test_repairs) == 1 and "mind_tick" in swarm._test_repairs[0], (
        "deferring the repair that would clear the lockdown is a deadlock"
    )


def test_non_contract_repair_still_defers_under_lockdown(swarm, monkeypatch):
    _lockdown_policy(monkeypatch)
    asyncio.run(swarm.attempt_repair("dream_journal", {"status": "STALE"}))
    assert swarm._test_repairs == [], "luxury repairs must respect the lockdown"


def test_other_policy_reasons_still_defer_contract_repairs(swarm, monkeypatch):
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "foreground_quiet_window",
    )
    asyncio.run(swarm.attempt_repair("mind_tick", {"status": "STALE"}))
    assert swarm._test_repairs == [], (
        "the immune-lane exemption is lockdown-specific, not a blanket bypass"
    )


def test_repair_cooldown_still_applies_through_lockdown(swarm, monkeypatch):
    _lockdown_policy(monkeypatch)
    asyncio.run(swarm.attempt_repair("mind_tick", {"status": "STALE"}))
    asyncio.run(swarm.attempt_repair("mind_tick", {"status": "STALE"}))
    assert len(swarm._test_repairs) == 1 and "mind_tick" in swarm._test_repairs[0], (
        "immune lane must not become a repair storm: cooldown still applies"
    )


def test_clear_policy_repairs_normally(swarm, monkeypatch):
    _clear_policy(monkeypatch)
    asyncio.run(swarm.attempt_repair("mind_tick", {"status": "DEGRADED"}))
    assert len(swarm._test_repairs) == 1 and "mind_tick" in swarm._test_repairs[0]
