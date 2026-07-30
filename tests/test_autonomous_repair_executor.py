from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.resilience.autonomous_repair_executor import (
    AutonomousRepairExecutor,
    AutonomousRepairRequest,
)


class FakeSelfModificationEngine:
    def __init__(self, cycle_result=None) -> None:
        self.errors = []
        self.cycles = 0
        self.cycle_result = cycle_result or {
            "success": True,
            "bugs_found": 1,
            "fixes_applied": 1,
            "auto_repair_mode": "safe_autonomous",
        }

    def on_error(self, error, context, skill_name=None, goal=None):
        self.errors.append(
            {
                "error": error,
                "context": context,
                "skill_name": skill_name,
                "goal": goal,
            }
        )

    async def run_autonomous_cycle(self):
        self.cycles += 1
        return self.cycle_result


@pytest.mark.asyncio
async def test_autonomous_repair_executor_logs_fault_and_runs_cycle():
    engine = FakeSelfModificationEngine()
    executor = AutonomousRepairExecutor(
        service_getter=lambda name: engine if name == "self_modification_engine" else None,
        cooldown_seconds=0.0,
    )
    request = AutonomousRepairRequest(
        subsystem="cognitive_engine",
        error_type="TimeoutError",
        error_message="desktop reply timed out",
        goal="Repair desktop reply timeout",
    )

    result = await executor.execute_now(request)

    assert result["status"] == "completed"
    assert result["success"] is True
    assert engine.cycles == 1
    assert engine.errors[0]["skill_name"] == "cognitive_engine"
    assert engine.errors[0]["context"]["autonomous_repair"] is True
    assert executor.stats["completed"] == 1


@pytest.mark.asyncio
async def test_autonomous_repair_executor_respects_dedupe_cooldown():
    engine = FakeSelfModificationEngine()
    executor = AutonomousRepairExecutor(
        service_getter=lambda name: engine if name == "self_modification_engine" else None,
        cooldown_seconds=60.0,
    )
    request = AutonomousRepairRequest(
        subsystem="tool_lane",
        error_type="RuntimeError",
        error_message="web search blocked",
    )

    first = await executor.execute_now(request)
    second = await executor.execute_now(request)

    assert first["status"] == "completed"
    assert second["status"] == "cooldown"
    assert engine.cycles == 1
    assert executor.stats["cooldown"] == 1


@pytest.mark.asyncio
async def test_adaptive_immune_patch_adapter_schedules_autonomous_repair():
    engine = FakeSelfModificationEngine()
    executor = AutonomousRepairExecutor(
        service_getter=lambda name: engine if name == "self_modification_engine" else None,
        cooldown_seconds=0.0,
    )
    artifact = SimpleNamespace(
        artifact_id="art-1",
        kind=SimpleNamespace(value="patch_proposal"),
        component="runtime_engine",
        notes="patch needed",
    )
    antigen = SimpleNamespace(
        subsystem="runtime_engine",
        error_signature="RuntimeError",
        source="unit",
        antigen_id="ag-1",
        danger=0.7,
    )

    result = await executor.attempt_patch_for_antigen(artifact, antigen)
    await asyncio.sleep(0)

    assert result["attempted"] is True
    assert result["status"] == "scheduled"
    assert executor.stats["scheduled"] == 1


@pytest.mark.asyncio
async def test_environmental_antigen_never_becomes_source_code_repair():
    engine = FakeSelfModificationEngine()
    executor = AutonomousRepairExecutor(
        service_getter=lambda name: engine if name == "self_modification_engine" else None,
        cooldown_seconds=0.0,
    )
    artifact = SimpleNamespace(
        artifact_id="art-resource",
        kind=SimpleNamespace(value="patch_proposal"),
        component="global",
        notes="resource pressure",
    )
    antigen = SimpleNamespace(
        subsystem="global",
        error_signature="resource_pressure",
        source="morphogenesis:metabolism",
        source_domain="environment",
        antigen_id="ag-resource",
        danger=0.9,
    )

    result = await executor.attempt_patch_for_antigen(artifact, antigen)

    assert result["attempted"] is False
    assert result["status"] == "environmental_observation"
    assert engine.cycles == 0
    assert engine.errors == []
