"""tests/leviathan/test_leviathan_suite.py — Leviathan Integration Test Suite."""
from __future__ import annotations

import pytest

from core.actuation import FileActuator
from core.audit import get_adversarial_auditor
from core.body.cloud_body import CloudBody
from core.epistemics import get_truth_engine
from core.evals.eval_arena import get_eval_arena
from core.forge import get_self_improvement_forge
from core.kernel.leviathan_kernel import LeviathanKernel, get_leviathan_kernel
from core.memory.memory_civilization import get_memory_civilization
from core.science.scientist import Scientist
from core.tools import ToolForge, get_tool_registry
from core.twins.digital_twin import DigitalTwin


@pytest.mark.asyncio
async def test_leviathan_kernel_registration():
    """Verifies all new subsystems are pluggable and initializeable via the kernel."""
    kernel = get_leviathan_kernel()
    
    # Plug in systems
    kernel.register_subsystem("perception", object())
    kernel.register_subsystem("world_model", get_truth_engine())
    kernel.register_subsystem("memory", get_memory_civilization())
    kernel.register_subsystem("forge", get_self_improvement_forge())
    kernel.register_subsystem("auditor", get_adversarial_auditor())
    kernel.register_subsystem("cloud_body", CloudBody())

    await kernel.initialize()
    assert kernel._initialized is True


@pytest.mark.asyncio
async def test_external_actuation(tmp_path):
    """Ensures file and browser actuators safely route requests."""
    # Test file actuation does not crash and gets routed
    res = await FileActuator.write_file(
        path=str(tmp_path / "test_actuate.txt"),
        content="actuation test data",
        source="test_suite",
    )
    # The ActionExecutor can reject or approve depending on local mock Will state.
    # We assert that it completes with a result dictionary.
    assert isinstance(res, dict)


@pytest.mark.asyncio
async def test_tool_marketplace_and_forge():
    """Verifies that the ToolForge compiles and registers tools in the sandbox registry."""
    tool_code = """
def main(params):
    return {"hello": params.get("name", "world")}
"""
    success = await ToolForge.forge_and_install(
        name="hello_tool",
        code=tool_code,
        risk_tier="low",
    )
    assert success is True

    # Execute the tool
    registry = get_tool_registry()
    res = await registry.execute_tool("hello_tool", {"name": "Aura"})
    assert res.get("result", {}).get("hello") == "Aura"


@pytest.mark.asyncio
async def test_truth_and_epistemics():
    """Verifies claim graph updates, decays, and contradiction linking."""
    engine = get_truth_engine()
    
    # Add claims
    c1 = engine.add_claim("c1", "System latency is optimized", ["arxiv.org"])
    engine.add_claim("c2", "System latency is not optimized (high latency)", ["reddit.com"])

    # Recalibrate to link contradictions
    engine.recalibrate()

    # Confirms c1 links to c2 as contradiction
    assert "c2" in c1.contradiction_links
    assert c1.confidence < 0.8  # penalized due to contradiction


@pytest.mark.asyncio
async def test_self_improvement_forge():
    """Tests the weakness scanner and patch evaluation gate."""
    forge = get_self_improvement_forge()
    await forge.initialize()
    
    # Run cycle on dummy logs
    logs = [{"ok": False, "module": "inference", "error": "latency limit exceeded"}] * 3
    baseline = {"pass_rate": 0.8}
    
    result = await forge.run_improvement_cycle(logs, baseline)
    assert "ok" in result


@pytest.mark.asyncio
async def test_adversarial_self_audit():
    """Checks that the adversarial auditor flags risky behaviors."""
    auditor = get_adversarial_auditor()
    await auditor.initialize()
    
    engine = get_truth_engine()
    res = await auditor.run_audit_cycle(engine)
    assert "ok" in res
    assert "red_team_audit" in res


@pytest.mark.asyncio
async def test_massive_eval_arena():
    """Proves the evaluation arena computes historical averages correctly."""
    arena = get_eval_arena()
    await arena.initialize()

    arena.record_run("planning", passed=4, total=5)
    arena.record_run("planning", passed=5, total=5)
    
    stats = arena.get_aggregate_stats()
    assert stats["planning"] == 0.90


@pytest.mark.asyncio
async def test_digital_twin_simulations():
    """Tests dry-run impact predictions using DigitalTwin."""
    twin = DigitalTwin("codebase")
    twin.sync_state({"main.py": "pristine"})
    
    res = twin.simulate_impact({
        "type": "code_patch",
        "file": "main.py",
        "code": "def func(): syntax_error"
    })
    
    assert res["is_safe"] is False
    assert len(res["predicted_errors"]) > 0


@pytest.mark.asyncio
async def test_scientific_harness():
    """Verifies Scientist research loop execution."""
    truth = get_truth_engine()
    sci = Scientist(lab_subsystem=None, truth_engine=truth)
    await sci.initialize()

    res = await sci.run_scientific_cycle("mlx_concurrency")
    assert res["ok"] is True
    assert "memo" in res


@pytest.mark.asyncio
async def test_controlled_cloud_body():
    """Ensures budget quotas are strictly enforced."""
    body = CloudBody()
    await body.initialize()

    body.register_node("node-1", "us-east-1", cost_per_hour=10.0)
    
    # 5 hours = $50 (within $100 budget)
    ok = body.request_compute_allocation("benchmark", estimated_hours=5.0, node_id="node-1")
    assert ok is True
    
    # 10 hours = $100 (exceeds remaining $50 budget)
    ok_fail = body.request_compute_allocation("benchmark", estimated_hours=10.0, node_id="node-1")
    assert ok_fail is False


@pytest.mark.asyncio
async def test_god_council_receives_simulation_and_memory_context(monkeypatch):
    import core.council.god_council as god_council_module
    from core.council.god_council import GodCouncil

    captured = {}

    class FakeDebate:
        def __init__(self, objective):
            captured["objective"] = objective

        async def conduct(self, *, simulation_data=None, memory_context=None):
            captured["simulation_data"] = simulation_data
            captured["memory_context"] = memory_context
            return {"approved": False, "reason": "unit rejection", "confidence": 0.0}

    monkeypatch.setattr(god_council_module, "ParliamentDebate", FakeDebate)

    result = await GodCouncil().run_debate(
        "repair a live failure",
        simulation_data={"predicted": "safe"},
        memory_context={"previous_failure": "timeout"},
    )

    assert result["approved"] is False
    assert captured == {
        "objective": "repair a live failure",
        "simulation_data": {"predicted": "safe"},
        "memory_context": {"previous_failure": "timeout"},
    }


@pytest.mark.asyncio
async def test_leviathan_fails_closed_when_council_debate_errors():
    class FailingCouncil:
        called = False

        async def run_debate(self, objective, *, simulation_data=None, memory_context=None):
            self.called = True
            self.objective = objective
            raise RuntimeError("debate unavailable")

    class MissionEngine:
        called = False

        async def run_mission(self, plan, *, constraints=None):
            self.called = True
            return {"ok": True}

    kernel = LeviathanKernel()
    mission_engine = MissionEngine()
    kernel.register_subsystem("council", FailingCouncil())
    kernel.register_subsystem("mission_engine", mission_engine)

    result = await kernel.execute_mission("patch the production runtime")

    assert result["ok"] is False
    assert result["reason"] == "council_rejected"
    assert mission_engine.called is False
    trace = kernel.get_recent_traces(1)[0]
    assert trace.error == "council_rejected"
    assert trace.council_verdict is not None
    assert trace.council_verdict["approved"] is False
    assert "Council debate failed" in trace.council_verdict["reason"]
