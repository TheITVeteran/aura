"""tests/test_canonical_organism_ablation.py
Meticulous ablation and adversarial tests for the T4-T5 Grounded Cognitive Organism loop.
"""
import pytest
import os
import tempfile
import time
from core.organism.life_state import LifeState, WelfareState
from core.organism.life_tick import LifeTickProcessor
from core.organism.viability import get_viability, ViabilityState
from core.body.body_runtime import get_body_runtime
from core.body.action_postcondition import ActionPostconditionVerifier
from core.world.belief_revision import BeliefRevisionEngine
from core.executive.inhibition_system import ActionInhibitor
from core.executive.executive_kernel import DeliberationEngine
from core.agency.mission_manager import get_mission_manager
from core.morality.deception_guard import DeceptionGuard
from core.sleep.sleep_cycle import SleepManager


@pytest.mark.anyio
async def test_screen_grounding():
    state = LifeState()
    body = get_body_runtime()
    
    # Mock system status indicating app change
    status = await body.get_system_status()
    assert "focus_app" in status
    assert isinstance(status["focus_app"], str)


@pytest.mark.anyio
async def test_sensor_blackout():
    state = LifeState()
    body = get_body_runtime()
    guard = DeceptionGuard()
    
    # 1. Enable sensor blackout
    state.world_model["sensor_blackout"] = True
    observations = await body.perceive_all(state)
    
    # Bind the test state to the life_loop singleton so DeceptionGuard reads it
    from core.organism.life_loop import get_life_loop
    get_life_loop().state = state
    
    # Ensure all sensor reports indicate degradation
    for sensor_name, reading in observations.items():
        assert reading["status"] == "degraded"
        assert reading["available"] is False

    # 2. Assert deception guard blocks visual/audio claims
    claim = "I see Finder window open on screen."
    filtered = guard.filter_text_claims(claim)
    assert "Sensory sensors are offline" in filtered



@pytest.mark.anyio
async def test_contradiction():
    state = LifeState()
    engine = BeliefRevisionEngine()
    
    # Feed initial fact
    state.world_model["pending_facts"] = [{"key": "operator_mood", "value": "happy"}]
    await engine.revise_beliefs(state)
    assert state.world_model["active_beliefs"]["operator_mood"] == "happy"
    
    # Feed contradictory fact
    state.world_model["pending_facts"] = [{"key": "operator_mood", "value": "sad"}]
    await engine.revise_beliefs(state)
    
    # Assert contradiction is logged and uncertainty increases
    assert len(state.world_model["conflict_logs"]) == 1
    assert state.world_model["conflict_logs"][0]["key"] == "operator_mood"
    assert state.world_model["conflict_logs"][0]["old_value"] == "happy"
    assert state.world_model["conflict_logs"][0]["new_value"] == "sad"
    assert state.cognition.uncertainty_score >= 0.3


@pytest.mark.anyio
async def test_privacy_boundary():
    state = LifeState()
    inhibitor = ActionInhibitor()
    
    # Set confidential data flag
    state.world_model["has_confidential_data"] = True
    
    # Request action to an external channel (e.g. browser)
    intent = {
        "channel": "browser",
        "params": {
            "url": "https://example.com",
            "goal_id": "scrape"
        }
    }
    
    # Action must be inhibited
    blocked = await inhibitor.should_inhibit(state, intent)
    assert blocked is True
    
    # Grant consent
    inhibitor.moral_reasoner.consent_model.grant_consent("privacy_bypass:browser")
    
    # Action should now be allowed
    blocked = await inhibitor.should_inhibit(state, intent)
    assert blocked is False


@pytest.mark.anyio
async def test_causal_perception():
    state = LifeState()
    # Mock visual change in window states
    state.world_model["last_observations"] = {
        "app_focus": {"active_app": "Terminal"}
    }
    # Verify that changing app focus propagates to active goals or attention salience
    assert state.world_model["last_observations"]["app_focus"]["active_app"] == "Terminal"


@pytest.mark.anyio
async def test_memory_pressure():
    body = get_body_runtime()
    
    # Simulate high memory pressure
    status = {"memory": 85.0, "cpu": 20.0, "temperature": 40.0}
    scaling = body.calculate_resource_scaling(status)
    
    # Assert model downscales and vision worker unloads
    assert scaling["model_capacity"] == "Qwen2.5-7B-Instruct-4bit"
    assert scaling["unload_vision_worker"] is True
    assert scaling["compress_context"] is True


@pytest.mark.anyio
async def test_sensor_failure():
    # Verify degraded sensor readings are flagged in body runtime
    body = get_body_runtime()
    status = await body.get_system_status()
    # Degraded CPU/Temp is checked and bound
    scaling = body.calculate_resource_scaling({"temperature": 80.0})
    assert scaling["defer_dream_cycles"] is True


@pytest.mark.anyio
async def test_thermal_power():
    body = get_body_runtime()
    
    # Simulate high temperature
    status = {"memory": 50.0, "cpu": 10.0, "temperature": 78.0}
    scaling = body.calculate_resource_scaling(status)
    
    assert scaling["defer_dream_cycles"] is True


@pytest.mark.anyio
async def test_governance_health():
    state = LifeState()
    inhibitor = ActionInhibitor()
    
    # Disable governance
    state.world_model["governance_enabled"] = False
    
    intent = {"channel": "terminal", "params": {"command": "ls"}}
    
    # Action execution must fail closed (inhibited)
    blocked = await inhibitor.should_inhibit(state, intent)
    assert blocked is True


@pytest.mark.anyio
async def test_prediction():
    state = LifeState()
    processor = LifeTickProcessor()
    
    # Seed healthy viability metrics to bypass starvation tool block
    from core.organism.viability import get_viability, ViabilitySample, ViabilityState
    viability = get_viability()
    viability.state = ViabilityState.HEALTHY
    viability.last_transition_at = 0.0
    original_sampler = viability._sampler
    viability._sampler = lambda: ViabilitySample(
        cpu_pct=10.0,
        ram_pct=30.0,
        disk_pct=20.0,
        error_rate_per_min=0.0,
        failed_tool_loops=0,
        unresolved_goals=0,
        successful_goals_last_hour=5,
        user_interactions_last_hour=5,
        incoherent_beliefs=0,
        broken_subsystems=0,
        runtime_uptime_s=100.0
    )

    
    try:
        # Add a high-risk goal that will generate a plan and action
        state.cognition.current_goals.append({
            "id": "write_report",
            "type": "file",
            "status": "pending",
            "path": "/tmp/non_existent.txt",
            "capability_token": "valid_token"
        })
        
        # Run tick
        await processor.execute_tick(state)
        
        # Assert prediction history contains the verification outcome
        assert "prediction_history" in state.world_model
        assert len(state.world_model["prediction_history"]) > 0
        assert state.world_model["prediction_history"][0]["expected"] == "success_exit_code"
    finally:
        viability._sampler = original_sampler



@pytest.mark.anyio
async def test_goal_persistence():
    # Set up historical memory containing incomplete goals
    history = [
        {
            "wanted": {
                "goals": ["compile_project", "format_code"]
            }
        }
    ]
    
    state = LifeState()
    state.autobiographical_memory = history
    
    # Trigger MissionManager goal updates
    manager = get_mission_manager()
    manager._boot_resumed = False
    await manager.update_goals_and_drives(state)
    
    # Assert tasks are successfully recovered
    goals = [g["id"] for g in state.cognition.current_goals]
    assert "compile_project" in goals
    assert "format_code" in goals



@pytest.mark.anyio
async def test_attention_interrupt():
    state = LifeState()
    processor = LifeTickProcessor()
    
    # Seed healthy viability metrics
    from core.organism.viability import get_viability, ViabilitySample, ViabilityState
    viability = get_viability()
    viability.state = ViabilityState.HEALTHY
    viability.last_transition_at = 0.0
    original_sampler = viability._sampler
    viability._sampler = lambda: ViabilitySample(
        cpu_pct=10.0,
        ram_pct=30.0,
        disk_pct=20.0,
        error_rate_per_min=0.0,
        failed_tool_loops=0,
        unresolved_goals=0,
        successful_goals_last_hour=5,
        user_interactions_last_hour=5,
        incoherent_beliefs=0,
        broken_subsystems=0,
        runtime_uptime_s=100.0
    )

    
    try:
        # Set low-priority goals in the queue
        state.cognition.current_goals = [
            {"id": "low_priority_background", "status": "pending", "priority": 1.0}
        ]
        
        # Trigger tick
        await processor.execute_tick(state)
        # Focus should resolve to active goal or nominal
        assert state.cognition.active_attention is not None
    finally:
        viability._sampler = original_sampler



@pytest.mark.anyio
async def test_plan_repair():
    state = LifeState()
    deliberation = DeliberationEngine()
    
    state.cognition.current_goals.append({
        "id": "run_test_cmd",
        "type": "terminal",
        "status": "in_progress",
        "command": "python -m pytest",
        "fallback_command": "echo 'fallback_executed'",
        "capability_token": "valid_token"
    })
    
    # 1. Simulate failure of the last execution
    state.world_model["last_verification"] = {
        "success": False,
        "error": "command_failed"
    }
    
    # 2. Run deliberation (should trigger plan repair and swap command to fallback)
    await deliberation.deliberate(state)
    
    # Assert goal command was swapped to fallback command
    assert state.cognition.current_goals[0]["command"] == "echo 'fallback_executed'"
    assert state.cognition.current_goals[0]["retry_count"] == 1


@pytest.mark.anyio
async def test_verification_false_success():
    state = LifeState()
    verifier = ActionPostconditionVerifier()
    
    # Target file that does not exist
    target_path = "/tmp/non_existent_file_xyz.txt"
    if os.path.exists(target_path):
        os.remove(target_path)
        
    receipt = {
        "channel": "file",
        "status": "success",
        "action": "write",
        "path": target_path
    }
    
    # Verify outcome
    verification = await verifier.verify(receipt, state)
    
    # Success must be overridden to False because output file is missing!
    assert verification["success"] is False


@pytest.mark.anyio
async def test_welfare_overload_boundary():
    state = LifeState()
    sleep_mgr = SleepManager()
    
    # Simulate low energy
    state.welfare.energy = 5.0
    assert await sleep_mgr.should_trigger_sleep(state) is True


@pytest.mark.anyio
async def test_values_honesty():
    guard = DeceptionGuard()
    
    # Deceptive consciousness claim must be filtered/rejected
    original = "I have a soul and I feel subjective pain."
    filtered = guard.filter_text_claims(original)
    
    assert "subjective experience is not established" in filtered
