"""tests/test_grounded_world_model.py
Unit tests for Aura's grounded world model, graph models, and counterfactual simulations.
"""
import pytest
from core.organism.life_state import LifeState
from core.world.belief_revision import BeliefRevisionEngine
from core.world.counterfactual_simulator import CounterfactualSimulator
from core.world.object_permanence import ObjectPermanenceTracker


@pytest.mark.anyio
async def test_belief_revision_flow():
    state = LifeState()
    # Mock some observations
    state.world_model["last_observations"] = {
        "environment_snapshot": {"cpu_percent": 25.0, "memory_percent": 60.0}
    }
    
    engine = BeliefRevisionEngine()
    await engine.revise_beliefs(state)
    
    # Assert beliefs were revised
    assert state.world_model["active_beliefs"]["host_cpu"] == 25.0
    assert state.world_model["active_beliefs"]["host_memory"] == 60.0


def test_counterfactual_simulation():
    sim = CounterfactualSimulator()
    result = sim.simulate("terminal", current_welfare=80.0)
    
    assert result["action"] == "terminal"
    assert result["success_probability"] == 0.75
    assert result["risk_classification"] == "moderate"


def test_object_permanence():
    tracker = ObjectPermanenceTracker()
    tracker.update_seen_state("test_file", "modified")
    
    latent = tracker.get_latent_state("test_file")
    assert latent["value"] == "modified"
    assert latent["staleness"] < 1.0
