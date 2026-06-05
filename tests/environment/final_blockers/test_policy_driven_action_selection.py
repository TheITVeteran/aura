import pytest

from core.environment.belief_graph import EnvironmentBeliefGraph
from core.environment.episode_manager import EpisodeManager
from core.environment.homeostasis import Homeostasis, Resource
from core.environment.ontology import ResourceState
from core.environment.parsed_state import ParsedState
from core.environment.policy.policy_orchestrator import PolicyOrchestrator


class LowHealthHomeostasis(Homeostasis):
    """Concrete homeostasis probe that exposes the low-health resource path."""

    def __init__(self):
        super().__init__()
        self.extracted_states = []

    def extract(self, parsed_state: ParsedState) -> list[Resource]:
        self.extracted_states.append(parsed_state)
        return [Resource(name="health", kind="health", value=2.0, max_value=25.0)]


@pytest.mark.asyncio
async def test_policy_low_hp_selects_survival_action_not_explore():
    orchestrator = PolicyOrchestrator()
    parsed_state = ParsedState(
        environment_id="test", 
        context_id="test", 
        sequence_id=1, 
        self_state={"hp": 2, "max_hp": 25},
        resources={"health": ResourceState(name="health", value=2.0, max_value=25.0)}
    )
    homeostasis = LowHealthHomeostasis()
    belief = EnvironmentBeliefGraph()
    episode = EpisodeManager(run_id="policy-survival", environment_id="test")
    
    intent = orchestrator.select_action(
        parsed_state=parsed_state,
        belief=belief,
        homeostasis=homeostasis,
        episode=episode,
        recent_frames=[]
    )
    
    assert homeostasis.extracted_states == [parsed_state]
    assert intent.name in ["wait", "retreat_to_safety", "stabilize_resource", "observe", "search", "inventory"]
