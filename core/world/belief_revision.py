"""core/world/belief_revision.py
Belief revision engine coordinating observations and updating LifeState beliefs.
"""
from typing import Dict, Any, Optional
import logging

from core.world.entity_graph import EntityGraph
from core.world.causal_graph import CausalGraph
from core.world.temporal_graph import TemporalGraph
from core.world.social_graph import SocialGraph
from core.world.project_graph import ProjectGraph
from core.world.location_graph import LocationGraph
from core.world.object_permanence import ObjectPermanenceTracker
from core.world.affordance_model import AffordanceModel
from core.world.counterfactual_simulator import CounterfactualSimulator
from core.world.uncertainty_model import UncertaintyModel

logger = logging.getLogger("World.BeliefRevision")


class BeliefRevisionEngine:
    """Consolidates inputs from all world graphs to revise world model beliefs."""

    def __init__(self):
        self.entity_graph = EntityGraph()
        self.causal_graph = CausalGraph()
        self.temporal_graph = TemporalGraph()
        self.social_graph = SocialGraph()
        self.project_graph = ProjectGraph()
        self.location_graph = LocationGraph()
        self.permanence = ObjectPermanenceTracker()
        self.affordance = AffordanceModel()
        self.simulator = CounterfactualSimulator()
        self.uncertainty = UncertaintyModel()

    async def revise_beliefs(self, state: Any) -> None:
        """Evaluates active observations and registers refined facts on state."""
        observations = state.world_model.get("last_observations", {})

        # Process system details
        env_data = observations.get("environment_snapshot", {})
        cpu = env_data.get("cpu_percent", 10.0)
        memory = env_data.get("memory_percent", 50.0)

        # Update entity nodes
        self.entity_graph.upsert_entity("host_cpu", "hardware", {"usage": cpu})
        self.entity_graph.upsert_entity("host_memory", "hardware", {"usage": memory})

        # Cache seen state for object permanence
        self.permanence.update_seen_state("host_cpu", cpu)
        
        # Calculate surprise / uncertainty
        expected = state.world_model.get("active_beliefs", {}).get("host_cpu", 10.0)
        surprise = self.uncertainty.calculate_uncertainty(expected, cpu)
        
        # Revise beliefs
        state.world_model["active_beliefs"] = {
            "host_cpu": cpu,
            "host_memory": memory,
            "surprise_index": surprise
        }
        
        state.cognition.uncertainty_score = surprise
        logger.info("Revised world model beliefs. Surprise index: %.2f", surprise)
