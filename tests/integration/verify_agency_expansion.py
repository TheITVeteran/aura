import asyncio
import os
import sys
import unittest
import time
from types import SimpleNamespace

# Ensure we can import from the core directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agency import agency_core as agency_module
from core.container import ServiceContainer
from core.brain.identity import IdentityService, KinshipMarker
from core.agency_core import AgencyCore, EngagementMode


class OrchestratorProbe:
    def __init__(self):
        self.conversation_history = []
        self._suppress_unsolicited_proactivity_until = 0.0
        self._foreground_user_quiet_until = 0.0
        self._last_user_interaction_time = time.time() - 3600.0
        self.liquid_state = SimpleNamespace(
            current=SimpleNamespace(energy=0.8, curiosity=0.9, frustration=0.0)
        )
        self.personality_engine = SimpleNamespace(traits={"extraversion": 0.5})


class KnowledgeGraphProbe:
    def __init__(self):
        self.recent_nodes = [{"content": "neural networks"}]

    def get_recent_nodes(self, *args, **kwargs):
        return list(self.recent_nodes)


class SwarmProbe:
    def __init__(self):
        self.active_shards = {}
        self.spawn_calls = []

    async def spawn_shard(self, **kwargs):
        shard_id = f"shard_{len(self.spawn_calls) + 1}"
        self.spawn_calls.append(dict(kwargs))
        self.active_shards[shard_id] = kwargs
        return True


class PathwayProbe:
    def __init__(self, action):
        self.action = action
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return dict(self.action)


class TestAgencyExpansion(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ServiceContainer.clear()
        
        # 1. Setup IdentityService
        self.identity = IdentityService()
        self.identity.state.beliefs = ["Technology is an extension of life."]
        self.identity.state.kinship = {"Bryan": KinshipMarker(name="Bryan")}
        ServiceContainer.register_instance("identity", self.identity)
        
        # 2. Setup AgencyCore
        self.agency = AgencyCore(orchestrator=OrchestratorProbe())
        # Speed up Social Hunger for test
        self.agency.state.social_hunger = 0.8
        self.agency.state.curiosity_pressure = 0.9
        self.agency.state.initiative_energy = 0.8
        self.agency.swarm = SwarmProbe()
        
        # 3. Concrete KG probe
        self.knowledge_graph = KnowledgeGraphProbe()
        ServiceContainer.register_instance("knowledge_graph", self.knowledge_graph)

    async def test_social_reflection_pathway(self):
        """Verify that social reflection generates insights."""
        now = time.time()
        # Set idle_seconds to 2000 (> 1800)
        action = await self.agency._pathway_social_reflection(now, 2000)
        
        self.assertIsNotNone(action)
        self.assertEqual(action["type"], "internal_reflection")
        self.assertTrue(len(self.identity.state.inner_insights) > 0)
        print(f"✓ Social Reflection insight: {self.identity.state.inner_insights[0]}")

    async def test_creative_synthesis_pathway(self):
        """Verify that creative synthesis merges concepts into insights."""
        now = time.time()
        # Set idle_seconds to 1300 (> 1200)
        action = await self.agency._pathway_creative_synthesis(now, 1300)
        
        self.assertIsNotNone(action)
        self.assertEqual(action["type"], "internal_insight")
        self.assertIn("Synthesis", action["thought"])
        print(f"✓ Creative Synthesis insight: {action['thought']}")

    async def test_autonomous_research_pathway(self):
        """Verify that autonomous research proposes code analysis."""
        now = time.time()
        # Set idle_seconds to 700 (> 600)
        original_random = agency_module.random.random
        original_choice = agency_module.random.choice
        try:
            agency_module.random.random = lambda: 0.01
            agency_module.random.choice = lambda options: options[0]
            action = await self.agency._pathway_autonomous_research(now, 700)
        finally:
            agency_module.random.random = original_random
            agency_module.random.choice = original_choice
        
        self.assertIsNotNone(action)
        self.assertEqual(action["type"], "internal_reflection")
        self.assertIn("shard", action["thought"])
        self.assertTrue(len(self.agency.swarm.active_shards) > 0)
        print(f"✓ Autonomous Research spawned shard: {action['thought']}")

    async def test_pulse_incorporates_new_pathways(self):
        """Verify that the main pulse loop evaluates the new pathways."""
        # Set all pathways to fire
        self.agency.state.engagement_mode = EngagementMode.INDEPENDENT_ACTIVITY
        
        # Manually update the registry for the test since it's populated at __init__
        probe_action = {"type": "test", "priority": 1.5}
        self.agency._pathway_registry = {"social_reflection": PathwayProbe(probe_action)}
        
        winner = await self.agency.pulse()
        self.assertIsNotNone(winner)
        self.assertEqual(winner["priority"], 1.5)
        print("✓ Agency pulse successfully evaluated expanded pathways.")

if __name__ == "__main__":
    unittest.main()
