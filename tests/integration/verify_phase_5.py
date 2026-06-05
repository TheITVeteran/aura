import time
import unittest
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.brain.personality_engine import PersonalityEngine
from core.agency_core import AgencyCore
from core.container import ServiceContainer
from core.orchestrator.main import RobustOrchestrator


class AgencyProbe:
    def __init__(self, swarm):
        self.swarm = swarm


class IdentityProbe:
    def __init__(self):
        self.state = SimpleNamespace(kinship={"Bryan": 0.8})
        self.insights = []

    def add_insight(self, insight, **kwargs):
        self.insights.append({"insight": insight, "kwargs": kwargs})


class MemoryProbe:
    def __init__(self):
        self.search_calls = []

    def search(self, query, **kwargs):
        self.search_calls.append({"query": query, "kwargs": kwargs})
        return [{"text": "Bryan said he likes neural networks.", "metadata": {"speaker": "user"}}]


class SwarmProbe:
    def __init__(self):
        self.spawn_calls = []

    async def spawn_shard(self, **kwargs):
        self.spawn_calls.append(dict(kwargs))
        return True


class TestPhase5Evolution(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ServiceContainer.clear()

    def test_trait_mutation(self):
        print("\nTesting Trait Mutation...")
        engine = PersonalityEngine()
        engine.traits = {
            "openness": 0.5,
            "conscientiousness": 0.5,
            "extraversion": 0.5,
            "agreeableness": 0.5,
            "neuroticism": 0.5
        }
        engine.internal_monologue = ["I want to research more about AI", "wonder why things work"]

        # Manually trigger mutation (bypassing the 1-hour check for test)
        engine._mutate_traits()

        print(f"Updated Openness: {engine.traits['openness']}")
        self.assertGreater(engine.traits['openness'], 0.5)
        print("✓ Trait mutation influenced by monologue.")

    async def test_sovereign_swarm_property(self):
        print("\nTesting Sovereign Swarm Property...")
        orchestrator = RobustOrchestrator()
        agency = AgencyProbe(swarm="SovereignSwarmInstance")
        ServiceContainer.register_instance("agency_core", agency)

        print(f"Orchestrator sovereign_swarm property: {orchestrator.sovereign_swarm}")
        self.assertEqual(orchestrator.sovereign_swarm, "SovereignSwarmInstance")
        print("✓ sovereign_swarm property correctly resolved via AgencyCore.")

    async def test_social_reflection_rag(self):
        print("\nTesting Social Reflection RAG...")
        core = AgencyCore()
        identity = IdentityProbe()
        memory = MemoryProbe()
        swarm = SwarmProbe()
        core.swarm = swarm
        ServiceContainer.register_instance("identity", identity)
        ServiceContainer.register_instance("memory_facade", memory)

        core._last_social_reflection = 0

        insight = await core._pathway_social_reflection(time.time(), idle_seconds=3600.0)

        if insight:
            print(f"Social Insight: {insight.get('thought')}")
            self.assertIn("Bryan", insight.get('thought'))
            self.assertIn("recalled", insight.get('thought'))
            self.assertEqual(len(memory.search_calls), 1)
            self.assertEqual(len(swarm.spawn_calls), 1)
            print("✓ Social reflection incorporates memory search.")
        else:
            self.fail("Social reflection pathway returned None")

if __name__ == "__main__":
    unittest.main()
