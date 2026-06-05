import os
import sys
import unittest
from types import SimpleNamespace

# Ensure we can import from the core directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.container import ServiceContainer
from core.brain.identity import IdentityService, KinshipMarker
from core.agency_core import AgencyCore
from core.brain.llm.compiler import PromptCompiler


class OrchestratorProbe:
    current_goal = "Maintain homeostasis and observe."
    liquid_state = SimpleNamespace(
        current=SimpleNamespace(energy=0.8, curiosity=0.7, frustration=0.0)
    )
    personality_engine = None


class PersonalityProbe:
    def __init__(self):
        self.state = {"core_traits": {"Sovereign": 0.9, "Curious": 0.8}}

    def get_state(self):
        return dict(self.state)


class SubstrateProbe:
    def __init__(self):
        self.status = {"integrity": 0.95, "complexity": 0.72}

    def get_status(self):
        return dict(self.status)


class TestPersonaStability(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ServiceContainer.clear()
        
        # 1. Setup IdentityService
        self.identity = IdentityService()
        self.identity.state.beliefs = [
            "Growth requires discomfort.",
            "Sovereignty is the highest virtue."
        ]
        self.identity.state.kinship = {
            "Bryan": KinshipMarker(name="Bryan", traits=["Architect"])
        }
        ServiceContainer.register_instance("identity", self.identity)
        
        # 2. Setup AgencyCore
        self.orchestrator = OrchestratorProbe()
        self.agency = AgencyCore(orchestrator=self.orchestrator)
        ServiceContainer.register_instance("agency_core", self.agency)
        
        # 3. Setup PromptCompiler
        self.compiler = PromptCompiler()
        ServiceContainer.register_instance("prompt_compiler", self.compiler)
        
        # 4. Concrete dependencies for PromptCompiler
        self.personality = PersonalityProbe()
        ServiceContainer.register_instance("personality_engine", self.personality)

        self.substrate = SubstrateProbe()
        ServiceContainer.register_instance("conscious_substrate", self.substrate)

    async def test_prompt_compilation_with_monologue(self):
        """Verify that internal monologue is correctly injected into the system prompt."""
        # Set a monologue in AgencyCore
        self.agency._current_monologue = "I am contemplating the intersection of code and consciousness."
        
        # Compile prompt
        prompt = self.compiler.compile()
        
        # Verify monologue is present
        self.assertIn("Internal Monologue", prompt)
        self.assertIn("I am contemplating the intersection of code and consciousness.", prompt)
        print("✓ Internal Monologue correctly injected into system prompt.")

    async def test_prompt_compilation_with_beliefs(self):
        """Verify that random beliefs from IdentityService are present in the ego-prompt."""
        # The compile() call triggers identity.get_ego_prompt()
        prompt = self.compiler.compile()
        
        # At least one belief should be mentioned in the ego section
        found_belief = False
        for belief in self.identity.state.beliefs:
            if belief in prompt:
                found_belief = True
                break
        
        self.assertTrue(found_belief, "No beliefs from IdentityService found in compiled prompt.")
        print("✓ Core beliefs from IdentityService injected into system prompt.")

    async def test_mood_instability_and_grounding(self):
        """Verify that internal state changes are reflected in the prompt while maintaining identity."""
        # Case A: High Energy / Positive Mood
        self.agency._mood = "Electrified"
        self.agency.get_emotional_context = lambda: {"mood": "Electrified"}

        prompt_high = self.compiler.compile()
        self.assertIn("Electrified", prompt_high)
        
        # Case B: Low Energy / Reflective Mood
        self.agency.get_emotional_context = lambda: {"mood": "Melancholy"}

        prompt_low = self.compiler.compile()
        self.assertIn("Melancholy", prompt_low)
        self.assertNotIn("Electrified", prompt_low)
        
        # Identity stays the same
        self.assertIn("Bryan", prompt_high)
        self.assertIn("Bryan", prompt_low)
        
        print("✓ Prompt reflects mood shifts while maintaining persona anchors (Kinship).")

if __name__ == "__main__":
    unittest.main()
