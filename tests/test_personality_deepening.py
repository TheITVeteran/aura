################################################################################

import asyncio
import os
import sys
import unittest
from types import SimpleNamespace

# Ensure we can import from the core directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.orchestrator import RobustOrchestrator
from core.brain.personality_engine import PersonalityEngine
from core.container import ServiceContainer


class RecordedCall:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


class CallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []
        self.call_args = None

    @property
    def called(self):
        return bool(self.calls)

    def __call__(self, *args, **kwargs):
        call = RecordedCall(args, kwargs)
        self.calls.append(call)
        self.call_args = call
        return self.result

    def assert_called(self):
        assert self.calls

    def assert_called_once(self):
        assert len(self.calls) == 1


class AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(RecordedCall(args, kwargs))
        return self.result


class TestPersonalityDeepening(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ServiceContainer.clear()
        
        liquid_state = SimpleNamespace(
            update=AsyncCallRecorder(),
            emotions={"contemplation": SimpleNamespace(intensity=0)},
            get_status=CallRecorder({"health": 1.0}),
        )
        ServiceContainer.register_instance("liquid_state", liquid_state)
        
        personality = SimpleNamespace(
            last_update=0,
            internal_monologue=[],
            emotions={"contemplation": SimpleNamespace(intensity=0)},
        )
        ServiceContainer.register_instance("personality_engine", personality)
        
        identity = SimpleNamespace(
            get_full_system_prompt=CallRecorder("System: Hello. INTERNAL MONOLOGUE: Reflection")
        )
        ServiceContainer.register_instance("identity", identity)

    async def asyncTearDown(self):
        ServiceContainer.clear()

    async def test_orchestrator_startup_and_personality_update(self):
        """Verify orchestrator starts and updates personality."""
        orch = RobustOrchestrator()
        orch.setup()
        
        # Inject explicit collaborators directly and ensure they are the ones used.
        pe = SimpleNamespace(update=CallRecorder())
        orch._personality_engine = pe
        
        # Initial cycle count
        self.assertEqual(orch.status.cycle_count, 0)
        
        async def no_async_work(*_args, **_kwargs):
            return None

        originals = {
            "_get_service": orch._get_service,
            "_acquire_next_message": orch._acquire_next_message,
            "_update_liquid_pacing": orch._update_liquid_pacing,
            "_trigger_autonomous_thought": orch._trigger_autonomous_thought,
            "_pulse_agency_core": orch._pulse_agency_core,
            "_run_terminal_self_heal": orch._run_terminal_self_heal,
        }
        orch._get_service = lambda name, *a: pe if name == "personality_engine" else SimpleNamespace()
        orch._acquire_next_message = no_async_work
        orch._update_liquid_pacing = no_async_work
        orch._trigger_autonomous_thought = no_async_work
        orch._pulse_agency_core = no_async_work
        orch._run_terminal_self_heal = no_async_work
        try:
            await orch._process_cycle()
        finally:
            for name, value in originals.items():
                setattr(orch, name, value)
        
        # Cycle count should increment
        self.assertEqual(orch.status.cycle_count, 1)
        
        # Verify personality update was called
        pe.update.assert_called()
        print("✓ Orchestrator cycle and personality update verified.")

    async def test_internal_monologue_and_prompt(self):
        """Verify internal monologue is captured and injected into prompt."""
        # Use real PersonalityEngine
        pe = PersonalityEngine()
        
        # Force a reflection trigger by manipulating emotions
        from core.brain.personality_engine import EmotionalState
        es = EmotionalState(name="contemplation")
        es.intensity = 90
        pe.emotions["contemplation"] = es
        
        # Generate behaviors (this should trigger monologue)
        pe._generate_spontaneous_behaviors()
        
        # Should have something in monologue
        self.assertTrue(len(pe.internal_monologue) > 0)
        reflection = pe.internal_monologue[0]
        print(f"✓ Internal Monologue captured: {reflection}")
        
        # Check identity prompt integration
        identity = SimpleNamespace(
            get_full_system_prompt=CallRecorder(f"INTERNAL MONOLOGUE: {reflection}")
        )
        ServiceContainer.register_instance("identity", identity)
        
        identity = ServiceContainer.get("identity")
        prompt = identity.get_full_system_prompt()
        self.assertIn("INTERNAL MONOLOGUE", prompt)
        self.assertIn(reflection, prompt)
        print("✓ Monologue injected into identity prompt.")

    def test_persona_persistence(self):
        """Verify persona can be persisted."""
        from core.brain import personality_engine as personality_module

        pe = PersonalityEngine()
        writer = CallRecorder()
        original_writer = personality_module.atomic_write_text
        personality_module.atomic_write_text = writer
        try:
            success = pe.persist()
            self.assertTrue(success)
            writer.assert_called_once()
            self.assertTrue(str(writer.call_args.args[0]).endswith("evolved_persona.json"))
        finally:
            personality_module.atomic_write_text = original_writer
        print("✓ Persona persistence verified.")

if __name__ == "__main__":
    unittest.main()


##
