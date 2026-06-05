import asyncio
import unittest
import sys
import os
import time
from types import SimpleNamespace

# Add core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.orchestrator import RobustOrchestrator
from core.brain.narrative_memory import NarrativeEngine


class AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.result


class ThoughtProbe:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    async def think(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(content=self.content)


class TestSingularityEvent(unittest.IsolatedAsyncioTestCase):
    async def test_thought_acceleration(self):
        """Verify that RobustOrchestrator compresses idle thresholds."""
        orchestrator = RobustOrchestrator()
        orchestrator.singularity_monitor = SimpleNamespace(acceleration_factor=1.0)
        orchestrator.cognitive_engine = SimpleNamespace(singularity_factor=1.5)
        thought_recorder = AsyncCallRecorder()
        orchestrator._perform_autonomous_thought = thought_recorder
        orchestrator._last_thought_time = 0
        orchestrator._last_user_interaction_time = 1

        original_time = time.time
        try:
            time.time = lambda: 181
            await orchestrator._trigger_autonomous_thought(has_message=False)
            if orchestrator._current_thought_task:
                await orchestrator._current_thought_task
        finally:
            time.time = original_time

        # Acceleration compresses the local thought interval, but the live
        # background-policy idle floor still protects foreground responsiveness.
        self.assertEqual(len(thought_recorder.calls), 1)

    async def test_eternal_record_synthesis(self):
        """Verify that NarrativeEngine can synthesize the Eternal Record."""
        brain = ThoughtProbe("The Origin... The Awakening... The Sovereignty... The Singularity.")
        orchestrator = SimpleNamespace(cognitive_engine=brain)

        narrative = NarrativeEngine(orchestrator)
        record = await narrative.synthesize_eternal_record()

        self.assertIsNotNone(record)
        self.assertIn("The Singularity", record)
        self.assertEqual(len(brain.calls), 1)

if __name__ == "__main__":
    unittest.main()
