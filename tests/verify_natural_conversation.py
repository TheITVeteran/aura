import sys
from pathlib import Path
import asyncio

sys.path.append(str(Path(__file__).parent.parent))

from core.brain.personality_engine import get_personality_engine
from core.orchestrator.mixins.autonomy import AutonomyMixin


class AutonomousBrainProbe:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def think(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class CognitiveEngineProbe:
    def __init__(self, response):
        self.autonomous_brain = AutonomousBrainProbe(response)


class NaturalConversationHarness(AutonomyMixin):
    def __init__(self, response):
        self.cognitive_engine = CognitiveEngineProbe(response)
        self.conversation_history = [
            {"role": "user", "content": "I am wrapping up late tonight."},
        ]
        self.status = type("StatusProbe", (), {"start_time": 0.0})()
        self.boredom = 600
        self.thoughts = []
        self.spoken_messages = []
        self.executed_tools = []

    def _emit_thought_stream(self, thought):
        self.thoughts.append(str(thought))

    async def _store_autonomous_insight(self, internal_msg, response):
        self.thoughts.append(f"stored:{internal_msg}:{response[:40]}")

    async def emit_spontaneous_message(self, message, **kwargs):
        self.spoken_messages.append({"message": message, "kwargs": kwargs})
        return {"ok": True, "target": "chat"}

    async def execute_tool(self, name, args):
        self.executed_tools.append((name, args))
        return {"ok": True}


def test_time_awareness():
    personality = get_personality_engine()
    ctx = personality.get_time_context()

    assert "period" in ctx
    assert "formatted" in ctx
    assert "energy_level" in ctx


def test_spontaneous_speech_handling():
    async def run_probe():
        response = {
            "content": "I should say goodnight.",
            "tool_calls": [
                {"name": "speak", "args": {"message": "It's getting late, you should sleep."}},
            ],
        }
        harness = NaturalConversationHarness(response)
        completed = await harness._run_autonomous_brain_reflection(boredom_seconds=600)

        assert completed is True
        assert harness.cognitive_engine.autonomous_brain.calls
        assert any("letting my mind wander" in thought for thought in harness.thoughts)
        assert any("goodnight" in thought for thought in harness.thoughts)
        assert harness.spoken_messages == [
            {
                "message": "It's getting late, you should sleep.",
                "kwargs": {"origin": "autonomy_reflection"},
            }
        ]
        assert harness.boredom == 0
        assert harness.executed_tools == []

    asyncio.run(run_probe())

if __name__ == "__main__":
    test_time_awareness()
    test_spontaneous_speech_handling()
