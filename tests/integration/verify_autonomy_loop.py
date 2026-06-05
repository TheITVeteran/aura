import time
from types import SimpleNamespace

import pytest

from core.orchestrator.mixins.autonomy import AutonomyMixin


class ThoughtStreamRecorder:
    def __init__(self):
        self.events = []

    def emit(self, title, message, **kwargs):
        self.events.append({"title": title, "message": message, "kwargs": kwargs})


class AutonomousBrainRecorder:
    def __init__(self):
        self.calls = []

    async def think(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "content": "I found a useful loose thread and want to investigate it further.",
            "tool_calls": [{"name": "web_search", "args": {"query": "current systems research"}}],
        }


class AutonomyRuntimeProbe(AutonomyMixin):
    def __init__(self, brain):
        self.cognitive_engine = SimpleNamespace(autonomous_brain=brain)
        self.conversation_history = [{"role": "user", "content": "keep improving the system"}]
        self.status = SimpleNamespace(cycle_count=0, start_time=time.time() - 240)
        self.boredom = 0
        self._last_thought_time = time.time() - 240
        self._last_user_interaction_time = 0
        self.goal_hierarchy = None
        self.state_repo = None
        self.liquid_state = None
        self.knowledge_graph = None
        self.drives = None
        self.streamed = []
        self.insights = []
        self.tools = []

    def _emit_thought_stream(self, message):
        self.streamed.append(message)

    async def _store_autonomous_insight(self, internal_msg, response):
        self.insights.append({"internal_msg": internal_msg, "response": response})

    async def execute_tool(self, name, args):
        self.tools.append({"name": name, "args": args})
        return {"ok": True}


@pytest.mark.asyncio
async def test_autonomy_loop_routes_reflection_through_brain_context_and_tools(monkeypatch):
    emitter = ThoughtStreamRecorder()
    brain = AutonomousBrainRecorder()
    runtime = AutonomyRuntimeProbe(brain)

    monkeypatch.setattr("core.thought_stream.get_emitter", lambda: emitter)
    monkeypatch.setattr("core.orchestrator.mixins.autonomy.background_activity_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr("core.orchestrator.mixins.autonomy.ServiceContainer.get", lambda name, default=None: default)

    await runtime._perform_autonomous_thought()

    assert emitter.events[0]["title"] == "Autonomous Drift"
    assert brain.calls
    assert brain.calls[0]["objective"] == "Reflect on current state."
    assert brain.calls[0]["context"]["boredom_level"] >= 200
    assert "keep improving the system" in brain.calls[0]["system_prompt"]
    assert any("loose thread" in item for item in runtime.streamed)
    assert runtime.tools == [{"name": "web_search", "args": {"query": "current systems research"}}]
    assert runtime.insights == [
        {
            "internal_msg": "Autonomous Reflection",
            "response": "I found a useful loose thread and want to investigate it further.",
        }
    ]
