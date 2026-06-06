import logging
import time
from types import SimpleNamespace

import pytest

from core.orchestrator import RobustOrchestrator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestAutonomy")


class AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return self.result


class CallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return self.result


@pytest.mark.asyncio
async def test_autonomy_loop(monkeypatch):
    """Autonomous thought should fall back to the direct autonomous brain."""
    orchestrator = RobustOrchestrator()
    brain_think = AsyncCallRecorder(
        result={
            "content": "I am bored.",
            "tool_calls": [{"name": "web_search", "args": {"query": "news"}}],
        }
    )
    brain = SimpleNamespace(think=brain_think)
    orchestrator.cognitive_engine = SimpleNamespace(autonomous_brain=brain)
    orchestrator.start_time = time.time() - 400.0
    orchestrator.status = SimpleNamespace(start_time=time.time() - 400.0)
    orchestrator._last_thought_time = time.time() - 200.0
    orchestrator._last_user_interaction_time = time.time() - 240.0
    emit_thought_stream = CallRecorder()
    execute_tool = AsyncCallRecorder()
    orchestrator._emit_thought_stream = emit_thought_stream
    orchestrator.execute_tool = execute_tool

    monkeypatch.setattr(
        "core.orchestrator.mixins.autonomy.ServiceContainer.get",
        staticmethod(lambda _name, default=None: default),
    )

    await orchestrator._perform_autonomous_thought()

    assert len(brain_think.calls) == 1
    kwargs = brain_think.calls[0].kwargs
    assert kwargs["objective"] == "Reflect on current state."
    assert kwargs["context"]["boredom_level"] >= 5
    assert len(execute_tool.calls) == 1
    assert execute_tool.calls[0].args == ("web_search", {"query": "news"})
    emitted_text = [call.args[0] for call in emit_thought_stream.calls]
    assert "...letting my mind wander..." in emitted_text
    assert "I am bored." in emitted_text
