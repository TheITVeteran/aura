from types import SimpleNamespace

import pytest

from core.brain.cognitive_engine import ThinkingMode
from core.orchestrator import RobustOrchestrator


class CognitiveEngineRecorder:
    def __init__(self):
        self.calls = []

    async def think(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(
            content="I am fully operational.",
            mode=kwargs.get("mode"),
            confidence=0.99,
            reasoning=["Systems check complete.", "All modules responding."],
            action=None,
        )


@pytest.mark.asyncio
async def test_public_input_entrypoint_is_async_and_delegates_to_priority_lane():
    orchestrator = RobustOrchestrator()
    calls = []

    async def priority_lane(message, origin="user"):
        calls.append({"message": message, "origin": origin})
        return "accepted"

    orchestrator.process_user_input_priority = priority_lane

    result = await orchestrator.process_user_input("system health report", origin="desktop")

    assert result == "accepted"
    assert calls == [{"message": "system health report", "origin": "desktop"}]


@pytest.mark.asyncio
async def test_cognitive_loop_returns_text_from_current_user_pipeline():
    orchestrator = RobustOrchestrator()
    engine = CognitiveEngineRecorder()
    orchestrator.cognitive_engine = engine
    orchestrator.conversation_history = []

    async def user_pipeline(message, metadata=None):
        thought = await orchestrator.cognitive_engine.think(
            objective=message,
            context={
                "history": list(orchestrator.conversation_history),
                "personality": {"mood": "focused"},
            },
            mode=ThinkingMode.CREATIVE,
        )
        orchestrator.conversation_history.append({"role": "user", "content": message})
        orchestrator.conversation_history.append({"role": "assistant", "content": thought.content})
        return {"ok": True, "response": thought.content}

    orchestrator._process_message = user_pipeline

    response = await orchestrator._run_cognitive_loop(
        "Please analyze the system health and summarize any bottlenecks.",
        origin="user",
    )

    assert response == "I am fully operational."
    assert orchestrator.conversation_history[-1] == {
        "role": "assistant",
        "content": "I am fully operational.",
    }
    assert engine.calls[0]["kwargs"]["mode"] is ThinkingMode.CREATIVE
    assert engine.calls[0]["kwargs"]["context"]["personality"]["mood"] == "focused"
