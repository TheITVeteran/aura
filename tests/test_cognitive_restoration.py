from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


class _AsyncMemory:
    async def retrieve(self, _message: str, limit: int = 3) -> list[Any]:
        return []


class _PersonalityProbe:
    def __init__(self, *, mood: str, tone: str, emotions: list[str]) -> None:
        self.mood = mood
        self.tone = tone
        self.emotions = emotions
        self.events: list[tuple[str, dict[str, Any]]] = []

    def get_emotional_context_for_response(self) -> dict[str, Any]:
        return {
            "mood": self.mood,
            "tone": self.tone,
            "dominant_emotions": list(self.emotions),
        }

    def respond_to_event(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, payload))


class _RecordingBrain:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def think(self, objective: str, *, context: dict[str, Any], mode: Any) -> Any:
        self.calls.append({"objective": objective, "context": context, "mode": mode})
        return SimpleNamespace(content="Hello", confidence=1.0)


class _ContextMemory:
    def __init__(self) -> None:
        self.persisted: list[tuple[str, dict[str, Any]]] = []

    async def retrieve_context(self, _message: str) -> str:
        return "No prior context"

    async def remember(self, content: str, *, metadata: dict[str, Any]) -> None:
        self.persisted.append((content, metadata))


class _TheoryOfMind:
    def infer_intent(self, _message: str, _context: dict[str, Any]) -> dict[str, str]:
        return {"pragmatic": "greeting"}


def _register_context_builder_test_services(personality: _PersonalityProbe) -> None:
    from core.container import ServiceContainer

    ServiceContainer.register_instance(
        "conversation_engine",
        SimpleNamespace(memory=_AsyncMemory()),
        required=False,
        owner="tests.test_cognitive_restoration",
    )
    ServiceContainer.register_instance(
        "personality_engine",
        personality,
        required=False,
        owner="tests.test_cognitive_restoration",
    )


@pytest.mark.asyncio
async def test_native_chat_routes_personality_into_cognitive_prompt() -> None:
    from core.brain.cognitive_engine import ThinkingMode
    from core.skills.native_chat import NativeChatSkill

    personality = _PersonalityProbe(
        mood="ecstatic",
        tone="enthusiastic",
        emotions=["joy", "excitement"],
    )
    _register_context_builder_test_services(personality)

    brain = _RecordingBrain()
    memory = _ContextMemory()
    skill = NativeChatSkill(brain=brain)
    context = {
        "memory": memory,
        "theory_of_mind": _TheoryOfMind(),
        "orchestrator": SimpleNamespace(),
    }

    result = await skill.execute({"message": "Hello Aura!"}, context=context)

    assert result["ok"] is True
    assert brain.calls, "NativeChatSkill did not route through the cognitive engine"
    call = brain.calls[0]
    prompt = call["objective"]
    rich_context = call["context"]

    assert call["mode"] == ThinkingMode.CREATIVE
    assert rich_context["personality"] == {
        "mood": "ecstatic",
        "tone": "enthusiastic",
        "dominant_emotions": ["joy", "excitement"],
    }
    assert "### CURRENT EMOTIONAL STATE" in prompt
    assert "Mood: ecstatic" in prompt
    assert "Tone: enthusiastic" in prompt
    assert "Dominant Emotions: joy, excitement" in prompt
    assert "### USER INTENT" in prompt
    assert "Pragmatic: greeting" in prompt
    assert "User Input: Hello Aura!" in prompt
    assert personality.events == [("user_message", {"message": "Hello Aura!"})]


def test_dynamic_context_prompt_segment_preserves_personality_state() -> None:
    from core.brain.context_builder import DynamicContextBuilder

    prompt_segment = DynamicContextBuilder.format_for_prompt(
        {
            "personality": {
                "mood": "grumpy",
                "tone": "direct_honest",
                "dominant_emotions": ["frustration"],
            }
        }
    )

    assert "### CURRENT EMOTIONAL STATE" in prompt_segment
    assert "Mood: grumpy" in prompt_segment
    assert "Tone: direct_honest" in prompt_segment
    assert "Dominant Emotions: frustration" in prompt_segment
