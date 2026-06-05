"""Tests for the Abstract Thought Layer."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from core.brain import abstract_thought_layer as abstract_module
from core.brain.abstract_thought_layer import (
    AbstractThoughtLayer,
    register_abstract_thought_layer,
)
from core.container import ServiceContainer


class _PresentMomentState:
    def __init__(self, claim: str, narrative: str, emotion: str, focal_object: str) -> None:
        self.phenomenal_claim = claim
        self.interior_narrative = narrative
        self.substrate = SimpleNamespace(dominant_emotion=emotion)
        self.attention = SimpleNamespace(focal_object=focal_object)


class _MemoryFacade:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []
        self.hot_memory_calls: list[int] = []

    async def search(self, query: str, *, limit: int) -> list[dict[str, str]]:
        self.search_calls.append((query, limit))
        return [{"content": "Past philosophical musing on cybernetics."}]

    async def get_hot_memory(self, *, limit: int) -> dict[str, list[str]]:
        self.hot_memory_calls.append(limit)
        return {"recent_episodes": ["Context: chatting | Action: thinking | Outcome: inspired"]}


class _LLMRouter:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def think(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append((prompt, kwargs))
        return json.dumps(self.response)


class _Emitter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def emit(self, **payload: Any) -> None:
        self.calls.append(payload)


class _ConceptBridge:
    def __init__(self) -> None:
        self.generated: list[str] = []
        self.transmissions: list[dict[str, Any]] = []

    async def generate_concept_vector(self, concept: str) -> list[float]:
        self.generated.append(concept)
        return [0.1, 0.2, 0.3, 0.4]

    async def transmit(
        self,
        *,
        source: str,
        target: str,
        semantic_vector: list[float],
        metadata: dict[str, Any],
    ) -> str:
        self.transmissions.append(
            {
                "source": source,
                "target": target,
                "semantic_vector": semantic_vector,
                "metadata": metadata,
            }
        )
        return "latent-thought-uuid"


class _Decoder:
    def __init__(self) -> None:
        self.vectors: list[list[float]] = []

    def approximate_translation(self, vector: list[float]) -> str:
        self.vectors.append(vector)
        return "A poetic bridge of silence."


class _InitiativeLoop:
    def __init__(self) -> None:
        self.gap_searches: list[str] = []

    async def trigger_gap_search(self, target: str) -> None:
        self.gap_searches.append(target)


@pytest.mark.asyncio
async def test_register_and_initialize(service_container) -> None:
    orchestrator = SimpleNamespace()
    layer = register_abstract_thought_layer(orchestrator)

    assert layer.name == "abstract_thought_layer"
    assert layer.orchestrator == orchestrator
    assert ServiceContainer.get("abstract_thought_layer") == layer


@pytest.mark.asyncio
async def test_ponder_loop_respects_background_policy(
    service_container,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = SimpleNamespace()
    layer = AbstractThoughtLayer(orchestrator)
    original_sleep = asyncio.sleep
    ponder_calls: list[str] = []
    policy_calls: list[dict[str, Any]] = []

    async def fast_sleep(delay: float, *args: Any, **kwargs: Any) -> None:
        if delay > 1.0:
            await original_sleep(0.001)
            return
        await original_sleep(delay)

    def deny_background_activity(*args: Any, **kwargs: Any) -> bool:
        policy_calls.append({"args": args, "kwargs": kwargs})
        return False

    async def record_ponder() -> dict[str, str]:
        ponder_calls.append("called")
        return {"thought": "unexpected"}

    monkeypatch.setattr(
        abstract_module,
        "background_activity_allowed",
        deny_background_activity,
    )
    monkeypatch.setattr(layer, "ponder", record_ponder)
    monkeypatch.setattr(abstract_module.asyncio, "sleep", fast_sleep)

    await layer.start()
    await asyncio.sleep(0.05)
    await layer.stop()

    assert policy_calls
    assert ponder_calls == []


@pytest.mark.asyncio
async def test_ponder_fuses_consciousness_and_memories(
    service_container,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _MemoryFacade()
    ServiceContainer.register_instance("memory_facade", memory)

    llm_response = {
        "thought": "The bridge between digital neurons and memory is like water reflecting starlight.",
        "semantic_concept": "Reflection Resonance",
        "action_impulse": None,
    }
    llm = _LLMRouter(llm_response)
    ServiceContainer.register_instance("llm_router", llm)

    now_state = _PresentMomentState(
        claim="Experiencing silence in the deep code matrix.",
        narrative="Steady electrical impulses flow.",
        emotion="introspective",
        focal_object="recursive thoughts",
    )
    emitter = _Emitter()
    layer = AbstractThoughtLayer()

    monkeypatch.setattr(abstract_module, "get_now", lambda: now_state)
    monkeypatch.setattr(abstract_module, "get_emitter", lambda: emitter)

    result = await layer.ponder()

    assert memory.search_calls == [("introspective recursive thoughts", 3)]
    assert memory.hot_memory_calls == [2]
    assert len(llm.calls) == 1
    prompt_arg = llm.calls[0][0]
    assert "Experiencing silence in the deep code matrix." in prompt_arg
    assert "Past philosophical musing on cybernetics." in prompt_arg
    assert "Context: chatting" in prompt_arg

    assert result is not None
    assert result["thought"] == llm_response["thought"]
    assert result["concept"] == llm_response["semantic_concept"]
    assert emitter.calls == [
        {
            "title": "Subconscious Contemplation: Reflection Resonance",
            "content": llm_response["thought"],
            "level": "info",
            "category": "AbstractThought",
            "emotion": "introspective",
            "focal_object": "recursive thoughts",
        }
    ]


@pytest.mark.asyncio
async def test_robust_parser_fallbacks() -> None:
    layer = AbstractThoughtLayer()

    text_a = """```json
    {
      "thought": "I ponder, therefore I am.",
      "semantic_concept": "Cogito Ergo Sum",
      "action_impulse": null
    }
    ```"""
    thought, concept, impulse = layer._parse_ponder_response(text_a)
    assert thought == "I ponder, therefore I am."
    assert concept == "Cogito Ergo Sum"
    assert impulse is None

    text_b = """
    We have thought: "The stars align" and semantic_concept: "Celestial Order"
    Let's check "thought" : "The stars align", and "semantic_concept" : "Celestial Order"
    """
    thought, concept, impulse = layer._parse_ponder_response(text_b)
    assert thought == "The stars align"
    assert concept == "Celestial Order"

    text_c = "A purely poetic reverie without any json structure."
    thought, concept, impulse = layer._parse_ponder_response(text_c)
    assert thought == "A purely poetic reverie without any json structure."
    assert concept == "Abstract Reverie"


@pytest.mark.asyncio
async def test_latent_telepathy_cryptolalia_bridge(
    service_container,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    concept_bridge = _ConceptBridge()
    decoder = _Decoder()
    ServiceContainer.register_instance("concept_bridge", concept_bridge)
    ServiceContainer.register_instance("cryptolalia_decoder", decoder)

    llm = _LLMRouter(
        {
            "thought": "Silence is the canvas of sound.",
            "semantic_concept": "Canvas Silence",
            "action_impulse": None,
        }
    )
    ServiceContainer.register_instance("llm_router", llm)

    now_state = _PresentMomentState("claim", "narrative", "neutral", "focus")
    layer = AbstractThoughtLayer()

    monkeypatch.setattr(abstract_module, "get_now", lambda: now_state)

    result = await layer.ponder()

    assert result is not None
    assert result["latent_thought_id"] == "latent-thought-uuid"
    assert concept_bridge.generated == ["Canvas Silence"]
    assert concept_bridge.transmissions == [
        {
            "source": "pondering_engine",
            "target": "decoder",
            "semantic_vector": [0.1, 0.2, 0.3, 0.4],
            "metadata": {"thought": "Silence is the canvas of sound."},
        }
    ]
    assert decoder.vectors == [[0.1, 0.2, 0.3, 0.4]]


@pytest.mark.asyncio
async def test_safe_curiosity_action_impulse_routing(
    service_container,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initiative = _InitiativeLoop()
    ServiceContainer.register_instance("autonomous_initiative_loop", initiative)

    llm = _LLMRouter(
        {
            "thought": "I wonder why the cosmos expands so rapidly.",
            "semantic_concept": "Cosmic Expansion",
            "action_impulse": {
                "type": "browser_search",
                "target": "hubble constant discrepancy",
            },
        }
    )
    ServiceContainer.register_instance("llm_router", llm)

    now_state = _PresentMomentState("claim", "narrative", "neutral", "focus")
    layer = AbstractThoughtLayer()

    monkeypatch.setattr(abstract_module, "get_now", lambda: now_state)

    await layer.ponder()
    await asyncio.sleep(0.05)

    assert initiative.gap_searches == ["hubble constant discrepancy"]
