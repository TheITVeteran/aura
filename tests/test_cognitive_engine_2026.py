"""tests/test_cognitive_engine_2026.py
A-Tier verification for the hardened Cognitive Engine.
Updated to match the modular-phase facade API.
"""

import pytest
import asyncio

from core.brain.cognitive_engine import CognitiveEngine
from core.brain.types import ThinkingMode, Thought
from core.container import ServiceContainer
from core.state.aura_state import AuraState


class CognitiveBackendProbe:
    async def generate(self, prompt, system_prompt, options=None):
        return '{"content": "I am thinking clearly.", "reasoning": ["Step 1"], "confidence": 0.9}'

    async def chat_stream_async(self, messages):
        yield "Thinking..."
        yield "Done."

    async def check_health_async(self):
        return True


class PhaseProbe:
    marker = "phase-probe"


class RecoveryRepositoryProbe:
    def __init__(self, rollback_started, release_rollback):
        self.rollback_started = rollback_started
        self.release_rollback = release_rollback
        self.rollback_reasons = []

    async def rollback(self, reason):
        self.rollback_reasons.append(reason)
        self.rollback_started.set()
        await self.release_rollback.wait()


class StateRepositoryProbe:
    def __init__(self):
        self.state = AuraState.default()

    async def get_current(self):
        return self.state


class StreamEventProbe:
    def __init__(self, content):
        self.content = content


class RouterStreamProbe:
    def __init__(self):
        self.messages = []

    async def think_stream(self, messages, **kwargs):
        self.messages.append({"messages": messages, "kwargs": kwargs})
        for token in ["Thinking...", "Done."]:
            yield StreamEventProbe(token)


class InteractionRecorderProbe:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    async def record_interaction(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if self.fail:
            raise RuntimeError("recording failed")


@pytest.fixture(autouse=True)
def clean_container():
    ServiceContainer.reset()
    yield
    ServiceContainer.reset()


@pytest.fixture
def engine():
    backend = CognitiveBackendProbe()
    engine = CognitiveEngine(backend=backend)
    return engine

@pytest.mark.asyncio
async def test_engine_initialization(engine):
    assert engine.backend is not None
    assert len(engine.thoughts) == 0

@pytest.mark.asyncio
async def test_engine_setup(engine, monkeypatch):
    monkeypatch.setattr(
        "core.brain.cognitive_engine.instantiate_legacy_runtime_phases",
        lambda _kernel, include_executive_closure=False: [("probe", PhaseProbe())],
    )

    engine.setup()
    assert len(engine._phases) == 1
    assert "PhaseProbe" in engine.phase_map

@pytest.mark.asyncio
async def test_engine_think_no_response(engine):
    """Test think() when no assistant response is generated."""
    engine._phases = []  # No phases — no response generated

    thought = await engine.think("Hello", origin="test")
    assert isinstance(thought, Thought)
    # Honest-suppression contract: when no answer-quality response is
    # produced, the engine returns an explicitly suppressed empty thought
    # (confidence 0.0) instead of fabricating a 0.5-confidence filler —
    # downstream layers own retry/fallback with receipts.
    assert thought.confidence == 0.0
    assert thought.metadata.get("suppressed") is True
    assert any("cycle_no_response" in r for r in (thought.reasoning or []))

@pytest.mark.asyncio
async def test_engine_health_check(engine):
    # CP126: check_health previously returned "healthy" unconditionally —
    # a zero-phase engine with no repository reported a full spectrum. The
    # honest contract names its gaps.
    health = await engine.check_health()
    assert health["status"] == "degraded"
    assert "no_phases_loaded" in health["issues"]
    assert "state_repository_absent" in health["issues"]
    assert health["modular"] is True
    assert "phases_count" in health

    # Equipped engine reports healthy with no issues.
    engine._phases = [object()]
    engine.state_repository = object()
    health = await engine.check_health()
    assert health["status"] == "healthy"
    assert health["issues"] == []

@pytest.mark.asyncio
async def test_reactive_recovery_does_not_hold_lock_while_rollback_runs(engine):
    rollback_started = asyncio.Event()
    release_rollback = asyncio.Event()

    engine.state_repository = RecoveryRepositoryProbe(rollback_started, release_rollback)

    first_recovery = asyncio.create_task(
        engine._reactive_recovery("Hello", ThinkingMode.FAST, "api", "test-failure")
    )

    await asyncio.wait_for(rollback_started.wait(), timeout=1.0)

    second = await asyncio.wait_for(
        engine._reactive_recovery("Hello again", ThinkingMode.FAST, "api", "test-failure-2"),
        timeout=1.0,
    )

    assert "still gathering" in second.content.lower()

    release_rollback.set()
    first = await asyncio.wait_for(first_recovery, timeout=1.0)
    assert isinstance(first, Thought)

@pytest.mark.asyncio
async def test_engine_think_stream(engine):
    """Test think_stream() using a concrete router probe."""
    engine.state_repository = StateRepositoryProbe()
    router = RouterStreamProbe()
    ServiceContainer.register_instance("llm_router", router)

    tokens = []
    async for token in engine.think_stream("Stream test"):
        tokens.append(token)

    assert tokens == ["Thinking...", "Done."]
    assert router.messages


@pytest.mark.asyncio
async def test_record_interaction_prefers_context_manager():
    engine = CognitiveEngine()
    context_manager = InteractionRecorderProbe()
    learning_engine = InteractionRecorderProbe()
    ServiceContainer.register_instance("context_manager", context_manager)
    ServiceContainer.register_instance("learning_engine", learning_engine)

    await engine.record_interaction("Hi", "Hey there", domain="relational")

    assert context_manager.calls == [
        {"args": ("Hi", "Hey there"), "kwargs": {"domain": "relational"}}
    ]
    assert learning_engine.calls == []


@pytest.mark.asyncio
async def test_record_interaction_falls_back_to_learning_engine():
    engine = CognitiveEngine()
    context_manager = InteractionRecorderProbe(fail=True)
    learning_engine = InteractionRecorderProbe()
    ServiceContainer.register_instance("context_manager", context_manager)
    ServiceContainer.register_instance("learning_engine", learning_engine)

    await engine.record_interaction("Hi", "Hey there", domain="relational")

    assert learning_engine.calls == [
        {
            "args": (),
            "kwargs": {
                "user_input": "Hi",
                "aura_response": "Hey there",
                "domain": "relational",
            },
        }
    ]
