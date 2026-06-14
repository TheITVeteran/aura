import asyncio
from types import SimpleNamespace

import pytest

from core.brain.llm.runtime_wiring import (
    build_agentic_tool_map,
    derive_substrate_generation_overrides,
    prepare_runtime_payload,
)
from core.runtime.errors import get_degradation_tracker
from core.state.aura_state import AuraState


@pytest.fixture(autouse=True)
def _reset_degradation_tracker():
    get_degradation_tracker().reset()
    yield
    get_degradation_tracker().reset()


@pytest.mark.asyncio
async def test_prepare_runtime_payload_preserves_prompt_when_contract_state_is_invalid():
    class _SealedCognition:
        def __setattr__(self, _name, _value):
            attempts = object.__getattribute__(self, "__dict__").get("_set_attempts", 0)
            object.__setattr__(self, "_set_attempts", attempts + 1)
            raise AttributeError("cognition is sealed")

        def __getattr__(self, _name):
            attempts = object.__getattribute__(self, "__dict__").get("_get_attempts", 0)
            object.__setattr__(self, "_get_attempts", attempts + 1)
            raise AttributeError("cognition is sealed")

    class _BrokenState:
        cognition = _SealedCognition()

    prompt, system_prompt, messages, contract, runtime_state = await prepare_runtime_payload(
        prompt="Can you still answer?",
        system_prompt=None,
        messages=None,
        state=_BrokenState(),
        origin="api",
        is_background=False,
    )

    assert prompt == "Can you still answer?"
    assert system_prompt is None
    assert messages is None
    assert contract is None
    assert runtime_state is not None
    actions = [
        record.action for record in get_degradation_tracker().recent(subsystem="runtime_wiring")
    ]
    assert (
        "continued with unstamped runtime state; response contract will be built from explicit objective"
        in actions
    )
    assert "continued without a response contract after contract construction failed" in actions
    assert "using raw prompt/messages because context assembler failed" in actions


@pytest.mark.asyncio
async def test_prepare_runtime_payload_records_memory_hydration_failure(monkeypatch):
    state = AuraState.default()

    class _BrokenMemoryFacade:
        async def search(self, _query, limit=5):
            self.search_calls = getattr(self, "search_calls", 0) + 1
            raise RuntimeError("vector store offline")

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: _BrokenMemoryFacade() if name == "memory_facade" else default
        ),
    )

    prompt, _, messages, contract, _ = await prepare_runtime_payload(
        prompt="What do you remember about our dynamic?",
        system_prompt=None,
        messages=[{"role": "user", "content": "What do you remember about our dynamic?"}],
        state=state,
        origin="api",
        is_background=False,
    )

    assert prompt == "User: What do you remember about our dynamic?"
    assert messages is not None
    assert contract is not None
    last = get_degradation_tracker().recent(subsystem="runtime_wiring")[-1]
    assert (
        last.action
        == "continued payload assembly with existing state memory after retrieval hydration failed"
    )


@pytest.mark.asyncio
async def test_prepare_runtime_payload_bounds_slow_memory_hydration(monkeypatch):
    state = AuraState.default()
    monkeypatch.setenv("AURA_RUNTIME_MEMORY_HYDRATION_TIMEOUT_S", "0.05")

    class _SlowMemoryFacade:
        async def search(self, _query, limit=5):
            await asyncio.sleep(10.0)
            return [{"content": "too late", "metadata": {"type": "fact"}}]

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: _SlowMemoryFacade() if name == "memory_facade" else default
        ),
    )

    prompt, _, messages, contract, _ = await prepare_runtime_payload(
        prompt="What do you remember about our plan?",
        system_prompt=None,
        messages=[{"role": "user", "content": "What do you remember about our plan?"}],
        state=state,
        origin="desktop",
        is_background=False,
    )

    assert prompt == "User: What do you remember about our plan?"
    assert messages is not None
    assert contract is not None
    last = get_degradation_tracker().recent(subsystem="runtime_wiring")[-1]
    assert (
        last.action
        == "continued payload assembly with existing state memory after retrieval hydration failed"
    )


@pytest.mark.asyncio
async def test_prepare_runtime_payload_returns_turn_consistent_state_snapshot(monkeypatch):
    state = AuraState.default()
    state.affect.dominant_emotion = "curious"
    state.cognition.attention_focus = "the user's current question"

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda _name, default=None: default),
    )

    _, _, messages, contract, payload_state = await prepare_runtime_payload(
        prompt="What are you noticing right now?",
        system_prompt=None,
        messages=None,
        state=state,
        origin="desktop",
        is_background=False,
    )

    assert payload_state is not state
    assert payload_state.affect.dominant_emotion == "curious"
    assert payload_state.cognition.attention_focus == "What are you noticing right now?"
    assert payload_state.cognition.current_objective == "What are you noticing right now?"
    assert state.cognition.attention_focus == "the user's current question"
    assert messages is not None
    assert contract is not None

    state.affect.dominant_emotion = "frustrated"
    state.cognition.attention_focus = "a later background event"

    assert payload_state.affect.dominant_emotion == "curious"
    assert payload_state.cognition.attention_focus == "What are you noticing right now?"


def test_build_agentic_tool_map_records_capability_registry_failure(monkeypatch):
    def _broken_get(name, default=None):
        if name == "capability_engine":
            raise RuntimeError("capability registry unavailable")
        return default

    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(_broken_get))

    assert (
        build_agentic_tool_map(
            required_skill="web_search",
            objective="Search, summarize, and save the result.",
        )
        is None
    )

    last = get_degradation_tracker().recent(subsystem="runtime_wiring")[-1]
    assert last.action == "returned no agentic tool map after capability registry lookup failed"


def test_build_agentic_tool_map_skips_capability_inventory_questions(monkeypatch):
    def _should_not_read_registry(name, default=None):
        if name == "capability_engine":
            raise AssertionError("capability inventory questions must not build tool maps")
        return default

    monkeypatch.setattr("core.container.ServiceContainer.get", staticmethod(_should_not_read_registry))

    assert (
        build_agentic_tool_map(
            objective="What tools can you hypothetically use externally on my computer?",
            max_tools=8,
        )
        is None
    )


def test_substrate_generation_overrides_reuse_fresh_turn_profile(monkeypatch):
    class VoiceEngine:
        def __init__(self):
            self.profile = SimpleNamespace(compilation_source="bounded_voice")
            self.calls = []
            self.compile_calls = 0

        def get_generation_params_for(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "temperature": 0.62,
                "top_p": 0.81,
                "substrate_profile_reused": True,
                "substrate_profile_age_s": 0.21,
            }

        def compile_profile(self, **_kwargs):
            self.compile_calls += 1
            return self.profile

        def get_current_profile(self):
            return self.profile

    engine = VoiceEngine()
    runtime_state = object()
    monkeypatch.setattr(
        "core.voice.substrate_voice_engine.get_substrate_voice_engine",
        lambda: engine,
    )

    overrides = derive_substrate_generation_overrides(
        runtime_state=runtime_state,
        objective="explain what tools you can use",
        origin="desktop",
        is_background=False,
    )

    assert engine.calls == [
        {
            "state": runtime_state,
            "user_message": "explain what tools you can use",
            "origin": "desktop",
        }
    ]
    assert engine.compile_calls == 0
    assert overrides["temperature"] == pytest.approx(0.62)
    assert overrides["top_p"] == pytest.approx(0.81)
    assert overrides["substrate_generation_source"] == "bounded_voice, reused_runtime_profile"
