import asyncio
from types import SimpleNamespace

import pytest

from core.brain.cognitive_engine import CognitiveEngine
from core.brain.types import ThinkingMode, Thought
from core.runtime.errors import get_degradation_tracker
from core.state.aura_state import AuraState


class StateRepositoryFixture:
    def __init__(self, state):
        self._current = state
        self.get_current_calls = 0
        self.commits = []

    async def get_current(self):
        self.get_current_calls += 1
        return self._current

    async def commit(self, state, *args, **kwargs):
        self.commits.append((state, args, kwargs))
        self._current = state


def test_cognitive_engine_treats_prefixed_user_origin_as_foreground():
    assert CognitiveEngine._is_background_request("routing_user", False) is False
    assert CognitiveEngine._is_background_request("routing_voice_command", False) is False
    assert CognitiveEngine._is_background_request("autonomous_thought", False) is True


def test_cognitive_engine_treats_live_desktop_origins_as_user_facing():
    assert CognitiveEngine._is_user_facing_origin("chat_api") is True
    assert CognitiveEngine._is_user_facing_origin("desktop_ui") is True
    assert CognitiveEngine._is_user_facing_origin("voice_bridge") is True
    assert CognitiveEngine._is_user_facing_origin("agency_core") is False


def test_cognitive_engine_live_desktop_origin_updates_working_memory(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default().derive("cognitive_intent: desktop_ui", origin="desktop_ui")
    engine.state_repository = None
    engine._phases = []

    monkeypatch.setenv("AURA_TESTING", "1")

    thought = asyncio.run(
        engine._run_thinking_loop(
            state,
            "Desktop live path should stay foreground.",
            ThinkingMode.FAST,
            "desktop_ui",
        )
    )

    assert state.transition_origin == "desktop_ui"
    assert state.cognition.working_memory[-1]["role"] == "user"
    assert state.cognition.working_memory[-1]["origin"] == "desktop_ui"
    assert thought.reasoning


def test_cognitive_engine_preserves_desktop_origin_after_phase_derives(monkeypatch):
    class _ResettingPhase:
        async def execute(self, state, objective=None, **_kwargs):
            derived = state.derive("phase_default_origin_reset")
            derived.cognition.working_memory.append(
                {
                    "role": "assistant",
                    "content": "I will keep the live desktop turn on the foreground path.",
                }
            )
            return derived

    engine = CognitiveEngine()
    repo = StateRepositoryFixture(AuraState.default())
    engine.state_repository = repo
    engine._phases = [_ResettingPhase()]

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(get=lambda name, default=None: repo if name == "state_repository" else default),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: default,
    )
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "",
    )

    thought = asyncio.run(
        engine.think(
            "Keep this live desktop turn foreground.",
            mode=ThinkingMode.FAST,
            origin="desktop_ui",
        )
    )

    assert thought.content == "I will keep the live desktop turn on the foreground path."
    assert repo.commits
    committed_state = repo.commits[-1][0]
    assert committed_state.transition_origin == "desktop_ui"
    assert committed_state.cognition.current_origin is None


def test_cognitive_engine_uses_canonical_context_assembler():
    from core.brain.llm.context_assembler import ContextAssembler

    CognitiveEngine()

    assert ContextAssembler.build_system_prompt.__module__ == "core.brain.llm.context_assembler"
    assert ContextAssembler.build_messages.__module__ == "core.brain.llm.context_assembler"
    assert not getattr(ContextAssembler, "_patched_v1", False)


@pytest.mark.asyncio
async def test_cognitive_engine_skips_identity_refresh_for_background_origin(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = StateRepositoryFixture(state)
    captured = {}

    async def _fake_run(state, objective, mode, origin, context=None, **kwargs):
        captured["objective"] = objective
        return Thought(id="bg-thought", content="ok", mode=mode)

    monitor = SimpleNamespace(needs_context_refresh=lambda *_args, **_kwargs: True)

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(get=lambda name, default=None: repo if name == "state_repository" else default),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: monitor if name == "drift_monitor" else default,
    )
    monkeypatch.setattr(
        "core.brain.cognitive_engine.ContextAssembler.build_system_prompt",
        staticmethod(lambda _state: "context" * 500),
    )
    monkeypatch.setattr(engine, "_run_thinking_loop", _fake_run)

    thought = await engine.think(
        "Summarize internal maintenance state.",
        mode=ThinkingMode.FAST,
        origin="autonomous",
        is_background=True,
    )

    assert thought.content == "ok"
    assert captured["objective"] == "Summarize internal maintenance state."


@pytest.mark.asyncio
async def test_cognitive_engine_suppresses_background_thoughts_when_background_policy_blocks(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = StateRepositoryFixture(state)

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(get=lambda name, default=None: repo if name == "state_repository" else SimpleNamespace()),
    )
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "failure_lockdown_0.20",
    )

    thought = await engine.think(
        "Distill this memory to its essential insight.",
        mode=ThinkingMode.FAST,
        origin="sovereign_pruner",
        is_background=True,
    )

    assert thought.metadata["suppressed"] is True
    assert "background_thought_suppressed" in thought.reasoning[0]


@pytest.mark.asyncio
async def test_cognitive_engine_background_no_response_is_quiet_noop(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = StateRepositoryFixture(state)
    engine.state_repository = repo
    engine._phases = []

    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "",
    )

    thought = await engine.think(
        "I was curious about the host environment, so I initiated a system scan.",
        mode=ThinkingMode.FAST,
        origin="agency_core_environmental_explorer",
        is_background=True,
    )

    assert thought.content == ""
    assert thought.metadata["suppressed"] is True
    assert "background_cycle_no_response" in thought.reasoning[0]


@pytest.mark.asyncio
async def test_cognitive_engine_resolves_missing_origin_from_orchestrator(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = StateRepositoryFixture(state)
    orchestrator = SimpleNamespace(_current_origin="terminal_monitor")
    captured = {}

    async def _fake_run(state, objective, mode, origin, context=None, **kwargs):
        captured["origin"] = origin
        return Thought(id="origin-from-orch", content="ok", mode=mode)

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: (
                orchestrator if name == "orchestrator" else repo if name == "state_repository" else default
            )
        ),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: default,
    )
    monkeypatch.setattr(
        "core.runtime.background_policy.background_activity_reason",
        lambda *args, **kwargs: "",
    )
    monkeypatch.setattr(engine, "_run_thinking_loop", _fake_run)

    thought = await engine.think("Investigate the timeout.", mode=ThinkingMode.FAST)

    assert thought.content == "ok"
    assert captured["origin"] == "terminal_monitor"


@pytest.mark.asyncio
async def test_cognitive_engine_defaults_missing_origin_to_system(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = StateRepositoryFixture(state)
    captured = {}

    async def _fake_run(state, objective, mode, origin, context=None, **kwargs):
        captured["origin"] = origin
        return Thought(id="origin-default", content="ok", mode=mode)

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(get=lambda name, default=None: repo if name == "state_repository" else default),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: default,
    )
    monkeypatch.setattr(engine, "_run_thinking_loop", _fake_run)

    thought = await engine.think("Perform internal maintenance.", mode=ThinkingMode.FAST)

    assert thought.content == "ok"
    assert captured["origin"] == "system"


@pytest.mark.asyncio
async def test_cognitive_engine_user_recovery_uses_bounded_primary_router(monkeypatch):
    engine = CognitiveEngine()
    captured = {}

    class _Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return "I am still on the live desktop thread and answering the user directly."

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(get=lambda name, default=None: _Router() if name == "llm_router" else default),
    )

    thought = await engine._reactive_recovery(
        "Answer directly: are you still on the live desktop thread?",
        ThinkingMode.FAST,
        "desktop_ui",
        "timeout",
    )

    assert thought.content.startswith("I am still on the live desktop thread")
    assert captured["prefer_tier"] == "primary"
    assert captured["foreground_request"] is True
    assert captured["skip_runtime_payload"] is True
    assert captured["allow_deep_handoff"] is False
    assert captured["max_tokens"] <= 384


@pytest.mark.asyncio
async def test_cognitive_engine_reactive_recovery_delegates_rollback_governance_to_repository(
    monkeypatch,
):
    from core.governance_context import get_active_governance

    engine = CognitiveEngine()
    captured = {}

    class _Repo:
        async def rollback(self, reason):
            token = get_active_governance()
            captured["reason"] = reason
            captured["token"] = token

    class _Router:
        async def think(self, **_kwargs):
            return "I recovered the live user-facing turn through the governed primary router."

    engine.state_repository = _Repo()
    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(get=lambda name, default=None: _Router() if name == "llm_router" else default),
    )

    thought = await engine._reactive_recovery(
        "Answer directly after a cognitive timeout.",
        ThinkingMode.FAST,
        "desktop_ui",
        "timeout",
    )

    assert thought.content.startswith("I recovered the live user-facing turn")
    assert captured["reason"] == "recovery: timeout"
    assert captured["token"] is None


@pytest.mark.asyncio
async def test_cognitive_engine_desktop_quick_reply_uses_governed_primary_router(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    repo = StateRepositoryFixture(state)
    engine.state_repository = repo
    engine._phases = [
        SimpleNamespace(
            execute=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("compact desktop quick reply should not enter phase loop")
            )
        )
    ]
    captured = {}

    class _Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return "I am on the live desktop path and answering this turn directly."

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router() if name == "llm_router" else default
        ),
    )

    thought = await engine._run_thinking_loop(
        state,
        "You ok?",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "visible_user_message": "You ok?",
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "max_tokens": 512,
        },
        is_background=False,
        timeout_s=42.0,
    )

    assert thought.content.startswith("I am on the live desktop path")
    assert captured["prefer_tier"] == "primary"
    assert captured["protected_foreground_lane"] is True
    assert captured["desktop_cognitive_engine_required"] is True
    assert captured["allow_cloud_fallback"] is False
    assert captured["allow_deep_handoff"] is False
    assert captured["skip_runtime_payload"] is True
    assert repo.commits
    committed = repo.commits[-1][0]
    assert committed.cognition.working_memory[-2]["role"] == "user"
    assert committed.cognition.working_memory[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_cognitive_engine_desktop_quick_reply_includes_recent_context(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)
    captured = {}

    class _Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return "I am carrying the recent context forward instead of losing the thread."

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router() if name == "llm_router" else default
        ),
    )

    thought = await engine._run_thinking_loop(
        state,
        "Continue from there.",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "visible_user_message": "Continue from there.",
            "recent_conversation_context": (
                "User: The live desktop lane lost context.\n"
                "Aura: I should preserve bounded recent exchanges through CognitiveEngine."
            ),
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "max_tokens": 512,
        },
        is_background=False,
        timeout_s=42.0,
    )

    user_message = captured["messages"][1]["content"]
    assert thought.content.startswith("I am carrying")
    assert "[CURRENT USER MESSAGE]" in user_message
    assert "[RECENT COMPLETED CONVERSATION FOR CONTINUITY ONLY]" in user_message
    assert "live desktop lane lost context" in user_message
    assert "Continue from there." in user_message
    assert user_message.index("[CURRENT USER MESSAGE]") < user_message.index(
        "[RECENT COMPLETED CONVERSATION FOR CONTINUITY ONLY]"
    )


@pytest.mark.asyncio
async def test_cognitive_engine_desktop_quick_keeps_runtime_payload_when_required(monkeypatch):
    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)
    captured = {}

    class _Router:
        async def think(self, **kwargs):
            captured.update(kwargs)
            return "I am answering from the live path with the current thread in view."

    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router() if name == "llm_router" else default
        ),
    )

    thought = await engine._run_thinking_loop(
        state,
        "Hey Aura, are you there?",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "live_runtime_payload_required": True,
            "visible_user_message": "Hey Aura, are you there?",
            "live_speech_grounding_frame": {
                "attention_focus": "Bryan's live desktop check",
                "dominant_action": "answer",
                "mood": "steady",
            },
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
            "max_tokens": 512,
        },
        is_background=False,
        timeout_s=42.0,
    )

    assert thought.content.startswith("I am answering")
    assert captured["skip_runtime_payload"] is False
    assert captured["allow_cloud_fallback"] is False
    assert "LIVE SPEECH GROUNDING" in captured["messages"][0]["content"]
    assert "not prose to repeat" in captured["messages"][0]["content"]


@pytest.mark.asyncio
async def test_cognitive_engine_desktop_quick_failure_does_not_enter_second_model_path(
    monkeypatch,
):
    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = StateRepositoryFixture(state)
    phase_calls = 0
    router_calls = 0

    class _Phase:
        async def execute(self, *_args, **_kwargs):
            nonlocal phase_calls
            phase_calls += 1
            raise AssertionError("failed compact desktop turn must not enter the full phase loop")

    class _Router:
        async def think(self, **_kwargs):
            nonlocal router_calls
            router_calls += 1
            raise TimeoutError("cold Cortex exceeded compact deadline")

    engine._phases = [_Phase()]
    monkeypatch.setattr(
        "core.brain.cognitive_engine.get_container",
        lambda: SimpleNamespace(
            get=lambda name, default=None: _Router() if name == "llm_router" else default
        ),
    )

    thought = await engine._run_thinking_loop(
        state,
        "Tell me about distributed systems.",
        ThinkingMode.FAST,
        "desktop_ui",
        context={
            "desktop_quick_reply_contract": True,
            "cognitive_engine_required": True,
            "desktop_cognitive_engine_required": True,
        },
        is_background=False,
        timeout_s=90.0,
    )

    assert router_calls == 1
    assert phase_calls == 0
    assert thought.metadata["desktop_cognitive_engine_failure"] is True
    assert thought.metadata["model_retry_suppressed"] is True
    assert "won't fabricate" in thought.content


@pytest.mark.asyncio
async def test_cognitive_engine_strict_answer_recovery_propagates_cancellation(monkeypatch):
    import core.brain.llm_health_router as router_module

    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = None
    engine._phases = []

    class _CancellingRouter:
        async def think(self, *args, **kwargs):
            await asyncio.sleep(0)
            raise asyncio.CancelledError()

    monkeypatch.setattr(router_module, "get_llm_router", lambda: _CancellingRouter())

    with pytest.raises(asyncio.CancelledError):
        await engine._run_thinking_loop(
            state,
            "Solve exactly. <answer>required</answer>",
            ThinkingMode.FAST,
            "user",
        )


@pytest.mark.asyncio
async def test_cognitive_engine_strict_answer_recovery_records_typed_failure(monkeypatch):
    import core.brain.llm_health_router as router_module

    tracker = get_degradation_tracker()
    tracker.reset()

    engine = CognitiveEngine()
    state = AuraState.default()
    engine.state_repository = None
    engine._phases = []

    class _FailingRouter:
        async def think(self, *args, **kwargs):
            await asyncio.sleep(0)
            raise RuntimeError("router offline")

    monkeypatch.setattr(router_module, "get_llm_router", lambda: _FailingRouter())

    thought = await engine._run_thinking_loop(
        state,
        "Solve exactly. <answer>required</answer>",
        ThinkingMode.FAST,
        "user",
    )

    assert thought.content == ""
    assert "strict_answer_recovery_failed" in thought.reasoning[0]
    records = tracker.recent(subsystem="cognitive_engine", limit=1)
    assert records
    assert records[-1].action == "returned strict answer recovery failure after direct recovery failed"
    tracker.reset()
