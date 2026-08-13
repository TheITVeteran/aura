from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.conversation.conversational_momentum_engine import (
    ConversationalMomentumEngine,
    ConversationThread,
)


class _AssistantIngress:
    def __init__(self) -> None:
        self.expressions: list[tuple[str, int]] = []

    async def __call__(self, content: str, urgency: int) -> None:
        self.expressions.append((content, urgency))


class _UserIngress:
    async def __call__(self, *_args, **_kwargs):
        raise AssertionError("assistant speech must never enter through user ingress")


@pytest.mark.asyncio
async def test_spontaneous_turn_uses_typed_assistant_expression(monkeypatch) -> None:
    assistant_ingress = _AssistantIngress()
    orchestrator = SimpleNamespace(
        _last_user_interaction_time=0.0,
        status=SimpleNamespace(is_processing=False),
        _proactive_notify_callback=assistant_ingress,
        process_user_input=_UserIngress(),
    )
    engine = ConversationalMomentumEngine(orchestrator)
    engine.running = True
    thread = ConversationThread(topic="causal memory", last_turn="causal memory", momentum=0.2)

    monkeypatch.setattr(
        "core.conversation.conversational_momentum_engine.background_activity_reason",
        lambda *_args, **_kwargs: "",
    )
    await engine._trigger_spontaneous_turn(thread)

    assert len(assistant_ingress.expressions) == 1
    content, urgency = assistant_ingress.expressions[0]
    assert "causal memory" in content
    assert urgency == 2


@pytest.mark.asyncio
async def test_generated_turn_uses_the_same_assistant_ingress(monkeypatch) -> None:
    assistant_ingress = _AssistantIngress()
    orchestrator = SimpleNamespace(
        _proactive_notify_callback=assistant_ingress,
        process_user_input=_UserIngress(),
    )
    router = SimpleNamespace(generate=lambda *_args, **_kwargs: None)

    async def generate(*_args, **_kwargs) -> str:
        return "A generated follow-up"

    router.generate = generate
    monkeypatch.setattr(
        "core.conversation.conversational_momentum_engine.get_runtime_service",
        lambda name, default=None: router if name == "llm_router" else default,
    )
    engine = ConversationalMomentumEngine(orchestrator)
    engine.active_threads = [
        ConversationThread(topic="latent search", last_turn="latent search", momentum=0.8)
    ]

    await engine.generate_spontaneous_turn()

    assert assistant_ingress.expressions == [("A generated follow-up", 2)]


@pytest.mark.asyncio
async def test_missing_assistant_ingress_fails_closed_without_user_reingest() -> None:
    engine = ConversationalMomentumEngine(SimpleNamespace(process_user_input=_UserIngress()))

    assert await engine._emit_assistant_expression("private thought", urgency=2) is False
    assert await engine._emit_assistant_expression("   ", urgency=2) is False
