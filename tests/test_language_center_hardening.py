from __future__ import annotations

from types import SimpleNamespace

import pytest


def _thought(**kwargs):
    from core.inner_monologue import ThoughtPacket

    return ThoughtPacket(
        stance=kwargs.get("stance", "direct"),
        tone=kwargs.get("tone", "direct"),
        model_tier=kwargs.get("model_tier", "local"),
        llm_briefing=kwargs.get("llm_briefing", "Say the important thing clearly."),
    )


@pytest.mark.asyncio
async def test_language_center_express_falls_back_when_router_lookup_fails(monkeypatch):
    import core.language_center as language

    records = []
    center = language.LanguageCenter()

    async def failing_router_lookup():
        message = "container unavailable"
        raise RuntimeError(message)

    monkeypatch.setattr(center, "_ensure_router", failing_router_lookup)
    monkeypatch.setattr(language, "record_degradation", lambda *args, **kwargs: records.append((args, kwargs)))

    response = await center.express(_thought(), "hello", history=[])

    assert "trouble putting them into words" in response
    assert records
    assert "fallback expression" in records[0][1]["action"]


@pytest.mark.asyncio
async def test_language_center_stream_falls_back_when_router_missing(monkeypatch):
    import core.language_center as language

    center = language.LanguageCenter()

    async def no_router():
        return False

    monkeypatch.setattr(center, "_ensure_router", no_router)

    chunks = [
        chunk
        async for chunk in center.express_stream(
            _thought(tone="warm"),
            "hello",
            history=[],
        )
    ]

    assert len(chunks) == 1
    assert "language center failed" in chunks[0]


@pytest.mark.asyncio
async def test_language_center_start_records_registration_failure(monkeypatch):
    import core.event_bus as event_bus
    import core.language_center as language

    records = []
    center = language.LanguageCenter()

    class BrokenBus:
        async def publish(self, *_args, **_kwargs):
            message = "event bus offline"
            raise RuntimeError(message)

    async def no_router():
        return False

    monkeypatch.setattr(center, "_ensure_router", no_router)
    monkeypatch.setattr(event_bus, "get_event_bus", lambda: BrokenBus())
    monkeypatch.setattr(language, "record_degradation", lambda *args, **kwargs: records.append((args, kwargs)))

    await center.start()

    assert center.get_status()["mycelium_registered"] is False
    assert records
    assert "mycelium registration" in records[0][1]["action"]


@pytest.mark.asyncio
async def test_language_center_dispatch_failure_attempts_prompt_route(monkeypatch):
    import core.language_center as language

    records = []
    center = language.LanguageCenter()
    calls = []

    class Router:
        async def generate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            if "messages" in kwargs:
                message = "message route unavailable"
                raise RuntimeError(message)
            return "Aura: clean answer"

    center._router = Router()
    monkeypatch.setattr(language, "record_degradation", lambda *args, **kwargs: records.append((args, kwargs)))

    response = await center.express(
        _thought(llm_briefing="Speak plainly."),
        "hello",
        history=[SimpleNamespace(role="user", content="bad entry")],
    )

    assert response == "clean answer"
    assert len(calls) == 2
    assert records
    assert records[0][1]["extra"]["dispatch"] == "messages"
