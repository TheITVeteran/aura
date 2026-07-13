import asyncio

import pytest

from core.runtime import conversation_support
from core.state.aura_state import AuraState


@pytest.mark.asyncio
async def test_record_conversation_experience_prefers_memory_facade_commit(monkeypatch):
    state = AuraState.default()
    captured = {}

    class DummyFacade:
        async def commit_interaction(self, **kwargs):
            captured.update(kwargs)
            return "episode-1"

    def fake_optional_service(*names, default=None):
        if "memory_facade" in names:
            return DummyFacade()
        if "episodic_memory" in names:
            raise AssertionError("record_conversation_experience should not bypass memory_facade when it exists")
        return default

    monkeypatch.setattr(conversation_support.service_access, "optional_service", fake_optional_service)
    monkeypatch.setattr(
        conversation_support,
        "update_conversational_intelligence",
        lambda *args, **kwargs: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        conversation_support,
        "record_shared_ground_callbacks",
        lambda *args, **kwargs: asyncio.sleep(0),
    )

    await conversation_support.record_conversation_experience(
        "Please explain this architecture in detail.",
        (
            "The architecture routes each request through one governed cognition path, "
            "then verifies effects before committing the result to durable state."
        ),
        state,
    )

    assert captured["action"] == "conversation_reply"
    assert captured["success"] is True
    assert captured["metadata"]["origin"] == "api"
    assert captured["metadata"]["domain"] == "conversation"
    assert captured["metadata"]["objective"] == "Please explain this architecture in detail."
    assert captured["metadata"]["semantic_mode"] == "technical"
    assert captured["metadata"]["learning_admission"] == "verified"


@pytest.mark.asyncio
async def test_record_conversation_experience_adds_searchable_continuity_memory(monkeypatch):
    state = AuraState.default()
    captured = {"commit": None, "continuity": None}

    class DummyFacade:
        async def commit_interaction(self, **kwargs):
            captured["commit"] = kwargs
            return "episode-1"

        async def add_memory(self, text, metadata=None):
            captured["continuity"] = {"text": text, "metadata": dict(metadata or {})}
            return True

    def fake_optional_service(*names, default=None):
        if "memory_facade" in names:
            return DummyFacade()
        return default

    monkeypatch.setattr(conversation_support.service_access, "optional_service", fake_optional_service)
    monkeypatch.setattr(
        conversation_support,
        "update_conversational_intelligence",
        lambda *args, **kwargs: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        conversation_support,
        "record_shared_ground_callbacks",
        lambda *args, **kwargs: asyncio.sleep(0),
    )

    await conversation_support.record_conversation_experience(
        "My laptop crashed when Aura used over 100GB of RAM.",
        "I will treat that as a live desktop reliability fault and preserve the context.",
        state,
    )

    assert captured["commit"] is not None
    continuity = captured["continuity"]
    assert continuity is not None
    assert continuity["text"].startswith("Conversation continuity memory.")
    assert "100GB of RAM" in continuity["text"]
    assert continuity["metadata"]["memory_type"] == "conversation_continuity"
    assert continuity["metadata"]["searchable_conversation_context"] is True
    assert continuity["metadata"]["preserve_for_continuity"] is True
    assert continuity["metadata"]["provenance_source"] == "live_conversation_turn"
    assert continuity["metadata"]["learning_admission"] == "verified"


@pytest.mark.asyncio
async def test_record_conversation_experience_rejects_misgrounded_self_condition(
    monkeypatch,
):
    calls = []

    class DummyFacade:
        async def commit_interaction(self, **kwargs):
            calls.append(("commit", kwargs))
            return "episode-should-not-exist"

        async def add_memory(self, text, metadata=None):
            calls.append(("memory", text, metadata))
            return True

    monkeypatch.setattr(
        conversation_support.service_access,
        "optional_service",
        lambda *names, default=None: (
            DummyFacade() if "memory_facade" in names else default
        ),
    )
    monkeypatch.setattr(
        conversation_support,
        "update_conversational_intelligence",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("misgrounded reply reached conversational intelligence")
        ),
    )

    await conversation_support.record_conversation_experience(
        "Are you okay though? Feeling fine?",
        (
            "I am with you. RAM pressure is 75.6% with 15.6 GB available; "
            "CPU load is 25.8% on this host."
        ),
        AuraState.default(),
    )

    assert calls == []
