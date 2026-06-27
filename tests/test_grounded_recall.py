"""Tests for positional/temporal grounded recall (anti-confabulation)."""
from __future__ import annotations

import pytest

from core.conversation import grounded_recall as gr


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("Do you remember what I first asked", "first"),
        ("what did I first ask", "first"),
        ("what was my first question?", "first"),
        ("the first thing I said to you", "first"),
        ("how did this conversation start", "first"),
        ("what did we start talking about", "first"),
        ("what did I just ask you", "last"),
        ("what was my previous question", "last"),
        # negatives
        ("what is the capital of France", None),
        ("can you open notes", None),
        ("tell me about yourself", None),
        ("how are you feeling", None),
    ],
)
def test_detect_positional_recall(msg, expected):
    assert gr.detect_positional_recall(msg) == expected


def test_resolve_uses_live_transcript_first_and_last(monkeypatch):
    class _Entry:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    class _FakeTranscript:
        _entries = [
            _Entry("aura", "Infinity online."),
            _Entry("user", "you with me, Aura?"),
            _Entry("aura", "Yeah, I'm here."),
            _Entry("user", "checking the convo lane"),
            _Entry("user", "Do you remember what I first asked"),
        ]

        @classmethod
        def get_instance(cls):
            return cls()

    import core.conversation.unified_transcript as ut
    monkeypatch.setattr(ut, "UnifiedTranscript", _FakeTranscript)

    # current turn is excluded; first real user turn is the grounding fact
    first = gr.resolve_positional_turn("Do you remember what I first asked", "first")
    assert first == "you with me, Aura?"
    last = gr.resolve_positional_turn("Do you remember what I first asked", "last")
    assert last == "checking the convo lane"


def test_build_context_block_contains_quote(monkeypatch):
    class _Entry:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    class _FakeTranscript:
        _entries = [_Entry("user", "you with me, Aura?"),
                    _Entry("user", "Do you remember what I first asked")]

        @classmethod
        def get_instance(cls):
            return cls()

    import core.conversation.unified_transcript as ut
    monkeypatch.setattr(ut, "UnifiedTranscript", _FakeTranscript)

    block = gr.build_grounded_recall_context("Do you remember what I first asked")
    assert block is not None
    assert "you with me, Aura?" in block
    assert "GROUNDED RECALL" in block
    assert "quoted speaker is the user, not you" in block
    assert "never as something you said" in block


def test_get_world_state_registers_canonical_singleton(monkeypatch):
    registrations = []

    from core import world_state as world_state_module

    monkeypatch.setattr(world_state_module, "_ws_instance", None)
    monkeypatch.setattr(
        world_state_module.ServiceContainer,
        "has",
        classmethod(lambda cls, name: bool(registrations)),
    )
    monkeypatch.setattr(
        world_state_module.ServiceContainer,
        "register_instance",
        classmethod(
            lambda cls, name, instance, **metadata: registrations.append(
                (name, instance, metadata)
            )
        ),
    )

    first = world_state_module.get_world_state()
    second = world_state_module.get_world_state()

    assert first is second
    assert len(registrations) == 1
    assert registrations[0][0] == "world_state"
    assert registrations[0][1] is first
    assert registrations[0][2]["required_for"] == "live environment grounding"


def test_build_context_none_when_no_prior_turn(monkeypatch):
    class _FakeTranscript:
        _entries = []

        @classmethod
        def get_instance(cls):
            return cls()

    import core.conversation.unified_transcript as ut
    monkeypatch.setattr(ut, "UnifiedTranscript", _FakeTranscript)

    assert gr.build_grounded_recall_context("what did I first ask") is None


def test_build_context_none_for_non_recall():
    assert gr.build_grounded_recall_context("what's the weather like") is None


@pytest.mark.parametrize(
    "reply,expected",
    [
        (
            "I care more about reliability than spectacle because trust is built slowly.",
            "You said you care more about reliability than spectacle because trust is built slowly.",
        ),
        (
            "I said reliability matters because consistency earns trust.",
            "You said reliability matters because consistency earns trust.",
        ),
        (
            "My point was that dependable behavior matters. Spectacle fades.",
            "You said your point was that dependable behavior matters. Spectacle fades.",
        ),
    ],
)
def test_repair_grounded_recall_speaker_attribution(reply, expected):
    repaired, changed = gr.repair_grounded_recall_speaker_attribution(
        "What did I just say?",
        reply,
    )
    assert changed is True
    assert repaired == expected


def test_repair_grounded_recall_does_not_rewrite_ordinary_first_person_reply():
    reply = "I am checking my uncertainty before I answer."
    repaired, changed = gr.repair_grounded_recall_speaker_attribution(
        "How does confusion affect your reasoning?",
        reply,
    )
    assert changed is False
    assert repaired == reply
