"""Generic-assistant-language repair salvages brief social replies.

The worker's quality gate correctly rejects servile tails ("...is there
anything else I can help with?"). But for a brief social turn ("thanks"),
stripping the tail leaves a complete correct reply ("You're welcome!") that
the old 8-word floor discarded back to the servile original — causing a retry
storm that ended in zero tokens and sustained lag. These tests pin the fix.
"""
from __future__ import annotations

from core.conversation.response_reliability import repair_generic_assistant_language


def test_brief_social_reply_is_salvaged():
    out = repair_generic_assistant_language(
        "thanks, that was really helpful",
        "You're welcome! Is there anything else I can help you with?",
    )
    assert out == "You're welcome!"


def test_greeting_reply_is_salvaged():
    out = repair_generic_assistant_language(
        "hey there",
        "Hello! How can I help you today?",
    )
    assert "how can i help" not in out.lower()
    assert "Hello!" in out


def test_substantive_turn_keeps_word_floor():
    # A long/substantive turn whose reply is ONLY servile has nothing real to
    # salvage — return the original so the gate still catches it (not a
    # non-answer masquerading as a real reply).
    out = repair_generic_assistant_language(
        "explain how quantum entanglement actually works in detail",
        "How can I help? I would be happy to help.",
    )
    assert out == "How can I help? I would be happy to help."


def test_clean_reply_passes_through_unchanged():
    clean = "The capital of France is Paris."
    assert repair_generic_assistant_language("what's the capital of france", clean) == clean


def test_worker_salvages_instead_of_empty_on_exhaustion():
    """The worker's exhaustion path must salvage a clean brief reply rather
    than yield zero tokens (source contract)."""
    import inspect

    import core.brain.llm.mlx_worker as worker

    src = inspect.getsource(worker)
    assert "repair_generic_assistant_language" in src
    assert "Salvaged a clean brief reply" in src
    # The salvage must be attempted BEFORE the empty fallback.
    idx_salvage = src.find("Salvaged a clean brief reply")
    idx_empty = src.find('response_text = ""\n                                            break')
    assert idx_salvage != -1
