"""Elasticity and compaction, measured against what they actually cost.

Both decided by counting. Elasticity dropped prompt blocks after fixed numbers
of turns, so forty one-line exchanges hit the deepest trimming level while using
a few percent of the window, and two pasted files sat at level 0 while
overflowing it. Compaction deleted every older tool result outright, which does
not make the model forget the topic — it makes the model answer from a gap.
"""
from __future__ import annotations

import pytest

from core.brain.llm.context_assembler import ContextAssembler
from core.state.aura_state import AuraState


def _state_with(messages: list[dict]) -> AuraState:
    state = AuraState.default()
    state.cognition.working_memory = messages
    return state


def test_many_tiny_turns_do_not_trim_the_prompt():
    """Forty exchanges of "ok" used to reach elasticity 3."""
    chatter = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "ok"}
        for i in range(80)
    ]

    assert ContextAssembler._conversation_depth(_state_with(chatter)) >= 40
    assert ContextAssembler._elasticity_level(_state_with(chatter)) == 0


def test_a_few_enormous_turns_do_trim_it():
    """Two pasted files are four turns deep and most of the window."""
    huge = [
        {"role": "user", "content": "x" * 200_000},
        {"role": "assistant", "content": "y" * 200_000},
    ]
    state = _state_with(huge)

    assert ContextAssembler._conversation_depth(state) == 2
    assert ContextAssembler._elasticity_level(state) > 0


def test_pressure_rises_with_content_not_with_turns():
    small = _state_with([{"role": "user", "content": "hi"}] * 50)
    large = _state_with([{"role": "user", "content": "z" * 50_000}] * 2)

    assert ContextAssembler._transcript_pressure(large) > ContextAssembler._transcript_pressure(small)


def test_the_steps_are_ordered_and_leave_the_first_half_alone():
    steps = ContextAssembler._PRESSURE_STEPS

    assert list(steps) == sorted(steps)
    assert steps[0] >= 0.5, "trimming before half the window is free"
    assert steps[-1] < 1.0


def test_compaction_leaves_a_receipt_where_a_tool_result_was():
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {
            "role": "user",
            "content": "the DNS lookup output",
            "metadata": {"type": "tool_result"},
        },
        {"role": "user", "content": "and then?"},
        {"role": "assistant", "content": "then this"},
        {"role": "user", "content": "current"},
    ]

    out = ContextAssembler.microcompact(messages, keep_recent=3)
    rendered = "\n".join(str(m.get("content", "")) for m in out)

    assert "COMPACTED TOOL_RESULT" in rendered, "the result was deleted, not compacted"
    receipts = [
        m for m in out
        if (m.get("metadata") or {}).get("type") == "compaction_receipt"
    ]
    assert receipts
    assert receipts[0]["metadata"]["original_chars"] == len("the DNS lookup output")
    assert len(receipts[0]["metadata"]["content_digest"]) == 12


def test_the_receipt_is_much_smaller_than_what_it_replaces():
    """Compaction still has to earn its name."""
    big = "result line\n" * 500
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": big, "metadata": {"type": "tool_result"}},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]

    out = ContextAssembler.microcompact(messages, keep_recent=3)
    receipt = next(
        m for m in out if (m.get("metadata") or {}).get("type") == "compaction_receipt"
    )

    assert len(receipt["content"]) < len(big) / 10


def test_a_truncated_assistant_turn_carries_a_digest():
    long_reply = "sentence. " * 400
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "assistant", "content": long_reply},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]

    out = ContextAssembler.microcompact(messages, keep_recent=3)
    truncated = next(m for m in out if m["role"] == "assistant" and "more characters" in m["content"])

    assert "digest " in truncated["content"]
    assert truncated["content"].startswith("sentence. ")


@pytest.mark.parametrize("content", ["", "   ", "..."])
def test_near_empty_messages_are_still_dropped(content):
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": content},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]

    out = ContextAssembler.microcompact(messages, keep_recent=3)

    assert all(str(m.get("content", "")).strip() not in {"", "...", "   "} for m in out)
