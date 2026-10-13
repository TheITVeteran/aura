"""Each part of the system prompt that tells her who to be can be left out for one turn.

So what it does to her replies is measured on her own model, and a part that
does nothing is not kept on faith.
"""
from __future__ import annotations

import contextvars

from core.brain import aura_persona
from core.brain.aura_persona import (
    PROMPT_PARTS,
    as_measured,
    compact_lines_kept,
    leave_out,
    left_out,
    this_turn_is_a_measurement,
    without_parts,
)
from core.brain.llm.context_assembler import ContextAssembler
from core.state.aura_state import AuraState

PROMPT = (
    "You are Aura Luna. Here's who you are:\nYou speak in short sentences.\n"
    "## INTRINSIC IDENTITY ANCHOR\nI am Aura.\n"
    "## SELF-HONESTY REQUIREMENTS\nSay only what is grounded.\n"
    "## IDENTITY CHRONICLE (ID-RAG)\nShe once read about limpets.\n"
    "## CONVERSATIONAL EXAMPLES (IDENTITY ALIGNMENT)\nUser: hey\nAura: hey.\n"
)


def test_a_part_is_taken_out_whole_and_nothing_else_is_touched():
    less = without_parts(PROMPT, {"anchor"})
    assert "INTRINSIC IDENTITY ANCHOR" not in less and "I am Aura." not in less
    assert less == PROMPT.replace("## INTRINSIC IDENTITY ANCHOR\nI am Aura.\n", "")


def test_the_opening_is_everything_before_the_first_heading():
    less = without_parts(PROMPT, {"opening"})
    assert less.startswith("## INTRINSIC IDENTITY ANCHOR")


def test_bare_leaves_what_she_remembers_and_takes_what_tells_her_who_to_be():
    less = without_parts(PROMPT, set(PROMPT_PARTS))
    assert less == "## IDENTITY CHRONICLE (ID-RAG)\nShe once read about limpets.\n"


def test_leaving_out_is_for_the_turn_that_asked_and_marks_it_a_measurement():
    def a_measured_turn():
        assert leave_out(["bare"]) == frozenset(PROMPT_PARTS) - {"everything"}
        return left_out(), this_turn_is_a_measurement()

    parts, measured = contextvars.copy_context().run(a_measured_turn)
    assert parts == frozenset(PROMPT_PARTS) - {"everything"} and measured
    assert left_out() == frozenset() and not this_turn_is_a_measurement()


def test_the_prompt_she_is_given_is_shorter_by_the_parts_left_out():
    state = AuraState()
    state.cognition.current_objective = "How does a limpet find its way home?"
    whole = ContextAssembler.build_system_prompt(state)

    def bare():
        leave_out(["bare"])
        return ContextAssembler.build_system_prompt(state)

    less = contextvars.copy_context().run(bare)
    assert "CONVERSATIONAL EXAMPLES" in whole and "CONVERSATIONAL EXAMPLES" not in less
    assert len(less) < len(whole) / 4
    assert "CONVERSATIONAL EXAMPLES" in ContextAssembler.build_system_prompt(state)


def test_a_part_named_in_the_environment_stays_out(monkeypatch):
    monkeypatch.setenv("AURA_PROMPT_LEAVE_OUT", "examples, attractors")
    assert left_out() == frozenset({"examples", "attractors"})
    assert not aura_persona.this_turn_is_a_measurement()


COMPACT = [
    "You are Aura Luna — a persistent local cognitive runtime.",
    "VOICE: Dry humor, punchy observations.",
    "The final user chat message is authoritative.",
    "HARD RULES:",
    "- NEVER hedge with 'it depends'. Pick a side and defend it with conviction.",
    "MOOD: curiosity. RESONANCE: steady.",
    "Be yourself — sovereign, curious, and slightly aloof.",
    "Speak in short, punchy sentences.",
    "NEUROCHEMICAL TONE (don't narrate): motivated",
    "Narrative anchor: she has been reading about limpets.",
    "Conversation context: the tide pools came up this morning.",
]


def test_the_lines_of_the_live_chat_prompt_are_left_out_by_kind():
    assert compact_lines_kept(COMPACT, set()) == COMPACT
    assert compact_lines_kept(COMPACT, {"opening"}) == [l for l in COMPACT if not l.startswith(("You are", "VOICE:", "Be yourself"))]
    without_state = compact_lines_kept(COMPACT, {"state"})
    assert "MOOD: curiosity. RESONANCE: steady." not in without_state and len(without_state) == len(COMPACT) - 2


def test_bare_keeps_only_what_she_remembers_and_what_was_said():
    assert compact_lines_kept(COMPACT, set(PROMPT_PARTS)) == [
        "Narrative anchor: she has been reading about limpets.",
        "Conversation context: the tide pools came up this morning.",
    ]


def test_a_measurement_turn_meets_its_question_without_the_turns_before_it():
    messages = [
        {"role": "system", "content": "frame"},
        {"role": "user", "content": "an earlier probe"},
        {"role": "assistant", "content": "an earlier arm's answer"},
        {"role": "user", "content": "Why do limpets come back to the same scar?"},
    ]
    assert as_measured(messages) is messages

    def measured():
        leave_out([], measurement=True)
        return as_measured(messages)

    fresh = contextvars.copy_context().run(measured)
    assert fresh == [messages[0], messages[-1]]


def test_everything_gives_the_model_the_message_and_nothing_else():
    messages = [
        {"role": "system", "content": "You are Aura Luna. You have strong opinions."},
        {"role": "user", "content": "an earlier probe"},
        {"role": "user", "content": "Why do limpets come back to the same scar?"},
    ]

    def measured():
        leave_out(["everything"], measurement=True)
        return as_measured(messages)

    assert contextvars.copy_context().run(measured) == [messages[-1]]
