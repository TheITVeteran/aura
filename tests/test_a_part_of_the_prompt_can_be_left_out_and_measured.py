"""Each part of the system prompt that tells her who to be can be left out for one turn.

So what it does to her replies is measured on her own model, and a part that
does nothing is not kept on faith.
"""
from __future__ import annotations

import contextvars

from core.brain import aura_persona
from core.brain.aura_persona import PROMPT_PARTS, leave_out, left_out, this_turn_is_a_measurement, without_parts
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
        assert leave_out(["bare"]) == frozenset(PROMPT_PARTS)
        return left_out(), this_turn_is_a_measurement()

    parts, measured = contextvars.copy_context().run(a_measured_turn)
    assert parts == frozenset(PROMPT_PARTS) and measured
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
