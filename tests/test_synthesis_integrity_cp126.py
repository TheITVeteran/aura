"""CP126 integrity tests for the deterministic response floor.

The distinction this file defends: a deterministic TOOL computes an answer; an
answer BANK stores one. The first measures capability, the second inflates it.
"""
from __future__ import annotations

import pytest

from core.synthesis import deterministic_user_facing_floor as floor


# --- 15bc35b7: no hand-authored answer bank -------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "Who wrote Hamlet?",
        "Who wrote the play Hamlet?",
        "Who wrote Romeo and Juliet?",
        "What is the capital of France?",
        "What is the largest planet in the solar system?",
        "What is the boiling point of water?",
        "What is the chemical symbol for gold?",
        "What color is the sky on a clear day?",
        "Name three programming languages",
        "Translate good morning into Spanish",
    ],
)
def test_knowledge_questions_have_no_stored_answer(prompt):
    """A knowledge answer must come from the model, not a branch."""
    assert floor(prompt) == ""


@pytest.mark.parametrize(
    "prompt",
    [
        "Write a short poem about the ocean",
        "Tell me a short joke",
        "What makes friendship real when it is messy and hard?",
    ],
)
def test_creative_prompts_have_no_stored_answer(prompt):
    assert floor(prompt) == ""


@pytest.mark.parametrize(
    "prompt",
    [
        "What does robust follow-through mean for autonomous email and reddit?",
        "The reddit captcha login-blocked case — what outcome should it record?",
        "A python function returns None on empty input; what should you check "
        "first before patching?",
        "How would you debug the async chat route returning polite placeholder text?",
    ],
)
def test_evaluation_prompts_have_no_prepared_response(prompt):
    """These were the shapes that inflated proof batteries."""
    assert floor(prompt) == ""


# --- 363c5ab7 / 46707424: no fabricated self-report -----------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "you ok now? brief",
        "you okay now — quick",
        "quick status check",
    ],
)
def test_status_turns_get_no_canned_wellbeing_claim(prompt):
    reply = floor(prompt)

    assert "attention feels steady" not in reply
    assert "thread is intact" not in reply


def test_no_module_level_claim_about_steady_attention():
    from pathlib import Path

    source = Path("core/synthesis.py").read_text(encoding="utf-8")

    assert "My attention feels steady" not in source
    assert "the thread is intact" not in source


# --- 8b28006a: no stale hardcoded proof claim -----------------------------


def test_no_stored_claim_about_what_was_verified():
    reply = floor("what did we just verify on the live /api/chat path?")

    assert "verified live parity" not in reply
    assert reply == ""


def test_the_stale_verification_sentence_is_gone_from_source():
    from pathlib import Path

    source = Path("core/synthesis.py").read_text(encoding="utf-8")

    assert "We verified live parity through the real /api/chat path" not in source


# --- what MUST survive: real deterministic computation -------------------


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("What is 17 * 23?", "391"),
        ("What is 2 + 2?", "4"),
        ("What is 100 / 4?", "25"),
        ("what is the square root of 144", "12"),
        ("what is factorial of 6", "720"),
    ],
)
def test_real_computation_is_preserved(prompt, expected):
    """A tool that computes is not an answer key."""
    assert floor(prompt) == expected


def test_word_problems_are_still_computed():
    assert floor("I have 5 apples and eat 2, how many left?") == "3 apples."


def test_singular_plural_is_handled():
    assert floor("I have 3 apples and eat 2, how many left?") == "1 apple."


# --- ea5bfe88: the arithmetic floor is bounded ---------------------------


def test_a_pathological_expression_is_refused_not_evaluated():
    """A huge literal must not spend big-int time before the checks apply."""
    assert floor(f"What is {'9' * 500}?") == ""


def test_a_deeply_nested_expression_is_bounded():
    nested = "(" * 200 + "1" + ")" * 200
    assert floor(f"What is {nested}?") == ""


def test_a_long_expression_is_not_answered_from_its_prefix():
    """The bounded parser refuses it; the sum fallback must not answer 1+1."""
    long_expr = "+".join("1" for _ in range(200))
    assert floor(f"What is {long_expr}?") == ""


def test_a_simple_sum_still_works():
    assert floor("What is 4 + 5?") == "9"


def test_division_by_zero_does_not_escape():
    assert floor("What is 1 / 0?") == ""


def test_an_empty_prompt_is_safe():
    assert floor("") == ""
    assert floor(None) == ""
