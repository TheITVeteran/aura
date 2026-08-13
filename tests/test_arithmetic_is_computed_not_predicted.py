"""Decidable arithmetic is computed, never predicted.

LIVE DEFECT, 2026-08-10: "what is 7919 times 6421? just the number." → 50864799.
The product is 50847899.

A transformer does not calculate, it predicts the next token, and a
four-by-four-digit product in one forward pass is unreliable at any parameter
count — "just the number" had also removed the intermediate steps that are the
only reason a model ever gets these right.

The runtime could already do the sum. requested_arithmetic_result is how a
later gate KNOWS such an answer is wrong. It returned None here: its matcher
wanted a lead-in verb AND symbol operators, so it computed nothing for "times",
nothing for "multiply X by Y", and nothing for a bare "2+2" — every phrasing a
person actually uses.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    _arithmetic_answer_missing,
    requested_arithmetic_result,
)

TRUE_PRODUCT = 50847899


@pytest.mark.parametrize(
    "question,expected",
    [
        ("what is 7919 times 6421? just the number.", TRUE_PRODUCT),
        ("multiply 7919 by 6421", TRUE_PRODUCT),
        ("work out 7919 x 6421", TRUE_PRODUCT),
        ("7919*6421", TRUE_PRODUCT),
        ("2+2", 4),
        ("  12 / 4 ", 3),
        ("compute 144 divided by 12", 12),
        ("add 12 and 30", 42),
        ("subtract 5 from 20", 15),
        ("how much is 45 minus 17", 28),
        ("what is 1,000 * 2?", 2000),
        ("what's 15% of 240?", 36),
        ("what is 12 * 12?", 144),
    ],
)
def test_the_runtime_computes_the_answer(question, expected):
    result = requested_arithmetic_result(question)
    assert result is not None, f"no computation attempted for {question!r}"
    assert abs(result - expected) < 1e-6


@pytest.mark.parametrize(
    "text",
    [
        "tell me about consciousness",
        "my version is 3.11.2 and yours is 3.12",
        "the meeting is 2026-08-10",
        "i have 3 cats and 2 dogs",
        "what is your name",
        "x marks the spot",
        "how are you doing today",
    ],
)
def test_ordinary_text_is_not_mistaken_for_a_sum(text):
    """A wrong value injected as authoritative is worse than none.

    Numbers with operators between them appear in version strings, dates and
    ranges, so computation requires the turn to actually be asking for one.
    """
    assert requested_arithmetic_result(text) is None


def test_the_live_wrong_answer_is_detected():
    """The exact failure: her number, against the real one."""
    question = "what is 7919 times 6421? just the number."
    assert _arithmetic_answer_missing(question, "50864799") is True
    assert _arithmetic_answer_missing(question, "50847899") is False
    assert _arithmetic_answer_missing(question, "It's 50,847,899.") is False


def test_integer_results_above_float_precision_remain_exact():
    question = "what is 9007199254740993 + 2?"

    assert requested_arithmetic_result(question) == 9007199254740995
    assert _arithmetic_answer_missing(question, "9007199254740995") is False
    assert _arithmetic_answer_missing(question, "9007199254740996") is True


def test_power_results_above_float_precision_remain_exact():
    question = "what is 3 to the 40th power?"
    expected = 3**40

    assert requested_arithmetic_result(question) == expected
    assert _arithmetic_answer_missing(question, str(expected)) is False
    assert _arithmetic_answer_missing(question, str(expected + 1)) is True
