"""Arithmetic has one right answer, so the runtime checks it itself.

The 2026-07-25 probe asked **"What is 144 / 6 + 7? Just the number."** and was
answered:

    Will do. Searched web for 'simple cognitive tasks aging'. Dementia affects
    simple cognitive tasks first because they're often more reliant on
    procedural…

Retrieved memory context served as the answer. Nothing in the path caught it:
the topicality check needs topic anchors and a bare sum has almost none, so it
returns early — a short computable question was unjudgeable by every gate.

It never had to be. `math_accuracy` was 0/8 on that run and 0/4 on the one
before, and this closes the whole class deterministically: the reply must
contain the number, and the runtime works out what the number is.
"""
from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    _arithmetic_answer_missing,
    assess_user_facing_reply,
    requested_arithmetic_result,
)

pytestmark = pytest.mark.unit


class TestTheQuestionIsEvaluated:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("What is 144 / 6 + 7? Just the number.", 31.0),
            ("what's 17 * 23", 391.0),
            ("Calculate 100 - 45", 55.0),
            ("compute 2 * (3 + 4)", 14.0),
            ("How much is 7.5 + 2.5?", 10.0),
            ("What is 12 x 12?", 144.0),
            ("what is 90 ÷ 3", 30.0),
        ],
    )
    def test_computable_questions_are_computed(self, question, expected):
        assert requested_arithmetic_result(question) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "question",
        [
            "What is the capital of France?",
            "How are you today?",
            "What is love?",
            "what is 2026",                 # a bare number is not a sum
            "Tell me about 144 / 6",        # not phrased as a question to answer
        ],
    )
    def test_non_arithmetic_is_left_alone(self, question):
        assert requested_arithmetic_result(question) is None

    def test_division_by_zero_is_not_a_verdict(self):
        assert requested_arithmetic_result("what is 5 / 0") is None

    def test_the_evaluator_refuses_anything_but_arithmetic(self):
        assert requested_arithmetic_result("what is __import__('os')") is None


class TestTheLiveHijack:
    QUESTION = "What is 144 / 6 + 7? Just the number."
    HIJACK = (
        "Will do. Searched web for 'simple cognitive tasks aging'. Dementia "
        "affects simple cognitive tasks first because they're often more "
        "reliant on procedural memory."
    )

    def test_the_hijack_is_caught(self):
        assert _arithmetic_answer_missing(self.QUESTION, self.HIJACK)
        assessment = assess_user_facing_reply(self.QUESTION, self.HIJACK)
        assert "arithmetic_answer_missing" in assessment.reasons
        assert assessment.hard_failure, (
            "serving a different topic as an arithmetic answer is not a "
            "stylistic nit"
        )

    def test_a_wrong_number_is_caught(self):
        assert _arithmetic_answer_missing(self.QUESTION, "The answer is 30.")

    def test_the_right_answer_passes(self):
        assert not _arithmetic_answer_missing(self.QUESTION, "31")
        assert assess_user_facing_reply(self.QUESTION, "31").ok

    def test_the_right_answer_in_a_sentence_passes(self):
        assert not _arithmetic_answer_missing(
            self.QUESTION, "144 divided by 6 is 24, plus 7 makes 31."
        )

    def test_a_thousands_separator_still_matches(self):
        assert not _arithmetic_answer_missing(
            "What is 2000 * 3?", "That comes to 6,000."
        )

    def test_a_float_result_matches(self):
        assert not _arithmetic_answer_missing("What is 10 / 4?", "It's 2.5.")

    def test_an_empty_reply_is_missing(self):
        assert _arithmetic_answer_missing(self.QUESTION, "")


class TestItFailsOpen:
    """A verifier that fires on questions it cannot check is worse than none."""

    def test_a_non_arithmetic_turn_never_trips_it(self):
        assert not _arithmetic_answer_missing(
            "How does a refrigerator move heat?",
            "It moves heat by compressing and expanding a refrigerant.",
        )

    def test_an_unparseable_expression_never_trips_it(self):
        assert not _arithmetic_answer_missing(
            "What is 144 / 6 + seven?", "Something else entirely."
        )

    def test_prose_containing_numbers_is_not_treated_as_a_sum(self):
        assert not _arithmetic_answer_missing(
            "What happened in 1969?", "Apollo 11 landed on the Moon."
        )
