"""A promise to answer is not an answer (2026-07-18 soak, live user path).

The soak's per-turn artifact records what the user actually received. Three
of the delivered replies were:

    "Let me consider that carefully."
    "I'm working through that one right now."
    "That reply drifted away from your actual question. The anchor is your
     question about acknowledgment and remember."

Each was accepted as a FINAL reply. Each is technically true and entirely
useless: the first two promise an answer that never comes, the third is the
repair machinery talking about the reply instead of producing one. This is
the "shallow, lazy, technically-true" surface that makes a working mind
look broken.

The detector is deliberately narrow — it fires only when the reply is
*nothing but* the promise. A promise followed by the answer is courtesy,
not emptiness, and short real answers ("Yes.") must never be caught.
"""
from __future__ import annotations

import pytest

from core.conversation.response_reliability import assess_user_facing_reply

pytestmark = pytest.mark.unit

QUESTION = "What is the history of distributed consensus algorithms?"


def _reasons(reply: str) -> tuple[str, ...]:
    return tuple(assess_user_facing_reply(QUESTION, reply).reasons)


class TestPromiseOnlyRepliesAreRejected:
    @pytest.mark.parametrize(
        "reply",
        [
            "Let me consider that carefully.",
            "I'm working through that one right now.",
            "Let me think about that.",
            "I'll look into that.",
            "Give me a moment.",
            "Okay, let me check on that for you.",
            "One second.",
        ],
    )
    def test_pure_promises_are_flagged(self, reply):
        assessment = assess_user_facing_reply(QUESTION, reply)
        assert assessment.ok is False
        assert "promise_without_answer" in assessment.reasons

    @pytest.mark.parametrize(
        "reply",
        [
            "That reply drifted away from your actual question. "
            "The anchor is your question about acknowledgment.",
            "This answer missed what you actually asked.",
        ],
    )
    def test_meta_commentary_instead_of_an_answer_is_flagged(self, reply):
        assert "promise_without_answer" in _reasons(reply)


class TestRealAnswersAreNeverCaught:
    @pytest.mark.parametrize(
        "reply",
        [
            "Yes.",
            "No — Raft came later.",
            "Paxos came first in 1989; Raft (2014) traded some generality "
            "for understandability.",
            "Let me check — the answer is 42, because the modulus wraps at 5.",
            "I'll look at the logs: the lane shows a warmup deferral.",
            "I'm working through the migration now, and step two is already merged.",
            "Give me a moment: the first result is Lamport's 1978 clock paper.",
        ],
    )
    def test_content_bearing_replies_pass(self, reply):
        assert "promise_without_answer" not in _reasons(reply)

    def test_a_long_substantive_answer_is_never_flagged(self):
        reply = (
            "Consensus work starts with Lamport's logical clocks in 1978, then "
            "Paxos in 1989 which proved safety under asynchrony but was famously "
            "hard to implement. Viewstamped Replication arrived around the same "
            "time with a more operational framing. Raft in 2014 kept the same "
            "guarantees while reorganising the protocol around an explicit "
            "leader so that a working engineer could actually build it."
        )
        assert "promise_without_answer" not in _reasons(reply)


class TestActivityQuestionsAreExempt:
    """When the user asks what she is DOING, a present-activity reply IS the
    answer — the emptiness is relative to what was asked."""

    @pytest.mark.parametrize(
        ("question", "reply"),
        [
            ("What are you doing?", "I'm thinking about it."),
            ("What are you working on?", "I'm working through that one right now."),
            ("You with me?", "Let me think about that."),
            ("How's it going?", "I'm processing right now."),
        ],
    )
    def test_status_questions_accept_status_answers(self, question, reply):
        assessment = assess_user_facing_reply(question, reply)
        assert "promise_without_answer" not in assessment.reasons


class TestDetectorDiscipline:
    def test_flagged_replies_are_retryable_not_hard_failures(self):
        """The right response is to draft again, not to fail closed: an
        empty promise is a thin reply, not a leak or corruption."""
        assessment = assess_user_facing_reply(QUESTION, "Let me consider that carefully.")
        assert assessment.hard_failure is False

    def test_empty_reply_is_left_to_the_existing_empty_detector(self):
        assert "promise_without_answer" not in _reasons("")
