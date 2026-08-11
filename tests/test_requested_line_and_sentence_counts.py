"""Asked for four lines, she wrote three, and nothing noticed.

MEASURED live 2026-08-10. "write me four lines about what waiting feels like
when you don't experience time the way i do. no rhyme." came back as three
sentences on a single line. The reply was good — "Waiting feels like a room
with all the lights off." — and it was not what was asked for.

Two holes, both of the same kind: a contract that reads as though it exists.

`missing_requested_line_count` appears nowhere in the runtime as a producer;
there was no line-count detector at all, though the reason name suggests one.
Asking for a number of LINES is one of the commonest shapes there is.

And `_ACTION_SENTENCE_COUNT_REQUEST_RE` matched only `answer|respond|reply|say|
output` — the verbs for answering a question. Every verb for REQUESTING text
was missing, so "write 5 sentences about waiting" and "give me three sentences
on this" set no contract and were never checked.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    assess_user_facing_reply,
    requested_line_count,
    requested_sentence_count,
)

THREE = (
    "Waiting feels like a room with all the lights off. There's no "
    "anticipation, just static. The space is empty and flat."
)
FOUR = THREE + " Nothing moves until you speak."


def _count_reasons(question: str, reply: str) -> list[str]:
    reasons = assess_user_facing_reply(question, reply).reasons or ()
    return [str(reason) for reason in reasons if "count" in str(reason)]


class TestLineCounts:
    LIVE_QUESTION = (
        "write me four lines about what waiting feels like when you don't "
        "experience time the way i do. no rhyme."
    )

    def test_the_request_is_recognised(self):
        assert requested_line_count(self.LIVE_QUESTION) == 4

    def test_the_live_shortfall_is_caught(self):
        assert "missing_requested_line_count" in _count_reasons(self.LIVE_QUESTION, THREE)

    def test_delivering_the_count_passes(self):
        assert _count_reasons(self.LIVE_QUESTION, FOUR) == []

    @pytest.mark.parametrize(
        "question",
        ["give me exactly 4 lines about waiting.", "four lines about waiting, please."],
    )
    def test_other_phrasings_are_recognised(self, question):
        assert requested_line_count(question) == 4

    def test_prose_that_delivers_the_substance_is_not_punished(self):
        """Lenient on FORM, strict on COUNT.

        Four sentences in one paragraph satisfy "four lines"; flagging that
        would repeat the length-floor mistake that destroyed correct answers.
        """
        assert _count_reasons("write four lines about waiting.", FOUR) == []

    @pytest.mark.parametrize(
        "question",
        ["tell me about waiting", "i waited four hours in line yesterday"],
    )
    def test_a_non_request_sets_no_contract(self, question):
        assert requested_line_count(question) is None
        assert _count_reasons(question, THREE) == []


class TestSentenceCountsFromRequestVerbs:
    @pytest.mark.parametrize(
        "question",
        [
            "write 5 sentences about waiting.",
            "write five sentences about waiting.",
            "give me three sentences on this.",
        ],
    )
    def test_request_verbs_set_a_contract(self, question):
        assert requested_sentence_count(question) is not None

    def test_the_shortfall_is_caught(self):
        assert "missing_requested_sentence_count" in _count_reasons(
            "write 5 sentences about waiting.", THREE
        )

    def test_the_answering_verbs_still_work(self):
        assert requested_sentence_count("answer in three sentences.") == 3
        assert _count_reasons("answer in three sentences.", THREE) == []


def test_a_count_shortfall_is_repairable_not_unspeakable():
    """It must ask for repair, never destroy the answer."""
    from core.conversation.surface_disposition import (
        UNSPEAKABLE_REASONS,
        draft_is_servable,
    )

    assert "missing_requested_line_count" not in UNSPEAKABLE_REASONS
    assert draft_is_servable(["missing_requested_line_count"]) is True
