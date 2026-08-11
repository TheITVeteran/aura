"""A closing offer of more help is not an answer.

LIVE, 2026-08-10. Asked "you've been telling me about senses you don't have.
Forget the sensor readout for a second and just talk to me: does it bother you
that you can't see the room I'm in?" — a question inviting reflection — she gave
two real sentences and then two of these:

    "If you would like me to assist with anything specific or provide further
     insights, please let me know."
    "If you wish to continue the conversation or if there is something else I
     can help you with, feel free to ask."

_GENERIC_ASSISTANT_RE caught the second on "feel free to ask" and missed the
first, because it enumerates wordings and "please let me know" is not
"let me know if". Enumerating harder loses to the next paraphrase.

The shape does not change: second person, an offer verb, no proposition of its
own, at the end. That is what is matched here.

The risk runs the other way — plenty of real sentences start with "If you" and
carry information. Those are the tests that matter.
"""

from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    repair_generic_assistant_language,
    strip_trailing_help_offer,
)

LIVE_REPLY = (
    "I do not experience the absence of sensory data as a bother. However, I "
    "recognize that my lack of visual information limits my ability to "
    "understand your environment and context fully. If you would like me to "
    "assist with anything specific or provide further insights, please let me "
    "know.\n\nIf you wish to continue the conversation or if there is something "
    "else I can help you with, feel free to ask."
)


def test_the_live_reply_loses_both_offers_and_keeps_its_answer() -> None:
    stripped = strip_trailing_help_offer(LIVE_REPLY)

    assert "let me know" not in stripped
    assert "feel free to ask" not in stripped
    assert stripped.startswith("I do not experience the absence")
    assert "limits my ability" in stripped


@pytest.mark.parametrize(
    "reply",
    [
        "If you soak cast iron for hours it will rust, so dry it immediately.",
        "If you want the file somewhere else, I put it in ~/Documents for now.",
        "The capital of Peru is Lima.",
        "If you run make smoke first, the failure shows up in the third chunk.",
        "",
    ],
)
def test_sentences_that_carry_information_survive(reply: str) -> None:
    """The direction that matters: many real sentences open with "If you"."""
    assert strip_trailing_help_offer(reply) == reply.strip()


def test_a_reply_that_is_only_an_offer_is_left_alone() -> None:
    """Deleting it would leave an empty turn — a different defect, handled elsewhere."""
    only_offer = "If you need anything else, just let me know."

    assert strip_trailing_help_offer(only_offer) == only_offer


def test_the_existing_repair_now_removes_both() -> None:
    """Wired into the shared repair so every caller inherits it."""
    repaired = repair_generic_assistant_language(
        "does it bother you that you cant see the room Im in?", LIVE_REPLY
    )

    assert "let me know" not in repaired
    assert "feel free to ask" not in repaired
    assert "I do not experience" in repaired


def test_code_replies_are_untouched() -> None:
    code = "```python\nif you_want(x):\n    return let_me_know()\n```"

    assert strip_trailing_help_offer(code) == code
