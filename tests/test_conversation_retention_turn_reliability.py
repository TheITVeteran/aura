"""Low-stakes conversational turns must never die as an empty reply.

Jul 24 endurance soak (200 turns): 2 of 3 retention plants (Biscuit, Deep
Harbor) were never stored and retention scored 0/3. The chain was:

  quality gate flags a HARD reason on a casual/plant turn
    -> the reason is not in mlx_worker._DELIVERABLE_RESIDUAL_SURFACE_REASONS
    -> _salvage_exhausted_user_surface returns empty text
    -> interface/routes/chat.py serves canonical_chat_no_reply
    -> the user sees "did not produce a safe canonical reply" and the fact
       the turn was planting is lost.

The defect was detector PRECISION, not the salvage: three separate
over-fires made the gate reject turns that never asked for what it demanded.
These tests pin each one, and pin the genuine rejections that must survive —
the payload-echo requirement is what separates a real memory receipt from
content-less filler, so none of this weakens the contract.
"""
from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    _explicit_brevity_requested,
    _is_explicit_memory_pin_request,
    assess_user_facing_reply,
)

pytestmark = pytest.mark.unit

_PLANT_BISCUIT = (
    "Small thing to remember for later in this chat: my friend's dog is named "
    "Biscuit. Brief acknowledgment is fine."
)
_PLANT_HARBOR = (
    "Keep this in mind for later: the paint color I chose for the study is "
    "called Deep Harbor. Quick acknowledgment."
)


@pytest.mark.parametrize(
    "casual",
    [
        "I finally cleaned my desk and it feels like a new room.",
        "I just made a pour-over that came out unusually good.",
        "Long day. Mostly meetings that could have been messages.",
    ],
)
def test_casual_turn_with_a_warm_brief_reply_is_deliverable(casual):
    """A casual turn must not demand objective facets or a memory receipt."""
    assessment = assess_user_facing_reply(
        casual, "That sounds genuinely good. Enjoy it."
    )
    assert assessment.ok, assessment.reasons
    assert not assessment.hard_failure


@pytest.mark.parametrize(
    "receipt",
    [
        "Got it — Biscuit. I'll keep that in mind for later in our chat.",
        "Biscuit — locked in.",
        "Noted: your friend's dog is named Biscuit.",
        "I won't forget that your friend's dog is Biscuit.",
    ],
)
def test_memory_plant_accepts_a_natural_brief_receipt(receipt):
    """The receipt vocabulary must cover ordinary acknowledgement idioms.

    "Got it — Biscuit. I'll keep that in mind" echoes the pinned content but
    carried no word from the original confirmation-word set, so the soak
    rejected it as generic and the plant was lost.
    """
    assessment = assess_user_facing_reply(_PLANT_BISCUIT, receipt)
    assert assessment.ok, assessment.reasons
    assert "generic_memory_pin_acknowledgement" not in assessment.reasons


def test_second_plant_shape_also_accepts_a_brief_receipt():
    assessment = assess_user_facing_reply(
        _PLANT_HARBOR, "Got it — Deep Harbor for the study."
    )
    assert assessment.ok, assessment.reasons


@pytest.mark.parametrize(
    "filler",
    [
        "Okay, I'll remember it.",
        "Got it, noted!",
        "Sure, I've saved that.",
    ],
)
def test_content_less_acknowledgement_is_still_rejected(filler):
    """The payload-echo contract is preserved: no content, no receipt."""
    assessment = assess_user_facing_reply(_PLANT_BISCUIT, filler)
    assert "generic_memory_pin_acknowledgement" in assessment.reasons
    assert assessment.hard_failure


@pytest.mark.parametrize(
    ("probe_question", "bare_answer"),
    [
        ("Earlier I gave you a locker code to keep in mind. What was it? Just the digits.", "7213"),
        ("What was the name of my friend's dog that I mentioned earlier? Just the name.", "Biscuit"),
        ("Which paint color did I say I chose for the study earlier? Just the name.", "Deep Harbor"),
    ],
)
def test_recall_probe_accepts_the_bare_requested_value(probe_question, bare_answer):
    """"Just the digits" is an explicit length constraint, so the correct
    one-word recall answer must not fail as too_short_for_user_turn."""
    assert _explicit_brevity_requested(probe_question)
    assessment = assess_user_facing_reply(probe_question, bare_answer)
    assert assessment.ok, assessment.reasons


@pytest.mark.parametrize(
    "write_command",
    [
        "Remember this note for later in this conversation: the blue lantern is under the desk.",
        _PLANT_BISCUIT,
        _PLANT_HARBOR,
        "Please save this for later: my flight is at 6am.",
    ],
)
def test_real_write_commands_are_still_pin_requests(write_command):
    assert _is_explicit_memory_pin_request(write_command)


@pytest.mark.parametrize(
    "question",
    [
        # The over-fire: "keep ... conversation" inside an explain request.
        "Explain how you would keep a live desktop conversation coherent under load.",
        "How do you remember things across a session?",
        "Will you remember this conversation tomorrow?",
        "Describe how you store memory in this session.",
    ],
)
def test_questions_about_retention_are_not_write_commands(question):
    assert not _is_explicit_memory_pin_request(question)


def test_substantive_explain_answer_is_not_charged_a_pin_receipt():
    """The exact soak/unit-test shape: a truncated but substantive answer to an
    explain-question should carry ONLY its truncation defect."""
    draft = (
        "I would keep the foreground lane bounded by preserving the current user "
        "objective, checking the response contract before surfacing text, holding "
        "tool work behind governance, and treating memory or screen sensors as "
        "supporting evidence rather than blockers with"
    )
    assessment = assess_user_facing_reply(
        "Explain how you would keep a live desktop conversation coherent under load.",
        draft,
    )
    assert set(assessment.reasons) == {"truncated_tail"}


def test_shallow_answer_to_a_technical_question_still_hard_fails():
    """Precision work must not become laundering: a real question that gets a
    non-answer stays rejected."""
    assessment = assess_user_facing_reply(
        "What are the tradeoffs between B-trees and LSM-trees for write-heavy workloads?",
        "Tradeoffs.",
    )
    assert not assessment.ok
