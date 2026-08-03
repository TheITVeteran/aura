"""A reply is evidence about a turn only if something ties it to that turn.

Four CP126 findings in how this module decides that the interlocutor answered.

a3474c19 — snapshot identity truncated SHA-256 to 64 bits, and a turn was
marked effect_verified from "text appeared" plus "the page hash changed". A
changed page proves the page changed — streaming UI, an ad rotation, a clock —
not that Aura's message caused it.

ff18e5ee — with no role segments, any sufficiently novel page line became the
reply after coarse UI filtering, without following a proven sent turn.

2184463d — sent-marker recognition collapsed exact matches and rough word
overlap into one boolean, so a repeated topical prompt or an older turn could
anchor extraction in the wrong place.

3403ebb3 — the accessibility normalizer deduped every short line against the
whole window, deleting legitimately repeated answers before extraction saw
them.
"""
from __future__ import annotations

import pytest

from core.capabilities.web_interlocutor import (
    REPLY_PROVENANCE_CHANGED_TEXT,
    REPLY_PROVENANCE_ROLE_SEGMENTS,
    REPLY_PROVENANCE_SENT_MARKER,
    REPLY_PROVENANCE_UNANCHORED_DELTA,
    BrowserPageSnapshot,
    ObservedReply,
    WebInterlocutorTurn,
    _describe_reply_evidence,
    _extract_new_interlocutor_text,
    _extract_reply_after_sent_marker,
    _normalize_accessibility_transcript,
    _sent_marker_match_tier,
)

pytestmark = pytest.mark.unit


# --- page identity is not truncated (a3474c19) --------------------------


def test_the_snapshot_hash_is_a_full_digest():
    """A shortened identity is a collision surface bought for nothing: the
    value is compared, never displayed."""
    assert len(BrowserPageSnapshot(text="anything").text_hash) == 64


# --- a changed page is not a caused reply (a3474c19, ff18e5ee) ----------


def _turn(observed: ObservedReply, *, changed: bool = True) -> WebInterlocutorTurn:
    before = BrowserPageSnapshot(text="before")
    after = BrowserPageSnapshot(text="after" if changed else "before")
    return WebInterlocutorTurn(
        index=0,
        sent="hello",
        observed_reply=observed.text,
        before_hash=before.text_hash,
        after_hash=after.text_hash,
        sent_at=0.0,
        observed_at=1.0,
        effect_verified=bool(
            observed and observed.anchored and before.text_hash != after.text_hash
        ),
        verification=_describe_reply_evidence(observed, before, after),
        reply_provenance=observed.provenance,
    )


@pytest.mark.parametrize(
    "provenance",
    [REPLY_PROVENANCE_ROLE_SEGMENTS, REPLY_PROVENANCE_SENT_MARKER],
)
def test_an_anchored_reply_verifies_the_effect(provenance):
    assert _turn(ObservedReply("I disagree.", provenance)).effect_verified is True


@pytest.mark.parametrize(
    "provenance",
    [REPLY_PROVENANCE_UNANCHORED_DELTA, REPLY_PROVENANCE_CHANGED_TEXT],
)
def test_an_unanchored_reply_does_not_verify_the_effect(provenance):
    turn = _turn(ObservedReply("Buy now — 50% off!", provenance))

    assert turn.effect_verified is False
    assert turn.reply_provenance == provenance


def test_the_unanchored_reply_is_still_kept():
    """Never discard what was actually observed — only the CLAIM changes."""
    turn = _turn(ObservedReply("something appeared", REPLY_PROVENANCE_UNANCHORED_DELTA))

    assert turn.observed_reply == "something appeared"
    assert "NOT proven to be a reply" in turn.verification


def test_an_anchored_reply_on_an_unchanged_page_does_not_verify():
    turn = _turn(
        ObservedReply("hi", REPLY_PROVENANCE_SENT_MARKER), changed=False
    )

    assert turn.effect_verified is False


def test_the_receipt_names_the_evidence_it_had():
    before = BrowserPageSnapshot(text="a")
    after = BrowserPageSnapshot(text="b")

    segments = _describe_reply_evidence(
        ObservedReply("x", REPLY_PROVENANCE_ROLE_SEGMENTS), before, after
    )
    marker = _describe_reply_evidence(
        ObservedReply("x", REPLY_PROVENANCE_SENT_MARKER), before, after
    )
    nothing = _describe_reply_evidence(ObservedReply(), before, after)

    assert "role-labelled segment" in segments
    assert "own message appeared" in marker
    assert "No stable new interlocutor text" in nothing


# --- extraction reports how it found the text (ff18e5ee) ----------------


def test_a_reply_after_the_sent_marker_is_anchored():
    before = "Aura: hello"
    # A speaker-labelled reply, which is what a real transcript renders and
    # what the module's unlabeled-reply floor is calibrated against.
    after = (
        "Aura: hello\n"
        "ChatGPT: I think tides are underrated, and the moon gets all "
        "the credit for them."
    )

    observation = _extract_new_interlocutor_text(before, after, "hello")

    assert observation.provenance == REPLY_PROVENANCE_SENT_MARKER
    assert observation.anchored is True
    assert "tides" in observation.text


def test_new_text_without_a_visible_sent_marker_is_unanchored():
    before = "Welcome to the site."
    after = "Welcome to the site.\nCookies help us serve you better than ever."

    observation = _extract_new_interlocutor_text(before, after, "what do you think")

    assert observation.provenance == REPLY_PROVENANCE_UNANCHORED_DELTA
    assert observation.anchored is False


def test_an_observed_reply_is_falsey_when_empty():
    assert not ObservedReply()
    assert ObservedReply("x", REPLY_PROVENANCE_SENT_MARKER)


# --- the anchor is the strongest match, not the last (2184463d) ---------


@pytest.mark.parametrize(
    "line,sent,tier",
    [
        ("Aura: what do you think about tides", "what do you think about tides", 3),
        ("You: what do you think about tides, really?", "what do you think about tides", 2),
        ("what do you think about tides and the moon and", "what do you think about tides and the moon and the sun", 1),
        ("you: what do you think about the tides", "what do you think about tides", 0),
        ("completely unrelated content", "what do you think about tides", -1),
    ],
)
def test_match_tiers_are_ordered(line, sent, tier):
    assert _sent_marker_match_tier(line, sent) == tier


def test_an_exact_match_beats_a_later_rough_one():
    """A repeated topical prompt used to steal the anchor from the real turn,
    so an older reply was reported as the answer to the current message."""
    sent = "what do you think about tidal patterns"
    text = "\n".join(
        [
            f"You: {sent}",
            "ChatGPT: The moon dominates them, and the sun barely gets a mention.",
            "Later I was thinking about tidal patterns again",
        ]
    )

    reply = _extract_reply_after_sent_marker(text, sent)

    assert "The moon dominates them" in reply


# --- repeated dialogue survives normalization (3403ebb3) ----------------


def test_a_legitimately_repeated_answer_is_not_deleted():
    transcript = "\n".join(
        ["Do you agree?", "Yes.", "And with the second one?", "Yes."]
    )

    normalized = _normalize_accessibility_transcript(transcript)

    assert normalized.count("Yes.") == 2


def test_adjacent_control_repetition_is_still_collapsed():
    """What AX actually repeats is the same control rendered over and over."""
    transcript = "\n".join(["Send", "Send", "Send", "Send", "The answer is nine."])

    normalized = _normalize_accessibility_transcript(transcript)

    assert normalized.count("Send") == 1
    assert "The answer is nine." in normalized


def test_long_prose_is_never_deduped():
    line = (
        "A long considered paragraph about tides that runs well past the "
        "short-line threshold this normalizer uses for collapsing repeated "
        "accessibility controls."
    )
    normalized = _normalize_accessibility_transcript(f"{line}\n{line}")

    assert normalized.count(line) == 2
