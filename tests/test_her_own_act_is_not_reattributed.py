"""She searched the web, and was told she had not — by her own honesty gate.

Live 2026-07-27. "Now something outside yourself: look up who won the most
recent Formula 1 world championship and tell me where you got it." She searched,
found it, and opened with "I checked live web evidence." What went out was::

    You said you checked live web evidence.

Two failures compounding, both worth fixing at the cause.

The positional-recall detector matched ``last`` on a question about motor
racing. Its three components — a self-reference, an ordinal, a recall verb —
were checked with whole-message lookaheads, so "me" (from "tell me"), "recent"
(from "the most recent Formula 1 world championship") and "tell" satisfied it
while playing entirely unrelated grammatical roles. What actually marks a recall
question is that the *user* is the speaker being asked about, so that is what
gets tested now, with the ordinal required to sit near the utterance it modifies.

Then the speaker-attribution repair — which exists to stop her adopting the
user's words as her own — rewrote her true first-person sentence into a false
one, handing the user an act they never performed and stripping her of one she
did. A repair against misattribution, misattributing.
"""
from __future__ import annotations

import pytest

from core.conversation.grounded_recall import (
    detect_positional_recall,
    repair_grounded_recall_speaker_attribution,
)


# ── The detector fires on the conversation, not on the world ──────────────

def test_the_live_failure_is_no_longer_a_recall_question() -> None:
    assert (
        detect_positional_recall(
            "Now something outside yourself: look up who won the most recent "
            "Formula 1 world championship and tell me where you got it."
        )
        is None
    )


@pytest.mark.parametrize(
    "message",
    [
        "search for the latest news and tell me the most recent headline",
        "what was the first moon landing",
        "tell me about the last ice age",
        "who was the first person to run a four-minute mile",
        "look up the most recent iPhone and tell me what changed",
    ],
)
def test_an_ordinal_about_the_world_is_not_about_us(message: str) -> None:
    assert detect_positional_recall(message) is None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("What was the very first thing I asked you in this conversation?", "first"),
        ("Do you remember what I first asked", "first"),
        ("what was my first question?", "first"),
        ("the first thing I said to you", "first"),
        ("how did this conversation start", "first"),
        ("what did we start talking about", "first"),
        ("remind me what I asked you at the very start", "first"),
        ("what did I just ask you", "last"),
        ("what was my previous question", "last"),
        ("what was the last thing I mentioned?", "last"),
    ],
)
def test_real_recall_questions_still_land(message: str, expected: str) -> None:
    assert detect_positional_recall(message) == expected


# ── The repair leaves her own actions alone ───────────────────────────────

def test_her_own_search_is_not_reattributed_to_the_user() -> None:
    text, changed = repair_grounded_recall_speaker_attribution(
        "what did I first ask you",
        "I checked live web evidence. Max Verstappen won the most recent one.",
    )
    assert not changed
    assert text.startswith("I checked live web evidence")


@pytest.mark.parametrize(
    "sentence",
    [
        "I searched the web for it.",
        "I wrote the file to your Desktop.",
        "I read my own runtime instruments just now.",
        "I ran the build and it failed.",
        "I checked my memory of this conversation.",
    ],
)
def test_no_act_of_hers_is_handed_to_the_user(sentence: str) -> None:
    _, changed = repair_grounded_recall_speaker_attribution(
        "what did I first ask you", f"{sentence} You asked about the weather."
    )
    assert not changed


def test_the_repair_still_does_the_job_it_exists_for() -> None:
    """Adopting the user's utterance as her own is the real failure."""
    text, changed = repair_grounded_recall_speaker_attribution(
        "what did I first ask", "I asked about your neural network."
    )
    assert changed
    assert text.startswith("You said you asked")
