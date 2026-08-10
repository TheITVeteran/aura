"""The anchor must name what was asked about, not the longest adverb in it.

LIVE DEFECT, 2026-08-10. Asked for three specific readings — "(1) how many
heartbeats are active, as a fraction, (2) your uptime in seconds, (3) the exact
action name that keeps getting refused… I will check all three" — the degraded
composer replied:

    I couldn't get a clear enough answer together, and I'd rather say that than
    hand you something thin. I understood you to be asking about heartbeats and
    actually. Ask me again and I should have it.

Refusing to guess was correct. Naming "actually" as a topic was not. It came
from "answer only from what you can actually read", and it won because
_select_anchor_topic_tokens ranks non-priority candidates by -len(token) —
longest word first — so an eight-letter stance adverb outranked every noun in
the question.

The category was already recognised: "really" was in _TOPIC_STOPWORDS. It had
simply never been filled in.
"""

from __future__ import annotations

import pytest


LIVE_MESSAGE = (
    "careful, I want to test that claim, not accept it. do not agree with me. "
    "answer only from what you can actually read: (1) how many heartbeats are "
    "active, as a fraction, (2) your uptime in seconds, (3) the exact action "
    "name that keeps getting refused. if any of those three is not readable to "
    "you, say not readable for that one instead of guessing."
)


def _pick(text: str) -> list[str]:
    from interface.routes.chat import _select_anchor_topic_tokens

    return _select_anchor_topic_tokens(text)


def test_live_message_no_longer_anchors_on_an_adverb() -> None:
    topics = _pick(LIVE_MESSAGE)

    assert "actually" not in topics
    assert "heartbeats" in topics


@pytest.mark.parametrize(
    "adverb",
    [
        "actually",
        "basically",
        "specifically",
        "exactly",
        "honestly",
        "simply",
        "instead",
        "literally",
        "obviously",
        "probably",
    ],
)
def test_stance_adverbs_are_never_topics(adverb: str) -> None:
    topics = _pick(f"tell me {adverb} what your uptime in seconds is")

    assert adverb not in topics


def test_real_nouns_still_win() -> None:
    """The fix must not hollow out the anchor."""
    assert set(_pick("open my notes and write a paragraph about yourself")) == {
        "notes",
        "paragraph",
    }
    assert "uptime" in _pick(
        "what is basically your uptime, specifically in seconds?"
    )
    assert "perception" in _pick(
        "compare the memory subsystem and the perception pipeline"
    )


def test_anchor_sentence_reads_as_english() -> None:
    """The end product a person actually sees."""
    topics = _pick(LIVE_MESSAGE)

    assert len(topics) >= 2
    sentence = (
        f"I understood you to be asking about {topics[0]} and {topics[1]}."
    )

    assert "actually" not in sentence
    assert sentence.endswith(".")
