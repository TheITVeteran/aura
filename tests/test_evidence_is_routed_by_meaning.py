"""Evidence follows the MEANING of a request, not its wording.

Bryan, live 2026-08-04: "a lot of these requests are tied into specific
phrases and that shouldn't be the case. part of her reasoning has to include
general associations and a general understanding of what is being asked."

He had the evidence for it. "Which file in your repository does that
function live in?" reached her source and was answered correctly; "What
python module is that from" — the same question — missed a regex and was
answered from her weights. A keyword gate that misses leaves her blind to
something she can actually see.
"""
from __future__ import annotations

import pytest

from core.cognition.evidence_relevance import (
    OWN_SOURCE,
    SCREEN_PERCEPTION,
    relevance,
    semantic_routing_available,
    wants_evidence,
)

pytestmark = pytest.mark.skipif(
    not semantic_routing_available(),
    reason="sentence-transformers unavailable; routing falls back to the lexical floor",
)


@pytest.mark.parametrize(
    "question",
    [
        "What python module is that from",
        "Which file in your repository does that function live in?",
        "Can you show me a snippet of your code that you're interested in?",
        "where can it be found?",
        # Phrasings that appear in no pattern anywhere in the codebase.
        "show me how you're actually built",
        "let me see a bit of what you're made of",
    ],
)
def test_a_question_about_her_code_finds_her_source(question: str) -> None:
    assert relevance(question, OWN_SOURCE) > 0.0, question
    assert wants_evidence(question, OWN_SOURCE)


@pytest.mark.parametrize(
    "question",
    [
        "Hey, Aura. Can you tell me what you see on the screen?",
        "what's on my screen right now?",
        "What's behind your window? Can you see what's underneath it?",
        "is there anything about UFC on my screen right now?",
        "what was that repo you saw?",
    ],
)
def test_a_question_about_the_screen_finds_the_perception(question: str) -> None:
    assert relevance(question, SCREEN_PERCEPTION) > 0.0, question
    assert wants_evidence(question, SCREEN_PERCEPTION)


@pytest.mark.parametrize(
    "question",
    ["what's 17 times 4?", "how are you feeling today?", "tell me a joke"],
)
def test_unrelated_turns_pull_no_evidence(question: str) -> None:
    assert not wants_evidence(question, OWN_SOURCE), question
    assert not wants_evidence(question, SCREEN_PERCEPTION), question


def test_writing_new_code_is_not_a_question_about_her_own():
    """"Write me a python module" shares vocabulary and shares no intent."""
    assert relevance("write me a python module for sorting", OWN_SOURCE) < 0.0
    assert not wants_evidence("write me a python module for sorting", OWN_SOURCE)


def test_the_lexical_floor_can_add_but_never_veto():
    """Meaning wins; the pattern is a floor under it, not a gate over it."""
    assert wants_evidence(
        "zzz unparseable zzz", OWN_SOURCE, lexical_floor=lambda _text: True
    )
    # A floor that says no cannot suppress a clear semantic match.
    assert wants_evidence(
        "Which file in your repository does that function live in?",
        OWN_SOURCE,
        lexical_floor=lambda _text: False,
    )


def test_an_empty_request_asks_for_nothing():
    assert not wants_evidence("", OWN_SOURCE)
    assert relevance("", SCREEN_PERCEPTION) == 0.0


def test_a_screen_question_does_not_drag_in_her_source():
    """Both concepts score positive; only one of them is the question."""
    from interface.routes.chat import (
        _turn_may_concern_own_source,
        _turn_may_concern_perception,
    )

    for question in ("what's on my screen?", "what do you see right now?"):
        assert _turn_may_concern_perception(question), question
        assert not _turn_may_concern_own_source(question), question


def test_a_request_to_write_prose_is_not_a_request_to_look():
    """Live: this pulled a screen reading into a request for two sentences."""
    question = "Give me two concise sentences about reliable desktop tool use."
    assert not wants_evidence(question, SCREEN_PERCEPTION)
    assert not wants_evidence(question, OWN_SOURCE)


def test_a_question_that_spans_both_keeps_both():
    """A repo seen on a screen is genuinely both, and she needs both."""
    question = "what was that repo you saw?"
    assert wants_evidence(question, SCREEN_PERCEPTION)
    assert wants_evidence(question, OWN_SOURCE)
