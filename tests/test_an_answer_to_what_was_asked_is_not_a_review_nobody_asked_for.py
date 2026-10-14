"""A reply that takes up what the message raised is not discarded for its wording.

Live, 13 October: asked what she made of a limpet that farms a patch of algae,
her answer was rejected as an "unrequested content review" by a list of words
(plot, piece, the story is) and a fallback that invented a genus was sent.
"""
from __future__ import annotations

import pytest

from core.conation import what_she_wants
from core.conation.engine import ConationEngine
from interface.routes.chat import _looks_like_unrequested_content_review

ASKED = (
    "I read that the owl limpet farms its own patch of algae and shoves other limpets off it. "
    "A limpet with a garden. What do you make of a limpet that defends territory?"
)
ANSWERED = (
    "The story is simpler than farming, I think. The limpet scrapes its plot bare in a way that "
    "lets a thin film of algae regrow, and a limpet that wanders onto the plot gets shoved off. "
    "The narrative of a garden is ours. Each piece of rock the limpet keeps is one the limpet "
    "can graze again, and that is the whole premise of holding territory."
)
A_REVIEW = (
    "The story is a classic setup: this narrative follows a character through each chapter, "
    "and the plot of the novel turns on the premise that the film never shows."
)


@pytest.fixture
def has_talked(tmp_path):
    record = what_she_wants.reset_for_test(tmp_path / "wants.json", engine=ConationEngine())
    record.talk.remember_talk(
        [
            ("What do you make of the plan, and has the time come to look at it?",
             "I think the plan is fine. There is a piece of it I would change, and I will say which."),
            ("Tell me what you think the weather will do this weekend.",
             "I think it turns on the front that is still off the coast, and that is all I can say."),
        ]
        * 30
    )
    yield record
    what_she_wants.reset_for_test(tmp_path / "after.json", engine=ConationEngine())


def test_her_answer_about_the_limpet_is_kept(has_talked):
    assert _looks_like_unrequested_content_review(ASKED, ANSWERED) == (False, "")


def test_a_review_of_a_story_nobody_mentioned_is_still_caught(has_talked):
    asked = "Tell me what you think the weather will do this weekend."
    assert _looks_like_unrequested_content_review(asked, A_REVIEW) == (True, "unrequested_content_review")


def test_asking_what_it_is_about_does_not_count_the_exchange(has_talked):
    before = has_talked.talk._exchanges
    assert "limpet" in has_talked.talk.subjects_of(ASKED, ANSWERED)
    assert has_talked.talk._exchanges == before
