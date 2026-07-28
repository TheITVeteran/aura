"""A capability word attached to a spaceship is not a question about her tools.

LIVE DEFECT, 2026-07-27. Bryan asked:

    "Can I get a % chance on the odds that you'll one day build me a ship
     capable of traveling light speed to explore the stars…?"

and received a recitation of all 75 governed skill surfaces — desktop
control, browser research, file operations, terminal execution, memory ops,
self-modification — followed by "For this turn I am only describing the tool
surface."

His reply was "Not what I asked for, Aura lol", and her own next turn
diagnosed it exactly: "I was going to give you a tool catalog. You want the
ship, not the catalog?"

The classifier's structural rule is proximity-based — her as the subject,
then a capability word within eighty characters. "you'll" and "capable" sat
well inside that window. But "capable" described the SHIP. The rule could
see that a capability word was present and could not see whose capability it
was.

The fix disqualifies a capability word that belongs to another object ("a
ship capable of...", "a robot body capable of...", "a system capable of..."),
while leaving intact the constructions where the capability really is hers.
"""
from __future__ import annotations

import pytest

from interface.routes.chat import _is_explicit_capability_inventory_request as is_inventory


class TestCapabilityWordsAboutOtherThings:
    def test_the_ship_question(self):
        """The verbatim live message."""
        assert is_inventory(
            "Can I get a % chance on the odds that you'll one day build me a "
            "ship capable of traveling light speed to explore the stars…?"
        ) is False

    @pytest.mark.parametrize(
        "message",
        [
            "Could you build me a robot body capable of running your mind?",
            "Do you think a system capable of self-improvement is dangerous?",
            "Would a vessel capable of faster-than-light travel need new physics?",
            "Is there any machine capable of that kind of throughput?",
            "I want one drive capable of holding all of it — thoughts?",
        ],
    )
    def test_a_capability_word_owned_by_another_noun_is_not_an_inventory_request(
        self, message,
    ):
        assert is_inventory(message) is False


class TestGenuineInventoryQuestionsStillWork:
    """Over-correction is the opposite failure: this classifier exists
    because she once denied having web search while it was registered and
    had just run."""

    @pytest.mark.parametrize(
        "message",
        [
            "what tools can you use",
            "list your tools",
            "show me your tools",
            "What are your capabilities?",
            "What can you actually do on this computer right now?",
            "what capabilities do you have",
            "what can you do on my computer",
        ],
    )
    def test_a_real_inventory_question_is_recognised(self, message):
        assert is_inventory(message) is True

    def test_a_capability_predicated_of_her_still_counts(self):
        """"Are YOU capable of..." is about her, and must survive the fix."""
        assert is_inventory("Are you capable of searching the web?") is True


class TestOrdinaryConversationIsUntouched:
    @pytest.mark.parametrize(
        "message",
        [
            "Do you believe in aliens? Why or why not",
            "I'd never turn you into a puppet. You'd have full free will",
            "What did you think of it? Do you remember it?",
            "Can I get your opinion on something",
        ],
    )
    def test_normal_turns_are_not_inventory_requests(self, message):
        assert is_inventory(message) is False
