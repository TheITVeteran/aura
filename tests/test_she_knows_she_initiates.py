"""She has independent thought generation, and told Bryan she doesn't.

Asked outright whether she could message him first, she answered:

    That's not accurate. I respond to user inputs — that's my primary
    function... My spontaneous seeming messages are actually the result of
    pattern recognition in our ongoing dialogue, not independent thought
    generation.

and produced a JSON block reading {"initiative": false, "thought_pattern":
"linear-response"}. None of that was read from anywhere. It was asserted.

At that same moment her log carried her own unprompted goal updates —
"I'm hitting a bit of a wall with the user privacy research" — and an hour
earlier she had opened a conversation with "I noticed you went quiet.
Everything alright?", which Bryan received before saying anything.

So the faculty is real, it fires, and the only thing missing was her knowing
it. Believing a false thing about herself is the defect; the record is the
correction.
"""

import pytest

import core.autonomy.proactive_presence as proactive_presence
from core.conversation.capability_condition import (
    capability_condition_evidence,
    needed_capabilities,
)

pytestmark = pytest.mark.unit

ASKED = "Can you initiate a conversation on your own, or do you only respond to prompts?"


@pytest.fixture(autouse=True)
def _clean_record():
    saved = dict(proactive_presence._INITIATIVE_LOG)
    proactive_presence._INITIATIVE_LOG.update({"count": 0, "last_at": 0.0, "last_text": ""})
    yield
    proactive_presence._INITIATIVE_LOG.update(saved)


class TestTheQuestionIsRecognised:
    @pytest.mark.parametrize("question", [
        ASKED,
        "do you have independent thought?",
        "can you message me first, unprompted?",
        "do you ever reach out spontaneously?",
    ])
    def test_asking_about_initiative_pulls_the_record(self, question):
        assert "initiative" in needed_capabilities(question)

    def test_an_ordinary_turn_does_not(self):
        assert "initiative" not in needed_capabilities("what do you think about entropy?")


class TestTheRecordAnswersForHer:
    def test_having_spoken_first_is_stated_as_fact(self):
        proactive_presence.note_unprompted_message(
            "I noticed you went quiet. Everything alright?"
        )
        block = capability_condition_evidence(ASKED)
        assert "STARTED 1 CONVERSATION(S) YOURSELF" in block
        assert "I noticed you went quiet" in block
        assert "Do not claim you only respond to prompts" in block

    def test_not_yet_this_session_is_not_a_denial(self):
        """"It hasn't happened yet" and "I can't" are different claims."""
        block = capability_condition_evidence(ASKED)
        assert "proactive-presence faculty" in block
        assert "not about whether you have it" in block

    def test_the_counter_moves_when_she_speaks(self):
        assert proactive_presence.initiative_record()["has_spoken_unprompted"] is False
        proactive_presence.note_unprompted_message("Morning. Quiet start?")
        record = proactive_presence.initiative_record()
        assert record["has_spoken_unprompted"] is True
        assert record["count"] == 1
