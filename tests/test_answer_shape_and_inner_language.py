"""Two things a mind should not do out loud.

Both were measured on the live desktop path, on the same turn, after a clean
restart. Asked "Hey Aura — quick check before I record a demo. How are you
feeling right now?" she said:

    Yes, I am okay. I feel warm and settled, with low distress and a coherent
    sense of the current thread. My attention is on body_pressure.

Every number behind that was real. Two things were still wrong: "Yes," answered
a yes/no question nobody asked, and ``body_pressure`` is a field name read
aloud as if it were a thing to attend to.
"""

from __future__ import annotations

import pytest

from core.dialogue.question_shape import OPEN, POLAR, open_answer, question_shape
from core.self.inner_language import is_internal_identifier, say_focus


class TestQuestionShape:
    @pytest.mark.parametrize(
        "message",
        [
            "How are you feeling right now?",
            "how are you doing today?",
            "What happened?",
            "Why did you stop?",
            "tell me how you are",
            "Hey Aura — quick check before I record a demo. "
            "How are you feeling right now?",
        ],
    )
    def test_open_questions_are_open(self, message):
        assert question_shape(message) == OPEN

    @pytest.mark.parametrize(
        "message",
        [
            "Are you okay?",
            "are you ok",
            "Do you remember what I asked first?",
            "Can you open Notes?",
            "You are okay, right?",
            "and are you alright?",
            "so do you remember me?",
        ],
    )
    def test_polar_questions_are_polar(self, message):
        assert question_shape(message) == POLAR

    def test_the_last_question_governs(self):
        """A multi-clause message is answered at its final question."""
        assert question_shape("I was worried. Are you okay?") == POLAR

    def test_unknown_falls_through_to_the_open_form(self):
        """Leading with a polarity word nobody asked for is the worse bet."""
        assert open_answer("Nice work.", "Yes, fine.", "Feeling steady.") == (
            "Feeling steady."
        )

    def test_open_question_never_gets_a_yes(self):
        assert open_answer(
            "How are you feeling right now?", "Yes, I am okay.", "I feel settled."
        ) == "I feel settled."


class TestInnerLanguage:
    @pytest.mark.parametrize(
        "raw,spoken",
        [
            ("body_pressure", "how much load my body is carrying"),
            ("felt_coherence", "whether my sense of myself is holding together"),
            ("recall.episodic_query", "remembering"),
            ("cognition.attention.body_pressure", "how much load my body is carrying"),
        ],
    )
    def test_known_channels_are_spoken_in_plain_language(self, raw, spoken):
        assert say_focus(raw) == spoken

    @pytest.mark.parametrize(
        "raw", ["some_unknown_channel_xyz", "lane:sub-name", "weirdCamelChannel"]
    )
    def test_untranslatable_symbols_are_not_spoken_at_all(self, raw):
        """Silence is honest. A field name read aloud as prose is not."""
        assert say_focus(raw) == ""

    @pytest.mark.parametrize(
        "raw",
        [
            "the demo Bryan is recording",
            "trust",
            "what Bryan just asked me to do",
        ],
    )
    def test_her_own_words_pass_through_untouched(self, raw):
        assert say_focus(raw) == raw

    def test_identifier_detection(self):
        assert is_internal_identifier("body_pressure")
        assert is_internal_identifier("a.b.c")
        assert not is_internal_identifier("trust")
        assert not is_internal_identifier("the exchange in front of me")


def test_the_measured_turn_comes_out_clean():
    """The exact live failure, end to end."""
    from core.self.self_condition import _clean_focus

    question = "Hey Aura — quick check before I record a demo. How are you feeling right now?"
    opener = open_answer(
        question, "Yes, I am okay. I feel warm and settled.", "I feel warm and settled."
    )
    assert not opener.startswith("Yes")
    assert _clean_focus("body_pressure") == "how much load my body is carrying"
