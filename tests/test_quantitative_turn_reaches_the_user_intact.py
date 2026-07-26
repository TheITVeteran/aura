"""Two live desktop defects from the 2026-07-26 soak, both about numbers.

Typed into the running desktop UI, not reproduced from a server-side harness.
"""

from core.conversation.response_reliability import (
    asks_for_a_number,
    normalize_user_facing_format,
    numeric_answer_missing,
    requires_reasoning_lane,
)

MARBLES = (
    "A bag has 3 red, 4 blue and 5 green marbles. I draw two without "
    "replacement. What's the probability both are the same colour? "
    "Show the reasoning, then give the exact fraction."
)


class TestDecimalSurvivesFormatRepair:
    """A decimal point is not a jammed list marker."""

    def test_trailing_decimal_is_not_split_onto_its_own_line(self):
        # Live: this reply reached the chat window as "...coherence 0." on one
        # line and "86" on the next.
        served = (
            "The current supported values are valence -0.04, arousal 0.48, "
            "distress 0.18, welfare 0.80, coherence 0.86."
        )
        assert normalize_user_facing_format(served) == served

    def test_decimals_mid_sentence_survive(self):
        for text in (
            "Pi is roughly 3.14159.",
            "The probability is 7/33, about 0.21.",
            "Score 4.5. Next item.",
            "It finished in 1999. 2000 was next.",
        ):
            assert normalize_user_facing_format(text) == text

    def test_a_genuinely_jammed_list_marker_is_still_repaired(self):
        assert normalize_user_facing_format("Here are the steps.1. Do X.2. Do Y.") == (
            "Here are the steps.\n1. Do X.\n2. Do Y."
        )


class TestWordProblemsAreDeterminateTurns:
    """Steering must stand down for a question with one right answer.

    Live: the marble question below carries no arithmetic operator and says
    "show the reasoning" rather than "show your work", so both classifiers
    said no. Steering stayed at the governor's alpha and the served reply was
    "Do product of multiple exponent term simplify reflexion".
    """

    def test_probability_word_problem_is_determinate(self):
        assert asks_for_a_number(MARBLES)
        assert requires_reasoning_lane(MARBLES)

    def test_a_reply_with_no_quantity_at_all_is_caught(self):
        assert numeric_answer_missing(
            MARBLES, "Do product of multiple exponent term simplify reflexion"
        )
        assert not numeric_answer_missing(
            MARBLES, "Same-colour pairs: 3+6+10 = 19 of 66, so 19/66."
        )

    def test_other_quantity_nouns_count(self):
        for text in (
            "What is the average of 4, 9 and 11?",
            "Two dice are rolled. What are the odds of 7 or 11?",
            "Out of 40 seats, 12 are taken. What fraction is that?",
        ):
            assert asks_for_a_number(text), text

    def test_expressive_turns_are_left_alone(self):
        for text in (
            "How are you feeling today?",
            "Tell me about your memory system.",
            "What do you think about honesty?",
            "What is the value of our friendship?",
            "Can you write me a poem about rain?",
            "How many episodes have you lived through?",
        ):
            assert not asks_for_a_number(text), text
            assert not requires_reasoning_lane(text), text
