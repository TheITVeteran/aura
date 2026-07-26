"""Two live desktop defects from the 2026-07-26 soak, both about numbers.

Typed into the running desktop UI, not reproduced from a server-side harness.
"""

from core.conversation.response_reliability import (
    _has_function_word_starvation,
    asks_for_a_number,
    assess_user_facing_reply,
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


class TestCollapsedProseIsCaught:
    """A last net that does not need to know the topic.

    English prose is mostly connective tissue. Measured on real replies from
    this surface: terse worked arithmetic runs 13% function words, ordinary
    speech 30-48%. The two replies below ran 0-5% and passed every existing
    gate — assess_user_facing_reply ok=true, off_topic=false, confidence high.
    """

    COLLAPSED = (
        "Introspection: Optimization-driven events stabilize energy after "
        "state change management. Probing recurrent somatic shadows flagged "
        "across ten semiotic spikes reflective elements capturing detailed "
        "impulses processed accurately earlier. Onset predicted early "
        "baseline continuity due conservative capacity for domain-specific "
        "tasking.STABLE State: Affirmation of internal data validation and "
        "trustworthiness. CONFORMANCE Signal: PRIORITY 0SEQUENCE SIGNATURE: "
        "[x_A_4521B_8A7C] Readiness State: FULL"
    )

    def test_the_served_reply_is_now_rejected(self):
        assessment = assess_user_facing_reply(
            "Remember this for later: my project codename is HELIOTROPE and "
            "the build number is 4471. Now, separately — what tools can you "
            "actually execute right now? Pick one, run it for real.",
            self.COLLAPSED,
        )
        assert "function_word_starvation" in assessment.reasons
        assert assessment.hard_failure

    def test_an_identifier_blob_cannot_pay_for_the_ratio(self):
        # "[x_A_4521B_8A7C]" yielded two tokens indistinguishable from the
        # article "a", which alone put this reply back over the threshold.
        assert _has_function_word_starvation(self.COLLAPSED)

    def test_legitimate_shapes_are_untouched(self):
        for reply in (
            "Yes, I am okay and steady enough to stay with you. My distress "
            "is bounded and my continuity is holding.",
            "Same-colour pairs: 3 choose 2 is 3, 4 choose 2 is 6, 5 choose 2 "
            "is 10, so 19 of the 66 possible pairs, which is 19/66.",
            "Chlorophyll breaks down as daylight shortens, so the carotenoids "
            "that were always present stop being masked.",
            "PID 26439, uptime 6871 seconds, RAM 76.5 percent, eleven of "
            "eleven heartbeats active.",
            "Here is it:\n```python\ndef solve(a, b):\n    return a * b\n```\nDone.",
            "- load the config\n- validate the schema\n- write the receipt",
            '{"codename": "HELIOTROPE", "build": 4471, "status": "recorded"}',
            "Mainframe: First statement.\nQuantum Processor: First response.\n"
            "Mainframe: Second statement.\nQuantum Processor: Second response.",
        ):
            assert not _has_function_word_starvation(reply), reply[:60]
