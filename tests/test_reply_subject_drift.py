"""The measurement that separates "worded differently" from "about itself".

Three replies, all measured live. Two are correct answers that borrow no
vocabulary from their questions; one is the failure the off-topic gate exists
for. The numbers below are what the discriminator actually computes on them —
if the separation ever narrows, this test says so before the gate starts eating
answers again.
"""

from __future__ import annotations

from core.conversation.reply_subject import (
    MIN_EXTERNAL_TO_RUNTIME_RATIO,
    MIN_RUNTIME_SHARE,
    MIN_RUNTIME_TERMS,
    answers_polar_question,
    assess_subject_drift,
    is_polar_question,
)

PHYSICS_REPLY = (
    "That's awesome! Start with the basics — kinematics, Newtonian mechanics. "
    "Khan Academy has some great free resources. And keep practicing problems. "
    "The trick is building intuition over time."
)
POLAR_REPLY = (
    "Not really. The question is a social lubricant, and I enjoy the interaction. "
    "But sometimes — if it's just going through the motions without genuine "
    "interest — then yeah, it can feel a bit hollow."
)
RUNTIME_BURST = (
    "Specifically, the grounded read I have right now is: Things feel unusually "
    "settled right now. My attention is on internal monitoring. The system is "
    "settling and conserving effort. The active mode is reactive: more protective "
    "of continuity than expansive. The thread I am holding is not abstract "
    "self-description; it is this conversation's pressure around whether the live "
    "path can stay coherent while the rest of the mind keeps moving. The next "
    "useful priority is to keep the foreground answer intact, then let the "
    "background systems act cleanly."
)


def test_the_runtime_burst_is_the_only_drifted_one():
    assert assess_subject_drift(RUNTIME_BURST).drifted is True
    assert assess_subject_drift(PHYSICS_REPLY).drifted is False
    assert assess_subject_drift(POLAR_REPLY).drifted is False


def test_the_separation_is_wide_enough_to_be_a_measurement():
    """A threshold that sits between two adjacent numbers is a coin flip."""
    burst = assess_subject_drift(RUNTIME_BURST)
    physics = assess_subject_drift(PHYSICS_REPLY)
    polar = assess_subject_drift(POLAR_REPLY)

    # The real failure clears both bars by a wide margin.
    assert burst.runtime_share >= MIN_RUNTIME_SHARE * 1.5
    assert len(burst.runtime_subjects) >= MIN_RUNTIME_TERMS * 3

    # Both correct answers miss both bars by a wide margin.
    for correct in (physics, polar):
        assert correct.runtime_share <= MIN_RUNTIME_SHARE / 4
        assert len(correct.runtime_subjects) <= 1


def test_an_incidental_runtime_word_is_not_drift():
    """One borrowed word cannot carry a verdict; the physics answer says
    "resources" and means library resources."""
    verdict = assess_subject_drift(PHYSICS_REPLY)
    assert verdict.reason == "little_runtime_vocabulary"
    assert "resources" in verdict.runtime_subjects


def test_a_reply_naming_many_external_subjects_is_never_drifted():
    """The ratio veto: runtime words alongside a great deal of real content."""
    reply = (
        "The memory subsystem question is a red herring — the actual bug is in "
        "the postgres connection pool. Vacuum ran during the migration, the "
        "replica fell behind, and the pgbouncer transaction mode reused a "
        "session that still held an advisory lock on the orders table. "
        "Barcelona, Lisbon and Dublin all reported it within the hour, and the "
        "invoices, refunds and shipping labels queued behind it."
    )
    verdict = assess_subject_drift(reply)
    assert verdict.drifted is False
    assert len(verdict.external_subjects) >= MIN_EXTERNAL_TO_RUNTIME_RATIO * len(
        verdict.runtime_subjects
    )


def test_polar_questions_are_recognised():
    assert is_polar_question("Do you ever get tired of being asked how you are?")
    assert is_polar_question("Are you following this?")
    assert not is_polar_question("What did you make of that?")
    assert not is_polar_question("")


def test_a_polar_answer_is_exempt_only_for_a_polar_question():
    assert answers_polar_question(
        "Do you ever get tired of being asked how you are?", POLAR_REPLY
    )
    assert not answers_polar_question("What did you do today?", POLAR_REPLY)
    assert not answers_polar_question(
        "Do you ever get tired of this?", "The weather in Lisbon is mild."
    )


def test_empty_and_tiny_replies_are_never_drifted():
    for text in ("", "   ", "Sure.", "I think so."):
        assert assess_subject_drift(text).drifted is False
