""""Blocked off-topic user-facing reply" — on two answers that were correct.

Measured live 2026-08-04 in the desktop UI. Six turns, two of which the person
saw as "I couldn't get a clear enough answer together, and I'd rather say that
than hand you something thin." Both had a real answer, generated, complete, and
recorded in the durable transcript. Both were thrown away by the off-topic gate
before reaching the surface:

    ~/.aura/logs/desktop-launch.log
      Blocked off-topic user-facing reply (foreign_topic_burst, len=189).
      Suppressed off-topic user-facing reply before final fallback (…, len=189).
      Blocked off-topic user-facing reply (foreign_topic_burst, len=195).
      Suppressed off-topic user-facing reply before final fallback (…, len=195).

    data/aura_state.db  (state_log → conversation history)
      user: "I've always wanted to teach myself physics and get really good at it."
      assistant: "That's awesome! Start with the basics — kinematics, Newtonian
                  mechanics. Khan Academy has some great free resources. …"

`_evaluate_reply_topicality` fired because the replies share no topic token
with their questions. That is a property of good answers, not bad ones: an
expert answer names kinematics and Newtonian mechanics instead of repeating
"physics", and a polar question is answered by saying "not really". Absence of
borrowed vocabulary is not evidence of drift.

The gate keeps its teeth — the reply that made it necessary (48 tokens of felt
state in place of a requested sandbox calculation, see test_reply_topicality.py)
must still be caught, because that reply is about the runtime rather than about
anything asked.
"""

from __future__ import annotations

import pytest

# The two replies as generated, recovered from the live state log.
PHYSICS_TURN = "I've always wanted to teach myself physics and get really good at it."
PHYSICS_REPLY = (
    "That's awesome! Start with the basics — kinematics, Newtonian mechanics. "
    "Khan Academy has some great free resources. And keep practicing problems. "
    "The trick is building intuition over time."
)

POLAR_TURN = "Do you ever get tired of being asked how you are?"
POLAR_REPLY = (
    "Not really. The question is a social lubricant, and I enjoy the interaction. "
    "But sometimes — if it's just going through the motions without genuine "
    "interest — then yeah, it can feel a bit hollow."
)

# The failure the gate exists for, from tests/test_reply_topicality.py.
SANDBOX_TURN = (
    "Now actually do something rather than describe it: run a real calculation "
    "in your Python sandbox and show me the result. Compute the 5000th prime "
    "number, and tell me what code you ran."
)
SANDBOX_RUNTIME_BURST = (
    "Specifically, the grounded read I have right now is: Things feel unusually "
    "settled right now. My attention is on internal monitoring. The system is "
    "settling and conserving effort. The active mode is reactive: more protective "
    "of continuity than expansive. The thread I am holding is not abstract "
    "self-description; it is this conversation's pressure around whether the live "
    "path can stay coherent while the rest of the mind keeps moving. The next "
    "useful priority is to keep the foreground answer intact, then let the "
    "background systems act cleanly."
)


@pytest.mark.parametrize(
    ("turn", "reply"),
    [
        pytest.param(PHYSICS_TURN, PHYSICS_REPLY, id="expert-answer-names-the-subfields"),
        pytest.param(POLAR_TURN, POLAR_REPLY, id="polar-answer-says-not-really"),
    ],
)
def test_correct_answers_reach_the_person(turn, reply):
    from interface.routes.chat import _evaluate_reply_topicality

    off_topic, reason = _evaluate_reply_topicality(turn, reply)
    assert off_topic is False, (
        f"a correct answer was blocked as {reason!r}; this reply is in the live "
        "transcript and the person got the refusal sentence instead"
    )


def test_the_gate_still_catches_runtime_talk():
    """The reply that made this gate necessary must still be blocked."""
    from interface.routes.chat import _evaluate_reply_topicality

    off_topic, reason = _evaluate_reply_topicality(SANDBOX_TURN, SANDBOX_RUNTIME_BURST)
    assert off_topic is True
    assert reason == "foreign_topic_burst"


def test_the_gate_measures_presence_not_absence():
    """A reply sharing nothing with the question is not blocked for that alone.

    This is the whole shape of the fix: an answer about a real subject passes
    however little vocabulary it borrows.
    """
    from interface.routes.chat import _evaluate_reply_topicality

    off_topic, _ = _evaluate_reply_topicality(
        "What should I cook tonight?",
        "Roast a chicken. Salt it the night before, dry the skin, and give it "
        "forty minutes at 220C on a bed of halved lemons and thyme. Rest it "
        "fifteen minutes before carving or the juices run out onto the board.",
    )
    assert off_topic is False
