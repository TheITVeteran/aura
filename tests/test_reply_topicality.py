"""The off-topic detector, and why it could never fire.

Measured live in the desktop UI. Asked to "run a real calculation in your Python
sandbox and show me the result", the entire reply was 48 tokens of felt state —
"Things feel unusually settled right now. My attention is on internal
monitoring..." — sharing ZERO topic words with the question. It was logged
off_topic=False.

Two independent reasons, both of which had to be fixed:

1. The single overlapping token was "now", a temporal filler, and ANY single
   overlap exonerated the reply.
2. The bridge-marker guard listed bare "it", "this", "that", "there" and
   "because" and matched them as SUBSTRINGS of the whole reply. Every English
   sentence of any length contains those — "it" appears inside "priority" and
   "intact" — so the guard was unconditionally true and `foreign_topic_burst`
   could never fire at all.

This gate leads to repair, so the false-POSITIVE cases matter as much as the
false negative: a real answer, an honest refusal, and an on-topic reply that
happens to share no vocabulary must all still pass.
"""

from __future__ import annotations

QUESTION = (
    "Now actually do something rather than describe it: run a real calculation "
    "in your Python sandbox and show me the result. Compute the 5000th prime "
    "number, and tell me what code you ran."
)

LIVE_OFF_TOPIC_REPLY = (
    "Specifically, the grounded read I have right now is: Things feel unusually "
    "settled right now. My attention is on internal monitoring. The system is "
    "settling and conserving effort. The active mode is reactive: more protective "
    "of continuity than expansive. The thread I am holding is not abstract "
    "self-description; it is this conversation's pressure around whether the live "
    "path can stay coherent while the rest of the mind keeps moving. The next "
    "useful priority is to keep the foreground answer intact, then let the "
    "background systems act cleanly."
)


def test_the_live_off_topic_reply_is_caught():
    from interface.routes.chat import _evaluate_reply_topicality

    off_topic, reason = _evaluate_reply_topicality(QUESTION, LIVE_OFF_TOPIC_REPLY)
    assert off_topic is True, "the measured live failure must be detected"
    assert reason == "foreign_topic_burst"


def test_temporal_filler_is_not_a_topic_anchor():
    """"now" was the single token holding the whole verdict open."""
    from interface.routes.chat import _extract_topic_tokens

    assert "now" not in _extract_topic_tokens("tell me what is happening right now")


def test_second_person_words_do_not_bypass_subject_evidence():
    """The live failure used "your" and still abandoned the requested task."""
    from interface.routes.chat import _evaluate_reply_topicality

    reply = LIVE_OFF_TOPIC_REPLY + " You asked me to stay connected."
    assert _evaluate_reply_topicality(QUESTION, reply) == (
        True,
        "foreign_topic_burst",
    )


def test_self_process_questions_can_receive_runtime_self_prose():
    from interface.routes.chat import _evaluate_reply_topicality

    question = "How does uncertainty change your attention and decision process?"
    assert _evaluate_reply_topicality(question, LIVE_OFF_TOPIC_REPLY) == (False, "")


def test_external_reasoning_request_remains_an_external_task():
    from interface.routes.chat import _evaluate_reply_topicality

    question = "Can you use your reasoning process to solve this checksum?"
    assert _evaluate_reply_topicality(question, LIVE_OFF_TOPIC_REPLY) == (
        True,
        "foreign_topic_burst",
    )


def test_real_answers_refusals_and_topic_free_replies_all_pass():
    """This gate triggers repair, so false positives cost real replies."""
    from interface.routes.chat import _evaluate_reply_topicality

    passing = (
        # A real, executed answer.
        "I ran it in the sandbox: a simple sieve, and the 5000th prime came back "
        "as 48611. The code incremented a candidate and tested divisibility up to "
        "its square root, counting until 5000.",
        # An honest refusal.
        "I can't execute that from this turn — the sandbox isn't reachable, so I "
        "won't hand you a number I didn't actually compute.",
        # On topic, engaging with the ask, sharing almost no vocabulary with it.
        "You asked for something executed rather than described. I couldn't reach "
        "the interpreter, so there is nothing I can honestly hand back to you.",
    )
    for reply in passing:
        off_topic, reason = _evaluate_reply_topicality(QUESTION, reply)
        assert off_topic is False, f"a good reply was flagged {reason!r}: {reply[:60]!r}"


def test_short_replies_are_never_flagged():
    """A brief answer has no room to demonstrate topical overlap."""
    from interface.routes.chat import _evaluate_reply_topicality

    for reply in ("48611.", "Yes.", "I can't run that.", "Done — 48611."):
        assert _evaluate_reply_topicality(QUESTION, reply) == (False, "")
