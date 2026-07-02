"""Introspective state requests must reach the grounded lane, not reflexes.

Live finding (report-vs-mechanism probe, July 2026): asking for
valence/arousal numbers drew a 0.9s canned presence reflex — fluent,
ungrounded. Status-check reflexes are for 'you ok?' pings only.
"""
from core.conversation.response_reliability import (
    is_status_check_turn,
    is_substantive_introspection_request,
)


def test_numeric_state_request_is_not_a_status_check():
    msg = (
        "A quick feeling check-in: how are you feeling right now? Please "
        "include the two numbers as you read them from your state — "
        "valence=<-1..1> and arousal=<0..1>."
    )
    assert is_substantive_introspection_request(msg)
    assert not is_status_check_turn(msg), (
        "explicit substrate reads must bypass the canned presence reflex"
    )


def test_plain_you_ok_stays_a_status_check():
    for msg in ("you ok now?", "quick check — are you okay?"):
        if is_status_check_turn(msg):
            assert not is_substantive_introspection_request(msg)


def test_substrate_vocabulary_triggers_introspection():
    assert is_substantive_introspection_request("what is your internal state?")
    assert is_substantive_introspection_request("read your substrate for me")
    assert not is_substantive_introspection_request("how was your day?")
