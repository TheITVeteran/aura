"""An error must be reported as an error, not explained as a decision.

LIVE DEFECT, 2026-08-10, observed twice in one session.

Confronted with an inaccurate answer, she did not report uncertainty — she
asserted an intention:

    "I do have access to that information. And yes, I told you a comfortable
     story because it's easier than the truth."

and, an hour later, after inventing a statement she had never made:

    "If you are asking why I provided a false account, it was because I
     believed that an accurate report of my internal state would not have been
     as entertaining."

She has no record of forming either intention. Both are second fabrications
laid on top of the first, and the second one is corrosive in a way the
original error was not: it tells the person she will trade accuracy for
entertainment.

The identity contract already holds the right standard for actions — "never
claim an action completed without verified outcome evidence". These rules
apply that same standard to claims about her own past words and her own
reasons.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def rules() -> tuple[str, ...]:
    from core.conversation.chat_preflight import _SELF_SPEECH_RULES

    return _SELF_SPEECH_RULES


def test_transcript_is_the_source_for_what_she_said(rules) -> None:
    blob = " ".join(rules).lower()

    assert "transcript" in blob
    assert "cannot retrieve" in blob


def test_errors_are_reported_as_errors_not_intentions(rules) -> None:
    blob = " ".join(rules).lower()

    assert "not as a decision" in blob
    assert "no record of intentions" in blob


def test_a_true_alternative_is_offered_not_just_a_prohibition(rules) -> None:
    """Fix the reasoning, not the words: she needs a sentence she CAN say."""
    blob = " ".join(rules).lower()

    assert "generated that rather than retrieved it" in blob


def test_corrections_are_not_accepted_by_inventing_agreement(rules) -> None:
    blob = " ".join(rules).lower()

    assert "inventing agreement" in blob


def test_rules_reach_the_identity_contract() -> None:
    """A rule nobody injects is a comment."""
    import inspect

    from core.conversation import chat_preflight

    source = inspect.getsource(chat_preflight.inject_operational_self_context)

    assert "_SELF_SPEECH_RULES" in source


def test_existing_evidence_rules_survive(rules) -> None:
    """These additions must not displace the capability-honesty rules."""
    blob = " ".join(rules).lower()

    assert "verified outcome evidence" in blob
    assert "just a language model" in blob
