"""A message that means nothing alone must inherit the turn that gives it meaning.

Every case here is from the live transcript of 2026-08-03, where Bryan was
refused three times for one request and had his answer to Aura's own question
ignored.
"""

from __future__ import annotations

import pytest

from core.runtime.desktop_objective_intent import looks_like_screen_observation
from core.runtime.referential_continuation import (
    answers_a_question,
    effective_message,
    is_referential_continuation,
)

SCREEN_REQUEST = "Hey, Aura can you tell me what you see on my screen currently?"


@pytest.mark.parametrize(
    "message", ["Can you do it now?", "Yes you can lol", "try again", "do it", "please"]
)
def test_follow_ups_inherit_the_request(message):
    """Each of these was refused live. "It" was the screen read."""
    assert not looks_like_screen_observation(message), "precondition: not self-contained"
    resolved = effective_message(message, previous_user_request=SCREEN_REQUEST)
    assert resolved.kind == "continuation"
    assert looks_like_screen_observation(resolved.text), message


def test_the_answer_to_her_own_question_is_bound_to_it():
    """Aura asked "Response from who?"; Bryan answered; she repeated herself."""
    resolved = effective_message(
        "From the grant research funds manager",
        previous_assistant_message="Tell me the good news first. Response from who?",
    )
    assert resolved.kind == "answer"
    assert "Response from who?" in resolved.text
    assert "grant research funds manager" in resolved.text


@pytest.mark.parametrize(
    "message",
    [
        "Can you open Chrome now?",                            # names its own object
        "yes, and also summarise the last three papers",        # new request
        "What do you think about emergence?",                   # standalone question
        "Tell me about corrigibility",                          # standalone
    ],
)
def test_standalone_messages_are_left_alone(message):
    """Attaching stale intent to a fresh request is worse than missing a follow-up."""
    assert not is_referential_continuation(message), message
    resolved = effective_message(message, previous_user_request=SCREEN_REQUEST)
    assert not resolved.resolved
    assert resolved.text == message


def test_a_fragment_after_a_statement_is_not_an_answer():
    """Only a QUESTION creates a slot for a fragment to fill."""
    assert not answers_a_question(
        "From the grant research funds manager",
        "I've been thinking about protocols for autonomous agent swarms.",
    )


def test_resolution_never_rewrites_the_visible_message():
    """The joined form is for routing. It is not what Bryan said."""
    resolved = effective_message("do it", previous_user_request=SCREEN_REQUEST)
    assert resolved.antecedent == SCREEN_REQUEST
    assert resolved.text != "do it"          # joined for routing
    assert "do it" in resolved.text          # his words preserved inside it


def test_empty_and_missing_context_are_safe():
    assert effective_message("").text == ""
    assert not effective_message("do it").resolved
    assert not effective_message("do it", previous_user_request="").resolved
