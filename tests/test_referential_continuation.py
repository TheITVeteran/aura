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


class TestTheGeneralReplyPathIsBound:
    """The resolver must reach ordinary conversation, not just desktop routing.

    Both cases below reached cognition as bare fragments live on 2026-08-03,
    because think() received one message and no turn before it.
    """

    @staticmethod
    def _engine():
        from core.brain.cognitive_engine import CognitiveEngine

        return CognitiveEngine.__new__(CognitiveEngine)

    @staticmethod
    def _transcript(turns):
        """A transcript containing exactly these turns.

        UnifiedTranscript is a process singleton, so without clearing it each
        test resolves against the previous test's conversation — which is the
        correct behaviour in production (the immediately preceding turn IS the
        antecedent) and pure cross-test pollution here.
        """
        from core.conversation.unified_transcript import UnifiedTranscript

        t = UnifiedTranscript.get_instance()
        with t._lock:
            t._entries.clear()
        for role, content in turns:
            t.add(role, content)
        return t

    def test_a_fragment_answering_her_question_is_bound_to_it(self):
        self._transcript([
            ("user", "I submitted you for a research grant. Want the response?"),
            ("aura", "Tell me the good news first. Response from who?"),
            ("user", "From the grant research funds manager"),
        ])
        resolved = self._engine()._objective_with_antecedent(
            "From the grant research funds manager"
        )
        assert "Response from who?" in resolved
        assert "grant research funds manager" in resolved

    def test_a_retry_is_bound_to_the_request_it_retries(self):
        self._transcript([
            ("user", "Hey, Aura can you tell me what you see on my screen currently?"),
            ("aura", "I can't see your screen right now."),
            ("user", "Can you do it now?"),
        ])
        resolved = self._engine()._objective_with_antecedent("Can you do it now?")
        assert "screen" in resolved
        assert "Can you do it now?" in resolved

    def test_a_standalone_message_is_returned_unchanged(self):
        self._transcript([
            ("user", "Hey, Aura can you tell me what you see on my screen currently?"),
            ("aura", "I can't see your screen right now."),
        ])
        message = "What do you think about emergence?"
        assert self._engine()._objective_with_antecedent(message) == message

    def test_it_never_resolves_a_message_against_itself(self):
        """The current turn is already in the transcript when think() runs."""
        self._transcript([("user", "do it")])
        resolved = self._engine()._objective_with_antecedent("do it")
        assert resolved == "do it"

    def test_empty_and_broken_lookups_are_safe(self):
        engine = self._engine()
        assert engine._objective_with_antecedent("") == ""
        assert engine._objective_with_antecedent("   ").strip() == ""
