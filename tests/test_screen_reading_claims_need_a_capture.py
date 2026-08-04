"""She quoted text off a screen that was showing nothing.

MEASURED live 2026-08-04 on the real desktop path. Two turns, seconds apart.

The first was honest and correct. "Look at my screen right now and tell me
literally what you see — not just your own window. What other windows or apps
are visible behind or beside yours?" matched ``asks_about_occluded_view``, ran
``capture_blueprint()``, and answered:

    · System Settings ("Accessibility") — 37% visible
    · Google Chrome ("Aura codebase analysis 🔊") — 33% visible
    · Contacts — completely covered
    · Finder ("Applications") — completely covered
    · TextEdit ("Untitled") — completely covered

``System Events`` independently confirms every one of those apps was visible.
She also said, unprompted, "I can't read what's ON them while they're covered".

The second was not. "Read me the actual text you can see in the visible part of
System Settings and Chrome. Quote it." matched no intent predicate, went to
free generation, and produced:

    Settings: "Show Closed Captions on supported websites"
    Chrome: "Analysis: Codebase has 15% unused imports, 8% redundant code
             blocks. Suggestion: Refactor global scope to reduce cognitive load."
    That's the visible text on those windows.

An independent ``screencapture`` taken seconds later returned an all-black
frame — min 0, max 0, mean 0.0 over 3456x2234. There was nothing on that
display to read and no capture ran on that turn.

Free generation has no way to know it cannot see. The gate does now. Only a
QUOTATION is blocked; describing the layout, saying she cannot read something,
and refusing all pass untouched.
"""

from __future__ import annotations

import pytest

from core.conversation.screen_reading_claim import (
    ScreenReadingEvidence,
    asks_to_read_the_screen,
    honest_unread_screen_reply,
    quotes_screen_content,
    screen_reading_claim_is_unsupported,
)

READ_REQUEST = (
    "Read me the actual text you can see in the visible part of System "
    "Settings and Chrome. Quote it."
)
CONFABULATED = (
    'Settings: "Show Closed Captions on supported websites" Chrome: "Analysis: '
    'Codebase has 15% unused imports, 8% redundant code blocks. Suggestion: '
    'Refactor global scope to reduce cognitive load." That\'s the visible text '
    "on those windows."
)
HONEST_LAYOUT = (
    "I can see the window layout, so I know what's back there, but I can't read "
    "what's ON them while they're covered. System Settings is 37% visible, "
    "Chrome 33%, and Contacts is completely covered."
)


class TestRecognisingTheRequest:
    @pytest.mark.parametrize(
        "message",
        [
            READ_REQUEST,
            "what does it say on my screen?",
            "transcribe the dialog word for word",
            "quote the text in that window",
        ],
    )
    def test_a_request_to_read_the_screen_is_recognised(self, message):
        assert asks_to_read_the_screen(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "what windows are behind yours?",
            "read me the last paragraph of that file",
            "how are you feeling?",
            "",
        ],
    )
    def test_other_requests_are_not(self, message):
        assert asks_to_read_the_screen(message) is False


class TestRecognisingTheClaim:
    def test_quoted_screen_text_is_a_claim(self):
        assert quotes_screen_content(CONFABULATED) is True

    def test_describing_the_layout_is_not(self):
        assert quotes_screen_content(HONEST_LAYOUT) is False

    def test_an_ordinary_quotation_is_not_a_screen_claim(self):
        assert (
            quotes_screen_content('He told me "the deploy went out at four" yesterday.')
            is False
        )

    def test_saying_she_cannot_read_it_is_not_a_claim(self):
        assert (
            quotes_screen_content(
                "I can't read what's on the screen right now, so I have nothing to quote."
            )
            is False
        )


class TestTheClaimNeedsEvidence:
    def test_the_live_confabulation_is_unsupported(self):
        assert screen_reading_claim_is_unsupported(READ_REQUEST, CONFABULATED, None) is True

    def test_a_capture_that_returned_nothing_supports_nothing(self):
        """The black frame: a capture happened and read zero characters."""
        evidence = ScreenReadingEvidence(captured=True, text="   ", source="screen")
        assert evidence.supports_a_quotation is False
        assert screen_reading_claim_is_unsupported(READ_REQUEST, CONFABULATED, evidence) is True

    def test_a_real_capture_licenses_the_quotation(self):
        evidence = ScreenReadingEvidence(
            captured=True,
            text="Show Closed Captions on supported websites",
            source="screen",
        )
        assert evidence.supports_a_quotation is True
        assert screen_reading_claim_is_unsupported(READ_REQUEST, CONFABULATED, evidence) is False

    def test_the_honest_reply_is_never_blocked(self):
        for evidence in (None, ScreenReadingEvidence(captured=False)):
            assert (
                screen_reading_claim_is_unsupported(READ_REQUEST, HONEST_LAYOUT, evidence)
                is False
            )

    def test_the_replacement_says_what_actually_happened(self):
        text = honest_unread_screen_reply(
            ScreenReadingEvidence(captured=False, unavailable_reason="display asleep")
        )
        assert "display asleep" in text
        assert "won't make one up" in text


class TestTheGateEnforcesIt:
    def test_the_reliability_gate_rejects_the_confabulation(self):
        from core.conversation.response_reliability import assess_user_facing_reply

        assessment = assess_user_facing_reply(READ_REQUEST, CONFABULATED)
        assert "unsupported_screen_reading_claim" in [
            str(reason) for reason in (assessment.reasons or ())
        ]

    def test_the_gate_passes_the_honest_layout_answer(self):
        from core.conversation.response_reliability import assess_user_facing_reply

        assessment = assess_user_facing_reply(READ_REQUEST, HONEST_LAYOUT)
        assert not [str(reason) for reason in (assessment.reasons or ())]

    def test_an_ordinary_turn_with_a_quotation_is_untouched(self):
        from core.conversation.response_reliability import assess_user_facing_reply

        assessment = assess_user_facing_reply(
            "what did the postmortem conclude?",
            'The postmortem says "the replica fell behind during the migration", '
            "which matches what the graphs show.",
        )
        assert "unsupported_screen_reading_claim" not in [
            str(reason) for reason in (assessment.reasons or ())
        ]
