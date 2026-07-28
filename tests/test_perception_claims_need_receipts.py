"""Perception and action are different claims with different evidence.

I got this wrong in a way worth writing down, because the failure mode is the
one this whole file exists to prevent.

Asked live to look at the screen, Aura named the applications that were open.
The log showed zero computer_use dispatches, so I concluded she had
fabricated it, and shipped a guard requiring a tool receipt for any claim
about a screen. Bryan — who can see his own screen — pointed out that she was
right. The apps were there.

She has a continuous vision feed that captures the screen every couple of
seconds. It is a SENSE, not a dispatch, and it files no per-turn receipt. So
a receipt-only check destroyed an accurate observation for arriving through
the wrong subsystem: the exact class of defect this session has been fixing,
committed by the person fixing it.

The corrected model:

    "I can see Chrome on your screen"   perception — a fresh frame backs it
    "I opened Chrome"                   action     — a receipt backs it

Nothing observes an action into existence, so a frame never excuses one. And
a fabricated screen with no frames behind it is still caught.
"""

import pytest

import core.senses.continuous_vision as continuous_vision
from core.conversation.response_reliability import (
    _has_unfounded_tool_execution_claim,
)

pytestmark = pytest.mark.unit

SEEING = "I can see Chrome, VS Code and Notepad++ on your screen."
ACTING = "I opened Chrome for you."
RECEIPT = ({"tool": "computer_use", "ok": True},)


@pytest.fixture
def no_frames(monkeypatch):
    monkeypatch.setattr(continuous_vision, "_LAST_SCREEN_FRAME_AT", 0.0)
    yield


@pytest.fixture
def fresh_frame():
    continuous_vision._note_screen_frame()
    yield


class TestPerceptionIsBackedByAFrame:
    def test_a_real_observation_survives(self, fresh_frame):
        """The regression this file was rewritten for."""
        assert not _has_unfounded_tool_execution_claim(SEEING, tool_receipts=())

    def test_a_screen_claim_with_no_frames_is_still_caught(self, no_frames):
        assert _has_unfounded_tool_execution_claim(SEEING, tool_receipts=())


class TestActionIsBackedByAReceipt:
    def test_a_frame_does_not_excuse_an_action(self, fresh_frame):
        """Seeing the screen is not the same as having clicked on it."""
        assert _has_unfounded_tool_execution_claim(ACTING, tool_receipts=())

    def test_a_receipt_permits_the_action_claim(self, no_frames):
        assert not _has_unfounded_tool_execution_claim(ACTING, tool_receipts=RECEIPT)

    def test_an_unbacked_action_is_caught(self, no_frames):
        assert _has_unfounded_tool_execution_claim(ACTING, tool_receipts=())


class TestTheOriginalGuardStillHolds:
    def test_fabricated_code_output_is_still_caught(self, no_frames):
        assert _has_unfounded_tool_execution_claim(
            "Python code: 2 + 2 Output: 4", tool_receipts=()
        )

    @pytest.mark.parametrize(
        "reply",
        [
            "I see what you mean about the tradeoff.",
            "I see the problem in your reasoning.",
            "Want me to take a screenshot?",
            "I cannot see your screen, so I would not guess.",
            "The result is 19/66.",
        ],
    )
    def test_ordinary_speech_is_untouched(self, reply, no_frames):
        assert not _has_unfounded_tool_execution_claim(reply, tool_receipts=())
