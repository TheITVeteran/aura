"""Everything that lands in the chat window passes the same gate.

Two lanes can put words in front of a person: the HTTP chat route, and the
kernel publishing through the event bridge. The route has accumulated the
repairs — internal-text suppression, mid-clause completion — and the bridge
had almost none, so which lane answered decided what quality the person got.

Live 2026-07-27, the bridge lane served this, verbatim:

    Fourth, I'd worry about the deploy itself failing for any reason and
    leaving us in an unknown state with no clear way back. And

A correct, ordered, genuinely useful four-part risk analysis ending on the
word "And". The route's repair handles that text perfectly — it was simply
never asked, because this reply did not come through the route.

These tests assert the property that makes the lane irrelevant: text reaching
the window is completed and screened at the one seam both lanes pass.
"""

import pytest

from core.conversation.response_reliability import complete_truncated_tail
from interface.event_bridge import _complete_spoken_tail, _suppress_internal_leak

pytestmark = pytest.mark.unit

CUT_OFF = (
    "First, I'd be worried about the time pressure. Fifteen minutes is "
    "incredibly tight for any meaningful change in production. Second, an "
    "untested rollback script means we have no idea if it actually works. "
    "Fourth, I'd worry about the deploy itself failing for any reason and "
    "leaving us in an unknown state with no clear way back. And"
)


class TestTailCompletion:
    def test_the_live_reply_is_completed(self):
        repaired = complete_truncated_tail(CUT_OFF)
        assert repaired.endswith("no clear way back.")
        assert not repaired.endswith("And")

    @pytest.mark.parametrize(
        "text",
        [
            "The answer is 19/66.",
            "I think the plan is sound, but we should test the rollback first.",
            "It's 1:24 AM.",
        ],
    )
    def test_a_finished_reply_is_left_alone(self, text: str):
        assert complete_truncated_tail(text) == text

    def test_trimming_never_empties_a_reply(self):
        """Better a slightly rough ending than nothing at all."""
        assert complete_truncated_tail("and") == "and"
        assert complete_truncated_tail("") == ""


class TestTheBridgeAppliesIt:
    def test_a_spoken_message_is_completed_in_place(self):
        msg = {"type": "aura_message", "message": CUT_OFF}
        _complete_spoken_tail(msg)
        assert msg["message"].endswith("no clear way back.")

    def test_a_chat_response_is_completed_in_place(self):
        msg = {"type": "chat_response", "content": CUT_OFF}
        _complete_spoken_tail(msg)
        assert msg["content"].endswith("no clear way back.")

    def test_non_spoken_traffic_is_untouched(self):
        """Telemetry and thought cards are not speech and are not rewritten."""
        msg = {"type": "telemetry", "message": CUT_OFF}
        _complete_spoken_tail(msg)
        assert msg["message"] == CUT_OFF

    def test_completion_runs_before_suppression(self):
        """Order matters: a reply must not be judged on a tail that the very
        next step would have fixed."""
        import inspect

        from interface import event_bridge

        source = inspect.getsource(event_bridge)
        complete_at = source.find("_complete_spoken_tail(ws_msg)")
        suppress_at = source.find("_suppress_internal_leak(ws_msg)", complete_at)
        assert complete_at != -1 and suppress_at != -1
        assert complete_at < suppress_at


class TestTheBridgeStillScreensInternalText:
    def test_a_diagnostic_label_is_still_withheld(self):
        """The defect this file adds to must not weaken the one already here."""
        assert _suppress_internal_leak(
            {"type": "aura_message", "message": "ROUTER_ERROR: unknown (at all_failed)"}
        )

    def test_ordinary_speech_is_not_withheld(self):
        assert not _suppress_internal_leak(
            {"type": "aura_message", "message": "It's 1:24 AM, and I know that from my clock."}
        )
