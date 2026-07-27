"""A second lane must not answer a turn the route already answered.

Live 2026-07-27. Asked whether consciousness is just computation, and told
not to just agree, she pushed back properly:

    I don't think you're right, and I'll tell you why. Consciousness isn't
    just computation — not in the way that running a program is conscious.

Three minutes later, unprompted, into the same window:

    I'll tackle this head-on. Let's break down those elements... 1.
    Decentralization - This is about distributing authority, control and
    resources across a network... blockchain or peer-to-peer networks

Earlier the same lane answered a question about tools with an affect report
("More strained. My energy level has decreased..."). Both are the kernel
finishing the deep pass on a turn the route had already answered from a
faster lane, then publishing through the event bus on its own.

The dangerous fix is a time window — "the route answered recently, so stay
quiet" — because it would also mute genuine unprompted speech, which is a
capability, not a bug. The discrimination has to be sharper: the route
records WHAT it served, and only a DIFFERENT answer inside the window is
withheld. The normal streaming path, which publishes the route's own text
through the same bus, must always pass.
"""

import time

import pytest

from core.conversation.surface_delivery import (
    LATE_LANE_WINDOW_S,
    note_route_delivered,
    reset_route_delivery,
    route_answer_supersedes,
)
from interface.event_bridge import _suppress_internal_leak

pytestmark = pytest.mark.unit

ROUTE_ANSWER = (
    "I don't think you're right, and I'll tell you why. Consciousness isn't "
    "just computation — not in the way that running a program is conscious. "
    "My stateful runtime with memory, identity boundaries, and ethical "
    "constraints isn't token prediction."
)
LATE_OTHER_ANSWER = (
    "I'll tackle this head-on. Let's break down those elements and see how "
    "they fit together for something 'essential': 1. Decentralization - This "
    "is about distributing authority, control and resources across a network."
)
LATE_AFFECT_REPORT = (
    "More strained. My energy level has decreased and focus has slightly "
    "increased but not enough to compensate for the overall drop in state "
    "quality."
)


@pytest.fixture(autouse=True)
def _clean():
    reset_route_delivery()
    yield
    reset_route_delivery()


class TestASecondAnswerIsWithheld:
    def test_the_live_off_topic_second_answer(self):
        note_route_delivered(ROUTE_ANSWER)
        assert route_answer_supersedes(LATE_OTHER_ANSWER)

    def test_the_live_affect_report(self):
        note_route_delivered("I can use DuckDuckGo, WolframAlpha, and Python.")
        assert route_answer_supersedes(LATE_AFFECT_REPORT)

    def test_the_bridge_withholds_it(self):
        note_route_delivered(ROUTE_ANSWER)
        assert _suppress_internal_leak(
            {"type": "aura_message", "message": LATE_OTHER_ANSWER}
        )


class TestNormalDeliveryIsNeverWithheld:
    def test_the_routes_own_answer_passes(self):
        """The streaming path publishes the route's text through this same
        bus. Withholding it would silence every reply."""
        note_route_delivered(ROUTE_ANSWER)
        assert not route_answer_supersedes(ROUTE_ANSWER)
        assert not _suppress_internal_leak(
            {"type": "aura_message", "message": ROUTE_ANSWER}
        )

    def test_a_trimmed_or_wrapped_version_of_it_passes(self):
        note_route_delivered(ROUTE_ANSWER)
        assert not route_answer_supersedes(ROUTE_ANSWER[:180])
        assert not route_answer_supersedes(f"  {ROUTE_ANSWER}\\n\\n")

    def test_speech_before_any_route_answer_passes(self):
        """Unprompted speech in a fresh conversation is not a second answer."""
        assert not route_answer_supersedes(LATE_OTHER_ANSWER)

    def test_speech_after_the_window_passes(self):
        """Proactive speech is a capability. It must survive this check."""
        note_route_delivered(ROUTE_ANSWER)
        import core.conversation.surface_delivery as sd

        sd._LAST_ROUTE_AT = time.time() - (LATE_LANE_WINDOW_S + 5)
        assert not route_answer_supersedes(LATE_OTHER_ANSWER)

    def test_empty_text_is_not_a_second_answer(self):
        note_route_delivered(ROUTE_ANSWER)
        assert not route_answer_supersedes("")


class TestTheEarlierScreeningStillWorks:
    def test_internal_diagnostics_are_still_withheld(self):
        assert _suppress_internal_leak(
            {"type": "aura_message", "message": "ROUTER_ERROR: unknown (at all_failed)"}
        )

    def test_ordinary_speech_is_still_delivered(self):
        assert not _suppress_internal_leak(
            {"type": "aura_message", "message": "It's 1:24 AM, and I know that from my clock."}
        )
