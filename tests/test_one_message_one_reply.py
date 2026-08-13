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

from core.conversation.session_scope import (
    conversation_session_scope,
    conversation_turn_var,
)
from core.conversation.surface_delivery import (
    LATE_LANE_WINDOW_S,
    note_route_delivered,
    note_turn_started,
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
CONVERSATION_ID = "conversation-a"
TURN_ID = "turn-a"


def _note(reply: str, *, conversation_id: str = CONVERSATION_ID, turn_id: str = TURN_ID):
    note_route_delivered(
        reply,
        conversation_id=conversation_id,
        turn_id=turn_id,
    )


def _supersedes(
    reply: str,
    *,
    conversation_id: str = CONVERSATION_ID,
    turn_id: str = TURN_ID,
) -> bool:
    return route_answer_supersedes(
        reply,
        conversation_id=conversation_id,
        turn_id=turn_id,
    )


def _spoken(reply: str, *, conversation_id: str = CONVERSATION_ID, turn_id: str = TURN_ID):
    return {
        "type": "aura_message",
        "message": reply,
        "metadata": {
            "conversation_id": conversation_id,
            "conversation_turn_id": turn_id,
        },
    }


@pytest.fixture(autouse=True)
def _clean():
    reset_route_delivery()
    yield
    reset_route_delivery()


class TestASecondAnswerIsWithheld:
    def test_the_live_off_topic_second_answer(self):
        _note(ROUTE_ANSWER)
        assert _supersedes(LATE_OTHER_ANSWER)

    def test_the_live_affect_report(self):
        _note("I can use DuckDuckGo, WolframAlpha, and Python.")
        assert _supersedes(LATE_AFFECT_REPORT)

    def test_the_bridge_withholds_it(self):
        _note(ROUTE_ANSWER)
        assert _suppress_internal_leak(_spoken(LATE_OTHER_ANSWER))

    def test_an_open_turn_withholds_unprompted_speech_for_that_turn(self):
        note_turn_started(conversation_id=CONVERSATION_ID, turn_id=TURN_ID)
        assert _suppress_internal_leak(_spoken(LATE_OTHER_ANSWER))

    def test_unscoped_autonomous_speech_waits_while_any_turn_is_open(self):
        note_turn_started(conversation_id=CONVERSATION_ID, turn_id=TURN_ID)
        assert route_answer_supersedes(LATE_OTHER_ANSWER)

    def test_another_sessions_turn_does_not_overwrite_or_suppress_this_one(self):
        _note(ROUTE_ANSWER)
        _note(
            "A separate session's answer.",
            conversation_id="conversation-b",
            turn_id="turn-b",
        )
        assert _suppress_internal_leak(_spoken(LATE_OTHER_ANSWER))
        assert not _suppress_internal_leak(
            _spoken(
                LATE_OTHER_ANSWER,
                conversation_id="conversation-c",
                turn_id="turn-c",
            )
        )


class TestNormalDeliveryIsNeverWithheld:
    def test_the_routes_own_answer_passes(self):
        """The streaming path publishes the route's text through this same
        bus. Withholding it would silence every reply."""
        _note(ROUTE_ANSWER)
        assert not _supersedes(ROUTE_ANSWER)
        assert not _suppress_internal_leak(_spoken(ROUTE_ANSWER))

    def test_a_trimmed_or_wrapped_version_of_it_passes(self):
        _note(ROUTE_ANSWER)
        assert not _supersedes(ROUTE_ANSWER[:180])
        assert not _supersedes(f"  {ROUTE_ANSWER}\\n\\n")

    def test_speech_before_any_route_answer_passes(self):
        """Unprompted speech in a fresh conversation is not a second answer."""
        assert not route_answer_supersedes(LATE_OTHER_ANSWER)

    def test_unscoped_proactive_speech_is_not_bound_to_another_settled_session(self):
        _note(ROUTE_ANSWER)
        assert not route_answer_supersedes(LATE_OTHER_ANSWER)

    def test_speech_after_the_window_passes(self):
        """Proactive speech is a capability. It must survive this check."""
        _note(ROUTE_ANSWER)
        import core.conversation.surface_delivery as sd

        sd._STATES[(CONVERSATION_ID, TURN_ID)].last_route_at = time.time() - (
            LATE_LANE_WINDOW_S + 5
        )
        assert not _supersedes(LATE_OTHER_ANSWER)

    def test_empty_text_is_not_a_second_answer(self):
        _note(ROUTE_ANSWER)
        assert not _supersedes("")

    def test_delivery_state_refuses_anonymous_writes(self):
        with pytest.raises(ValueError, match="exact conversation and turn"):
            note_route_delivered(ROUTE_ANSWER, conversation_id="", turn_id="")


class TestTheEarlierScreeningStillWorks:
    def test_internal_diagnostics_are_still_withheld(self):
        assert _suppress_internal_leak(
            {"type": "aura_message", "message": "ROUTER_ERROR: unknown (at all_failed)"}
        )

    def test_ordinary_speech_is_still_delivered(self):
        assert not _suppress_internal_leak(
            {"type": "aura_message", "message": "It's 1:24 AM, and I know that from my clock."}
        )


@pytest.mark.asyncio
async def test_output_gate_preserves_conversation_turn_lineage_on_the_bus(
    monkeypatch: pytest.MonkeyPatch,
    service_container,
):
    from core import event_bus as event_bus_module
    from core.utils.output_gate import AutonomousOutputGate

    published: list[tuple[str, object]] = []

    class _Bus:
        def publish_threadsafe(self, topic, payload):
            published.append((topic, payload))

    monkeypatch.setattr(event_bus_module, "get_event_bus", lambda: _Bus())
    gate = AutonomousOutputGate()

    async def _no_receipt(*_args, **_kwargs):
        return None

    monkeypatch.setattr(gate, "_emit_output_receipt", _no_receipt)

    with conversation_session_scope(CONVERSATION_ID):
        token = conversation_turn_var.set(TURN_ID)
        try:
            await gate._send_to_primary(
                "A complete answer.",
                origin="user",
                metadata={"voice": False},
            )
        finally:
            conversation_turn_var.reset(token)

    aura_payloads = [
        payload for topic, payload in published if topic == "aura_message"
    ]
    assert len(aura_payloads) == 1
    assert aura_payloads[0]["metadata"]["conversation_id"] == CONVERSATION_ID
    assert aura_payloads[0]["metadata"]["conversation_turn_id"] == TURN_ID
