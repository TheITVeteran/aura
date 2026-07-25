"""A repeated degraded status must name its own repetition.

The 2026-07-18 soak flagged ``identical_reply_repeated_x32``: Aura sent the
same status sentence 32 turns in a row. Every individual sentence was true,
but delivered verbatim 32 times it reads as a broken loop and tells the user
nothing new — the honest fact by turn three is *that this keeps happening*.

Contract: consecutive identical degraded statuses escalate into an honest
"this is the Nth turn in a row" clause; a real answer resets the streak; and
the underlying status text is never altered, only annotated.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_streak():
    from interface.routes import chat as chat_routes

    chat_routes._reset_lane_status_repeat_state()
    yield
    chat_routes._reset_lane_status_repeat_state()


_RECOVERING = {
    "state": "recovering",
    "conversation_ready": False,
    "last_failure_reason": "foreground_warmup_deferred_memory_pressure",
}


def _message(chat_routes, lane=None):
    return chat_routes._conversation_lane_user_message(lane or dict(_RECOVERING))


class TestRepeatEscalation:
    def test_first_occurrences_are_not_annotated(self):
        from interface.routes import chat as chat_routes

        first = _message(chat_routes)
        second = _message(chat_routes)
        assert "in a row" not in first
        assert "in a row" not in second

    def test_third_consecutive_repeat_names_the_streak(self):
        from interface.routes import chat as chat_routes

        for _ in range(2):
            _message(chat_routes)
        third = _message(chat_routes)
        assert "3th turn in a row" in third or "in a row" in third
        assert "worth looking at" in third

    def test_streak_count_keeps_climbing(self):
        from interface.routes import chat as chat_routes

        last = ""
        for _ in range(6):
            last = _message(chat_routes)
        assert "6" in last and "in a row" in last

    def test_the_underlying_status_text_is_preserved(self):
        from interface.routes import chat as chat_routes

        body = chat_routes._lane_status_message_body(dict(_RECOVERING))
        for _ in range(4):
            annotated = _message(chat_routes)
        assert annotated.startswith(body), "annotation must never rewrite the status"

    def test_a_real_answer_resets_the_streak(self):
        from interface.routes import chat as chat_routes

        for _ in range(4):
            _message(chat_routes)
        chat_routes._record_recent_response("Here is a real answer.", "a question")
        assert "in a row" not in _message(chat_routes)

    def test_a_different_status_resets_the_streak(self):
        """Only CONSECUTIVE identical statuses count — a changed situation is
        new information, not a repeat."""
        from interface.routes import chat as chat_routes

        for _ in range(4):
            _message(chat_routes)
        other = _message(chat_routes, {"state": "cold", "conversation_ready": False})
        assert "in a row" not in other
