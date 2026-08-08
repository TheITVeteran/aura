from __future__ import annotations

from contextlib import contextmanager

import pytest

pytestmark = pytest.mark.unit


@contextmanager
def _quality_scope(chat_routes, *, session: str, principal: str = "owner:bryan"):
    principal_token = chat_routes._CHAT_REQUEST_PRINCIPAL.set(principal)
    surface_token = chat_routes._CHAT_REQUEST_SURFACE.set("owner")
    session_token = chat_routes._CHAT_REQUEST_SESSION.set(session)
    try:
        yield
    finally:
        chat_routes._CHAT_REQUEST_SESSION.reset(session_token)
        chat_routes._CHAT_REQUEST_SURFACE.reset(surface_token)
        chat_routes._CHAT_REQUEST_PRINCIPAL.reset(principal_token)


@pytest.fixture(autouse=True)
def _clean_quality_state():
    from interface.routes import chat as chat_routes

    chat_routes._reset_conversation_quality_registry()
    chat_routes._conversation_log.clear()
    yield
    chat_routes._conversation_log.clear()
    chat_routes._reset_conversation_quality_registry()


def test_repetition_and_same_answer_state_stays_inside_one_conversation():
    from interface.routes import chat as chat_routes

    reply = "The answer belongs to this conversation only."
    with _quality_scope(chat_routes, session="conversation-a"):
        for _ in range(chat_routes._STALE_REPEAT_THRESHOLD):
            chat_routes._record_recent_response(reply, "Tell me about alpha.")
        assert chat_routes._is_stale_repeated_response(reply) is True
        assert chat_routes._is_same_answer_different_prompt(
            "Tell me about beta.", reply
        ) is True

    with _quality_scope(chat_routes, session="conversation-b"):
        assert chat_routes._is_stale_repeated_response(reply) is False
        assert chat_routes._is_same_answer_different_prompt(
            "Tell me about beta.", reply
        ) is False


def test_same_session_id_is_still_isolated_by_authenticated_principal():
    from interface.routes import chat as chat_routes

    reply = "A principal-bound answer."
    with _quality_scope(
        chat_routes, session="shared-label", principal="owner:bryan"
    ):
        for _ in range(chat_routes._STALE_REPEAT_THRESHOLD):
            chat_routes._record_recent_response(reply, "Owner prompt")
        assert chat_routes._is_stale_repeated_response(reply) is True

    with _quality_scope(
        chat_routes, session="shared-label", principal="paired:guest"
    ):
        assert chat_routes._is_stale_repeated_response(reply) is False


def test_lane_status_and_degradation_streaks_are_conversation_scoped():
    from interface.routes import chat as chat_routes

    status = "The local answer lane is recovering."
    with _quality_scope(chat_routes, session="conversation-a"):
        assert chat_routes._lane_status_repeat_suffix(status) == ""
        assert chat_routes._lane_status_repeat_suffix(status) == ""
        assert "3rd turn in a row" in chat_routes._lane_status_repeat_suffix(status)
        assert chat_routes._increment_conversation_degradation_streak() == 1
        assert chat_routes._increment_conversation_degradation_streak() == 2

    with _quality_scope(chat_routes, session="conversation-b"):
        assert chat_routes._lane_status_repeat_suffix(status) == ""
        assert chat_routes._conversation_degradation_streak() == 0

    with _quality_scope(chat_routes, session="conversation-a"):
        assert chat_routes._conversation_degradation_streak() == 2


@pytest.mark.asyncio
async def test_relevance_history_reads_only_the_active_conversation():
    from interface.routes import chat as chat_routes

    chat_routes._conversation_log.extend(
        [
            {
                "user": "Discuss volcanic glass.",
                "aura": "Obsidian is volcanic glass.",
                "session_id": "conversation-a",
            },
            {
                "user": "Discuss orbital mechanics.",
                "aura": "Orbits exchange kinetic and potential energy.",
                "session_id": "conversation-b",
            },
        ]
    )

    with _quality_scope(chat_routes, session="conversation-a"):
        recent = await chat_routes._gather_recent_user_messages_for_relevance(
            "Why does it fracture?"
        )

    assert recent == ["Discuss volcanic glass.", "Why does it fracture?"]


def test_quality_registry_is_bounded_and_retains_default_compatibility_bucket():
    from interface.routes import chat as chat_routes

    for index in range(chat_routes._CONVERSATION_QUALITY_STATE_LIMIT + 20):
        with _quality_scope(chat_routes, session=f"conversation-{index}"):
            chat_routes._record_recent_response(f"reply-{index}", f"prompt-{index}")

    assert (
        len(chat_routes._conversation_quality_states)
        <= chat_routes._CONVERSATION_QUALITY_STATE_LIMIT
    )
    assert (
        chat_routes._DEFAULT_CONVERSATION_QUALITY_KEY
        in chat_routes._conversation_quality_states
    )


def test_quality_identity_uses_the_same_bounded_session_key_as_the_log():
    from interface.routes import chat as chat_routes

    prefix = "s" * chat_routes._CHAT_SESSION_ID_MAX_CHARS
    first = chat_routes._conversation_quality_key(session_id=prefix + "-first")
    second = chat_routes._conversation_quality_key(session_id=prefix + "-second")

    assert first == second
