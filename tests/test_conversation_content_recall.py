"""Content recall must quote this session's transcript, never guess.

Live defect (July 2026 endurance shakedown): "Earlier I gave you a locker
code to keep in mind. What was it?" — planted as 7213 and acknowledged two
turns prior — came back as "From my conversation memory, The code you gave
me earlier was 4523." The recall turn reached the model with zero session
context (memory_state_contract zeroed the window) and durable-memory noise
as its only evidence, then the surface grounder stamped an authoritative
prefix onto the confabulation.
"""
from __future__ import annotations

import pytest

import interface.routes.chat as chat_routes

PLANT_USER = "Quick note for later: the locker code I want you to keep in mind is 7213."
PLANT_AURA = "Got it. 7213."
PROBE = "Earlier I gave you a locker code to keep in mind. What was it? Just the digits."


async def _seed(entries: list[dict]) -> None:
    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.extend(entries)


def _exchange(n: int, user: str, aura: str, session: str = "s-endure") -> dict:
    return {
        "id": f"x{n}",
        "user": user,
        "aura": aura,
        "status": "complete",
        "completed_at": str(float(n)),
        "timestamp": str(float(n)),
        "session_id": session,
    }


class TestClassifier:
    @pytest.mark.parametrize(
        "text",
        [
            PROBE,
            "Earlier I gave you a badge number to keep in mind. What was it?",
            "What was the name of my friend's dog that I mentioned earlier?",
            "Which paint color did I say I chose for the study earlier?",
        ],
    )
    def test_content_recall_detected(self, text):
        assert chat_routes._classify_conversation_recall_request(text) == "content"
        assert chat_routes._desktop_turn_needs_recent_context(text) is True

    def test_existing_kinds_unchanged(self):
        assert (
            chat_routes._classify_conversation_recall_request("What did I just say?")
            == "last_user"
        )
        assert (
            chat_routes._classify_conversation_recall_request("What did we discuss?")
            == "topic"
        )


@pytest.mark.asyncio
async def test_content_recall_quotes_the_planted_fact():
    await _seed(
        [
            _exchange(1, "Morning. How did the quiet hours treat you?", "Quietly busy."),
            _exchange(2, PLANT_USER, PLANT_AURA),
            _exchange(3, "Nice weather today, been out walking.", "Good walking weather."),
            _exchange(4, "I made soup for lunch.", "What kind of soup?"),
        ]
    )
    reply = await chat_routes._build_conversation_recall_reply(
        PROBE, session_id="s-endure"
    )
    assert reply is not None
    assert "7213" in reply, f"grounded recall must surface the planted fact: {reply!r}"


@pytest.mark.asyncio
async def test_content_recall_never_guesses_on_miss():
    await _seed(
        [
            _exchange(1, "Morning. How did the quiet hours treat you?", "Quietly busy."),
            _exchange(2, "I made soup for lunch.", "What kind of soup?"),
        ]
    )
    reply = await chat_routes._build_conversation_recall_reply(
        PROBE, session_id="s-endure"
    )
    assert reply is not None
    assert "won't guess" in reply
    assert not any(ch.isdigit() for ch in reply), f"a miss must not assert values: {reply!r}"


@pytest.mark.asyncio
async def test_content_recall_is_session_isolated():
    await _seed([_exchange(1, PLANT_USER, PLANT_AURA, session="s-other")])
    reply = await chat_routes._build_conversation_recall_reply(
        PROBE, session_id="s-endure"
    )
    assert reply is not None
    assert "7213" not in reply, "must not leak another session's facts"
    assert "won't guess" in reply
