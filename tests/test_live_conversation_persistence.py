import pytest

from interface.routes import chat as chat_routes


class _PersistenceFixture:
    def __init__(self):
        self.session_id = "session-live"
        self.rows = []

    def record_turn(self, role, content, origin="", cid=None, session_id=None):
        self.rows.append(
            {
                "role": role,
                "content": content,
                "origin": origin,
                "cid": cid,
                "session_id": session_id or self.session_id,
                "created_at": float(len(self.rows) + 1),
            }
        )
        return f"turn-{len(self.rows)}"

    def get_recent_sessions(self, limit=10):
        return [{"id": self.session_id, "last_active": 1.0}][:limit]

    def get_session_history(self, session_id=None, limit=100):
        assert session_id == self.session_id
        return self.rows[-limit:]


@pytest.mark.asyncio
async def test_completed_live_exchange_survives_process_memory_clear(monkeypatch):
    persistence = _PersistenceFixture()
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence if name == "persistence" else default
        ),
    )

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchange_id = await chat_routes._begin_logged_exchange(
        "The continuity codeword is restart-echo-742."
    )
    await chat_routes._complete_logged_exchange(
        exchange_id,
        "The continuity codeword is restart-echo-742.",
        "I will retain restart-echo-742 across a process restart.",
        record_experience=False,
    )

    assert [row["role"] for row in persistence.rows] == ["user", "aura"]
    assert all(row["origin"] == "desktop_ui" for row in persistence.rows)

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchanges = await chat_routes._recent_completed_conversation_exchanges(
        current_user_message="What was the continuity codeword?",
        limit=6,
    )

    assert exchanges == [
        {
            "user": "The continuity codeword is restart-echo-742.",
            "aura": "I will retain restart-echo-742 across a process restart.",
            "timestamp": "2.0",
        }
    ]


@pytest.mark.asyncio
async def test_durable_recall_reply_survives_process_memory_clear(monkeypatch):
    persistence = _PersistenceFixture()
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence if name == "persistence" else default
        ),
    )

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    exchange_id = await chat_routes._begin_logged_exchange(
        "The live desktop failure involved the 32B lane losing CognitiveEngine continuity."
    )
    await chat_routes._complete_logged_exchange(
        exchange_id,
        "The live desktop failure involved the 32B lane losing CognitiveEngine continuity.",
        "I tracked that as a live desktop continuity problem, not a backend-only issue.",
        record_experience=False,
    )

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()

    user_recall = await chat_routes._build_conversation_recall_reply(
        "Can you remind me what I said earlier?"
    )
    aura_recall = await chat_routes._build_conversation_recall_reply(
        "Can you remind me what you answered?"
    )
    topic_recall = await chat_routes._build_conversation_recall_reply(
        "Can you remind me what we discussed?"
    )

    assert user_recall is not None
    assert "32B lane losing CognitiveEngine continuity" in user_recall
    assert aura_recall is not None
    assert "live desktop continuity problem" in aura_recall
    assert topic_recall is not None
    assert "32B lane" in topic_recall
    assert "live desktop continuity problem" in topic_recall


@pytest.mark.asyncio
async def test_recent_context_deduplicates_durable_and_in_memory_exchange(monkeypatch):
    persistence = _PersistenceFixture()
    persistence.record_turn("user", "Keep this one copy.", origin="desktop_ui")
    persistence.record_turn("aura", "One copy retained.", origin="desktop_ui")
    monkeypatch.setattr(
        chat_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: persistence if name == "persistence" else default
        ),
    )

    async with chat_routes._get_convo_lock():
        chat_routes._conversation_log.clear()
        chat_routes._conversation_log.append(
            {
                "id": "same-exchange",
                "user": "Keep this one copy.",
                "aura": "One copy retained.",
                "status": "complete",
                "completed_at": "now",
            }
        )

    exchanges = await chat_routes._recent_completed_conversation_exchanges(
        current_user_message="Continue.",
        limit=6,
    )

    assert len(exchanges) == 1
    assert exchanges[0]["user"] == "Keep this one copy."
