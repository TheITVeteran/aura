from __future__ import annotations

import asyncio
import json

from core.memory import conversation_persistence as memory_persistence
from core.memory.conversation_persistence import ConversationMessage, ConversationPersistence


def test_memory_conversation_persistence_records_blank_response_turn(tmp_path):
    store = ConversationPersistence(tmp_path)

    asyncio.run(store._middleware_hook("please remember this", "", origin="text"))

    assert len(store._current_session_messages) == 1
    assert store._current_session_messages[0].role == "user"
    assert store._current_session_messages[0].content == "please remember this"
    assert store.save_sync() is True
    assert (tmp_path / f"session_{store.session_id}.json").exists()


def test_memory_conversation_persistence_quarantines_corrupt_session(tmp_path):
    good_payload = {
        "session_id": "valid-session",
        "started_at": 1.0,
        "message_count": 1,
        "messages": [
            {
                "role": "user",
                "content": "restore this context",
                "timestamp": 1.0,
                "session_id": "valid-session",
                "origin": "chat",
            }
        ],
    }
    (tmp_path / "session_valid-session.json").write_text(json.dumps(good_payload), encoding="utf-8")
    (tmp_path / "session_broken.json").write_text("{bad json", encoding="utf-8")

    store = ConversationPersistence(tmp_path)
    recent = store.load_recent(max_messages="5000")

    assert recent == good_payload["messages"]
    assert not (tmp_path / "session_broken.json").exists()
    assert list(tmp_path.glob("corrupt_*_session_broken.json"))


def test_memory_conversation_persistence_summary_failure_uses_deterministic_receipt(monkeypatch, tmp_path):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    class Engine:
        async def think(self, **_kwargs):
            attempted_objective = _kwargs.get("objective", "")
            assert attempted_objective
            raise RuntimeError("cognitive engine offline")

    class Orchestrator:
        cognitive_engine = Engine()

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    monkeypatch.setattr(memory_persistence, "record_degradation", record_degradation)

    store = ConversationPersistence(tmp_path)
    store._orchestrator = Orchestrator()
    for index in range(3):
        store._record(ConversationMessage(role="user", content=f"user topic {index}", session_id=store.session_id))
        store._record(ConversationMessage(role="assistant", content=f"assistant note {index}", session_id=store.session_id))

    summary = asyncio.run(store._generate_summary())

    assert summary is not None
    assert "User topics" in summary
    assert "assistant note" in summary
    assert recorded[0][0] == "conversation_persistence"
    assert recorded[0][1] == "RuntimeError"
    assert recorded[0][2]["receipt_required"] is True
    assert "deterministic session summary" in str(recorded[0][2]["action"])


def test_memory_conversation_persistence_save_failure_preserves_memory(monkeypatch, tmp_path):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    def failing_atomic_write(*_args, **_kwargs):
        assert _args or _kwargs
        raise OSError("disk full")

    monkeypatch.setattr(memory_persistence, "record_degradation", record_degradation)
    monkeypatch.setattr(memory_persistence, "atomic_write_text", failing_atomic_write)

    store = ConversationPersistence(tmp_path)
    store._record(ConversationMessage(role="user", content="must not disappear", session_id=store.session_id))

    assert store.save_sync() is False
    assert store._last_save_ok is False
    assert store._current_session_messages[0].content == "must not disappear"
    assert recorded[0][0] == "conversation_persistence"
    assert recorded[0][1] == "OSError"
    assert "kept conversation in memory" in str(recorded[0][2]["action"])


def test_memory_conversation_persistence_uses_filename_session_id_for_rotation(tmp_path):
    payload = {
        "session_id": "unsafe/../../outside",
        "started_at": 1.0,
        "message_count": 0,
        "messages": [],
    }
    (tmp_path / "session_safe-id.json").write_text(json.dumps(payload), encoding="utf-8")

    store = ConversationPersistence(tmp_path)
    sessions = store._list_sessions()

    assert sessions[0]["session_id"] == "safe-id"
    assert store._session_path(sessions[0]["session_id"]).parent == tmp_path
