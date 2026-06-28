from __future__ import annotations

import asyncio
import sys
import types
from concurrent.futures import ThreadPoolExecutor

from core.conversation import persistence as persistence_module
from core.conversation.persistence import ConversationPersistence


def _install_event_bus(monkeypatch, bus):
    module = types.ModuleType("core.event_bus")
    module.get_event_bus = lambda: bus
    monkeypatch.setitem(sys.modules, "core.event_bus", module)


def test_conversation_persistence_records_turn_and_publishes_threadsafe(monkeypatch, tmp_path):
    published: list[tuple[str, dict[str, object]]] = []

    class Bus:
        def publish_threadsafe(self, topic, payload):
            published.append((topic, payload))

    _install_event_bus(monkeypatch, Bus())

    store = ConversationPersistence(tmp_path / "conversations.db")
    session_id = store.start_session({"non_json": object()})
    turn_id = store.record_turn(
        "user\x00",
        "hello from persistence",
        origin="text",
        cid="cid-123",
    )

    history = store.get_session_history(session_id, limit="10000")

    assert turn_id
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello from persistence"
    assert len(history) == 1
    assert published[0][0] == "turn_recorded"
    assert published[0][1]["session_id"] == session_id
    assert published[0][1]["turn_id"] == turn_id
    assert published[0][1]["content_chars"] == len("hello from persistence")


def test_conversation_persistence_records_exchange_atomically(monkeypatch, tmp_path):
    published: list[tuple[str, dict[str, object]]] = []

    class Bus:
        def publish_threadsafe(self, topic, payload):
            published.append((topic, payload))

    _install_event_bus(monkeypatch, Bus())

    store = ConversationPersistence(tmp_path / "conversations.db")
    session_id = store.start_session()
    turn_ids = store.record_exchange(
        "Remember the live desktop path.",
        "I will carry it across restart.",
        origin="desktop_ui",
        cid="exchange-42",
    )

    history = store.get_session_history(session_id)

    assert len(turn_ids) == 2
    assert [row["role"] for row in history] == ["user", "aura"]
    assert [row["cid"] for row in history] == [
        "exchange-42:user",
        "exchange-42:aura",
    ]
    assert len(published) == 2


def test_conversation_persistence_deduplicates_turn_by_cid(monkeypatch, tmp_path):
    published: list[tuple[str, dict[str, object]]] = []

    class Bus:
        def publish_threadsafe(self, topic, payload):
            published.append((topic, payload))

    _install_event_bus(monkeypatch, Bus())

    store = ConversationPersistence(tmp_path / "conversations.db")
    session_id = store.start_session()
    first_id = store.record_turn("user", "same live prompt", origin="desktop_ui", cid="live-1:user")
    second_id = store.record_turn("user", "same live prompt", origin="desktop_ui", cid="live-1:user")

    history = store.get_session_history(session_id)

    assert second_id == first_id
    assert len(history) == 1
    assert history[0]["cid"] == "live-1:user"
    assert len(published) == 1


def test_conversation_persistence_exchange_reuses_prelogged_user_by_cid(monkeypatch, tmp_path):
    published: list[tuple[str, dict[str, object]]] = []

    class Bus:
        def publish_threadsafe(self, topic, payload):
            published.append((topic, payload))

    _install_event_bus(monkeypatch, Bus())

    store = ConversationPersistence(tmp_path / "conversations.db")
    session_id = store.start_session()
    prelogged_user_id = store.record_turn(
        "user",
        "foreground prompt",
        origin="desktop_ui",
        cid="race-42:user",
    )
    user_id, aura_id = store.record_exchange(
        "foreground prompt",
        "foreground answer",
        origin="desktop_ui",
        cid="race-42",
    )

    history = store.get_session_history(session_id)

    assert user_id == prelogged_user_id
    assert aura_id
    assert [row["role"] for row in history] == ["user", "aura"]
    assert [row["cid"] for row in history] == ["race-42:user", "race-42:aura"]
    assert [event[1]["role"] for event in published] == ["user", "aura"]


def test_conversation_persistence_bounded_history_returns_newest_turns(tmp_path):
    store = ConversationPersistence(tmp_path / "conversations.db")
    session_id = store.start_session()
    for index in range(8):
        store.record_turn("user", f"turn-{index}")

    history = store.get_session_history(session_id, limit=3)

    assert [row["content"] for row in history] == ["turn-5", "turn-6", "turn-7"]


def test_conversation_persistence_serializes_concurrent_writers(monkeypatch, tmp_path):
    class Bus:
        def publish_threadsafe(self, _topic, _payload):
            return None

    _install_event_bus(monkeypatch, Bus())
    store = ConversationPersistence(tmp_path / "concurrent-conversations.db")
    session_id = store.start_session()

    def write_turn(index: int) -> str:
        return store.record_turn(
            "user",
            f"concurrent-turn-{index}",
            origin="desktop_ui",
            cid=f"concurrent-{index}",
            session_id=session_id,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        turn_ids = list(pool.map(write_turn, range(40)))

    history = store.get_session_history(session_id, limit=100)

    assert len(set(turn_ids)) == 40
    assert len(history) == 40
    assert {row["cid"] for row in history} == {
        f"concurrent-{index}" for index in range(40)
    }


def test_conversation_persistence_async_publish_is_scheduled(monkeypatch, tmp_path):
    published: list[tuple[str, dict[str, object]]] = []
    scheduled: list[str] = []

    class Bus:
        async def publish(self, topic, payload):
            await asyncio.sleep(0)
            published.append((topic, payload))

    class Tracker:
        def create_task(self, coro, name=None):
            scheduled.append(name or "")
            return asyncio.create_task(coro)

    _install_event_bus(monkeypatch, Bus())
    monkeypatch.setattr(persistence_module, "get_task_tracker", lambda: Tracker())

    async def scenario():
        store = ConversationPersistence(tmp_path / "async-conversations.db")
        turn_id = store.record_turn("aura", "scheduled event", cid="cid-async")
        await asyncio.sleep(0.01)
        return turn_id

    turn_id = asyncio.run(scenario())

    assert turn_id
    assert scheduled == ["conversation.turn_recorded.publish"]
    assert published[0][0] == "turn_recorded"
    assert published[0][1]["cid"] == "cid-async"


def test_conversation_persistence_scheduler_failure_records_receipt(monkeypatch, tmp_path):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    class TaskSpec:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Scheduler:
        async def register(self, _spec):
            self.attempted = True
            raise RuntimeError("scheduler unavailable")

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    scheduler_module = types.ModuleType("core.scheduler")
    scheduler_module.TaskSpec = TaskSpec
    scheduler_module.scheduler = Scheduler()
    monkeypatch.setitem(sys.modules, "core.scheduler", scheduler_module)
    monkeypatch.setattr(persistence_module, "record_degradation", record_degradation)

    store = ConversationPersistence(tmp_path / "scheduler-conversations.db")
    asyncio.run(store.on_start_async())

    assert store.get_retention_status()["last_persist_error_at"] > 0
    assert recorded
    assert recorded[0][0] == "persistence"
    assert recorded[0][1] == "RuntimeError"
    assert recorded[0][2]["receipt_required"] is True
    assert "scheduled conversation pruning" in str(recorded[0][2]["action"])
