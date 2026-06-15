import asyncio

import pytest

from core.memory import chat_turn_logger
from core.memory.chat_turn_logger import ChatTurnLogger


class _ClosingTracker:
    def create_task(self, coro, **_kwargs):
        coro.close()
        return None


@pytest.mark.asyncio
async def test_chat_turn_logger_rebinds_episodic_memory_after_late_boot(monkeypatch):
    services = {}
    recorded = {}

    class _EpisodicMemory:
        def record_episode(self, **kwargs):
            recorded.update(kwargs)
            return "episode-late-boot"

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: services.get(name, default)),
    )
    monkeypatch.setattr(chat_turn_logger, "get_task_tracker", lambda: _ClosingTracker())

    turn_logger = ChatTurnLogger()
    await turn_logger._initialize()
    assert turn_logger._episodic_memory is None

    services["episodic_memory"] = _EpisodicMemory()
    result = await turn_logger.log_chat_turn(
        "Please remember this after boot completes.",
        "I will preserve it through the live conversation memory path.",
        session_id="desktop-session",
        metadata={"origin": "desktop_ui"},
    )

    assert result is True
    assert recorded["metadata"]["session_id"] == "desktop-session"
    assert recorded["metadata"]["conversation_lane"] is True


@pytest.mark.asyncio
async def test_chat_turn_logger_rebinds_memory_facade_after_late_boot(monkeypatch):
    services = {}
    recorded = {}

    class _MemoryFacade:
        async def add_memory(self, text, metadata=None):
            recorded["text"] = text
            recorded["metadata"] = dict(metadata or {})
            return True

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: services.get(name, default)),
    )

    turn_logger = ChatTurnLogger()
    await turn_logger._initialize()
    assert turn_logger._memory_facade is None

    services["memory_facade"] = _MemoryFacade()
    result = await turn_logger.log_user_message(
        "This fact arrived after memory initialization.",
        session_id="desktop-session",
        metadata={"origin": "desktop_ui"},
    )

    assert result is True
    assert recorded["text"].startswith("User said:")
    assert recorded["metadata"]["session_id"] == "desktop-session"
    assert recorded["metadata"]["conversation_lane"] is True
