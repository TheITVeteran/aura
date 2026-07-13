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
    assert recorded["metadata"]["learning_admission"] == "verified"


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


@pytest.mark.asyncio
async def test_chat_turn_logger_forwards_exact_identity_to_profile_learning(monkeypatch):
    from core.memory import profile_manager

    calls = []
    tasks = []

    class _EpisodicMemory:
        def record_episode(self, **_kwargs):
            return "episode-profile"

    class _Tracker:
        def create_task(self, coro, **_kwargs):
            task = asyncio.create_task(coro)
            tasks.append(task)
            return task

    async def _learn_from_turn_auto(**kwargs):
        calls.append(kwargs)
        return (1, 0)

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: (
                _EpisodicMemory() if name == "episodic_memory" else default
            )
        ),
    )
    monkeypatch.setattr(chat_turn_logger, "get_task_tracker", lambda: _Tracker())
    monkeypatch.setattr(profile_manager, "learn_from_turn_auto", _learn_from_turn_auto)
    turn_logger = ChatTurnLogger()

    assert await turn_logger.log_chat_turn(
        "I prefer concise progress summaries.",
        "I will keep the next progress summary concise and evidence-based.",
        session_id="desktop-session",
        metadata={"origin": "desktop_ui", "user_id": "bryan"},
    )
    await asyncio.gather(*tasks)

    assert calls[0]["user_id"] == "bryan"
    assert calls[0]["session_id"] == "desktop-session"


@pytest.mark.asyncio
async def test_chat_turn_logger_never_schedules_unscoped_profile_learning(monkeypatch):
    scheduled_names = []

    class _EpisodicMemory:
        def record_episode(self, **_kwargs):
            return "episode-unscoped"

    class _Tracker:
        def create_task(self, coro, **kwargs):
            scheduled_names.append(kwargs.get("name"))
            coro.close()
            return None

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: (
                _EpisodicMemory() if name == "episodic_memory" else default
            )
        ),
    )
    monkeypatch.setattr(chat_turn_logger, "get_task_tracker", lambda: _Tracker())
    turn_logger = ChatTurnLogger()

    assert await turn_logger.log_chat_turn(
        "I prefer concise progress summaries.",
        "I will keep the next progress summary concise and evidence-based.",
        session_id="desktop-session",
        metadata={"origin": "desktop_ui"},
    )

    assert scheduled_names == []


@pytest.mark.asyncio
async def test_chat_turn_logger_learns_exact_profile_when_episodic_memory_is_unavailable(
    monkeypatch,
):
    from core.memory import profile_manager

    calls = []
    tasks = []

    class _Tracker:
        def create_task(self, coro, **_kwargs):
            task = asyncio.create_task(coro)
            tasks.append(task)
            return task

    async def _learn_from_turn_auto(**kwargs):
        calls.append(kwargs)
        return (1, 0)

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda _name, default=None: default),
    )
    monkeypatch.setattr(chat_turn_logger, "get_task_tracker", lambda: _Tracker())
    monkeypatch.setattr(profile_manager, "learn_from_turn_auto", _learn_from_turn_auto)
    turn_logger = ChatTurnLogger()

    assert await turn_logger.log_chat_turn(
        "I prefer concise progress summaries.",
        "I will keep this summary concise.",
        session_id="desktop-session",
        metadata={"origin": "desktop_ui", "user_id": "bryan"},
    ) is False
    await asyncio.gather(*tasks)

    assert calls[0]["user_id"] == "bryan"
    assert calls[0]["session_id"] == "desktop-session"


@pytest.mark.asyncio
async def test_chat_turn_logger_rejects_misgrounded_self_condition_before_learning(
    monkeypatch,
):
    class _EpisodicMemory:
        def record_episode(self, **_kwargs):
            raise AssertionError("misgrounded condition reply reached episodic memory")

    class _Tracker:
        def create_task(self, coro, **_kwargs):
            coro.close()
            raise AssertionError("misgrounded condition reply reached profile learning")

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: (
                _EpisodicMemory() if name == "episodic_memory" else default
            )
        ),
    )
    monkeypatch.setattr(chat_turn_logger, "get_task_tracker", lambda: _Tracker())

    turn_logger = ChatTurnLogger()
    stored = await turn_logger.log_chat_turn(
        "Are you okay though? Feeling fine?",
        (
            "I am with you. RAM pressure is 75.6% with 15.6 GB available; "
            "CPU load is 25.8% on this host."
        ),
        session_id="desktop-session",
        metadata={"origin": "desktop_ui", "user_id": "bryan"},
    )

    assert stored is False
