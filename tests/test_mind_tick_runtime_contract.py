import asyncio
import sqlite3
import time

import pytest

import core.mind_tick as mind_module
from core.mind_tick import MindTick, _schedule_mind_task
from core.runtime.errors import get_degradation_tracker
from core.state.state_repository import StateRepository


class ClosingAwaitable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def __await__(self):
        if False:
            yield None
        return None


class FailingTracker:
    def create_task(self, _awaitable, *, name=None):
        self.last_name = name
        raise RuntimeError(f"{name}: loop unavailable")


class Watchdog:
    def register_component(self, *_args, **_kwargs):
        return None


def test_mind_scheduler_closes_unscheduled_awaitable():
    awaitable = ClosingAwaitable()

    task = _schedule_mind_task(awaitable, name="mind.contract", tracker=FailingTracker())

    assert task is None
    assert awaitable.closed is True


def test_mind_tick_liveness_requires_supervised_progress():
    class RunningTask:
        @staticmethod
        def done():
            return False

    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._task = RunningTask()
    tick._started_at = time.time()
    tick._last_successful_tick_at = time.time()
    tick._consecutive_loop_failures = 0
    tick._tick_count = 4

    assert tick.is_alive() is True
    assert tick.get_health_status()["healthy"] is True

    tick._consecutive_loop_failures = 3
    assert tick.is_alive() is False


@pytest.mark.asyncio
async def test_mind_tick_liveness_probe_repairs_dead_supervised_loop():
    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._started_at = time.time()
    tick._last_successful_tick_at = 0.0
    tick._consecutive_loop_failures = 3
    tick._last_liveness_repair_at = 0.0
    tick._liveness_repair_count = 0

    async def failed_loop():
        raise RuntimeError("background loop died")

    tick._task = asyncio.create_task(failed_loop())
    await asyncio.sleep(0)

    release = asyncio.Event()

    async def recovered_loop():
        await release.wait()

    tick._run_loop = recovered_loop

    assert tick.is_alive() is True
    assert tick._task is not None
    assert not tick._task.done()
    assert tick.get_health_status()["liveness_repair_count"] == 1

    release.set()
    await tick._task


@pytest.mark.asyncio
async def test_mind_tick_start_rolls_back_when_loop_cannot_be_scheduled(monkeypatch):
    monkeypatch.setattr(mind_module, "get_task_tracker", lambda: FailingTracker())
    monkeypatch.setattr("infrastructure.watchdog.get_watchdog", lambda: Watchdog())

    tick = MindTick.__new__(MindTick)
    tick._running = False
    tick._task = None

    async def run_loop():
        return None

    tick._run_loop = run_loop

    await tick.start()

    assert tick._running is False
    assert tick._task is None


@pytest.mark.asyncio
async def test_mind_tick_stop_drains_closed_db_task_failure():
    tracker = get_degradation_tracker()
    tracker.reset()
    tick = MindTick.__new__(MindTick)
    tick._running = True
    drain_failures: list[str] = []

    async def failed_loop():
        drain_failures.append("closed_database")
        raise sqlite3.ProgrammingError("Cannot operate on a closed database.")

    tick._task = asyncio.create_task(failed_loop())
    await asyncio.sleep(0)

    await tick.stop()

    assert tick._running is False
    assert drain_failures == ["closed_database"]
    assert any(
        "background loop failed while draining" in record.action
        for record in tracker.recent(subsystem="mind_tick")
    )
    tracker.reset()


@pytest.mark.asyncio
async def test_state_repository_clear_pending_proxy_commit_handles_closed_db(monkeypatch):
    tracker = get_degradation_tracker()
    tracker.reset()
    repo = StateRepository(is_vault_owner=False)

    class ClosedDB:
        async def execute(self, *_args, **_kwargs):
            self.execute_calls = getattr(self, "execute_calls", 0) + 1
            raise sqlite3.ProgrammingError("Cannot operate on a closed database.")

        async def commit(self):
            self.commit_calls = getattr(self, "commit_calls", 0) + 1
            raise AssertionError("commit should not run after closed execute")

    async def closed_db():
        return ClosedDB()

    monkeypatch.setattr(repo, "_ensure_db", closed_db)

    await repo._clear_pending_proxy_commit()

    assert any(
        "outbox clear failed" in record.action
        for record in tracker.recent(subsystem="state_repository")
    )
    tracker.reset()
