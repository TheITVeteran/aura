import asyncio
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

import core.mind_tick as mind_module
from core.container import ServiceContainer
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
    assert tick.is_alive() is True

    tick._last_successful_tick_at = time.time() - 601
    tick._last_loop_progress_at = time.time() - 601
    tick._last_liveness_repair_at = time.monotonic()
    assert tick.is_alive() is False


def test_mind_tick_liveness_allows_active_bounded_tick_progress():
    class RunningTask:
        @staticmethod
        def done():
            return False

    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._task = RunningTask()
    tick._started_at = time.time() - 700
    tick._active_tick_started_at = time.time() - 700
    tick._active_tick_stage = "kernel_tick"
    tick._last_progress_label = "kernel_tick"
    tick._last_successful_tick_at = time.time() - 700
    tick._last_loop_progress_at = time.time() - 700
    tick._consecutive_loop_failures = 0
    tick._tick_count = 5
    tick._last_liveness_repair_at = 0.0

    assert tick.is_alive() is True
    status = tick.get_health_status()
    assert status["healthy"] is True
    assert status["active_tick_stage"] == "kernel_tick"
    assert status["last_progress_label"] == "kernel_tick"


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
        await asyncio.sleep(0)  # yield once like a real loop, then die
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
async def test_mind_tick_done_callback_repairs_failed_loop_without_health_poll():
    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._started_at = time.time()
    tick._last_successful_tick_at = 0.0
    tick._consecutive_loop_failures = 0
    tick._last_liveness_repair_at = 0.0
    tick._liveness_repair_count = 0
    tick._owner_loop = asyncio.get_running_loop()
    tick.orchestrator = SimpleNamespace()

    release = asyncio.Event()

    async def failed_loop():
        await asyncio.sleep(0)
        raise TypeError("expected string or bytes-like object, got 'NoneType'")

    async def recovered_loop():
        await release.wait()

    tick._run_loop = recovered_loop
    failed_task = asyncio.create_task(failed_loop())
    tick._task = failed_task
    tick._install_loop_done_callback(failed_task, name="test.failed")

    deadline = time.monotonic() + 1.0
    while tick._task is failed_task and time.monotonic() < deadline:
        await asyncio.sleep(0.01)

    assert tick._task is not failed_task
    assert tick._task is not None
    assert not tick._task.done()
    assert tick._liveness_repair_count == 1

    release.set()
    await tick._task


@pytest.mark.asyncio
async def test_mind_tick_liveness_repairs_stale_alive_task():
    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._started_at = time.time() - 600
    tick._last_successful_tick_at = time.time() - 601
    tick._last_loop_progress_at = time.time() - 601
    tick._consecutive_loop_failures = 0
    tick._last_liveness_repair_at = 0.0
    tick._liveness_repair_count = 0
    tick._owner_loop = asyncio.get_running_loop()
    tick.orchestrator = SimpleNamespace()

    release_old = asyncio.Event()
    release_new = asyncio.Event()

    async def stale_loop():
        await release_old.wait()

    async def recovered_loop():
        await release_new.wait()

    stale_task = asyncio.create_task(stale_loop())
    tick._task = stale_task
    tick._run_loop = recovered_loop

    assert tick.is_alive() is False
    assert stale_task.cancelled() or stale_task.done() or stale_task.cancelling()

    # CP126 e98446be. The replacement is CHAINED to the stale loop actually
    # unwinding, not started the instant cancel() returns. `cancel()` only
    # requests cancellation — the old coroutine runs on until its next await
    # — so starting the replacement immediately meant two _run_loop
    # coroutines alive at once, both mutating the same state and committing
    # over each other. A repair that produces two minds is worse than the
    # stall it repairs.
    #
    # While the old loop unwinds the repair is IN FLIGHT and says so, which
    # is also the fix for CP126 c76abf56: "recovering" and "broken and
    # unattended" are different operational states.
    assert tick._repair_pending is True
    assert tick._task is stale_task, (
        "a replacement loop started before the stale one had unwound"
    )

    release_old.set()
    for _ in range(50):
        await asyncio.sleep(0)
        if tick._task is not stale_task:
            break

    assert tick._task is not stale_task
    assert tick._task is not None
    assert not tick._task.done()
    assert tick._repair_pending is False
    assert tick._last_loop_progress_at > time.time() - 5
    assert tick._liveness_repair_count == 1

    release_new.set()
    await tick._task


@pytest.mark.asyncio
async def test_mind_tick_never_runs_two_cognitive_loops_at_once():
    """The concurrency the repair used to create.

    Both loops mutate one state object and commit over each other, so the
    invariant is not "the replacement starts quickly" — it is that exactly
    one _run_loop exists at every instant.
    """
    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._started_at = time.time() - 600
    tick._last_successful_tick_at = time.time() - 601
    tick._last_loop_progress_at = time.time() - 601
    tick._consecutive_loop_failures = 0
    tick._last_liveness_repair_at = 0.0
    tick._liveness_repair_count = 0
    tick._repair_pending = False
    tick._owner_loop = asyncio.get_running_loop()
    tick.orchestrator = SimpleNamespace()

    live = {"count": 0, "peak": 0}
    release_old = asyncio.Event()
    release_new = asyncio.Event()

    async def stale_loop():
        live["count"] += 1
        live["peak"] = max(live["peak"], live["count"])
        try:
            await release_old.wait()
        finally:
            live["count"] -= 1

    async def recovered_loop():
        live["count"] += 1
        live["peak"] = max(live["peak"], live["count"])
        try:
            await release_new.wait()
        finally:
            live["count"] -= 1

    stale_task = asyncio.create_task(stale_loop())
    await asyncio.sleep(0)
    tick._task = stale_task
    tick._run_loop = recovered_loop

    tick.is_alive()
    release_old.set()
    for _ in range(50):
        await asyncio.sleep(0)
        if tick._task is not stale_task:
            break

    assert live["peak"] == 1, (
        f"{live['peak']} cognitive loops were alive at once; the repair "
        "started a replacement before the cancelled loop had unwound"
    )

    release_new.set()
    if tick._task is not None and not tick._task.done():
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
async def test_mind_tick_start_republishes_authoritative_running_instance(monkeypatch):
    monkeypatch.setattr("infrastructure.watchdog.get_watchdog", lambda: Watchdog())
    ServiceContainer.clear()

    stale_tick = SimpleNamespace(_running=False, _task=None, is_alive=lambda: False)
    ServiceContainer.register_instance("mind_tick", stale_tick, required=False)

    tick = MindTick.__new__(MindTick)
    tick.orchestrator = SimpleNamespace()
    tick._running = False
    tick._task = None
    tick._started_at = 0.0
    tick._last_successful_tick_at = 0.0
    tick._last_loop_progress_at = 0.0
    tick._last_progress_label = "not_started"
    tick._active_tick_started_at = 0.0
    tick._active_tick_stage = "idle"
    tick._consecutive_loop_failures = 0
    tick._last_liveness_repair_at = 0.0
    tick._liveness_repair_count = 0

    release = asyncio.Event()

    async def run_loop():
        await release.wait()

    tick._run_loop = run_loop

    await tick.start()

    assert ServiceContainer.get("mind_tick") is tick
    assert tick._task is not None
    assert not tick._task.done()

    release.set()
    await tick._task
    ServiceContainer.clear()


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


def test_stale_rhythm_verdict_names_the_wedged_stage():
    """'is_alive() returned False' told an operator nothing for two hours
    live (Jul 7): the rhythm was wedged at one stage with no receipt. Every
    stale verdict must now record WHERE the loop is stuck."""
    from core.health.degraded_events import isolated_degraded_event_scope

    class RunningTask:
        @staticmethod
        def done():
            return False

    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._task = RunningTask()
    tick._started_at = time.time() - 7000
    tick._last_successful_tick_at = time.time() - 4600
    tick._last_loop_progress_at = time.time() - 4600
    tick._active_tick_started_at = time.time() - 4600
    tick._active_tick_stage = "llm_health"
    tick._consecutive_loop_failures = 0
    tick._tick_count = 40
    tick._last_liveness_repair_at = time.monotonic()  # repair rate-limited off

    with isolated_degraded_event_scope("stale-stage-test"):
        from core.health.degraded_events import get_recent_degraded_events

        assert tick.is_alive() is False
        events = get_recent_degraded_events(limit=10)
    stale = [e for e in events if e.get("reason") == "rhythm_stale"]
    assert stale, events
    assert "stage=llm_health" in stale[0].get("detail", "")


def test_unreachable_liveness_repair_is_never_silent():
    """A repair that cannot run must say so — the silent no-op branch left
    the runtime DEGRADED for hours with 'repair machinery present'."""
    from core.health.degraded_events import (
        get_recent_degraded_events,
        isolated_degraded_event_scope,
    )

    class RunningTask:
        @staticmethod
        def done():
            return False

        @staticmethod
        def cancel():
            return True

    tick = MindTick.__new__(MindTick)
    tick._running = True
    tick._task = RunningTask()
    tick._active_tick_stage = "llm_health"
    tick._last_liveness_repair_at = 0.0
    tick._owner_loop = None  # the dead-end: no usable loop from a thread

    with isolated_degraded_event_scope("repair-unreachable-test"):
        result_holder = {}

        def probe():
            result_holder["repaired"] = tick._attempt_liveness_repair(
                reason="test", cancel_existing=False
            )

        worker = threading.Thread(target=probe)
        worker.start()
        worker.join(timeout=5)
        events = get_recent_degraded_events(limit=10)

    assert result_holder.get("repaired") is False
    unreachable = [e for e in events if e.get("reason") == "liveness_repair_unreachable"]
    assert unreachable, events
    assert "no usable owner loop" in unreachable[0].get("detail", "")


def test_tick_llm_health_await_is_bounded():
    """The rhythm loop must never hand its liveness to a dependency: both
    remaining bare awaits (state read, tier health sweep) carry timeouts."""
    import inspect

    src = inspect.getsource(MindTick._run_loop)
    assert "wait_for(\n                        self.orchestrator.state_repo.get_current()" in src.replace("  ", "  ") or "state_repo.get_current(), timeout=" in src
    assert "ensure_all_tiers_healthy(), timeout=" in src
