"""Task ownership includes a coroutine whose wrapper was cancelled before it ran."""
import asyncio
import inspect

import pytest

from core.runtime import task_ownership

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_cancelled_wrapper_closes_the_coroutine_it_never_started(monkeypatch):
    class Tracker:
        def bounded_track(self, awaitable, **kwargs):
            async def wrapped():
                await asyncio.sleep(0)
                await awaitable
            return asyncio.create_task(wrapped())

    monkeypatch.setattr(task_ownership, '_get_tracker', lambda: Tracker())
    ran = []
    async def work():
        ran.append(True)
    coroutine = work()
    task = task_ownership.create_tracked_task(coroutine, bounded=True)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED and not ran


@pytest.mark.asyncio
async def test_started_work_still_receives_cancellation_and_its_finally_runs(monkeypatch):
    monkeypatch.setattr(task_ownership, '_get_tracker', lambda: None)
    started, finalized = asyncio.Event(), []
    async def work():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            finalized.append(True)
    coroutine = work()
    task = task_ownership.create_tracked_task(coroutine)
    await started.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert finalized == [True] and inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


@pytest.mark.asyncio
async def test_a_normal_task_keeps_its_result_and_completion_callback(monkeypatch):
    monkeypatch.setattr(task_ownership, '_get_tracker', lambda: None)
    results = []
    async def work():
        return 7
    task = task_ownership.create_tracked_task(work(), on_done=lambda t: results.append(t.result()))
    assert await task == 7
    await asyncio.sleep(0)
    assert results == [7]


def test_unstarted_task_cleanup_has_a_registered_measured_claim():
    from core.organism.claims_realtime_control import install_realtime_control_claims
    from core.organism.model_validation import ValidationSuite

    assert task_ownership._unstarted_task_cleanup_invariant() == ()
    suite = ValidationSuite()
    install_realtime_control_claims(suite)
    check = next(t for t in suite.tests() if t.name == 'completed_task_closes_its_unstarted_coroutine')
    assert check.predict(None) is True and any(c.test == check.name for c in suite.claims())
