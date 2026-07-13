from __future__ import annotations

import asyncio
import gc
import time

import pytest

from core.utils.task_tracker import TaskRecord, TaskTracker


def test_bounded_tracking_uses_event_loop_local_semaphores() -> None:
    tracker = TaskTracker(name="loop-local-semaphore-test")

    async def capture() -> tuple[asyncio.Semaphore, asyncio.Semaphore]:
        return tracker._get_semaphore(), tracker._get_semaphore()

    first, first_again = asyncio.run(capture())
    second, second_again = asyncio.run(capture())

    assert first is first_again
    assert second is second_again
    assert first is not second


@pytest.mark.asyncio
async def test_reused_python_task_id_gets_a_new_observed_lifecycle() -> None:
    tracker = TaskTracker(name="terminal-collision-test")
    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(dict(context)))

    async def fail() -> None:
        raise ValueError("owned failure")

    task = asyncio.create_task(fail(), name="fresh-task")
    stale_lifecycle_id = "terminal-collision-test:stale"
    tracker._records[id(task)] = TaskRecord(
        lifecycle_id=stale_lifecycle_id,
        task_id=id(task),
        name="stale-task",
        owner="stale-owner",
        tracker=tracker.name,
        supervision="explicit",
        source="test-fixture",
        created_at=time.monotonic() - 1.0,
        done=True,
        finished_at=time.monotonic() - 0.5,
        outcome="succeeded",
    )

    try:
        tracker.observe(
            task,
            name="fresh-task",
            source="collision-regression",
            owner="unit-test",
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        stats = tracker.get_stats()
        receipt = stats["recently_completed"][-1]
        assert stats["failed_total"] == 1
        assert receipt["lifecycle_id"] != stale_lifecycle_id
        assert receipt["name"] == "fresh-task"
        assert receipt["owner"] == "unit-test"
        assert receipt["outcome"] == "failed"
        assert receipt["exception"] == "ValueError: owned failure"

        del task
        gc.collect()
        await asyncio.sleep(0)
        assert not any(
            "Task exception was never retrieved" in str(item.get("message", ""))
            for item in loop_errors
        )
    finally:
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_cancellation_emits_structured_terminal_receipt() -> None:
    tracker = TaskTracker(name="terminal-cancellation-test")
    started = asyncio.Event()

    async def wait_until_cancelled() -> None:
        started.set()
        await asyncio.Event().wait()

    task = tracker.create_task(
        wait_until_cancelled(),
        name="conversation_support.turn_updates",
        owner="response_generation",
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)

    receipt = tracker.get_stats()["recently_completed"][-1]
    assert receipt["lifecycle_id"].startswith("terminal-cancellation-test:")
    assert receipt["name"] == "conversation_support.turn_updates"
    assert receipt["owner"] == "response_generation"
    assert receipt["outcome"] == "cancelled"
    assert receipt["cancelled"] is True
    assert receipt["failed"] is False


@pytest.mark.asyncio
async def test_unnamed_task_uses_coroutine_identity_in_terminal_receipt() -> None:
    tracker = TaskTracker(name="terminal-success-test")

    async def complete() -> str:
        return "done"

    task = tracker.create_task(complete(), owner="unit-test")
    assert await task == "done"
    await asyncio.sleep(0)

    receipt = tracker.get_stats()["recently_completed"][-1]
    assert receipt["name"].endswith("complete")
    assert receipt["owner"] == "unit-test"
    assert receipt["outcome"] == "succeeded"
    assert receipt["failed"] is False
