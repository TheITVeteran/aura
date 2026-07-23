"""CP126 contract tests for the background reasoning queue."""
from __future__ import annotations

import asyncio

import pytest

from core.brain.reasoning_queue import (
    BackgroundReasoningQueue,
    QueueFull,
    ReasoningPriority,
)


def _queue(**kwargs) -> BackgroundReasoningQueue:
    queue = BackgroundReasoningQueue(**kwargs)
    queue._schedule_registry_size_update = lambda *, reason: None
    return queue


async def _stop(queue: BackgroundReasoningQueue) -> None:
    await queue.aclose(timeout=2.0)


# --- b3925b0c: bounded, with deadlines ------------------------------------


@pytest.mark.asyncio
async def test_the_queue_is_bounded():
    queue = _queue(maxsize=2)

    await queue.submit(lambda: 1, priority=ReasoningPriority.NORMAL)
    await queue.submit(lambda: 2, priority=ReasoningPriority.NORMAL)

    with pytest.raises(QueueFull):
        await queue.submit(lambda: 3, priority=ReasoningPriority.NORMAL)

    assert queue.stats()["queued"] == 2


@pytest.mark.asyncio
async def test_a_higher_priority_task_sheds_a_lower_one():
    queue = _queue(maxsize=2)

    await queue.submit(lambda: 1, priority=ReasoningPriority.HIGH, description="keep")
    low_id = await queue.submit(lambda: 2, priority=ReasoningPriority.LOW, description="shed")
    critical_id = await queue.submit(
        lambda: 3, priority=ReasoningPriority.CRITICAL, description="urgent"
    )

    assert queue.task_status(low_id) == "shed"
    assert queue.get_result(low_id)["reason"].startswith("displaced by")
    assert queue.task_status(critical_id) == "pending"
    assert queue.stats()["shed"] == 1


@pytest.mark.asyncio
async def test_a_refused_submission_leaves_a_receipt():
    queue = _queue(maxsize=1)
    await queue.submit(lambda: 1, priority=ReasoningPriority.CRITICAL)

    with pytest.raises(QueueFull):
        await queue.submit(lambda: 2, priority=ReasoningPriority.LOW, description="late")

    # The rejected id is not returned, but the queue did not grow.
    assert queue.stats()["queued"] == 1


@pytest.mark.asyncio
async def test_a_hung_task_does_not_stop_the_queue():
    queue = _queue()

    async def hang():
        await asyncio.sleep(30)

    await queue.start()
    hung_id = await queue.submit(hang, description="hung", timeout_s=0.05)
    ok_id = await queue.submit(lambda: "ok", description="after")

    await asyncio.wait_for(queue._queue.join(), timeout=5.0)

    assert queue.get_result(hung_id)["status"] == "timeout"
    assert queue.get_result(ok_id) == "ok"
    await _stop(queue)


@pytest.mark.asyncio
async def test_an_expired_task_is_not_executed():
    queue = _queue()
    ran = []

    await queue.submit(lambda: ran.append(1), description="stale", ttl_s=0.01)
    await asyncio.sleep(0.05)
    await queue.start()
    await asyncio.wait_for(queue._queue.join(), timeout=3.0)

    assert ran == []
    await _stop(queue)


# --- 7c56145d: workers survive, and are replaced ---------------------------


@pytest.mark.asyncio
async def test_an_unlisted_exception_does_not_kill_the_worker():
    queue = _queue()

    def boom():
        raise KeyError("not in the old recoverable tuple")

    await queue.start()
    failed_id = await queue.submit(boom, description="keyerror")
    ok_id = await queue.submit(lambda: "still working", description="after")
    await asyncio.wait_for(queue._queue.join(), timeout=5.0)

    assert queue.get_result(failed_id)["error_type"] == "KeyError"
    assert queue.get_result(ok_id) == "still working"
    assert queue.stats()["workers"] >= 1
    await _stop(queue)


@pytest.mark.asyncio
async def test_start_tops_up_a_missing_worker():
    queue = _queue(max_concurrent=2)
    await queue.start()
    assert queue.stats()["workers"] == 2

    victim = next(iter(queue._worker_tasks))
    victim.cancel()
    await asyncio.sleep(0.05)

    await queue.start()

    assert queue.stats()["workers"] == 2
    await _stop(queue)


@pytest.mark.asyncio
async def test_os_error_is_recorded_not_fatal():
    queue = _queue()

    def io_fail():
        raise OSError("disk went away")

    await queue.start()
    task_id = await queue.submit(io_fail)
    await asyncio.wait_for(queue._queue.join(), timeout=5.0)

    assert queue.get_result(task_id)["error_type"] == "OSError"
    await _stop(queue)


# --- ddab4f5a: pruning must not strand a parked worker --------------------


@pytest.mark.asyncio
async def test_prune_preserves_the_queue_object():
    queue = _queue()
    original = queue._queue

    await queue.submit(lambda: "keep", priority=ReasoningPriority.HIGH)
    await queue.submit(lambda: "drop", priority=ReasoningPriority.LOW)
    await queue.prune_low_priority(threshold_priority=ReasoningPriority.HIGH.value)

    assert queue._queue is original


@pytest.mark.asyncio
async def test_work_submitted_after_a_prune_still_reaches_a_parked_worker():
    queue = _queue()
    await queue.start()
    await asyncio.sleep(0.05)  # let the worker park in get()

    await queue.submit(lambda: "low", priority=ReasoningPriority.LOW)
    await queue.prune_low_priority(threshold_priority=ReasoningPriority.NORMAL.value)

    after_id = await queue.submit(lambda: "after prune", description="after")
    await asyncio.wait_for(queue._queue.join(), timeout=5.0)

    assert queue.get_result(after_id) == "after prune"
    await _stop(queue)


@pytest.mark.asyncio
async def test_prune_keeps_higher_priority_work():
    queue = _queue()
    keep_id = await queue.submit(lambda: "keep", priority=ReasoningPriority.HIGH)
    drop_id = await queue.submit(lambda: "drop", priority=ReasoningPriority.LOW)

    dropped = await queue.prune_low_priority(threshold_priority=ReasoningPriority.HIGH.value)

    assert dropped == 1
    assert queue._queue.get_nowait().task_id == keep_id
    assert queue.task_status(drop_id) == "pruned"


# --- 9056ce92: dropped work has a terminal envelope ------------------------


@pytest.mark.asyncio
async def test_a_pruned_task_gets_a_result_and_its_callback():
    queue = _queue()
    seen = []

    task_id = await queue.submit(
        lambda: "x",
        priority=ReasoningPriority.LOW,
        callback=lambda result: seen.append(result),
        description="doomed",
    )
    await queue.prune_low_priority(threshold_priority=ReasoningPriority.NORMAL.value)
    await asyncio.sleep(0.05)

    envelope = queue.get_result(task_id)
    assert envelope["status"] == "pruned"
    assert envelope["task_id"] == task_id
    assert queue.last_pruned_ids == [task_id]
    assert seen and seen[0]["status"] == "pruned"


@pytest.mark.asyncio
async def test_status_distinguishes_pending_done_and_unknown():
    queue = _queue()

    pending_id = await queue.submit(lambda: "v")
    assert queue.task_status(pending_id) == "pending"
    assert queue.task_status("nope") == "evicted_or_unknown"

    await queue.start()
    await asyncio.wait_for(queue._queue.join(), timeout=3.0)

    assert queue.task_status(pending_id) == "done"
    await _stop(queue)


@pytest.mark.asyncio
async def test_a_real_none_result_is_not_confused_with_absence():
    queue = _queue()
    await queue.start()
    task_id = await queue.submit(lambda: None, description="returns none")
    await asyncio.wait_for(queue._queue.join(), timeout=3.0)

    assert queue.get_result(task_id) is None
    assert queue.task_status(task_id) == "done"
    await _stop(queue)


# --- 99c0be71: shutdown resolves in-flight and queued work ----------------


@pytest.mark.asyncio
async def test_stop_gives_queued_tasks_a_terminal_result():
    queue = _queue()
    task_id = await queue.submit(lambda: "never runs", description="queued")

    queue.stop()

    assert queue.task_status(task_id) == "cancelled"
    assert queue.get_result(task_id)["reason"].startswith("queue stopped")
    assert queue.stats()["queued"] == 0


@pytest.mark.asyncio
async def test_in_flight_work_is_receipted_on_cancellation():
    queue = _queue()
    running = asyncio.Event()

    async def slow():
        running.set()
        await asyncio.sleep(30)

    await queue.start()
    task_id = await queue.submit(slow, description="in flight")
    await asyncio.wait_for(running.wait(), timeout=3.0)

    await queue.aclose(timeout=2.0)

    assert queue.get_result(task_id)["status"] == "cancelled"


@pytest.mark.asyncio
async def test_aclose_reports_whether_workers_stopped():
    queue = _queue()
    await queue.start()

    assert await queue.aclose(timeout=2.0) is True


# --- db963c45: callbacks cannot own a worker ------------------------------


@pytest.mark.asyncio
async def test_a_hung_callback_does_not_hold_the_worker():
    queue = _queue(callback_timeout_s=0.05)

    async def hang(_result):
        await asyncio.sleep(30)

    await queue.start()
    first = await queue.submit(lambda: "one", callback=hang, description="hung callback")
    second = await queue.submit(lambda: "two", description="after")

    await asyncio.wait_for(queue._queue.join(), timeout=5.0)

    assert queue.get_result(first) == "one"
    assert queue.get_result(second) == "two"
    await _stop(queue)


@pytest.mark.asyncio
async def test_a_failing_callback_still_preserves_the_result():
    queue = _queue()

    def boom(_result):
        raise IndexError("callback broke")

    await queue.start()
    task_id = await queue.submit(lambda: "kept", callback=boom)
    await asyncio.wait_for(queue._queue.join(), timeout=3.0)

    assert queue.get_result(task_id) == "kept"
    await _stop(queue)
