from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_owned_task_drain_preserves_result_and_captures_repeated_cancellation():
    from core.runtime.task_ownership import drain_owned_awaitable

    started = asyncio.Event()
    release = asyncio.Event()

    async def _operation():
        started.set()
        await release.wait()
        return "durable-result"

    owner = asyncio.create_task(
        drain_owned_awaitable(
            _operation(),
            name="unit.drain",
            owner="tests.task_ownership",
        )
    )
    await started.wait()
    owner.cancel("first cancellation")
    await asyncio.sleep(0)
    owner.cancel("second cancellation")
    await asyncio.sleep(0)
    assert owner.done() is False

    release.set()
    drained = await owner

    assert drained.task.done() is True
    assert drained.task.result() == "durable-result"
    assert drained.cancellation is not None


@pytest.mark.asyncio
async def test_owned_task_drain_forwards_shutdown_critical_ownership(monkeypatch):
    from core.runtime import task_ownership

    observed: dict[str, object] = {}

    class _Tracker:
        def create_task(self, awaitable, **kwargs):
            observed.update(kwargs)
            return asyncio.create_task(awaitable, name=kwargs.get("name"))

    monkeypatch.setattr(task_ownership, "_get_tracker", lambda: _Tracker())

    drained = await task_ownership.drain_owned_awaitable(
        asyncio.sleep(0, result=7),
        name="unit.shutdown-critical",
        owner="tests.task_ownership",
        allow_during_shutdown=True,
    )

    assert drained.task.result() == 7
    assert observed == {
        "name": "unit.shutdown-critical",
        "owner": "tests.task_ownership",
        "allow_during_shutdown": True,
    }


def test_shutdown_new_work_refusal_records_admission_receipt():
    from core.runtime.shutdown_coordinator import (
        clear_shutdown_request,
        request_shutdown,
        shutdown_admission_snapshot,
    )
    from core.runtime.task_ownership import runtime_shutdown_blocks_new_work

    clear_shutdown_request()
    request_shutdown("task-ownership-unit")
    try:
        before = int(shutdown_admission_snapshot()["counts"]["suppressed"])
        assert runtime_shutdown_blocks_new_work(
            "unit.refused",
            resource_kind="unit_work",
        )
        after = shutdown_admission_snapshot()
        assert int(after["counts"]["suppressed"]) == before + 1
        assert after["recent_events"][-1]["operation"] == "unit.refused"
    finally:
        clear_shutdown_request()


@pytest.mark.asyncio
async def test_task_tracker_suppresses_late_runtime_work_after_shutdown_request():
    from core.runtime.shutdown_coordinator import clear_shutdown_request, request_shutdown
    from core.utils.task_tracker import TaskTracker

    tracker = TaskTracker(name="shutdown-suppression-test")
    ran = False

    async def _late_runtime_work():
        nonlocal ran
        ran = True

    request_shutdown("unit-test")
    try:
        task = tracker.create_task(_late_runtime_work(), name="late-runtime-work")
        assert isinstance(task, asyncio.Task)
        assert await task is None
        assert ran is False
        assert getattr(task, "_aura_shutdown_suppressed", False) is True
        assert tracker.get_stats()["shutdown_suppressed_total"] == 1
    finally:
        clear_shutdown_request()


@pytest.mark.asyncio
async def test_shutdown_coordinator_handlers_are_allowed_after_shutdown_request():
    from core.runtime.shutdown_coordinator import (
        clear_shutdown_request,
        get_shutdown_coordinator,
        reset_shutdown_coordinator,
    )

    reset_shutdown_coordinator()
    coordinator = get_shutdown_coordinator()
    calls: list[str] = []

    async def _flush_handler():
        await asyncio.sleep(0)
        calls.append("flushed")

    coordinator.register(
        _flush_handler,
        phase="output_flush",
        name="test_flush_handler",
        timeout=1.0,
    )

    try:
        report = await coordinator.shutdown(timeout_per_phase=1.0)
        assert report.clean is True
        assert calls == ["flushed"]
    finally:
        reset_shutdown_coordinator()
        clear_shutdown_request()


@pytest.mark.asyncio
async def test_task_ownership_fallback_is_allowed_in_strict_runtime(monkeypatch):
    from core.runtime import strict_task_owner, task_ownership

    strict_task_owner.reset_violations()
    loop = asyncio.get_running_loop()
    strict_task_owner.install_strict_task_owner(loop)
    monkeypatch.setenv("AURA_STRICT_RUNTIME", "1")
    monkeypatch.setattr(task_ownership, "_get_tracker", lambda: None)

    async def _owned_fallback():
        await asyncio.sleep(0)
        return "ok"

    try:
        task = task_ownership.create_tracked_task(
            _owned_fallback(),
            name="task_ownership.fallback.strict",
        )
        assert await task == "ok"
        assert strict_task_owner.violations() == []
    finally:
        strict_task_owner.restore_strict_task_owner(loop)
        strict_task_owner.reset_violations()


@pytest.mark.asyncio
async def test_task_ownership_fallback_does_not_let_children_inherit_skip(monkeypatch):
    from core.runtime import strict_task_owner, task_ownership

    strict_task_owner.reset_violations()
    loop = asyncio.get_running_loop()
    strict_task_owner.install_strict_task_owner(loop)
    monkeypatch.setenv("AURA_STRICT_RUNTIME", "1")
    monkeypatch.setattr(task_ownership, "_get_tracker", lambda: None)

    async def _owned_parent():
        async def _unowned_child():
            await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="AURA_STRICT_RUNTIME"):
            asyncio.create_task(_unowned_child())

    try:
        await task_ownership.create_tracked_task(
            _owned_parent(),
            name="task_ownership.fallback.parent",
        )
        violations = strict_task_owner.violations()
        assert len(violations) == 1
        assert "_unowned_child" in violations[0]["coro"]
    finally:
        strict_task_owner.restore_strict_task_owner(loop)
        strict_task_owner.reset_violations()
