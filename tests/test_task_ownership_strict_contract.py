from __future__ import annotations

import asyncio

import pytest


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
    from core.runtime import strict_task_owner
    from core.runtime import task_ownership

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
    from core.runtime import strict_task_owner
    from core.runtime import task_ownership

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
