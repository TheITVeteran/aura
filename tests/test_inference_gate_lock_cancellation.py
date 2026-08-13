"""Cancellation must never leave the foreground-ready lock ownerless."""

from __future__ import annotations

import asyncio
import threading

import pytest

from core.brain.inference_gate import _thread_lock_context


@pytest.mark.asyncio
async def test_cancelled_waiter_cannot_acquire_after_its_coroutine_is_gone() -> None:
    lock = threading.Lock()
    assert lock.acquire(blocking=False)
    entered = asyncio.Event()

    async def _waiter() -> None:
        async with _thread_lock_context(lock, label="foreground_ready"):
            entered.set()

    waiter = asyncio.create_task(_waiter())
    await asyncio.sleep(0.03)
    assert not entered.is_set()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    lock.release()
    await asyncio.sleep(0.03)
    assert lock.acquire(blocking=False), (
        "a cancelled acquisition continued in a worker thread and stole the lock"
    )
    lock.release()

    # The lock remains usable by more than one subsequent owner; this catches
    # delayed acquisition as well as an immediate leak.
    for _ in range(2):
        async with _thread_lock_context(lock, label="foreground_ready"):
            assert lock.locked()
        assert not lock.locked()


@pytest.mark.asyncio
async def test_thread_lock_context_timeout_does_not_change_ownership() -> None:
    lock = threading.Lock()
    assert lock.acquire(blocking=False)

    with pytest.raises(TimeoutError, match="foreground_ready_timeout"):
        async with _thread_lock_context(
            lock,
            timeout_s=0.02,
            label="foreground_ready",
        ):
            raise AssertionError("a held lock must not be entered")

    assert lock.locked()
    lock.release()
