from __future__ import annotations

import asyncio
import threading

import pytest

from core.runtime.model_lane_control import run_owned_model_thread_call


@pytest.mark.asyncio
async def test_cancellation_waits_for_owned_native_model_call() -> None:
    entered = threading.Event()
    release = threading.Event()

    def _operation() -> str:
        entered.set()
        assert release.wait(2.0)
        return "complete"

    caller = asyncio.create_task(
        run_owned_model_thread_call(
            _operation,
            operation_name="cancelled-test",
        )
    )
    assert await asyncio.to_thread(entered.wait, 1.0)
    caller.cancel()
    await asyncio.sleep(0.02)
    assert caller.done() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller


@pytest.mark.asyncio
async def test_timeout_waits_for_owned_native_model_call_before_raising() -> None:
    entered = threading.Event()
    release = threading.Event()

    def _operation() -> str:
        entered.set()
        assert release.wait(2.0)
        return "complete"

    caller = asyncio.create_task(
        run_owned_model_thread_call(
            _operation,
            operation_name="timeout-test",
            timeout_s=0.01,
        )
    )
    assert await asyncio.to_thread(entered.wait, 1.0)
    await asyncio.sleep(0.03)
    assert caller.done() is False

    release.set()
    with pytest.raises(TimeoutError, match="owned_model_call_timed_out:timeout-test"):
        await caller
