"""ActorBus drop visibility and non-blocking delivery contracts."""
from __future__ import annotations

import asyncio

import pytest

from core.bus.actor_bus import ActorBus


@pytest.mark.asyncio
async def test_actor_bus_unknown_send_is_visible_drop() -> None:
    await ActorBus.reset_singleton()
    bus = ActorBus()
    bus.start()
    try:
        ok = await bus.send("missing_actor", "tick", {"value": 1})
        status = bus.get_status()

        assert ok is False
        assert status["send_drops"] == 1
        assert status["last_drop"]["kind"] == "send"
        assert status["last_drop"]["reason"] == "unknown_actor"
        assert status["last_drop"]["actor"] == "missing_actor"
    finally:
        await bus.stop()
        await ActorBus.reset_singleton()


@pytest.mark.asyncio
async def test_actor_bus_telemetry_overwrite_is_visible_drop() -> None:
    await ActorBus.reset_singleton()
    bus = ActorBus()
    bus._is_running = True
    bus._telemetry_queue = asyncio.Queue(maxsize=1)
    try:
        assert await bus.broadcast_telemetry("first", {"value": 1}) is True
        assert await bus.broadcast_telemetry("second", {"value": 2}) is True

        status = bus.get_status()
        assert status["telemetry_drops"] == 1
        assert status["last_drop"]["reason"] == "queue_full_overwrite"
        assert status["telemetry_queue_size"] == 1
        assert bus._telemetry_queue.get_nowait() == ("second", {"value": 2})
        bus._telemetry_queue.task_done()
    finally:
        bus._is_running = False
        await ActorBus.reset_singleton()


@pytest.mark.asyncio
async def test_actor_bus_transport_send_failure_is_visible_drop() -> None:
    await ActorBus.reset_singleton()
    bus = ActorBus()
    bus._is_running = True

    class FailingTransport:
        _is_running = True
        _pending_requests: dict[str, object] = {}
        write_conn = None
        _pipe_broken = False
        calls = 0

        async def send(self, *_args, **_kwargs):
            self.calls += 1
            raise BrokenPipeError("pipe closed")

    bus._transports["worker"] = FailingTransport()  # type: ignore[assignment]
    try:
        ok = await bus.send("worker", "tick", {"value": 1})
        status = bus.get_status()

        assert ok is False
        assert status["send_drops"] == 1
        assert status["last_drop"]["reason"] == "transport_send_failed"
        assert status["last_drop"]["actor"] == "worker"
    finally:
        bus._is_running = False
        await ActorBus.reset_singleton()


@pytest.mark.asyncio
async def test_actor_bus_health_fails_when_telemetry_task_dies() -> None:
    await ActorBus.reset_singleton()
    bus = ActorBus()
    bus.start()
    task = bus._telemetry_broadcaster_task
    assert task is not None
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    try:
        status = bus.get_status()

        assert bus.is_alive() is False
        assert status["healthy"] is False
        assert status["telemetry_task_alive"] is False
        assert status["telemetry_task"]["done"] is True
    finally:
        await bus.stop()
        await ActorBus.reset_singleton()
