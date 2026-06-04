import asyncio
import time

from core.bus.local_pipe_bus import LocalPipeBus


class _FakeConnection:
    def __init__(self, *, delay_s: float = 0.0):
        self.closed = False
        self.delay_s = delay_s
        self.sent: list[str] = []

    def send(self, raw: str) -> None:
        if self.delay_s:
            time.sleep(self.delay_s)
        self.sent.append(raw)

    def close(self) -> None:
        self.closed = True


def test_fire_and_forget_pipe_send_drops_during_backpressure():
    async def scenario():
        read_conn = _FakeConnection()
        write_conn = _FakeConnection()
        bus = LocalPipeBus(read_conn=read_conn, write_conn=write_conn, start_reader=False)

        lock = bus._get_write_lock()
        await lock.acquire()
        try:
            await bus._send_local("telemetry", {"value": 1})
        finally:
            lock.release()
            bus._shutdown_executor()

        assert write_conn.sent == []
        assert bus._write_backpressure_drops == 1

    asyncio.run(scenario())


def test_fire_and_forget_pipe_timeout_suppresses_future_writes(monkeypatch):
    async def scenario():
        read_conn = _FakeConnection()
        write_conn = _FakeConnection(delay_s=0.5)
        bus = LocalPipeBus(read_conn=read_conn, write_conn=write_conn, start_reader=False)
        monkeypatch.setenv("AURA_PIPE_FF_WRITE_TIMEOUT_S", "0.25")
        monkeypatch.setenv("AURA_PIPE_SUPPRESS_AFTER_TIMEOUT_S", "1.0")

        started_at = time.monotonic()
        await bus._send_local("telemetry", {"value": 1})

        try:
            assert bus._write_timeout_count == 1
            assert bus._write_suppressed_until > started_at
            assert bus.is_alive() is False
            status = bus.get_status()
            assert status["degraded"] is True
            assert status["alive"] is False
            assert status["write_suppressed_for_s"] > 0
            assert "TimeoutError" in str(status["last_error"])
            await bus._send_local("telemetry", {"value": 2})
            assert len(write_conn.sent) <= 1
        finally:
            bus._shutdown_executor()

    asyncio.run(scenario())


def test_actor_bus_reports_transport_health():
    async def scenario():
        from core.bus.actor_bus import ActorBus

        await ActorBus.reset_singleton()
        read_conn = _FakeConnection()
        write_conn = _FakeConnection()
        pipe = LocalPipeBus(read_conn=read_conn, write_conn=write_conn, start_reader=False)
        bus = ActorBus()
        bus._is_running = True
        bus._transports["gui"] = pipe

        try:
            assert bus.is_alive() is True
            assert bus.is_actor_usable("gui") is True
            pipe._mark_transport_degraded(
                TimeoutError("pipe saturated"),
                "test transport health propagation",
            )

            status = bus.get_status()
            assert bus.is_alive() is False
            assert bus.is_actor_usable("gui") is False
            assert status["healthy"] is False
            assert status["transports"]["gui"]["alive"] is False
            assert "pipe saturated" in str(status["transports"]["gui"]["last_error"])
        finally:
            pipe._shutdown_executor()
            await ActorBus.reset_singleton()

    asyncio.run(scenario())
