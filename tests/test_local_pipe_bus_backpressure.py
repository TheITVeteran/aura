import asyncio
import json
import time

from core.bus.local_pipe_bus import LocalPipeBus


class _FakeConnection:
    def __init__(self, *, delay_s: float = 0.0, fail_send: bool = False):
        self.closed = False
        self.delay_s = delay_s
        self.fail_send = fail_send
        self.sent: list[str] = []

    def send(self, raw: str) -> None:
        if self.fail_send:
            raise OSError("pipe response write failed")
        if self.delay_s:
            time.sleep(self.delay_s)
        self.sent.append(raw)

    def close(self) -> None:
        self.closed = True


class _ScriptedReadConnection(_FakeConnection):
    def __init__(self, messages):
        super().__init__()
        self._messages = list(messages)

    def recv(self):
        if self._messages:
            return self._messages.pop(0)
        raise EOFError


class _FakeTask:
    def __init__(self):
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


def test_fire_and_forget_pipe_send_drops_during_backpressure():
    async def scenario():
        from core.bus import local_pipe_bus as module

        records = []
        original_record = module.record_degradation
        module.record_degradation = lambda *args, **kwargs: records.append((args, kwargs))
        read_conn = _FakeConnection()
        write_conn = _FakeConnection()
        bus = LocalPipeBus(read_conn=read_conn, write_conn=write_conn, start_reader=False)
        bus._is_running = True

        lock = bus._get_write_lock()
        await lock.acquire()
        try:
            await bus._send_local("telemetry", {"value": 1})
        finally:
            lock.release()
            bus._shutdown_executor()
            module.record_degradation = original_record

        assert write_conn.sent == []
        assert bus._write_backpressure_drops == 1
        assert records == []
        assert bus.is_alive() is True
        status = bus.get_status()
        assert status["alive"] is True
        assert status["degraded"] is False
        assert status["write_backpressure_drops"] == 1
        assert "fire-and-forget pipe write blocked" in str(status["last_error"])

    asyncio.run(scenario())


def test_fire_and_forget_pipe_default_timeout_is_subsecond(monkeypatch):
    read_conn = _FakeConnection()
    write_conn = _FakeConnection()
    bus = LocalPipeBus(read_conn=read_conn, write_conn=write_conn, start_reader=False)
    monkeypatch.delenv("AURA_PIPE_FF_WRITE_TIMEOUT_S", raising=False)

    try:
        assert 0.05 <= bus._fire_and_forget_write_timeout_s() <= 0.5
    finally:
        bus._shutdown_executor()


def test_fire_and_forget_pipe_timeout_suppresses_future_writes(monkeypatch):
    async def scenario():
        from core.bus import local_pipe_bus as module

        records = []
        monkeypatch.setattr(module, "record_degradation", lambda *args, **kwargs: records.append((args, kwargs)))
        read_conn = _FakeConnection()
        write_conn = _FakeConnection(delay_s=0.5)
        bus = LocalPipeBus(read_conn=read_conn, write_conn=write_conn, start_reader=False)
        bus._is_running = True
        monkeypatch.setenv("AURA_PIPE_FF_WRITE_TIMEOUT_S", "0.25")
        monkeypatch.setenv("AURA_PIPE_SUPPRESS_AFTER_TIMEOUT_S", "1.0")

        started_at = time.monotonic()
        await bus._send_local("telemetry", {"value": 1})

        try:
            assert bus._write_timeout_count == 1
            assert bus._write_suppressed_until > started_at
            assert records == []
            assert bus.is_alive() is True
            status = bus.get_status()
            assert status["degraded"] is False
            assert status["alive"] is True
            assert status["write_suppressed_for_s"] > 0
            assert "TimeoutError" in str(status["last_error"])
            await bus._send_local("telemetry", {"value": 2})
            assert len(write_conn.sent) <= 1
        finally:
            bus._shutdown_executor()

    asyncio.run(scenario())


def test_local_pipe_bus_health_requires_running_transport():
    read_conn = _FakeConnection()
    write_conn = _FakeConnection()
    bus = LocalPipeBus(read_conn=read_conn, write_conn=write_conn, start_reader=False)

    try:
        status = bus.get_status()

        assert bus.is_alive() is False
        assert status["alive"] is False
        assert status["running"] is False
    finally:
        bus._shutdown_executor()


def test_local_pipe_bus_reader_mode_health_requires_background_tasks():
    class DoneTask:
        def done(self):
            return True

        def cancelled(self):
            return False

        def exception(self):
            return RuntimeError("reader crashed")

    read_conn = _FakeConnection()
    write_conn = _FakeConnection()
    bus = LocalPipeBus(read_conn=read_conn, write_conn=write_conn, start_reader=True)
    bus._is_running = True
    bus._dispatch_queue = asyncio.Queue(maxsize=1)
    bus._reader_task = DoneTask()  # type: ignore[assignment]
    bus._dispatcher_task = DoneTask()  # type: ignore[assignment]

    try:
        status = bus.get_status()

        assert bus.is_alive() is False
        assert status["background_tasks_alive"] is False
        assert status["reader_task"]["failed"] is True
        assert "reader crashed" in str(status["reader_task"]["exception"])
    finally:
        bus._shutdown_executor()


def test_local_pipe_bus_start_repairs_dead_background_worker(monkeypatch):
    async def scenario():
        from core.bus import local_pipe_bus as module

        records = []
        created = []

        class _Tracker:
            def create_task(self, coro, name=None):
                task = asyncio.create_task(coro, name=name)
                created.append((name, task))
                return task

        async def _alive():
            await asyncio.sleep(10)

        async def _dead():
            await asyncio.sleep(0)
            raise RuntimeError("dispatcher crashed")

        monkeypatch.setattr(module, "get_task_tracker", lambda: _Tracker())
        monkeypatch.setattr(
            module,
            "record_degradation",
            lambda *args, **kwargs: records.append((args, kwargs)),
        )

        read_conn = _FakeConnection()
        write_conn = _FakeConnection()
        bus = LocalPipeBus(read_conn=read_conn, write_conn=write_conn, start_reader=True)
        bus._is_running = True
        bus._loop = asyncio.get_running_loop()
        bus._dispatch_queue = asyncio.Queue(maxsize=1)
        reader_task = asyncio.create_task(_alive(), name="existing-reader")
        dispatcher_task = asyncio.create_task(_dead(), name="dead-dispatcher")
        bus._reader_task = reader_task
        bus._dispatcher_task = dispatcher_task
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        try:
            bus.start()

            assert bus._reader_task is reader_task
            assert bus._dispatcher_task is not dispatcher_task
            assert [name for name, _task in created] == ["local_pipe_bus.dispatch"]
            assert records
            assert records[-1][0][0] == "local_pipe_bus"
            assert records[-1][1]["action"] == "restarting_dead_background_workers"
            assert records[-1][1]["extra"]["dispatcher_task"]["failed"] is True
            assert bus.is_alive() is True
        finally:
            await bus.stop()

    asyncio.run(scenario())


def test_local_pipe_bus_health_fails_on_saturated_pending_requests():
    read_conn = _FakeConnection()
    write_conn = _FakeConnection()
    bus = LocalPipeBus(read_conn=read_conn, write_conn=write_conn, start_reader=False)
    bus._is_running = True
    loop = asyncio.new_event_loop()
    try:
        for idx in range(bus._max_pending_requests):
            bus._pending_requests[str(idx)] = loop.create_future()

        status = bus.get_status()

        assert bus.is_alive() is False
        assert status["pending_requests_saturated"] is True
        assert status["pending_requests"] == status["pending_request_limit"]
    finally:
        loop.close()
        bus._shutdown_executor()


def test_stop_treats_task_cancellation_as_normal_shutdown(monkeypatch):
    async def scenario():
        from core.bus import local_pipe_bus as module

        records = []
        monkeypatch.setattr(
            module,
            "record_degradation",
            lambda *args, **kwargs: records.append((args, kwargs)),
        )

        async def fake_wait_for(_task, timeout):
            assert timeout == 1.0
            raise asyncio.CancelledError()

        monkeypatch.setattr(module.asyncio, "wait_for", fake_wait_for)

        read_conn = _FakeConnection()
        write_conn = _FakeConnection()
        bus = LocalPipeBus(read_conn=read_conn, write_conn=write_conn, start_reader=False)
        bus._reader_task = _FakeTask()

        await bus.stop()

        assert bus._reader_task.cancelled is True
        assert records == []
        assert read_conn.closed is True
        assert write_conn.closed is True

    asyncio.run(scenario())


def test_stop_records_shutdown_timeouts_as_degradation(monkeypatch):
    async def scenario():
        from core.bus import local_pipe_bus as module

        records = []
        monkeypatch.setattr(
            module,
            "record_degradation",
            lambda *args, **kwargs: records.append((args, kwargs)),
        )

        async def fake_wait_for(_task, timeout):
            assert timeout == 1.0
            raise TimeoutError("shutdown wait timed out")

        monkeypatch.setattr(module.asyncio, "wait_for", fake_wait_for)

        read_conn = _FakeConnection()
        write_conn = _FakeConnection()
        bus = LocalPipeBus(read_conn=read_conn, write_conn=write_conn, start_reader=False)
        bus._reader_task = _FakeTask()
        bus._dispatcher_task = _FakeTask()

        await bus.stop()

        assert bus._reader_task.cancelled is True
        assert bus._dispatcher_task.cancelled is True
        assert [record[0][0] for record in records] == ["local_pipe_bus", "local_pipe_bus"]
        assert [record[1]["action"] for record in records] == [
            "reader task did not stop before shutdown timeout",
            "dispatcher task did not stop before shutdown timeout",
        ]

    asyncio.run(scenario())


def test_reader_survives_failed_dispatch_saturation_error_response():
    async def scenario():
        message = {
            "type": "work",
            "payload": {"value": 1},
            "trace_id": "trace-1",
            "request_id": "request-1",
            "is_request": True,
        }
        read_conn = _ScriptedReadConnection([json.dumps(message)])
        write_conn = _FakeConnection(fail_send=True)
        bus = LocalPipeBus(read_conn=read_conn, write_conn=write_conn, start_reader=True)
        bus._is_running = True
        bus._loop = asyncio.get_running_loop()
        bus._dispatch_queue = asyncio.Queue(maxsize=1)
        await bus._dispatch_queue.put((lambda *_args: None, {"type": "queued"}))
        bus.register_handler("work", lambda *_args: {"ok": True})

        try:
            await asyncio.wait_for(bus._read_loop(), timeout=2.0)

            status = bus.get_status()
            assert status["degraded"] is True
            assert "pipe response write failed" in str(status["last_error"])
            assert write_conn.sent == []
            assert bus._dispatch_queue.qsize() == 1
        finally:
            bus._shutdown_executor()

    asyncio.run(scenario())


def test_reader_marks_bus_stopped_on_peer_eof():
    async def scenario():
        read_conn = _ScriptedReadConnection([])
        write_conn = _FakeConnection()
        bus = LocalPipeBus(read_conn=read_conn, write_conn=write_conn, start_reader=True)
        bus._is_running = True
        bus._loop = asyncio.get_running_loop()

        try:
            await bus._read_loop()

            assert bus._is_running is False
            assert bus.is_alive() is False
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
        pipe._is_running = True
        bus = ActorBus()
        bus._is_running = True
        bus._telemetry_queue = asyncio.Queue(maxsize=1)
        bus._telemetry_broadcaster_task = asyncio.create_task(asyncio.sleep(10))
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
            bus._telemetry_broadcaster_task.cancel()
            try:
                await bus._telemetry_broadcaster_task
            except asyncio.CancelledError:
                pass
            pipe._shutdown_executor()
            await ActorBus.reset_singleton()

    asyncio.run(scenario())
