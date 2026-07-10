from __future__ import annotations

import asyncio
import signal
from types import SimpleNamespace

import pytest

from core.ops import process_manager as process_manager_module
from core.ops.process_manager import ManagedProcess, ProcessConfig, ProcessManager, ProcessState
from core.runtime import shutdown_coordinator


def _target() -> None:
    return None


def test_shutdown_latch_is_visible_before_grace_flag_io(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[bool] = []

    def _write_grace_flag(**_kwargs) -> None:
        observed.append(shutdown_coordinator.is_shutdown_requested())

    monkeypatch.setattr(shutdown_coordinator, "_write_grace_flag", _write_grace_flag)

    snapshot = shutdown_coordinator.request_shutdown("first")
    second = shutdown_coordinator.request_shutdown("second")

    assert observed == [True]
    assert snapshot["first_reason"] == "first"
    assert second["first_reason"] == "first"
    assert second["last_reason"] == "second"
    assert second["request_count"] == 2


@pytest.mark.asyncio
async def test_shutdown_coordinator_concurrent_and_repeated_calls_run_handlers_once() -> None:
    coordinator = shutdown_coordinator.ShutdownCoordinator()
    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def _handler() -> None:
        calls.append("handler")
        entered.set()
        await release.wait()

    coordinator.register(_handler, phase="actors", name="single-owner", timeout=1.0)

    first = asyncio.create_task(coordinator.shutdown(timeout_per_phase=1.0))
    await entered.wait()
    status = coordinator.get_status()
    assert status["running"] is True
    assert status["report"]["current_phase"] == "actors"

    second = asyncio.create_task(coordinator.shutdown(timeout_per_phase=1.0))
    await asyncio.sleep(0)
    release.set()
    first_report, second_report = await asyncio.gather(first, second)
    replay_report = await coordinator.shutdown(timeout_per_phase=1.0)

    assert calls == ["handler"]
    assert first_report.clean is True
    assert second_report.clean is True
    assert replay_report.clean is True
    assert replay_report.repeated_call_count == 2
    assert coordinator.get_status()["lifecycle_state"] == "TERMINATED"


@pytest.mark.asyncio
async def test_shutdown_coordinator_rejects_late_handler_registration() -> None:
    coordinator = shutdown_coordinator.ShutdownCoordinator()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _handler() -> None:
        entered.set()
        await release.wait()

    coordinator.register(_handler, phase="actors", name="owner", timeout=1.0)
    shutdown_task = asyncio.create_task(coordinator.shutdown(timeout_per_phase=1.0))
    await entered.wait()
    with pytest.raises(RuntimeError, match="teardown has started"):
        coordinator.register(lambda: None, phase="actors", name="late")
    release.set()
    assert (await shutdown_task).clean is True


def test_process_manager_does_not_install_signal_handlers_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: list[tuple[object, object]] = []
    monkeypatch.setattr(
        process_manager_module.signal,
        "signal",
        lambda sig, handler: installed.append((sig, handler)),
    )

    ProcessManager(register_atexit=False)

    assert installed == []


def test_process_manager_signal_latches_global_shutdown_before_cleanup() -> None:
    manager = ProcessManager(register_atexit=False)
    observed: list[bool] = []
    manager.cleanup = lambda **_kwargs: observed.append(  # type: ignore[method-assign]
        shutdown_coordinator.is_shutdown_requested()
    )

    manager._signal_handler(signal.SIGTERM, None)

    assert observed == [True]
    assert manager.shutdown_event.is_set()


def test_process_manager_cleanup_runs_once_even_when_stop_event_is_already_set() -> None:
    manager = ProcessManager(register_atexit=False, cleanup_timeout_s=1.0)
    calls: list[tuple[bool, float | None]] = []

    class _Process:
        def stop(self, force: bool = False, *, timeout_s: float | None = None) -> bool:
            calls.append((force, timeout_s))
            return True

    manager.processes["worker"] = _Process()  # type: ignore[assignment]
    manager.shutdown_event.set()

    first = manager.cleanup(timeout_s=1.0)
    second = manager.cleanup(timeout_s=1.0)

    assert first["status"] == "complete"
    assert second == first
    assert calls == [(True, pytest.approx(1.0, abs=0.1))]
    assert manager._cleanup_complete.is_set()


def test_process_manager_monitor_never_restarts_after_global_latch() -> None:
    manager = ProcessManager(register_atexit=False)
    process = SimpleNamespace(
        stats=SimpleNamespace(restarts=0),
        config=SimpleNamespace(max_restarts=1),
        state=ProcessState.RUNNING,
        get_status=lambda: (_ for _ in ()).throw(
            AssertionError("shutdown check must precede status/restart work")
        ),
    )
    manager.processes["worker"] = process

    shutdown_coordinator.request_shutdown("unit-test")
    manager._check_all_processes()

    assert process.state == ProcessState.RUNNING


@pytest.mark.asyncio
async def test_managed_process_start_refuses_before_process_factory_after_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = ManagedProcess(ProcessConfig(name="late-worker", target=_target))

    def _must_not_construct(*_args, **_kwargs):
        raise AssertionError("process factory reached after shutdown latch")

    monkeypatch.setattr(process_manager_module.mp, "Process", _must_not_construct)
    shutdown_coordinator.request_shutdown("unit-test")

    assert await managed.start() is False
    assert managed.state == ProcessState.STOPPED


def test_actor_spawn_is_reaped_when_shutdown_crosses_process_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.supervisor import tree as tree_module
    from core.supervisor.tree import ActorSpec, SupervisionTree

    class _Pipe:
        def close(self) -> None:
            return None

    class _Process:
        pid = None
        exitcode = None

        def __init__(self) -> None:
            self.alive = False
            self.terminated = False
            self.closed = False

        def start(self) -> None:
            self.alive = True
            shutdown_coordinator.request_shutdown("crossed-actor-start")

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

        def kill(self) -> None:
            self.alive = False

        def join(self, timeout: float | None = None) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    process = _Process()

    class _Context:
        def Pipe(self, *, duplex: bool):  # noqa: N802 - multiprocessing test double
            assert duplex is False
            return _Pipe(), _Pipe()

        def Process(self, **_kwargs):  # noqa: N802 - multiprocessing test double
            return process

    monkeypatch.setattr(tree_module.multiprocessing, "get_context", lambda _kind: _Context())
    tree = SupervisionTree()
    tree.add_actor(ActorSpec(name="worker", entry_point=lambda *_args: None))

    assert tree.start_actor("worker") is None
    assert process.terminated is True
    assert process.closed is True
    assert tree._actors["worker"].process is None
    assert tree._actors["worker"].next_restart_time == 0.0


@pytest.mark.asyncio
async def test_sensory_spawn_is_reaped_when_shutdown_crosses_process_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.senses import sensory_client as sensory_module

    class _Queue:
        def put(self, _item) -> None:
            return None

        def empty(self) -> bool:
            return True

        def close(self) -> None:
            return None

        def join_thread(self) -> None:
            return None

    class _Process:
        pid = 5432

        def __init__(self) -> None:
            self.alive = False
            self.terminated = False

        def start(self) -> None:
            self.alive = True
            shutdown_coordinator.request_shutdown("crossed-sensory-start")

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float | None = None) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

    process = _Process()

    class _Context:
        def Queue(self):  # noqa: N802 - multiprocessing test double
            return _Queue()

        def Process(self, **_kwargs):  # noqa: N802 - multiprocessing test double
            return process

    monkeypatch.setattr(sensory_module.mp, "get_context", lambda _kind: _Context())
    client = sensory_module.SensoryLocalClient()

    assert await client.start() is False
    assert process.terminated is True
    assert client._process is None


def test_vision_spawn_is_reaped_when_shutdown_crosses_process_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.brain.llm import mlx_vision_client as vision_module

    class _Queue:
        def close(self) -> None:
            return None

        def join_thread(self) -> None:
            return None

    class _Process:
        pid = 6543
        name = "MLX-Vision-Worker"

        def __init__(self) -> None:
            self.alive = False
            self.terminated = False

        def start(self) -> None:
            self.alive = True
            shutdown_coordinator.request_shutdown("crossed-vision-start")

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True
            self.alive = False

        def kill(self) -> None:
            self.alive = False

        def join(self, timeout: float | None = None) -> None:
            return None

    process = _Process()

    class _Context:
        def Queue(self, maxsize: int = 0):  # noqa: N802 - multiprocessing test double
            return _Queue()

        def Process(self, **_kwargs):  # noqa: N802 - multiprocessing test double
            return process

    monkeypatch.setattr(vision_module.mp, "get_context", lambda _kind: _Context())
    client = vision_module.MLXVisionClient("/models/vision-test")

    assert client.start() is False
    assert process.terminated is True
    assert client._process is None


@pytest.mark.asyncio
async def test_shadow_validation_refuses_before_queue_or_process_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.kernel import shadow_kernel

    def _must_not_create(*_args, **_kwargs):
        raise AssertionError("shadow process resources created after shutdown")

    monkeypatch.setattr(shadow_kernel.multiprocessing, "Queue", _must_not_create)
    instance = shadow_kernel.ShadowExecutionPhase.__new__(
        shadow_kernel.ShadowExecutionPhase
    )
    shutdown_coordinator.request_shutdown("unit-test")

    result = await instance._validate_mutation_detailed("x = 1", "result = True")

    assert result == (False, "runtime_shutdown")


@pytest.mark.asyncio
async def test_boot_and_readiness_health_surfaces_report_stopping() -> None:
    from interface.routes import system

    shutdown_coordinator.request_shutdown("health-surface-test")

    boot_payload, boot_status = system._build_boot_health_payload_sync(
        is_gui_proxy=False
    )
    ready_response = await system.readyz(None)

    assert boot_status == 503
    assert boot_payload["status"] == "stopping"
    assert boot_payload["blockers"] == ["runtime_shutdown"]
    assert boot_payload["shutdown"]["request"]["first_reason"] == (
        "health-surface-test"
    )
    assert ready_response.status_code == 503


@pytest.mark.asyncio
async def test_cognitive_daemon_stop_latches_before_owner_cleanup_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from core.ops import daemon as daemon_module

    observed: list[bool] = []

    class _Orchestrator:
        async def stop(self) -> None:
            observed.append(shutdown_coordinator.is_shutdown_requested())

    monkeypatch.setattr(daemon_module, "DAEMON_PID_FILE", tmp_path / "daemon.pid")
    monkeypatch.setattr(daemon_module, "DAEMON_SOCKET", tmp_path / "daemon.sock")
    daemon = daemon_module.CognitiveDaemon()
    daemon.orchestrator = _Orchestrator()

    await daemon.stop()
    await daemon.stop()

    assert observed == [True]
    assert daemon.orchestrator is None
    assert daemon._stopped is True
    assert daemon._stop_event.is_set()


@pytest.mark.asyncio
async def test_cognitive_daemon_reaps_socket_created_across_shutdown_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from core.ops import daemon as daemon_module

    class _Server:
        def __init__(self) -> None:
            self.closed = False
            self.waited = False

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            self.waited = True

    server = _Server()

    async def _start_unix_server(*_args, **_kwargs):
        shutdown_coordinator.request_shutdown("crossed-daemon-socket-start")
        return server

    monkeypatch.setattr(daemon_module, "DAEMON_SOCKET", tmp_path / "daemon.sock")
    monkeypatch.setattr(daemon_module.asyncio, "start_unix_server", _start_unix_server)
    daemon = daemon_module.CognitiveDaemon()

    with pytest.raises(RuntimeError, match="runtime_shutdown"):
        await daemon._start_socket_server()

    assert server.closed is True
    assert server.waited is True
    assert daemon._socket_server is None
