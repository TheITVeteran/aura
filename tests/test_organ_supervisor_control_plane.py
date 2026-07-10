from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.runtime.control_plane import (
    ObservedServiceState,
    PressureSnapshot,
    ResourceAdmissionController,
    RuntimeControlPlane,
)
from core.runtime.organ_supervisor import OrganSupervisor, RestartPolicy


def _plane() -> RuntimeControlPlane:
    return RuntimeControlPlane(
        admission=ResourceAdmissionController(
            pressure_provider=lambda: PressureSnapshot(memory_percent=40.0),
        )
    )


class _Process:
    def __init__(self) -> None:
        self.pid = 1234
        self.returncode = None
        self.signals = []
        self.killed = False

    def send_signal(self, signal) -> None:
        self.signals.append(signal)

    async def wait(self) -> int:
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


@pytest.mark.asyncio
async def test_command_organ_lifecycle_is_owned_by_control_plane(monkeypatch, tmp_path):
    from core.runtime import organ_supervisor

    process = _Process()

    class _Gateway:
        async def spawn_async(self, *_args, **_kwargs):
            return process

    monkeypatch.setattr(
        organ_supervisor,
        "get_subprocess_gateway",
        lambda: _Gateway(),
    )
    plane = _plane()
    supervisor = OrganSupervisor(control_plane=plane)
    supervisor.register_organ(
        "voice",
        cmd=["python", "voice.py"],
        cwd=str(tmp_path),
    )

    initial = plane.service_status()["organ:voice"]
    assert initial["desired_state"] == "stopped"
    assert initial["observed_state"] == "stopped"
    assert not hasattr(supervisor, "_watchdog_task")

    started = await supervisor.start_all()
    assert started["services"]["organ:voice"]["observed_state"] == "ready"
    assert supervisor.health()["voice"]["alive"] is True

    stopped = await supervisor.stop_all()
    assert stopped["services"]["organ:voice"]["observed_state"] == "stopped"
    assert process.signals
    assert process.returncode == 0
    assert supervisor.health()["voice"]["alive"] is False


@pytest.mark.asyncio
async def test_failed_organ_start_uses_control_plane_circuit(monkeypatch):
    from core.runtime import organ_supervisor

    class _Gateway:
        async def spawn_async(self, *_args, **_kwargs):
            raise OSError("spawn failed")

    monkeypatch.setattr(
        organ_supervisor,
        "get_subprocess_gateway",
        lambda: _Gateway(),
    )
    plane = _plane()
    supervisor = OrganSupervisor(control_plane=plane)
    supervisor.register_organ(
        "motor",
        cmd=["motor"],
        policy=RestartPolicy(max_restarts=1),
    )

    report = await supervisor.start_all()

    status = report["services"]["organ:motor"]
    assert status["observed_state"] == ObservedServiceState.CIRCUIT_OPEN.value
    assert status["last_error"] == "spawn failed"
    assert supervisor.get_status()["summary"]["open_circuits"] == 1


def test_organ_registration_rejects_ambiguous_or_unsafe_contracts():
    supervisor = OrganSupervisor(control_plane=_plane())

    with pytest.raises(ValueError, match="organ name"):
        supervisor.register_organ("../escape", cmd=["worker"])
    with pytest.raises(ValueError, match="command must be non-empty"):
        supervisor.register_organ("empty", cmd=[])

    supervisor.register_organ("voice", cmd=["worker"])
    with pytest.raises(ValueError, match="already registered"):
        supervisor.register_organ("voice", cmd=["other"])


@pytest.mark.asyncio
async def test_organ_ipc_rejects_oversized_request_before_socket_open(monkeypatch):
    monkeypatch.setenv("AURA_ORGAN_IPC_MAX_BYTES", "1024")
    supervisor = OrganSupervisor(control_plane=_plane())
    supervisor.register_organ("voice", cmd=["worker"])
    supervisor._organs["voice"].proc = SimpleNamespace(returncode=None, pid=1)

    with pytest.raises(ValueError, match="exceeds frame limit"):
        await supervisor.ipc_call("voice", {"payload": "x" * 2048})


def test_restart_policy_rejects_invalid_envelopes():
    with pytest.raises(ValueError, match="max_restarts"):
        RestartPolicy(max_restarts=-1)
    with pytest.raises(ValueError, match="window_s"):
        RestartPolicy(window_s=0)
    with pytest.raises(ValueError, match="backoff_factor"):
        RestartPolicy(backoff_factor=0.5)


def test_aura_main_resolves_canonical_actor_supervision_singleton(monkeypatch):
    import aura_main
    from core.supervisor.tree import get_tree

    monkeypatch.setattr(aura_main, "_supervisor_tree", None)

    assert aura_main.get_supervisor_tree() is get_tree()


def test_actor_spawn_failure_closes_pipes_and_enters_bounded_backoff(monkeypatch):
    import core.supervisor.tree as tree_module
    from core.supervisor.tree import ActorSpec, SupervisionTree

    class _Pipe:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _Process:
        pid = None

        def __init__(self) -> None:
            self.closed = False

        def start(self) -> None:
            raise OSError("spawn unavailable")

        def close(self) -> None:
            self.closed = True

    class _Context:
        def __init__(self) -> None:
            self.pipes: list[tuple[_Pipe, _Pipe]] = []
            self.process = _Process()

        def Pipe(self, *, duplex: bool):
            assert duplex is False
            pair = (_Pipe(), _Pipe())
            self.pipes.append(pair)
            return pair

        def Process(self, **_kwargs):
            return self.process

    context = _Context()
    monkeypatch.setattr(tree_module.multiprocessing, "get_context", lambda _kind: context)
    tree = SupervisionTree()
    tree.add_actor(
        ActorSpec(
            name="worker",
            entry_point=lambda *_args: None,
            restart_delay=0.1,
        )
    )

    assert tree.start_actor("worker") is None

    actor = tree._actors["worker"]
    assert actor.process is None
    assert actor.pipe is None
    assert actor.desired_running is True
    assert actor.consecutive_failures == 1
    assert actor.next_restart_time > 0
    assert actor.last_error == "spawn unavailable"
    assert context.process.closed is True
    assert all(endpoint.closed for pair in context.pipes for endpoint in pair)


def test_actor_spec_rejects_invalid_recovery_envelope():
    from core.supervisor.tree import ActorSpec

    with pytest.raises(ValueError, match="restart_policy"):
        ActorSpec(name="worker", entry_point=lambda: None, restart_policy="sometimes")
    with pytest.raises(ValueError, match="max_restarts"):
        ActorSpec(name="worker", entry_point=lambda: None, max_restarts=-1)
    with pytest.raises(ValueError, match="backoff_factor"):
        ActorSpec(name="worker", entry_point=lambda: None, backoff_factor=0.5)


def test_stale_organ_socket_cleanup_uses_file_gateway(monkeypatch, tmp_path):
    from core.runtime import organ_supervisor

    class _Gateway:
        def __init__(self) -> None:
            self.calls = []

        def delete_file(self, path, *, source):
            self.calls.append((path, source))
            path.unlink()
            return True

    gateway = _Gateway()
    monkeypatch.setattr(organ_supervisor, "_SOCK_DIR", tmp_path)
    monkeypatch.setattr(organ_supervisor, "get_file_write_gateway", lambda: gateway)
    stale = tmp_path / f"aura-{organ_supervisor.os.getpid()}-voice.sock"
    stale.write_text("stale", encoding="utf-8")

    OrganSupervisor(control_plane=_plane()).register_organ("voice", cmd=["worker"])

    assert not stale.exists()
    assert gateway.calls == [
        (stale, "runtime.organ_supervisor.stale_socket"),
    ]


@pytest.mark.asyncio
async def test_only_one_actor_supervision_monitor_can_be_live(monkeypatch):
    import core.supervisor.tree as tree_module
    from core.runtime.control_plane import reset_runtime_control_plane
    from core.supervisor.tree import SupervisionTree

    monkeypatch.setattr(tree_module.multiprocessing, "active_children", lambda: [])
    reset_runtime_control_plane()
    first = SupervisionTree()
    second = SupervisionTree()
    try:
        await first.start()
        with pytest.raises(RuntimeError, match="already owns"):
            await second.start()
        assert first.is_alive() is True
        assert second.is_alive() is False
    finally:
        await first.stop()
        reset_runtime_control_plane()


@pytest.mark.asyncio
async def test_actor_monitor_start_failure_releases_singleton_claim(monkeypatch):
    import core.supervisor.tree as tree_module
    from core.supervisor.tree import SupervisionTree

    class _Tracker:
        def create_task(self, *_args, **_kwargs):
            raise RuntimeError("task ownership unavailable")

    monkeypatch.setattr(tree_module, "get_task_tracker", lambda: _Tracker())
    tree = SupervisionTree()

    with pytest.raises(RuntimeError, match="task ownership unavailable"):
        await tree.start()

    assert tree.is_alive() is False
    assert SupervisionTree._active_instance is None
