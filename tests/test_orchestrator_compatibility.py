import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from core.orchestrator.handlers.shutdown import _gracefully_stop_actor_via_bus, orchestrator_shutdown
from core.runtime.errors import get_degradation_tracker
from core.orchestrator.mixins.boot.boot_cognitive import BootCognitiveMixin
from core.orchestrator.mixins.boot.boot_resilience import BootResilienceMixin
from core.orchestrator.mixins.output_formatter import OutputFormatterMixin


class _BootProbe(BootCognitiveMixin):
    cognition = None


class _ResilienceProbe(BootResilienceMixin):
    _actor_bus = None
    actor_bus = None
    supervisor = None
    reply_queue = True
    output_gate = None
    emit_spontaneous_message = None

    def __init__(self):
        self.status = SimpleNamespace(temporal_drift_s=0.0)


class _OutputProbe(OutputFormatterMixin):
    def __init__(self, emitter):
        self.cognitive_engine = SimpleNamespace(_emit_thought=emitter)


class AsyncCallFixture:
    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.side_effect is not None:
            result = self.side_effect(*args, **kwargs)
            if hasattr(result, "__await__"):
                return await result
            return result
        return self.return_value

    @property
    def await_args(self):
        return self.calls[-1] if self.calls else None

    @property
    def await_count(self):
        return len(self.calls)

    def assert_awaited_once(self):
        assert len(self.calls) == 1

    def assert_awaited_once_with(self, *args, **kwargs):
        assert self.calls == [(args, kwargs)]

    def assert_not_called(self):
        assert self.calls == []


@pytest.mark.asyncio
async def test_init_cognitive_core_awaits_async_setup(monkeypatch):
    setup = AsyncCallFixture()
    cognitive_engine = SimpleNamespace(setup=setup)
    capability_engine = object()

    def _get(name, default=None):
        if name == "cognitive_engine":
            return cognitive_engine
        if name == "capability_engine":
            return capability_engine
        return default

    monkeypatch.setattr(
        "core.orchestrator.mixins.boot.boot_cognitive.ServiceContainer.get",
        staticmethod(_get),
    )

    await _BootProbe()._init_cognitive_core()

    setup.assert_awaited_once_with(registry=capability_engine, router=capability_engine)


@pytest.mark.asyncio
async def test_emit_thought_stream_schedules_async_emitters():
    emitter = AsyncCallFixture()
    probe = _OutputProbe(emitter)

    probe._emit_thought_stream("hello")
    await asyncio.sleep(0)

    emitter.assert_awaited_once_with("hello")


@pytest.mark.asyncio
async def test_start_state_vault_actor_uses_actor_bus_request_for_handshake(monkeypatch):
    started = object()

    class _Supervisor:
        def __init__(self):
            self.specs = []

        def add_actor(self, spec):
            self.specs.append(spec)

        def start_actor(self, name):
            assert name == "state_vault"
            return started

    class _Bus:
        def __init__(self):
            self.actors = {}
            self.requests = []

        def has_actor(self, name):
            return name in self.actors

        def add_actor(self, name, connection):
            self.actors[name] = connection

        async def request(self, actor, msg_type, payload, timeout=0):
            self.requests.append((actor, msg_type, payload, timeout))
            return {"type": "pong", "ts": 123.0}

    bus = _Bus()
    supervisor = _Supervisor()

    monkeypatch.setattr(
        "core.orchestrator.mixins.boot.boot_resilience.ServiceContainer.get",
        staticmethod(lambda name, default=None: bus if name == "actor_bus" else default),
    )
    monkeypatch.setattr(
        "core.supervisor.tree.ActorSpec",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )

    probe = _ResilienceProbe()
    probe.supervisor = supervisor

    await probe._start_state_vault_actor()

    assert bus.actors["state_vault"] is started
    assert bus.requests == [
        ("state_vault", "ping", {"source": "boot_resilience", "attempt": 1}, 2.0)
    ]


@pytest.mark.asyncio
async def test_start_state_vault_actor_fallback_ping_uses_request_wire_format(monkeypatch):
    class _Pipe:
        def __init__(self):
            self.sent = []
            self.last_request_id = None

        def send(self, raw):
            payload = json.loads(raw)
            self.sent.append(payload)
            self.last_request_id = payload["request_id"]

        def poll(self, timeout):
            return True

        def recv(self):
            return json.dumps(
                {
                    "response_to": self.last_request_id,
                    "payload": {"type": "pong", "ts": 456.0},
                }
            )

    pipe = _Pipe()

    class _Supervisor:
        def add_actor(self, spec):
            self.spec = spec

        def start_actor(self, name):
            assert name == "state_vault"
            return pipe

    monkeypatch.setattr(
        "core.orchestrator.mixins.boot.boot_resilience.ServiceContainer.get",
        staticmethod(lambda _name, default=None: default),
    )
    monkeypatch.setattr(
        "core.supervisor.tree.ActorSpec",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )

    probe = _ResilienceProbe()
    probe.supervisor = _Supervisor()

    await probe._start_state_vault_actor()

    assert pipe.sent
    assert pipe.sent[0]["type"] == "ping"
    assert pipe.sent[0]["is_request"] is True
    assert pipe.sent[0]["payload"]["source"] == "boot_resilience"


@pytest.mark.asyncio
async def test_start_state_vault_actor_fallback_ping_supports_split_pipe_pairs(monkeypatch):
    class _ReadPipe:
        def __init__(self, write_pipe):
            self.write_pipe = write_pipe

        def poll(self, timeout):
            return True

        def recv(self):
            return json.dumps(
                {
                    "response_to": self.write_pipe.last_request_id,
                    "payload": {"type": "pong", "ts": 789.0},
                }
            )

    class _WritePipe:
        def __init__(self):
            self.sent = []
            self.last_request_id = None

        def send(self, raw):
            payload = json.loads(raw)
            self.sent.append(payload)
            self.last_request_id = payload["request_id"]

    write_pipe = _WritePipe()
    pipe_pair = (_ReadPipe(write_pipe), write_pipe)

    class _Supervisor:
        def add_actor(self, spec):
            self.spec = spec

        def start_actor(self, name):
            assert name == "state_vault"
            return pipe_pair

    monkeypatch.setattr(
        "core.orchestrator.mixins.boot.boot_resilience.ServiceContainer.get",
        staticmethod(lambda _name, default=None: default),
    )
    monkeypatch.setattr(
        "core.supervisor.tree.ActorSpec",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )

    probe = _ResilienceProbe()
    probe.supervisor = _Supervisor()

    await probe._start_state_vault_actor()

    assert write_pipe.sent
    assert write_pipe.sent[0]["type"] == "ping"
    assert write_pipe.sent[0]["is_request"] is True


@pytest.mark.asyncio
async def test_start_state_vault_actor_strict_runtime_fails_when_handshake_never_succeeds(monkeypatch):
    class _Pipe:
        def __init__(self):
            self.sent = []

        def send(self, raw):
            self.sent.append(json.loads(raw))

        def poll(self, timeout):
            return False

    class _Supervisor:
        def add_actor(self, spec):
            self.spec = spec

        def start_actor(self, name):
            assert name == "state_vault"
            return _Pipe()

    sleep_calls = []

    async def _fake_sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setenv("AURA_STRICT_RUNTIME", "1")
    monkeypatch.setattr(
        "core.orchestrator.mixins.boot.boot_resilience.ServiceContainer.get",
        staticmethod(lambda _name, default=None: default),
    )
    monkeypatch.setattr(
        "core.orchestrator.mixins.boot.boot_resilience.asyncio.sleep",
        _fake_sleep,
    )
    monkeypatch.setattr(
        "core.supervisor.tree.ActorSpec",
        lambda **kwargs: SimpleNamespace(**kwargs),
        raising=False,
    )

    probe = _ResilienceProbe()
    probe.supervisor = _Supervisor()

    with pytest.raises(RuntimeError, match="StateVaultActor failed to respond to ping"):
        await probe._start_state_vault_actor()

    assert sleep_calls


@pytest.mark.asyncio
async def test_calculate_temporal_drift_routes_recovery_through_unified_will(tmp_path, monkeypatch):
    from core.orchestrator.mixins.boot import boot_resilience

    heartbeat_path = tmp_path / "heartbeat"
    heartbeat_path.write_text(str(time.time() - 7200.0))

    monkeypatch.setattr(
        boot_resilience.config,
        "paths",
        SimpleNamespace(home_dir=tmp_path),
    )

    probe = _ResilienceProbe()
    probe.emit_spontaneous_message = AsyncCallFixture(
        return_value={"ok": True, "action": "released", "target": "secondary"}
    )
    probe.output_gate = SimpleNamespace(emit=AsyncCallFixture())

    probe._calculate_temporal_drift()
    await asyncio.sleep(0)

    probe.emit_spontaneous_message.assert_awaited_once()
    _, kwargs = probe.emit_spontaneous_message.await_args
    assert kwargs["origin"] == "recovery"
    assert kwargs["metadata"]["visible_presence"] is True
    assert kwargs["metadata"]["trigger"] == "temporal_drift_recovery"
    probe.output_gate.emit.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrator_shutdown_requests_graceful_state_vault_stop_before_bus_stop(monkeypatch):
    class _Bus:
        def __init__(self):
            self.calls = []

        def has_actor(self, name):
            return name == "state_vault"

        async def request(self, actor, msg_type, payload, timeout=0):
            self.calls.append(("request", actor, msg_type, payload, timeout))
            return None

        async def stop(self):
            self.calls.append(("stop",))

    class _Supervisor:
        def __init__(self):
            self.stop_calls = 0
            self._running = True

        def is_actor_running(self, name):
            if name != "state_vault":
                return False
            was_running = self._running
            self._running = False
            return was_running

        async def stop(self):
            self.stop_calls += 1

    bus = _Bus()
    supervisor = _Supervisor()
    state_repo = SimpleNamespace(
        get_current=AsyncCallFixture(return_value=None),
        close=AsyncCallFixture(),
        _transport_has_vault=lambda: True,
        is_vault_owner=False,
    )
    service_shutdown = AsyncCallFixture()
    event_bus_shutdown = AsyncCallFixture()

    monkeypatch.setattr(
        "core.resilience.snapshot_manager.SnapshotManager",
        lambda _orch: SimpleNamespace(freeze=lambda: None),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.shutdown",
        service_shutdown,
    )
    monkeypatch.setattr(
        "core.event_bus.get_event_bus",
        lambda: SimpleNamespace(shutdown=event_bus_shutdown),
    )
    monkeypatch.setattr(
        "core.utils.task_tracker.get_task_tracker",
        lambda: SimpleNamespace(shutdown=lambda timeout=3.0: None),
    )

    orch = SimpleNamespace(
        status=SimpleNamespace(running=True, is_processing=True),
        state_repo=state_repo,
        _actor_bus=bus,
        _supervisor_tree=supervisor,
        _publish_status=lambda _payload: None,
        _save_state=lambda _cause: None,
        _stop_event=None,
        kernel_interface=None,
    )

    await orchestrator_shutdown(orch)

    assert bus.calls[0] == (
        "request",
        "state_vault",
        "stop",
        {"source": "orchestrator_shutdown", "reason": "graceful_shutdown"},
        2.0,
    )
    assert bus.calls[1] == ("stop",)
    state_repo.close.assert_awaited_once()
    service_shutdown.assert_awaited_once()
    event_bus_shutdown.assert_awaited_once()
    assert supervisor.stop_calls == 1


def test_graceful_state_vault_stop_continues_when_bus_already_closed():
    tracker = get_degradation_tracker()
    tracker.reset()

    class _ClosedBus:
        def __init__(self):
            self.request_calls = []

        def has_actor(self, name):
            return name == "state_vault"

        async def request(self, *_args, **_kwargs):
            self.request_calls.append((_args, _kwargs))
            raise BrokenPipeError("Connection is closed")

    closed_bus = _ClosedBus()
    orch = SimpleNamespace(_actor_bus=closed_bus, _supervisor_tree=None)

    asyncio.run(_gracefully_stop_actor_via_bus(orch, "state_vault", stop_budget_s=0.01))

    assert len(closed_bus.request_calls) == 1
    assert any(
        "actor bus was already closed for state_vault" in record.action
        for record in tracker.recent(subsystem="shutdown")
    )
    tracker.reset()


def test_graceful_state_vault_stop_continues_when_actor_bus_is_degraded():
    from core.bus.actor_bus import BusDegraded

    tracker = get_degradation_tracker()
    tracker.reset()

    class _DegradedBus:
        def __init__(self):
            self.request_calls = []

        def has_actor(self, name):
            return name == "state_vault"

        async def request(self, *_args, **_kwargs):
            self.request_calls.append((_args, _kwargs))
            raise BusDegraded("Bus degraded or congested for state_vault")

    degraded_bus = _DegradedBus()
    orch = SimpleNamespace(_actor_bus=degraded_bus, _supervisor_tree=None)

    stopped = asyncio.run(
        _gracefully_stop_actor_via_bus(orch, "state_vault", stop_budget_s=0.01)
    )

    assert stopped is False
    assert len(degraded_bus.request_calls) == 1
    assert not tracker.recent(subsystem="shutdown")
    tracker.reset()


def test_graceful_state_vault_stop_uses_supervisor_when_actor_bus_unusable():
    class _UnusableBus:
        def __init__(self):
            self.request_calls = []

        def has_actor(self, name):
            return name == "state_vault"

        def is_actor_usable(self, name):
            return False

        async def request(self, *_args, **_kwargs):
            self.request_calls.append((_args, _kwargs))

    class _Supervisor:
        def __init__(self):
            self.stop_calls = []

        def stop_actor(self, name, **kwargs):
            self.stop_calls.append((name, kwargs))

    bus = _UnusableBus()
    supervisor = _Supervisor()
    orch = SimpleNamespace(_actor_bus=bus, _supervisor_tree=supervisor)

    stopped = asyncio.run(
        _gracefully_stop_actor_via_bus(orch, "state_vault", stop_budget_s=0.01)
    )

    assert stopped is True
    assert bus.request_calls == []
    assert supervisor.stop_calls[0][0] == "state_vault"


@pytest.mark.asyncio
async def test_orchestrator_shutdown_continues_when_final_state_commit_is_cancelled(
    monkeypatch, caplog
):
    tracker = get_degradation_tracker()
    tracker.reset()

    class _State:
        def derive(self, _cause):
            return self

    cancelled_commits = []

    async def cancelled_commit(*_args, **_kwargs):
        cancelled_commits.append((_args, _kwargs))
        raise asyncio.CancelledError()

    service_shutdown = AsyncCallFixture()
    event_bus_shutdown = AsyncCallFixture()

    monkeypatch.setattr(
        "core.resilience.snapshot_manager.SnapshotManager",
        lambda _orch: SimpleNamespace(freeze=lambda: None),
    )
    monkeypatch.setattr(
        "core.container.ServiceContainer.shutdown",
        service_shutdown,
    )
    monkeypatch.setattr(
        "core.event_bus.get_event_bus",
        lambda: SimpleNamespace(shutdown=event_bus_shutdown),
    )
    monkeypatch.setattr(
        "core.utils.task_tracker.get_task_tracker",
        lambda: SimpleNamespace(shutdown=lambda timeout=3.0: None),
    )

    orch = SimpleNamespace(
        status=SimpleNamespace(running=True, is_processing=True),
        state_repo=SimpleNamespace(
            get_current=AsyncCallFixture(return_value=_State()),
            commit=cancelled_commit,
            close=AsyncCallFixture(),
            _transport_has_vault=lambda: True,
            is_vault_owner=False,
        ),
        _actor_bus=None,
        _supervisor_tree=None,
        _publish_status=lambda _payload: None,
        _save_state=lambda _cause: None,
        _stop_event=None,
        kernel_interface=None,
    )

    with caplog.at_level("DEBUG", logger="Aura.Core.Orchestrator.Shutdown"):
        await orchestrator_shutdown(orch)

    assert "Shutdown state commit cancelled during process teardown" in caplog.text
    assert len(cancelled_commits) == 1
    service_shutdown.assert_awaited_once()
    event_bus_shutdown.assert_awaited_once()
    tracker.reset()
