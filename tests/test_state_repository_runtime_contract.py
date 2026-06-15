import asyncio
import logging
from types import SimpleNamespace

import pytest

import core.state.state_repository as state_module
from core.state.state_repository import StateRepository, _schedule_state_task


class ClosingAwaitable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def __await__(self):
        if False:
            yield None
        return None


class FailingTracker:
    def create_task(self, _awaitable, *, name=None):
        self.last_name = name
        raise RuntimeError(f"{name}: loop unavailable")


def test_state_scheduler_closes_unscheduled_awaitable():
    awaitable = ClosingAwaitable()

    task = _schedule_state_task(awaitable, name="state.contract", tracker=FailingTracker())

    assert task is None
    assert awaitable.closed is True


def test_shutdown_commit_bus_quieting_is_limited_to_state_vault_shutdown() -> None:
    from core.bus.actor_bus import _is_shutdown_commit_request as actor_shutdown_commit
    from core.bus.local_pipe_bus import _is_shutdown_commit_request as local_shutdown_commit

    payload = {"state": {"version": 1}, "cause": "shutdown", "trace_id": "shutdown-test"}

    assert local_shutdown_commit("commit", payload) is True
    assert actor_shutdown_commit("state_vault", "commit", payload) is True
    assert local_shutdown_commit("commit", {"cause": "foreground_commit"}) is False
    assert actor_shutdown_commit("state_vault", "commit", {"cause": "foreground_commit"}) is False
    assert actor_shutdown_commit("other_actor", "commit", payload) is False
    assert actor_shutdown_commit("state_vault", "get_state", payload) is False


@pytest.mark.asyncio
async def test_state_repair_reports_deferred_consumer_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(state_module, "get_task_tracker", lambda: FailingTracker())
    repo = StateRepository(db_path=str(tmp_path / "state.db"), is_vault_owner=True)
    repo._is_processing = True

    result = await repo.repair_runtime()

    assert result["actions"] == ["consumer_restart_deferred", "reconnected_db"]
    assert result["status"]["local_consumer_alive"] is False

    await repo.close()


@pytest.mark.asyncio
async def test_state_repository_rollback_self_governs_recovery_commit(monkeypatch):
    from core.governance_context import get_active_governance
    from core.state.aura_state import AuraState

    previous = AuraState.default()
    current = previous.derive("current", origin="test")
    repo = StateRepository(is_vault_owner=True)
    repo._current = current
    observed_tokens = []

    async def fake_history(limit=2):
        return [current, previous][:limit]

    async def fake_commit(_state, _serialized):
        token = get_active_governance()
        observed_tokens.append(token)

    monkeypatch.setattr(repo, "get_history", fake_history)
    monkeypatch.setattr(repo, "_commit_to_db", fake_commit)

    result = await repo.rollback("recovery: timeout")

    assert result is repo._current
    assert observed_tokens
    assert observed_tokens[0] is not None
    assert observed_tokens[0].domain == "state_mutation"
    assert observed_tokens[0].source == "state_repository.rollback"


@pytest.mark.asyncio
async def test_shutdown_proxy_commit_deferral_logs_as_lifecycle_event(tmp_path, caplog):
    repo = StateRepository(db_path=str(tmp_path / "state.db"), is_vault_owner=False)
    payload = {"state": {"version": 1}, "cause": "shutdown", "trace_id": "shutdown-test"}

    with caplog.at_level(logging.INFO, logger=state_module.logger.name):
        await repo._defer_proxy_commit(payload, BrokenPipeError("pipe closed"))

    messages = [record.getMessage() for record in caplog.records]
    assert any("Graceful-shutdown state commit stored for boot replay" in msg for msg in messages)
    assert not any("Deferred proxy commit for replay" in msg for msg in messages)

    await repo.close()


@pytest.mark.asyncio
async def test_shutdown_proxy_commit_bus_degraded_defers_instead_of_raising(tmp_path, caplog):
    from core.bus.actor_bus import BusDegraded

    class DegradedTransport:
        def __init__(self):
            self.calls = 0

        async def request(self, *_args, **_kwargs):
            self.calls += 1
            raise BusDegraded("Bus degraded or congested for state_vault")

    repo = StateRepository(db_path=str(tmp_path / "state.db"), is_vault_owner=False)
    payload = {"state": {"version": 1}, "cause": "shutdown", "trace_id": "shutdown-test"}
    transport_probe = DegradedTransport()

    with caplog.at_level(logging.INFO, logger=state_module.logger.name):
        ok, transport, error = await repo._send_proxy_commit_request(transport_probe, payload)
        await repo._defer_proxy_commit(payload, error)

    assert ok is False
    assert isinstance(error, BusDegraded)
    assert transport is None
    assert transport_probe.calls == 1
    messages = [record.getMessage() for record in caplog.records]
    assert not any("Vault transport closed during shutdown" in msg for msg in messages)
    assert any("Graceful-shutdown state commit stored for boot replay" in msg for msg in messages)

    await repo.close()


@pytest.mark.asyncio
async def test_state_vault_actor_closes_repository_on_shutdown(monkeypatch):
    import core.bus.local_pipe_bus as local_pipe_bus_module
    from core.state.vault import StateVaultActor

    class FakeBus:
        def __init__(self, *args, **kwargs):
            self._is_running = True
            self.handlers = {}
            self.stopped = False
            self.started = False
            self.sent = []

        def register_handler(self, name, handler):
            self.handlers[name] = handler

        def start(self):
            self.started = True

        async def send(self, *_args, **_kwargs):
            self.sent.append((_args, _kwargs))
            return None

        async def stop(self):
            self.stopped = True
            self._is_running = False

    class FakeRepo:
        def __init__(self, shm):
            self._shm = shm
            self.initialized = False
            self.closed = False

        async def initialize(self):
            self.initialized = True
            actor._is_running = False

        async def close(self):
            self.closed = True
            self._shm.close()

    class FakeShm:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(local_pipe_bus_module, "LocalPipeBus", FakeBus)

    actor = StateVaultActor.__new__(StateVaultActor)
    actor.db_path = "unused.db"
    shm = FakeShm()
    actor.repo = FakeRepo(shm)
    actor.shm_transport = shm
    actor._is_running = False
    actor._bus = None
    actor._heartbeat_interval = 0.01
    actor._heartbeat_task = None
    actor._background_tasks = set()

    await StateVaultActor.run(actor, pipe=None)

    assert actor.repo.initialized is True
    assert actor.repo.closed is True
    assert actor._bus.stopped is True
    assert shm.closed is True


@pytest.mark.asyncio
async def test_state_vault_stop_handler_wakes_actor_loop_immediately():
    from core.state.vault import StateVaultActor

    actor = StateVaultActor.__new__(StateVaultActor)
    actor._is_running = True
    actor._stop_event = asyncio.Event()

    result = await StateVaultActor._process_stop_bus(actor, {}, "trace-stop")

    assert result == {"ok": True, "stopping": True}
    assert actor._is_running is False
    assert actor._stop_event.is_set() is True


def test_state_vault_hard_exit_guard_is_disabled_for_tests(monkeypatch):
    import core.state.vault as vault_module

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "state-vault-test")
    monkeypatch.delenv("AURA_DISABLE_VAULT_HARD_EXIT", raising=False)

    assert vault_module._should_force_vault_process_exit() is False


def test_state_vault_hard_exit_guard_respects_disable_env(monkeypatch):
    import core.state.vault as vault_module

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("AURA_DISABLE_VAULT_HARD_EXIT", "1")

    assert vault_module._should_force_vault_process_exit() is False


def test_state_vault_process_exit_uses_os_exit_in_live_child(monkeypatch):
    import core.state.vault as vault_module

    calls = []
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("AURA_DISABLE_VAULT_HARD_EXIT", raising=False)
    monkeypatch.setattr(vault_module.os, "_exit", lambda code: calls.append(code))

    vault_module._finalize_vault_process_exit(7)

    assert calls == [7]


def test_state_queue_coalescing_is_bounded_and_keeps_latest(tmp_path):
    repo = StateRepository(db_path=str(tmp_path / "state.db"), is_vault_owner=True)
    repo._mutation_queue.put_nowait({"version": 1})
    repo._mutation_queue.put_nowait({"version": 2})
    repo._mutation_queue.put_nowait({"version": 3})

    dropped = repo._coalesce_pending_mutations(keep_latest=True)

    assert dropped == 2
    assert repo._mutation_queue.qsize() == 1
    assert repo._mutation_queue.get_nowait() == {"version": 3}


def test_state_repository_liveness_requires_owner_state_db_and_consumer(tmp_path):
    repo = StateRepository(db_path=str(tmp_path / "state.db"), is_vault_owner=True)
    assert repo.is_initialized() is False

    repo._current = object()
    repo._db = object()
    repo._consumer_task = SimpleNamespace(done=lambda: False)

    assert repo.is_initialized() is True


def test_state_repository_liveness_requires_proxy_transport_path():
    repo = StateRepository(is_vault_owner=False)
    repo._current = object()

    assert repo.is_initialized() is False

    repo._shm = object()

    assert repo.is_initialized() is True
