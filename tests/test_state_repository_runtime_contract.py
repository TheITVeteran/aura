import asyncio
import json
import logging
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

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


def _allowing_state_gate():
    async def approve_state_mutation(*_args, **_kwargs):
        decision = SimpleNamespace(
            receipt_id="state-test-receipt",
            domain="state_mutation",
            source="state-test",
            constraints={},
        )
        return True, "approved_by_test", decision

    return SimpleNamespace(
        approve_state_mutation=approve_state_mutation,
        record_external_decision=lambda **_kwargs: None,
    )


def test_state_scheduler_closes_unscheduled_awaitable():
    awaitable = ClosingAwaitable()

    task = _schedule_state_task(awaitable, name="state.contract", tracker=FailingTracker())

    assert task is None
    assert awaitable.closed is True


@pytest.mark.asyncio
async def test_strict_owner_commit_queues_real_transition_without_dummy_gateway_marker(
    monkeypatch,
) -> None:
    from core.state.aura_state import AuraState

    repo = StateRepository(is_vault_owner=True)
    repo._current = AuraState.default()
    enqueue = AsyncMock()
    monkeypatch.setattr(repo, "_enqueue_owner_commit", enqueue)
    monkeypatch.setenv("AURA_STRICT_RUNTIME", "1")
    monkeypatch.setattr(
        "core.state.state_gateway.get_state_gateway",
        lambda: (_ for _ in ()).throw(AssertionError("dummy gateway marker is forbidden")),
    )
    candidate = repo._current.derive("real-transition", origin="test")

    returned = await repo.commit(candidate, "real-transition", trace_id="trace-real")

    assert returned is candidate
    enqueue.assert_awaited_once()
    payload = enqueue.await_args.args[0]
    assert payload["state"] is candidate
    assert payload["cause"] == "real-transition"
    assert payload["trace_id"] == "trace-real"


@pytest.mark.asyncio
async def test_state_commit_admission_failure_never_publishes_or_persists(
    monkeypatch,
) -> None:
    from core.state.aura_state import AuraState

    async def fail_admission(*_args, **_kwargs):
        raise RuntimeError("constitution unavailable")

    gate = SimpleNamespace(approve_state_mutation=fail_admission)
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda: gate)
    repo = StateRepository(is_vault_owner=True)
    original = AuraState.default()
    repo._current = original
    repo._shm = object()
    repo._commit_to_db = AsyncMock()
    repo._sync_to_shm = AsyncMock()
    candidate = original.derive("rejected", origin="test")

    committed = await repo._process_commit(candidate, "rejected")

    assert committed is False
    assert repo._current is original
    assert repo.get_runtime_status()["failed_commit_count"] == 1
    assert "governance_unavailable:RuntimeError" in repo.get_runtime_status()[
        "last_commit_error"
    ]
    assert repo.get_runtime_status()["last_commit_at"] == 0.0
    repo._commit_to_db.assert_not_awaited()
    repo._sync_to_shm.assert_not_awaited()


@pytest.mark.asyncio
async def test_state_commit_persistence_failure_keeps_previous_visible_state(
    monkeypatch,
) -> None:
    from core.state.aura_state import AuraState

    gate = _allowing_state_gate()
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda: gate)
    repo = StateRepository(is_vault_owner=True)
    original = AuraState.default()
    repo._current = original
    repo._shm = object()
    repo._commit_to_db = AsyncMock(side_effect=OSError("disk unavailable"))
    repo._sync_to_shm = AsyncMock()
    candidate = original.derive("not-durable", origin="test")

    with pytest.raises(OSError, match="disk unavailable"):
        await repo._process_commit(candidate, "not-durable")

    assert repo._current is original
    assert repo.get_runtime_status()["failed_commit_count"] == 1
    assert "persistence_failed:OSError" in repo.get_runtime_status()["last_commit_error"]
    assert repo.get_runtime_status()["last_commit_at"] == 0.0
    repo._sync_to_shm.assert_not_awaited()


@pytest.mark.asyncio
async def test_state_commit_persists_before_shm_and_memory_publication(monkeypatch) -> None:
    from core.state.aura_state import AuraState

    gate = _allowing_state_gate()
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda: gate)
    repo = StateRepository(is_vault_owner=True)
    original = AuraState.default()
    repo._current = original
    repo._shm = object()
    events: list[str] = []

    async def persist(state, serialized):
        assert repo._current is original
        payload = json.loads(serialized)
        assert payload["transition_cause"] == "durable"
        assert payload["updated_at"] == state.updated_at
        events.append("db")

    async def publish_shm(*_args):
        assert repo._current is original
        events.append("shm")

    repo._commit_to_db = persist
    repo._sync_to_shm = publish_shm
    candidate = original.derive("durable", origin="test")

    committed = await repo._process_commit(candidate, "durable")

    assert committed is True
    assert events == ["db", "shm"]
    assert repo._current is candidate
    assert repo.get_runtime_status()["failed_commit_count"] == 0
    assert repo.get_runtime_status()["last_commit_error"] == ""
    assert repo.get_runtime_status()["last_commit_at"] > 0.0


@pytest.mark.asyncio
async def test_state_commit_transactions_cannot_interleave(monkeypatch) -> None:
    from core.state.aura_state import AuraState

    gate = _allowing_state_gate()
    monkeypatch.setattr("core.constitution.get_constitutional_core", lambda: gate)
    repo = StateRepository(is_vault_owner=True)
    original = AuraState.default()
    first = original.derive("first", origin="test")
    second = first.derive("second", origin="test")
    repo._current = original
    entered_first = asyncio.Event()
    release_first = asyncio.Event()
    persisted: list[int] = []

    async def persist(state, _serialized):
        persisted.append(state.version)
        if state is first:
            entered_first.set()
            await release_first.wait()

    repo._commit_to_db = persist
    first_task = asyncio.create_task(repo._process_commit(first, "first"))
    await asyncio.wait_for(entered_first.wait(), timeout=1.0)
    second_task = asyncio.create_task(repo._process_commit(second, "second"))
    await asyncio.sleep(0)

    assert persisted == [first.version]
    assert repo._current is original

    release_first.set()
    first_result, second_result = await asyncio.gather(first_task, second_task)

    assert first_result is True
    assert second_result is True
    assert persisted == [first.version, second.version]
    assert repo._current is second


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


def test_state_repository_sanitizes_only_explicit_restored_snapshot():
    from core.state.aura_state import AuraState

    prompt = "Ok. Once more. You with me?"
    state = AuraState.default()
    state.cognition.current_objective = prompt
    state.cognition.current_origin = "api"
    state.cognition.active_goals = [
        {
            "description": prompt,
            "source": "executive_closure",
            "metadata": {"foreground_turn": True},
        }
    ]
    repo = StateRepository(is_vault_owner=True)
    repo._current = state

    assert repo._sanitize_restored_state() is True
    assert state.cognition.current_objective is None
    assert state.cognition.active_goals == []


def test_state_repository_reuses_aiosqlite_connection_across_event_loops(tmp_path):
    """aiosqlite 0.20+ is loop-agnostic; access must not spawn worker churn."""

    repo = StateRepository(db_path=str(tmp_path / "state.db"), is_vault_owner=True)

    first = asyncio.run(repo._ensure_db())
    second = asyncio.run(repo._ensure_db())

    assert second is first
    asyncio.run(repo.close())
    assert repo._db is None


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
async def test_shutdown_direct_snapshot_does_not_reopen_aiosqlite_worker(
    monkeypatch,
    tmp_path,
):
    from core.state.aura_state import AuraState

    repo = StateRepository(db_path=str(tmp_path / "state.db"), is_vault_owner=False)
    state = AuraState.default().derive("shutdown", origin="system")

    async def forbidden_async_db_open():
        raise RuntimeError("runtime_shutdown")

    monkeypatch.setattr(repo, "_ensure_db", forbidden_async_db_open)

    committed = await repo._commit_shutdown_direct_snapshot(
        state,
        BrokenPipeError("vault transport closed"),
    )

    assert committed is True
    with sqlite3.connect(repo.db_path) as connection:
        row = connection.execute(
            "SELECT transition_cause FROM state_log ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        outbox_count = connection.execute(
            "SELECT COUNT(*) FROM proxy_commit_outbox"
        ).fetchone()
    assert row == ("shutdown",)
    assert outbox_count == (0,)


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
