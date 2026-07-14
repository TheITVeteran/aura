from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest

from core.health.read_model import HealthReadModelConfig, HealthSnapshotReadModel


def _wait_until(predicate, *, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true before timeout")


def _fallback() -> dict[str, object]:
    return {"status": "booting", "healthy": False, "blockers": ["initializing"]}


def test_read_model_normalizes_retry_bounds():
    model = HealthSnapshotReadModel(
        lambda: {"status": "ok"},
        _fallback,
        config=HealthReadModelConfig(
            refresh_interval_s=-1.0,
            max_stale_s=-1.0,
            collection_timeout_s=-1.0,
            retry_base_s=-1.0,
            retry_max_s=-2.0,
        ),
    )

    assert model.config.refresh_interval_s == 0.05
    assert model.config.max_stale_s == 0.05
    assert model.config.collection_timeout_s == 0.05
    assert model.config.retry_base_s == 0.05
    assert model.config.retry_max_s == 0.05


def test_read_model_supports_named_snapshot_identity():
    model = HealthSnapshotReadModel(
        lambda: {"healthy": True},
        lambda: {"healthy": False},
        config=HealthReadModelConfig(
            refresh_interval_s=1.0,
            schema_version="aura.integrity.snapshot.v1",
            metadata_key="integrity_read_model",
            worker_name_prefix="AuraIntegritySnapshot",
            incident_prefix="integrity-refresh",
            log_label="Integrity snapshot",
        ),
    )

    assert model.start() is True
    _wait_until(lambda: model.read().get("healthy") is True)
    payload = model.read()

    assert "health_read_model" not in payload
    assert payload["integrity_read_model"]["schema_version"] == (
        "aura.integrity.snapshot.v1"
    )
    assert payload["integrity_read_model"]["fresh"] is True


def test_read_model_never_joins_blocked_collector_and_keeps_one_worker():
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def collect() -> dict[str, object]:
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(1.0)
        return {"status": "ok", "healthy": True, "blockers": []}

    model = HealthSnapshotReadModel(
        collect,
        _fallback,
        config=HealthReadModelConfig(
            refresh_interval_s=0.05,
            max_stale_s=0.1,
            collection_timeout_s=0.02,
            retry_base_s=0.01,
            retry_max_s=0.05,
        ),
    )

    started_at = time.monotonic()
    first = model.read()
    elapsed = time.monotonic() - started_at
    assert elapsed < 0.05
    assert first["status"] == "booting"
    assert first["health_read_model"]["serving"] == "initializing"
    assert first["health_read_model"]["refresh_in_flight"] is True
    assert started.wait(0.2)

    for _ in range(20):
        assert model.read()["health_read_model"]["refresh_in_flight"] is True
    assert calls == 1

    time.sleep(0.06)
    timed_out = model.read()["health_read_model"]
    assert timed_out["refresh_timed_out"] is True
    assert timed_out["total_timeouts"] == 1
    assert timed_out["consecutive_failures"] == 1
    assert timed_out["incident_id"] == "health-refresh-000001"

    release.set()
    _wait_until(lambda: model.read()["health_read_model"]["fresh"] is True)
    recovered = model.read()["health_read_model"]
    assert recovered["total_refreshes"] == 1
    assert recovered["consecutive_failures"] == 0
    assert recovered["last_recovery"]["incident_id"] == "health-refresh-000001"
    assert recovered["last_recovery"]["failed_refreshes"] == 1


def test_close_and_restart_do_not_overlap_collectors():
    first_started = threading.Event()
    release_first = threading.Event()
    calls = 0

    def collect() -> dict[str, object]:
        nonlocal calls
        calls += 1
        current_call = calls
        if current_call == 1:
            first_started.set()
            assert release_first.wait(1.0)
        return {"status": "ok", "call": current_call}

    model = HealthSnapshotReadModel(collect, _fallback)
    assert model.start() is True
    assert first_started.wait(0.2)

    model.close()
    assert model.start() is False
    assert calls == 1

    release_first.set()
    _wait_until(lambda: model.read().get("call") == 2)
    assert calls == 2


def test_read_model_marks_old_success_expired_during_singleflight_refresh():
    block_refresh = threading.Event()
    refresh_started = threading.Event()
    calls = 0

    def collect() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls > 1:
            refresh_started.set()
            assert block_refresh.wait(1.0)
        return {
            "status": "ok",
            "healthy": True,
            "connected": True,
            "conversation_ready": True,
            "runtime_probe_healthy": True,
            "certification_ready": True,
            "required_probes": {"all_passed": True},
            "blockers": [],
            "readiness_contract": {
                "healthy": True,
                "system_ready": True,
                "conversation_ready": True,
                "runtime_probe_healthy": True,
                "certification_ready": True,
                "required_probes": {"all_passed": True},
                "blockers": [],
            },
            "boot": {
                "status": "ready",
                "ready": True,
                "system_ready": True,
                "conversation_ready": True,
                "required_probes": {"all_passed": True},
                "blockers": [],
            },
        }

    model = HealthSnapshotReadModel(
        collect,
        _fallback,
        config=HealthReadModelConfig(
            refresh_interval_s=0.05,
            max_stale_s=0.06,
            collection_timeout_s=0.2,
            retry_base_s=0.01,
            retry_max_s=0.05,
        ),
    )
    model.start()
    _wait_until(lambda: model.read()["health_read_model"]["fresh"] is True)
    time.sleep(0.07)

    expired = model.read()
    assert refresh_started.wait(0.2)
    assert calls == 2
    assert expired["healthy"] is True
    assert expired["health_read_model"]["serving"] == "expired"
    assert expired["health_read_model"]["refresh_in_flight"] is True

    from interface.routes.system import _apply_health_read_model_truth

    truthful = _apply_health_read_model_truth(expired)
    assert truthful["status"] == "stale"
    assert truthful["healthy"] is False
    assert truthful["connected"] is False
    assert truthful["conversation_ready"] is False
    assert truthful["runtime_probe_healthy"] is False
    assert truthful["certification_ready"] is False
    assert truthful["required_probes"]["all_passed"] is False
    assert truthful["readiness_contract"]["healthy"] is False
    assert truthful["boot"]["ready"] is False
    assert truthful["blockers"][0] == "health_snapshot_expired"

    block_refresh.set()


def test_read_model_coalesces_failure_episode_and_reports_one_recovery():
    outcomes: list[object] = [RuntimeError("collector unavailable"), {"status": "ok"}]

    def collect() -> dict[str, object]:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    model = HealthSnapshotReadModel(
        collect,
        _fallback,
        config=HealthReadModelConfig(
            refresh_interval_s=0.01,
            max_stale_s=0.03,
            collection_timeout_s=0.1,
            retry_base_s=0.01,
            retry_max_s=0.02,
        ),
    )
    model.start()
    _wait_until(
        lambda: model.read()["health_read_model"]["consecutive_failures"] == 1
    )
    failed = model.read()["health_read_model"]
    assert failed["incident_id"] == "health-refresh-000001"
    assert failed["total_failures"] == 1

    time.sleep(0.02)
    model.request_refresh()
    _wait_until(lambda: model.read()["health_read_model"]["fresh"] is True)
    recovered = model.read()["health_read_model"]
    assert recovered["incident_id"] is None
    assert recovered["consecutive_failures"] == 0
    assert recovered["last_recovery"] == {
        "incident_id": "health-refresh-000001",
        "failed_refreshes": 1,
        "recovered_at_unix": recovered["last_recovery"]["recovered_at_unix"],
    }


@pytest.mark.asyncio
async def test_api_health_returns_initial_snapshot_while_collector_is_blocked(monkeypatch):
    from interface.routes import system as system_routes

    release = threading.Event()
    started = threading.Event()

    def collect() -> dict[str, object]:
        started.set()
        assert release.wait(1.0)
        return {"status": "ok", "healthy": True}

    model = HealthSnapshotReadModel(
        collect,
        system_routes._health_snapshot_fallback,
        config=HealthReadModelConfig(
            refresh_interval_s=1.0,
            max_stale_s=2.0,
            collection_timeout_s=0.5,
        ),
    )
    monkeypatch.setattr(system_routes, "_HEALTH_READ_MODEL", model)
    monkeypatch.setattr(
        system_routes,
        "_restore_owner_session_from_request",
        lambda _request: None,
    )
    monkeypatch.setattr(
        system_routes,
        "_mark_runtime_service_progress",
        lambda _source: None,
    )

    started_at = time.monotonic()
    response = await system_routes.api_health(SimpleNamespace(headers={}))
    elapsed = time.monotonic() - started_at
    payload = json.loads(response.body)

    assert elapsed < 0.05
    assert started.wait(0.2)
    assert payload["status"] == "booting"
    assert payload["healthy"] is False
    assert payload["blockers"][0] == "health_snapshot_initializing"
    assert payload["health_read_model"]["refresh_in_flight"] is True
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-aura-health-generation"] == "0"
    assert response.headers["x-aura-health-serving"] == "initializing"
    release.set()


@pytest.mark.asyncio
async def test_readyz_uses_cached_canonical_readiness_without_inline_probe(monkeypatch):
    from core.runtime.health_contract import REQUIRED_HEALTH_PROBE_GROUPS
    from interface.routes import system as system_routes

    required_probes = {
        group: {
            "ok": True,
            "components": {component: True for component in components},
        }
        for group, components in REQUIRED_HEALTH_PROBE_GROUPS.items()
    }
    required_probes["all_passed"] = True
    payload = {
        "status": "ok",
        "healthy": True,
        "uptime": 321.5,
        "blockers": [],
        "required_probes": required_probes,
        "readiness_contract": {
            "healthy": True,
            "system_ready": True,
            "conversation_ready": True,
            "runtime_probe_healthy": True,
            "required_probes": required_probes,
            "blockers": [],
        },
        "health_read_model": {
            "expired": False,
            "snapshot_generation": 9,
            "age_s": 1.25,
            "serving": "fresh",
        },
    }

    class ReadModel:
        def read(self):
            return payload

    monkeypatch.setattr(system_routes, "_HEALTH_READ_MODEL", ReadModel())
    monkeypatch.setattr(system_routes, "is_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        system_routes,
        "_shutdown_health_status",
        lambda: {"running": False, "request": {"requested": False}},
    )

    started = time.monotonic()
    response = await system_routes.readyz(SimpleNamespace(headers={}))
    elapsed = time.monotonic() - started
    result = json.loads(response.body)

    assert elapsed < 0.05
    assert response.status_code == 200
    assert result == {
        "status": "ready",
        "ready": True,
        "issues": [],
        "uptime_s": 321.5,
        "conversation_ready": True,
        "runtime_probe_healthy": True,
        "required_probes_passed": True,
        "snapshot_generation": 9,
        "snapshot_age_s": 1.25,
        "serving": "fresh",
    }


@pytest.mark.asyncio
async def test_readyz_fails_closed_when_health_snapshot_is_expired(monkeypatch):
    from interface.routes import system as system_routes

    payload = system_routes._health_snapshot_fallback()
    payload["health_read_model"] = {
        "expired": True,
        "captured_at_unix": 1.0,
        "snapshot_generation": 3,
        "age_s": 31.0,
        "serving": "expired",
    }

    class ReadModel:
        def read(self):
            return payload

    monkeypatch.setattr(system_routes, "_HEALTH_READ_MODEL", ReadModel())
    monkeypatch.setattr(system_routes, "is_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        system_routes,
        "_shutdown_health_status",
        lambda: {"running": False, "request": {"requested": False}},
    )

    response = await system_routes.readyz(SimpleNamespace(headers={}))
    result = json.loads(response.body)

    assert response.status_code == 503
    assert result["ready"] is False
    assert result["issues"][0] == "health_snapshot_expired"
    assert result["serving"] == "expired"


def test_public_health_route_applies_shutdown_truth_to_cached_success(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(
        system_routes,
        "_shutdown_health_status",
        lambda: {"running": True, "request": {"requested": True}},
    )
    payload = {
        "status": "ok",
        "healthy": True,
        "connected": True,
        "conversation_ready": True,
        "runtime_probe_healthy": True,
        "certification_ready": True,
        "required_probes": {"all_passed": True},
        "blockers": [],
        "readiness_contract": {
            "healthy": True,
            "system_ready": True,
            "conversation_ready": True,
            "runtime_probe_healthy": True,
            "certification_ready": True,
            "required_probes": {"all_passed": True},
            "blockers": [],
        },
        "boot": {
            "status": "ready",
            "ready": True,
            "system_ready": True,
            "conversation_ready": True,
            "required_probes": {"all_passed": True},
            "blockers": [],
        },
    }

    result = system_routes._apply_current_shutdown_truth(payload)

    assert result["status"] == "stopping"
    assert result["healthy"] is False
    assert result["connected"] is False
    assert result["conversation_ready"] is False
    assert result["required_probes"]["all_passed"] is False
    assert result["blockers"][0] == "runtime_shutdown"
    assert result["shutdown"]["request"]["requested"] is True


@pytest.mark.asyncio
async def test_snapshot_collector_observes_without_constructing_services(monkeypatch):
    from interface.routes import system as system_routes

    def forbidden_get(*_args, **_kwargs):
        raise AssertionError("health snapshot collection must not construct a service")

    monkeypatch.setattr(system_routes.ServiceContainer, "get", forbidden_get)
    monkeypatch.setattr(
        system_routes.ServiceContainer,
        "peek",
        staticmethod(lambda _name, default=None: default),
    )
    monkeypatch.setattr(system_routes, "get_runtime_state", lambda: {"state": {}})
    monkeypatch.setattr(
        system_routes,
        "_collect_runtime_integrity_report",
        lambda: {
            "healthy": False,
            "concerns": ["not_sampled"],
            "advisory": [],
        },
    )
    monkeypatch.setattr(
        system_routes.psutil,
        "cpu_percent",
        lambda interval=None, percpu=False: [0.0] if percpu else 0.0,
    )
    monkeypatch.setattr(
        system_routes.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=0.0),
    )
    monkeypatch.setattr(
        system_routes.psutil,
        "disk_usage",
        lambda _path: SimpleNamespace(percent=0.0),
    )

    payload = await system_routes._collect_api_health_payload(
        allow_owner_loop_reads=False
    )

    assert isinstance(payload, dict)
    assert isinstance(payload["healthy"], bool)
    assert isinstance(payload["conversation_ready"], bool)
    assert isinstance(payload["readiness_contract"], dict)


def test_legacy_shell_health_poll_is_single_scheduled_incident_loop():
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    source = (project_root / "interface/static/aura.js").read_text(encoding="utf-8")
    service_worker = (project_root / "interface/static/service-worker.js").read_text(
        encoding="utf-8"
    )

    assert "function scheduleHealthPoll" in source
    assert "function recordHealthPollFailure" in source
    assert "function recordHealthPollSuccess" in source
    assert "HEALTH_POLL_JITTER_RATIO" in source
    assert "HEALTH_POLL_REMINDER_MS" in source
    assert "setInterval(pollHealth" not in source
    assert "health endpoint unavailable; retaining last known state" in source
    assert "endpoint recovered after" in source
    assert "fetch('/api/health')" not in service_worker
    assert service_worker.count("fetch('/api/health/heartbeat')") == 2
