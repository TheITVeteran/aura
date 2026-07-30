import asyncio
import time
from types import SimpleNamespace

import pytest


def _install_fail_closed_event_bus(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime.errors import get_subsystem_registry

    monkeypatch.setenv("AURA_MODE", "production")
    monkeypatch.setattr(ServiceContainer, "_services", dict(ServiceContainer._services))
    ServiceContainer._services["event_bus"] = SimpleNamespace(failure_policy="fail-closed")
    registry = get_subsystem_registry()
    with registry._lock:
        registry._systems.clear()
    return registry


def test_fail_closed_degradation_records_before_optional_raise(monkeypatch):
    from core.runtime.errors import get_degradation_tracker, record_degradation

    registry = _install_fail_closed_event_bus(monkeypatch)
    get_degradation_tracker().reset()

    record_degradation(
        "event_bus",
        RuntimeError("publish timeout"),
        severity="degraded",
        action="marked unhealthy for runtime health contract",
        enforce_failure_policy=False,
    )

    health = registry.get("event_bus")
    assert health is not None
    assert health.status == "degraded"
    assert get_degradation_tracker().count("event_bus", "critical") == 0
    assert get_degradation_tracker().count("event_bus", "degraded") == 1

    with pytest.raises(RuntimeError, match="CRITICAL SERVICE FAILURE"):
        record_degradation(
            "event_bus",
            RuntimeError("second timeout"),
            severity="degraded",
            action="fail closed after recording",
        )
    assert get_degradation_tracker().count("event_bus", "critical") == 1


def test_expected_background_warning_opt_out_keeps_service_healthy(monkeypatch):
    from core.runtime.errors import get_degradation_tracker, record_degradation

    registry = _install_fail_closed_event_bus(monkeypatch)
    get_degradation_tracker().reset()

    record_degradation(
        "event_bus",
        RuntimeError("background process lacks accessibility context"),
        severity="warning",
        action="fail-closed service probe failed",
        enforce_failure_policy=False,
    )

    health = registry.get("event_bus")
    assert health is not None
    assert health.status == "healthy"
    assert get_degradation_tracker().count("event_bus", "critical") == 0
    assert get_degradation_tracker().count("event_bus", "debug") == 1


def test_event_bus_records_degraded_health_without_callback_raise(monkeypatch):
    from core.event_bus import AuraEventBus

    registry = _install_fail_closed_event_bus(monkeypatch)
    bus = AuraEventBus()

    loop = asyncio.new_event_loop()
    try:
        bus.set_loop(loop)
        bus._record_error(
            TimeoutError("redis publish stalled"),
            "AuraEventBus: Redis publish stalled; switching to local-only mode: %s",
            degraded=True,
        )
    finally:
        loop.close()

    health = registry.get("event_bus")
    assert health is not None
    # A redis-publish TimeoutError → local-only fallback is graceful
    # degradation, not a service death: the bus keeps working. A bare timeout
    # no longer marks a fail-closed subsystem failed_closed (it would have
    # cascaded to a mind-wide lockdown; see test_runtime_error_architecture).
    assert health.status == "degraded"
    assert bus.get_status()["degraded"] is True
    assert bus.get_status()["alive"] is False


def test_boot_snapshot_rejects_important_contract_failure(monkeypatch):
    from core.health import boot_status

    def service(container_key: str) -> dict:
        return {
            "name": container_key,
            "container_key": container_key,
            "tier": "critical",
            "present": True,
            "liveness": "ok",
            "liveness_check": "is_ready",
            "error": None,
        }

    required_keys = (
        "kernel_interface",
        "inference_gate",
        "llm_router",
        "state_repository",
        "memory_facade",
        "scheduler",
        "unified_will",
        "authority_gateway",
        "capability_engine",
    )
    contract = {
        "status": "degraded",
        "healthy": False,
        "operational": True,
        "status_code": 200,
        "services": [service(key) for key in required_keys],
        "failures": {
            "critical": [],
            "important": [
                {
                    "name": "Event Bus",
                    "container_key": "event_bus",
                    "tier": "important",
                    "present": True,
                    "liveness": "failed",
                    "error": "is_alive() returned False",
                }
            ],
            "optional": [],
        },
    }
    monkeypatch.setattr(boot_status, "_runtime_contract_snapshot", lambda: contract)

    orchestrator = SimpleNamespace(
        status=SimpleNamespace(
            initialized=True,
            running=True,
            healthy=True,
            last_error="",
            cycle_count=2,
            start_time=time.time() - 5.0,
        ),
        health_check=lambda: True,
    )
    runtime_state = {
        "state": {"heartbeat_tick": time.time()},
        "sha256": "abc123",
        "signature": "sig",
    }

    snapshot, status_code = boot_status.build_boot_health_snapshot(
        orchestrator,
        runtime_state,
        is_gui_proxy=False,
        conversation_lane={"conversation_ready": True, "state": "ready"},
    )

    assert status_code == 503
    assert snapshot["ready"] is False
    assert snapshot["checks"]["runtime_contract_operational"] is True
    assert snapshot["checks"]["runtime_contract_healthy"] is False
    assert "runtime_contract_healthy" in snapshot["blockers"]
    assert "important:event_bus" in snapshot["blockers"]


def test_runtime_hygiene_treats_zombie_process_introspection_as_dead():
    import psutil

    from core.runtime.runtime_hygiene import _process_cmdline, _process_name

    class ZombieProc:
        def __init__(self):
            self.calls = []

        def cmdline(self):
            self.calls.append("cmdline")
            raise psutil.ZombieProcess(pid=1234, name="Python")

        def name(self):
            self.calls.append("name")
            raise psutil.ZombieProcess(pid=1234, name="Python")

    proc = ZombieProc()
    assert _process_cmdline(proc) == []
    assert _process_name(proc) == ""
    assert proc.calls == ["cmdline", "name"]


def test_dnu_runtime_health_blockers_reject_unhealthy_snapshots():
    from tools.agi.run_dnu_agi_proof_battery import proof_runtime_health_blockers

    assert proof_runtime_health_blockers(
        {
            "runtime_health_contract": {
                "healthy": True,
                "status": "healthy",
                "required_probes": {"all_passed": True},
            }
        }
    ) == []
    blockers = proof_runtime_health_blockers(
        {
            "runtime_health_contract": {
                "healthy": False,
                "status": "degraded",
                "required_probes": {
                    "all_passed": False,
                    "kernel": {"ok": True},
                    "inference": {"ok": False},
                },
            }
        }
    )
    assert any("runtime health status" in item for item in blockers)
    assert any("inference" in item for item in blockers)


def test_runtime_pressure_probe_rejects_high_existential_threat(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime import health_contract

    class HighThreat:
        def get_status(self):
            return {
                "existential_threat": 0.91,
                "lag_threat": 0.91,
                "memory_threat": 0.1,
            }

    def fake_get(cls, name, default=None):
        if name == "existential_stakes":
            return HighThreat()
        return default

    monkeypatch.setenv("AURA_HEALTH_EXISTENTIAL_THREAT_UNHEALTHY", "0.75")
    monkeypatch.setattr(
        "core.runtime.health_contract.get_runtime_service",
        lambda name, default=None: fake_get(ServiceContainer, name, default),
    )

    status = health_contract._runtime_pressure_status()

    assert status.present is True
    assert status.liveness_ok is False
    assert "existential_threat" in (status.error or "")


def test_runtime_pressure_probe_ignores_steady_state_memory_threat(monkeypatch):
    """A loaded model (high memory_threat, low lag) must stay healthy here.

    Steady-state memory pressure is owned by _unified_memory_pressure_status and
    the out-of-band watchdog. A ~20GB local model on a 64GB box sits near
    memory_threat 0.77 while serving requests fine; the runtime-pressure probe
    targets the *stuck/overloaded* runtime (lag), so it must not double-count
    memory and mark a healthy, request-serving runtime degraded.
    """
    from core.container import ServiceContainer
    from core.runtime import health_contract

    class LoadedModelThreat:
        def get_status(self):
            return {
                "existential_threat": 0.77,
                "lag_threat": 0.22,
                "memory_threat": 0.77,
            }

    def fake_get(cls, name, default=None):
        if name == "existential_stakes":
            return LoadedModelThreat()
        return default

    monkeypatch.setenv("AURA_HEALTH_EXISTENTIAL_THREAT_UNHEALTHY", "0.75")
    monkeypatch.setattr(
        "core.runtime.health_contract.get_runtime_service",
        lambda name, default=None: fake_get(ServiceContainer, name, default),
    )

    status = health_contract._runtime_pressure_status()

    assert status.present is True
    assert status.liveness_ok is True
    assert status.error is None


def test_runtime_pressure_probe_rejects_hard_event_loop_lag(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime import health_contract

    class HealthyThreat:
        def get_status(self):
            return {
                "existential_threat": 0.05,
                "lag_threat": 0.0,
                "memory_threat": 0.0,
            }

    class LagMonitor:
        def get_status(self):
            return {
                "last_lag_s": 12.5,
                "last_failure_reason": "",
            }

    def fake_get(cls, name, default=None):
        if name == "existential_stakes":
            return HealthyThreat()
        if name == "event_loop_monitor":
            return LagMonitor()
        return default

    monkeypatch.setenv("AURA_HEALTH_EVENT_LOOP_LAG_UNHEALTHY_S", "5.0")
    monkeypatch.setattr(
        "core.runtime.health_contract.get_runtime_service",
        lambda name, default=None: fake_get(ServiceContainer, name, default),
    )

    status = health_contract._runtime_pressure_status()

    assert status.liveness_ok is False
    assert "event_loop_monitor.last_lag_s" in (status.error or "")


def test_runtime_pressure_probe_does_not_reuse_stale_event_loop_lag(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime import health_contract

    class LagMonitor:
        def get_status(self):
            return {
                "last_lag_s": 12.5,
                "last_failure_reason": "",
                "sample_fresh": False,
                "sample_age_s": 20.0,
            }

    def fake_get(cls, name, default=None):
        if name == "event_loop_monitor":
            return LagMonitor()
        return default

    monkeypatch.setattr(
        "core.runtime.health_contract.get_runtime_service",
        lambda name, default=None: fake_get(ServiceContainer, name, default),
    )

    status = health_contract._runtime_pressure_status()

    assert status.present is True
    assert status.liveness_ok is True
    assert status.error is None


def test_runtime_pressure_boot_grace_defers_only_explicit_warmup_lag(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime import health_contract

    class LagMonitor:
        def get_status(self):
            return {
                "last_lag_s": 12.5,
                "last_failure_reason": "",
            }

    def fake_get(cls, name, default=None):
        if name == "event_loop_monitor":
            return LagMonitor()
        return default

    monkeypatch.setattr(
        "core.runtime.health_contract.get_runtime_service",
        lambda name, default=None: fake_get(ServiceContainer, name, default),
    )
    monkeypatch.setattr(health_contract, "_process_uptime_seconds", lambda: 30.0)
    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    monkeypatch.setenv("AURA_HEALTH_RUNTIME_PRESSURE_BOOT_GRACE_S", "180")

    status = health_contract._runtime_pressure_status()

    assert status.present is True
    assert status.liveness_ok is True
    assert status.error is None


def test_runtime_pressure_probe_rejects_recent_inference_saturation(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime import health_contract
    from core.runtime.errors import get_degradation_tracker, record_degradation

    def fake_get(cls, name, default=None):
        return default

    get_degradation_tracker().reset()
    monkeypatch.setattr(ServiceContainer, "get", classmethod(fake_get))
    monkeypatch.setenv("AURA_HEALTH_RECENT_DEGRADATION_WINDOW_S", "180")

    try:
        record_degradation(
            "llm_health_router",
            RuntimeError("generation gate saturated"),
            severity="degraded",
            action="refused to stack another concurrent generation",
        )

        status = health_contract._runtime_pressure_status()

        assert status.liveness_ok is False
        assert "recent_llm_health_router" in (status.error or "")
        assert "generation gate saturated" in (status.error or "")
    finally:
        get_degradation_tracker().reset()


def test_runtime_pressure_ignores_explicit_background_gate_contention(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime import health_contract
    from core.runtime.errors import get_degradation_tracker, record_degradation

    def fake_get(cls, name, default=None):
        return default

    get_degradation_tracker().reset()
    monkeypatch.setattr(ServiceContainer, "get", classmethod(fake_get))
    monkeypatch.setenv("AURA_HEALTH_RECENT_DEGRADATION_WINDOW_S", "180")

    try:
        record_degradation(
            "llm_health_router",
            RuntimeError("generation gate saturated"),
            severity="degraded",
            action=(
                "refused to stack another background concurrent generation; "
                "origin=system purpose=healing_shard"
            ),
        )

        status = health_contract._runtime_pressure_status()

        assert status.liveness_ok is True
        assert status.error is None
    finally:
        get_degradation_tracker().reset()


def test_runtime_pressure_boot_grace_does_not_hide_inference_saturation(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime import health_contract
    from core.runtime.errors import get_degradation_tracker, record_degradation

    def fake_get(cls, name, default=None):
        return default

    get_degradation_tracker().reset()
    monkeypatch.setattr(ServiceContainer, "get", classmethod(fake_get))
    monkeypatch.setattr(health_contract, "_process_uptime_seconds", lambda: 30.0)
    monkeypatch.setenv("AURA_PROOF_RUN", "1")
    monkeypatch.setenv("AURA_HEALTH_RUNTIME_PRESSURE_BOOT_GRACE_S", "180")
    monkeypatch.setenv("AURA_HEALTH_RECENT_DEGRADATION_WINDOW_S", "180")

    try:
        record_degradation(
            "llm_health_router",
            RuntimeError("generation gate saturated"),
            severity="degraded",
            action="refused to stack another concurrent generation",
        )

        status = health_contract._runtime_pressure_status()

        assert status.liveness_ok is False
        assert "generation gate saturated" in (status.error or "")
    finally:
        get_degradation_tracker().reset()


def test_runtime_pressure_ignores_background_brainstem_timeout(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime import health_contract
    from core.runtime.errors import get_degradation_tracker, record_degradation

    def fake_get(cls, name, default=None):
        return default

    get_degradation_tracker().reset()
    monkeypatch.setattr(ServiceContainer, "get", classmethod(fake_get))
    monkeypatch.setenv("AURA_HEALTH_RECENT_DEGRADATION_WINDOW_S", "180")

    try:
        record_degradation(
            "inference_gate",
            TimeoutError("inference_gate_generation_timeout:Brainstem:7.0s"),
            severity="critical",
            action="returned control after local generation exceeded inference-gate timeout",
        )

        status = health_contract._runtime_pressure_status()

        assert status.liveness_ok is True
        assert status.error is None
    finally:
        get_degradation_tracker().reset()


def test_runtime_pressure_rejects_foreground_cortex_timeout(monkeypatch):
    from core.container import ServiceContainer
    from core.runtime import health_contract
    from core.runtime.errors import get_degradation_tracker, record_degradation

    def fake_get(cls, name, default=None):
        return default

    get_degradation_tracker().reset()
    monkeypatch.setattr(ServiceContainer, "get", classmethod(fake_get))
    monkeypatch.setenv("AURA_HEALTH_RECENT_DEGRADATION_WINDOW_S", "180")

    try:
        record_degradation(
            "inference_gate",
            TimeoutError("inference_gate_generation_timeout:Cortex:76.0s"),
            severity="critical",
            action="foreground user-facing generation exceeded inference-gate timeout",
        )

        status = health_contract._runtime_pressure_status()

        assert status.liveness_ok is False
        assert "inference_gate_generation_timeout:Cortex" in (status.error or "")
    finally:
        get_degradation_tracker().reset()


def test_stability_guardian_initializing_summary_is_not_healthy():
    from core.resilience.stability_guardian import StabilityGuardian

    guardian = StabilityGuardian(SimpleNamespace(start_time=0.0))

    summary = guardian.get_health_summary()

    assert summary["status"] == "initializing"
    assert summary["healthy"] is False
    assert summary["active_issues"]
    assert "no stability report yet" in summary["message"]


def test_lock_watchdog_snapshot_survives_concurrent_mutation():
    import threading

    from core.resilience.lock_watchdog import LockWatchdog

    watchdog = LockWatchdog(check_interval=0.01, threshold=0.01)
    with watchdog._active_locks_guard:
        watchdog._active_locks.clear()

    stop = threading.Event()
    failures: list[BaseException] = []

    def mutate_locks() -> None:
        index = 0
        while not stop.is_set():
            lock_id = f"test-lock-{index % 8}"
            try:
                watchdog.report_acquire_start(lock_id, "test")
                if index % 2 == 0:
                    watchdog.report_release(lock_id)
            except (AssertionError, RuntimeError, ValueError, TypeError, OSError, KeyError, IndexError) as exc:  # pragma: no cover - copied back to main assertion
                failures.append(exc)
                stop.set()
            index += 1

    worker = threading.Thread(target=mutate_locks, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            snapshot = watchdog.get_snapshot()
            assert "locks" in snapshot
    finally:
        stop.set()
        worker.join(timeout=1.0)
        with watchdog._active_locks_guard:
            watchdog._active_locks.clear()

    assert failures == []


def test_autonomous_brain_enters_safe_mode_when_stability_health_is_unknown(monkeypatch):
    from core.brain.llm.autonomous_brain_integration import AutonomousCognitiveEngine
    from core.container import ServiceContainer

    class GuardianWithoutHealth:
        def get_health_summary(self):
            return {"status": "initializing"}

    def fake_get(cls, name, default=None):
        if name == "stability_guardian":
            return GuardianWithoutHealth()
        return default

    monkeypatch.setattr(ServiceContainer, "get", classmethod(fake_get))
    engine = AutonomousCognitiveEngine.__new__(AutonomousCognitiveEngine)

    assert engine._is_safe_mode() is True


def test_autonomous_brain_enters_safe_mode_when_stability_lookup_fails(monkeypatch):
    from core.brain.llm.autonomous_brain_integration import AutonomousCognitiveEngine
    from core.container import ServiceContainer

    def fake_get(cls, name, default=None):
        if name == "stability_guardian":
            raise RuntimeError("container unavailable")
        return default

    monkeypatch.setattr(ServiceContainer, "get", classmethod(fake_get))
    engine = AutonomousCognitiveEngine.__new__(AutonomousCognitiveEngine)

    assert engine._is_safe_mode() is True
