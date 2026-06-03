import asyncio
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
    assert health.status == "failed_closed"
    assert get_degradation_tracker().count("event_bus", "critical") == 1

    with pytest.raises(RuntimeError, match="CRITICAL SERVICE FAILURE"):
        record_degradation(
            "event_bus",
            RuntimeError("second timeout"),
            severity="degraded",
            action="fail closed after recording",
        )
    assert get_degradation_tracker().count("event_bus", "critical") == 2


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
    assert health.status == "failed_closed"
    assert bus.get_status()["degraded"] is True
    assert bus.get_status()["alive"] is False


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


def test_stability_guardian_initializing_summary_is_not_healthy():
    from core.resilience.stability_guardian import StabilityGuardian

    guardian = StabilityGuardian(SimpleNamespace(start_time=0.0))

    summary = guardian.get_health_summary()

    assert summary["status"] == "initializing"
    assert summary["healthy"] is False
    assert summary["active_issues"]
    assert "no stability report yet" in summary["message"]
