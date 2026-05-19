from __future__ import annotations

from types import SimpleNamespace

from core.resilience.degradation_repair import (
    DegradationRepairRouter,
    set_degradation_repair_router_for_tests,
)
from core.runtime.errors import get_subsystem_registry, record_degradation


class FakeResilience:
    def __init__(self) -> None:
        self.failures: list[dict[str, float | str]] = []

    def record_failure(self, domain: str, severity: float, stakes: float):
        self.failures.append({"domain": domain, "severity": severity, "stakes": stakes})
        return SimpleNamespace(value="friction")


class FakeSelfModification:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def on_error(self, error, context, skill_name=None, goal=None):
        self.calls.append(
            {
                "error": error,
                "context": context,
                "skill_name": skill_name,
                "goal": goal,
            }
        )


def _record(subsystem: str = "router_unit", severity: str = "degraded"):
    return SimpleNamespace(
        subsystem=subsystem,
        severity=severity,
        error_type="RuntimeError",
        error_message="route failed",
        action="route repair",
    )


def test_degradation_router_feeds_resilience_without_unneeded_code_repair():
    resilience = FakeResilience()
    router = DegradationRepairRouter(
        service_getter=lambda name: {"resilience_engine": resilience}.get(name),
        cooldown_seconds=0.0,
    )

    action = router.route(
        record=_record(),
        error=RuntimeError("route failed"),
        incident=SimpleNamespace(incident_id="inc-1", occurrence_count=1),
    )

    assert action.resilience_state == "friction"
    assert action.self_modification_status == "not_eligible"
    assert resilience.failures == [
        {
            "domain": "degradation:router_unit",
            "severity": 0.55,
            "stakes": 0.6,
        }
    ]


def test_degradation_router_dispatches_critical_repair_with_cooldown():
    resilience = FakeResilience()
    self_modification = FakeSelfModification()
    services = {
        "resilience_engine": resilience,
        "self_modification_engine": self_modification,
    }
    router = DegradationRepairRouter(
        service_getter=lambda name: services.get(name),
        cooldown_seconds=999.0,
    )

    first = router.route(
        record=_record(severity="critical"),
        error=RuntimeError("route failed"),
        incident=SimpleNamespace(incident_id="inc-2", occurrence_count=1),
    )
    second = router.route(
        record=_record(severity="critical"),
        error=RuntimeError("route failed"),
        incident=SimpleNamespace(incident_id="inc-2", occurrence_count=2),
    )

    assert first.self_modification_dispatched is True
    assert first.self_modification_status == "dispatched"
    assert second.self_modification_status == "cooldown"
    assert len(self_modification.calls) == 1
    assert self_modification.calls[0]["skill_name"] == "router_unit"


def test_record_degradation_updates_health_incident_and_repair_route(monkeypatch):
    import core.resilience.incident_manager as incident_module

    resilience = FakeResilience()
    self_modification = FakeSelfModification()
    services = {
        "resilience_engine": resilience,
        "self_modification_engine": self_modification,
    }
    router = DegradationRepairRouter(
        service_getter=lambda name: services.get(name),
        cooldown_seconds=0.0,
    )
    monkeypatch.setattr(
        incident_module,
        "_incident_manager",
        incident_module.IncidentManager(),
    )
    set_degradation_repair_router_for_tests(router)
    subsystem = "record_degradation_contract_unit"

    try:
        record = record_degradation(
            subsystem,
            RuntimeError("critical route failed"),
            severity="critical",
            action="dispatch repair",
            extra={"repair_requested": True},
        )
        health = get_subsystem_registry().get(subsystem)
        incident = incident_module.get_incident_manager()._active[f"degradation:{subsystem}"]
    finally:
        set_degradation_repair_router_for_tests(None)

    assert record.subsystem == subsystem
    assert health is not None
    assert health.status == "unavailable"
    assert "critical route failed" in health.last_error
    assert incident.metadata["repair_router"]["self_modification_status"] == "dispatched"
    assert resilience.failures[0]["severity"] == 0.95
    assert len(self_modification.calls) == 1
