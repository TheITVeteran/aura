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


class FakeAutonomousRepairExecutor:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def enqueue_background(self, request):
        self.requests.append(request)
        return {"status": "scheduled", "fingerprint": request.fingerprint}


class FakeImmune:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def observe_event(self, event, **kwargs):
        self.events.append({"event": event, "kwargs": kwargs})
        return SimpleNamespace(selected_artifact=None)


def _record(subsystem: str = "router_unit", severity: str = "degraded"):
    return SimpleNamespace(
        subsystem=subsystem,
        severity=severity,
        error_type="RuntimeError",
        error_message="route failed",
        action="route repair",
    )


def test_degradation_router_feeds_resilience_without_engine_available():
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
    assert action.self_modification_status == "engine_unavailable"
    assert resilience.failures == [
        {
            "domain": "degradation:router_unit",
            "severity": 0.55,
            "stakes": 0.6,
        }
    ]


def test_degradation_router_dispatches_critical_repair_with_cooldown():
    from core.resilience.autonomous_repair_executor import (
        set_autonomous_repair_executor_for_tests,
    )

    resilience = FakeResilience()
    self_modification = FakeSelfModification()
    autonomous = FakeAutonomousRepairExecutor()
    services = {
        "resilience_engine": resilience,
        "self_modification_engine": self_modification,
    }
    router = DegradationRepairRouter(
        service_getter=lambda name: services.get(name),
        cooldown_seconds=999.0,
    )

    set_autonomous_repair_executor_for_tests(autonomous)
    try:
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
    finally:
        set_autonomous_repair_executor_for_tests(None)

    assert first.self_modification_dispatched is True
    assert first.self_modification_status == "dispatched"
    assert first.autonomous_repair_status == "scheduled"
    assert second.self_modification_status == "cooldown"
    assert len(self_modification.calls) == 1
    assert len(autonomous.requests) == 1
    assert self_modification.calls[0]["skill_name"] == "router_unit"


def test_record_degradation_updates_health_incident_and_repair_route(monkeypatch):
    import core.resilience.incident_manager as incident_module
    from core.resilience.autonomous_repair_executor import (
        set_autonomous_repair_executor_for_tests,
    )

    resilience = FakeResilience()
    self_modification = FakeSelfModification()
    autonomous = FakeAutonomousRepairExecutor()
    services = {
        "resilience_engine": resilience,
        "self_modification_engine": self_modification,
        "adaptive_immune_system": FakeImmune(),
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
    set_autonomous_repair_executor_for_tests(autonomous)
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
        set_autonomous_repair_executor_for_tests(None)

    assert record.subsystem == subsystem
    assert health is not None
    assert health.status == "unavailable"
    assert "critical route failed" in health.last_error
    assert incident.metadata["repair_router"]["self_modification_status"] == "dispatched"
    assert incident.metadata["repair_router"]["autonomous_repair_status"] == "scheduled"
    assert resilience.failures[0]["severity"] == 0.95
    assert len(self_modification.calls) == 1
    assert len(autonomous.requests) == 1


def test_degradation_router_sends_warnings_to_immune_and_safe_repair():
    from core.resilience.autonomous_repair_executor import (
        set_autonomous_repair_executor_for_tests,
    )

    resilience = FakeResilience()
    self_modification = FakeSelfModification()
    autonomous = FakeAutonomousRepairExecutor()
    immune = FakeImmune()
    services = {
        "resilience_engine": resilience,
        "self_modification_engine": self_modification,
        "adaptive_immune_system": immune,
    }
    router = DegradationRepairRouter(
        service_getter=lambda name: services.get(name),
        cooldown_seconds=0.0,
    )

    set_autonomous_repair_executor_for_tests(autonomous)
    try:
        action = router.route(
            record=_record(subsystem="cognitive_engine", severity="warning"),
            error=TimeoutError("slow full-mind reply"),
            incident=None,
            extra={"repair_requested": True},
        )
    finally:
        set_autonomous_repair_executor_for_tests(None)

    assert action.self_modification_status == "dispatched"
    assert action.autonomous_repair_status == "scheduled"
    assert action.immune_status == "scheduled"
    assert self_modification.calls[0]["context"]["error_already_logged"] is True
    assert autonomous.requests[0].subsystem == "cognitive_engine"
