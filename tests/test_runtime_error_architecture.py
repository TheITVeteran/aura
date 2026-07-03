import pytest


def test_record_degradation_does_not_import_service_container():
    import inspect
    import core.runtime.errors as errors

    source = inspect.getsource(errors.record_degradation)
    assert "from core.container import ServiceContainer" not in source
    assert "core.observability.metrics" not in source
    assert "core.runtime.service_registry" in source


def test_record_degradation_uses_low_level_metric_sink():
    from core.runtime.errors import record_degradation
    from core.runtime.service_registry import install_metric_counter_sink

    counters: list[tuple[str, int]] = []
    install_metric_counter_sink(lambda name, amount=1: counters.append((name, amount)))

    try:
        record_degradation(
            "metric_bridge",
            RuntimeError("sample"),
            severity="warning",
            action="verify low-level metric sink",
        )
    finally:
        install_metric_counter_sink(None)

    assert ("degradation_metric_bridge_warning", 1) in counters


def test_record_degradation_preserves_fail_closed_policy(monkeypatch):
    from core.runtime.errors import record_degradation
    from core.runtime.service_registry import install_failure_policy_resolver

    install_failure_policy_resolver(
        lambda name: "fail-closed" if name == "critical_service" else None
    )
    monkeypatch.setenv("AURA_MODE", "production")

    try:
        with pytest.raises(RuntimeError, match="CRITICAL SERVICE FAILURE"):
            record_degradation(
                "critical_service",
                RuntimeError("boom"),
                severity="warning",
                action="test failure-policy bridge",
            )
    finally:
        install_failure_policy_resolver(None)


def test_metrics_reads_services_through_low_level_registry(monkeypatch):
    import inspect
    import numpy as np
    import core.observability.metrics as metrics_module
    from core.runtime.service_registry import install_service_resolver

    class Substrate:
        def get_state_summary(self):
            return {"valence": 0.25, "step_count": 7}

        def get_state_vector(self):
            return np.array([0.1, 0.2, 0.3])

    class DriveEngine:
        def get_state(self):
            return {"drives": {"curiosity": 0.8}}

    services = {
        "continuous_substrate": Substrate(),
        "drive_engine": DriveEngine(),
    }

    install_service_resolver(lambda name, default=None: services.get(name, default))
    monkeypatch.setattr(metrics_module, "_metrics_instance", metrics_module.MetricsCollector())

    try:
        samples = metrics_module.get_metrics().collect()
        sample_map = {(sample.name, tuple(sample.labels.items())): sample.value for sample in samples}
        assert sample_map[("aura_substrate_valence", ())] == 0.25
        assert sample_map[("aura_substrate_step_count", ())] == 7.0
        assert sample_map[("aura_drive_level", (("drive", "curiosity"),))] == 0.8
        assert metrics_module.check_readiness()["status"] in {"ready", "not_ready"}
    finally:
        install_service_resolver(None)

    source = inspect.getsource(metrics_module)
    assert "from core.container import ServiceContainer" not in source


def test_degradation_repair_default_lookup_uses_low_level_registry():
    import inspect
    import core.resilience.degradation_repair as repair_module
    from core.runtime.service_registry import install_service_resolver

    class Resilience:
        def record_failure(self, **_kwargs):
            return "recorded"

    install_service_resolver(
        lambda name, default=None: Resilience() if name == "resilience_engine" else default
    )

    try:
        router = repair_module.DegradationRepairRouter(cooldown_seconds=0)
        action = router.route(
            record=type("Record", (), {"subsystem": "demo", "severity": "degraded"})(),
            error=RuntimeError("boom"),
            incident=None,
            extra={},
        )
        assert action.resilience_state == "recorded"
    finally:
        install_service_resolver(None)

    assert "from core.container import ServiceContainer" not in inspect.getsource(
        repair_module._service_get
    )


def test_health_contract_uses_low_level_runtime_registry():
    import inspect
    import time
    import core.runtime.health_contract as health_contract
    from core.runtime.service_registry import install_service_resolver

    class Orchestrator:
        start_time = time.time() - 12.0

    class Stakes:
        def get_status(self):
            return {
                "existential_threat": 0.01,
                "lag_threat": 0.0,
                "memory_threat": 0.0,
            }

    class Monitor:
        def get_status(self):
            return {"last_lag_s": 0.0, "last_failure_reason": ""}

    services = {
        "orchestrator": Orchestrator(),
        "existential_stakes": Stakes(),
        "event_loop_monitor": Monitor(),
        "hypervisor": Monitor(),
    }
    install_service_resolver(lambda name, default=None: services.get(name, default))

    try:
        assert health_contract._process_uptime_seconds() >= 0.0
        status = health_contract._runtime_pressure_status()
        assert status.present is True
        assert status.liveness_ok is True
    finally:
        install_service_resolver(None)

    source = inspect.getsource(health_contract)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_governance_context_uses_runtime_registry_predicates(monkeypatch):
    import inspect
    import core.governance_context as governance_context
    from core.runtime.service_registry import (
        install_registration_locked_resolver,
        install_service_presence_resolver,
    )

    monkeypatch.delenv("AURA_GOVERNANCE_MODE", raising=False)
    monkeypatch.delenv("AURA_REQUIRE_GOVERNANCE", raising=False)

    install_service_presence_resolver(lambda name: name == "kernel_interface")
    install_registration_locked_resolver(lambda: False)
    try:
        assert governance_context.governance_runtime_active() is True
    finally:
        install_service_presence_resolver(None)
        install_registration_locked_resolver(None)

    install_service_presence_resolver(lambda _name: False)
    install_registration_locked_resolver(lambda: True)
    try:
        assert governance_context.governance_runtime_active() is True
    finally:
        install_service_presence_resolver(None)
        install_registration_locked_resolver(None)

    source = inspect.getsource(governance_context.governance_runtime_active)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_event_loop_monitor_uses_runtime_registry_for_foreground_status():
    import inspect
    import core.utils.concurrency as concurrency
    from core.runtime.service_registry import install_service_resolver

    class Gate:
        def get_conversation_status(self):
            return {
                "active": False,
                "foreground_owned": True,
                "warmup_in_flight": False,
                "active_generations": 0,
                "current_request_started_at": 0.0,
                "state": "ready",
            }

    install_service_resolver(lambda name, default=None: Gate() if name == "inference_gate" else default)
    try:
        monitor = concurrency.EventLoopMonitor()
        assert monitor._active_runtime_reason() == "foreground_generation"
    finally:
        install_service_resolver(None)

    source = inspect.getsource(concurrency.EventLoopMonitor._active_runtime_reason)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source

    robust_lock_source = inspect.getsource(concurrency.RobustLock.acquire_robust)
    assert "core.observability.metrics" not in robust_lock_source
    assert "core.container" not in robust_lock_source


def test_degraded_events_forwarding_uses_runtime_registry():
    import inspect
    import core.health.degraded_events as degraded_events
    from core.runtime.service_registry import install_service_resolver

    calls: list[tuple[BaseException, dict, str, str]] = []

    class SelfModifier:
        def on_error(self, error, context, *, skill_name, goal):
            calls.append((error, context, skill_name, goal))
            return None

    class Orchestrator:
        self_modifier = SelfModifier()

    event = {
        "classification": "foreground_blocking",
        "subsystem": "demo",
        "reason": "blocked",
        "detail": "detail",
        "severity": "critical",
        "context": {"source": "test"},
    }
    degraded_events._LAST_FORWARDED.clear()
    install_service_resolver(lambda name, default=None: Orchestrator() if name == "orchestrator" else default)

    try:
        degraded_events._forward_to_error_intelligence(
            ("demo", "blocked", "critical", "foreground_blocking"),
            event,
            exc=RuntimeError("boom"),
        )
    finally:
        install_service_resolver(None)
        degraded_events._LAST_FORWARDED.clear()

    assert len(calls) == 1
    error, context, skill_name, goal = calls[0]
    assert isinstance(error, RuntimeError)
    assert context["source"] == "test"
    assert skill_name == "demo"
    assert goal == "blocked"

    source = inspect.getsource(degraded_events._forward_to_error_intelligence)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_terminal_monitor_reliability_lookup_uses_runtime_registry():
    import inspect
    import core.terminal_monitor as terminal_monitor

    source = inspect.getsource(terminal_monitor.TerminalMonitor.check_for_errors)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_terminal_monitor_world_state_publish_uses_runtime_registry():
    import inspect
    import core.terminal_monitor as terminal_monitor

    source = inspect.getsource(terminal_monitor.TerminalMonitor._ingest_error)
    assert "from core.world_state import get_world_state" not in source
    assert "core.runtime.service_registry" in source
