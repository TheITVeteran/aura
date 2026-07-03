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


def test_world_state_registration_uses_runtime_registry(monkeypatch):
    import inspect
    import core.world_state as world_state
    from core.runtime.service_registry import (
        install_service_presence_resolver,
        install_service_registration_sink,
    )

    registered: list[tuple[str, object, bool, dict[str, str | None]]] = []
    monkeypatch.setattr(world_state, "_ws_instance", None)
    install_service_presence_resolver(lambda _name: False)
    install_service_registration_sink(
        lambda name, instance, required, metadata: registered.append(
            (name, instance, required, metadata)
        )
    )
    try:
        instance = world_state.get_world_state()
    finally:
        install_service_presence_resolver(None)
        install_service_registration_sink(None)
        monkeypatch.setattr(world_state, "_ws_instance", None)

    assert registered
    name, registered_instance, required, metadata = registered[0]
    assert name == "world_state"
    assert registered_instance is instance
    assert required is False
    assert metadata["owner"] == "core/world_state.py"

    source = inspect.getsource(world_state)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_ice_sentinel_registration_uses_runtime_registry(monkeypatch):
    import inspect
    import core.security.ice_sentinel as ice_sentinel
    from core.runtime.service_registry import (
        install_service_registration_sink,
        install_service_resolver,
    )

    registered: list[tuple[str, object, bool, dict[str, str | None]]] = []
    monkeypatch.setattr(ice_sentinel, "_INSTANCE", None)
    install_service_resolver(lambda _name, default=None: default)
    install_service_registration_sink(
        lambda name, instance, required, metadata: registered.append(
            (name, instance, required, metadata)
        )
    )
    try:
        instance = ice_sentinel.register_ice_sentinel()
    finally:
        install_service_resolver(None)
        install_service_registration_sink(None)
        monkeypatch.setattr(ice_sentinel, "_INSTANCE", None)

    names = [item[0] for item in registered]
    assert "ice" in names
    assert any(item[1] is instance and item[2] is False for item in registered)

    source = inspect.getsource(ice_sentinel.register_ice_sentinel)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_engine_support_brain_resolution_uses_runtime_registry():
    import inspect
    import core.utils.engine_support as engine_support
    from core.runtime.service_registry import install_service_resolver

    class Brain:
        pass

    class Orchestrator:
        brain = Brain()

    install_service_resolver(lambda name, default=None: Orchestrator() if name == "orchestrator" else default)
    try:
        assert isinstance(engine_support.resolve_brain(), Brain)
    finally:
        install_service_resolver(None)

    source = inspect.getsource(engine_support.resolve_brain)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_hierarchical_agency_handlers_use_runtime_registry():
    import inspect
    import core.agency.hierarchical_agency as hierarchical_agency
    from core.runtime.service_registry import install_service_resolver

    class GoalEngine:
        pass

    signals: list[tuple[str, str]] = []

    class RsiLoop:
        def record_signal(self, source, kind, **_kwargs):
            signals.append((source, kind))

    def resolver(name, default=None):
        if name == "goal_engine":
            return GoalEngine()
        if name == "recursive_self_improvement":
            return RsiLoop()
        return default

    agency = hierarchical_agency.HierarchicalAgency(ledger_enabled=False)
    install_service_resolver(resolver)
    try:
        strategic = agency._strategic(hierarchical_agency.Situation("plan", goal_horizon=0.8))
        self_improve = agency._self_improvement(
            hierarchical_agency.Situation("missing skill", capability_gap=0.9)
        )
    finally:
        install_service_resolver(None)

    assert strategic.detail["routed_to"] == "goal_engine"
    assert self_improve.detail["recorded"] == "capability_gap_signal"
    assert signals == [("hierarchical_agency", "capability_gap")]

    source = inspect.getsource(hierarchical_agency.HierarchicalAgency._strategic)
    source += inspect.getsource(hierarchical_agency.HierarchicalAgency._self_improvement)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_emergency_protocol_minimal_mode_uses_runtime_registry():
    import inspect
    import core.security.emergency_protocol as emergency_protocol
    from core.runtime.service_registry import install_service_resolver

    class BackgroundService:
        running = True

    services = {
        "curiosity_explorer": BackgroundService(),
        "skill_synthesizer": BackgroundService(),
    }
    install_service_resolver(lambda name, default=None: services.get(name, default))
    try:
        protocol = emergency_protocol.EmergencyProtocol()
        protocol._enter_minimal_mode()
    finally:
        install_service_resolver(None)

    assert services["curiosity_explorer"].running is False
    assert services["skill_synthesizer"].running is False

    source = inspect.getsource(emergency_protocol.EmergencyProtocol._enter_minimal_mode)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_scientific_engine_belief_publish_uses_runtime_registry():
    import inspect
    import core.cognition.scientific_engine as scientific_engine
    from core.runtime.service_registry import install_service_resolver

    beliefs: list[tuple[str, object, float, str]] = []

    class WorldState:
        def set_belief(self, key, value, *, confidence, source):
            beliefs.append((key, value, confidence, source))

    engine = scientific_engine.ScientificEngine.__new__(scientific_engine.ScientificEngine)
    hypothesis = scientific_engine.Hypothesis(
        hypothesis_id="hyp-test",
        claim="runtime registry routes belief updates",
        predicted_observable="belief_written",
        expected=1.0,
        confidence=0.8,
        status="supported",
    )
    install_service_resolver(lambda name, default=None: WorldState() if name == "world_state" else default)
    try:
        engine._publish_belief(hypothesis)
    finally:
        install_service_resolver(None)

    assert beliefs == [
        (
            "hypothesis:runtime registry routes belief updates",
            {"status": "supported", "expected": 1.0},
            0.8,
            "scientific_engine",
        )
    ]

    source = inspect.getsource(scientific_engine.ScientificEngine._publish_belief)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_intention_loop_uses_runtime_registry(tmp_path, monkeypatch):
    import inspect
    import core.agency.intention_loop as intention_loop
    from core.runtime.service_registry import (
        install_service_registration_sink,
        install_service_resolver,
    )

    class Embedder:
        def similarity(self, _expected, _actual):
            return 0.75

    services = {
        "cognitive_ledger": object(),
        "belief_revision_engine": object(),
        "embedding_engine": Embedder(),
    }
    original_class = intention_loop.IntentionLoop
    install_service_resolver(lambda name, default=None: services.get(name, default))
    try:
        loop = intention_loop.IntentionLoop(db_path=str(tmp_path / "intentions.db"))
        assert loop._get_ledger() is services["cognitive_ledger"]
        assert loop._get_belief_engine() is services["belief_revision_engine"]
        assert loop._calculate_surprise("expected outcome", "actual outcome") == 0.25
    finally:
        install_service_resolver(None)
        loop.close()

    registered: list[tuple[str, object, bool, dict[str, str | None]]] = []
    instance = object()
    monkeypatch.setattr(intention_loop, "_instance", None)
    monkeypatch.setattr(intention_loop, "IntentionLoop", lambda: instance)
    install_service_registration_sink(
        lambda name, value, required, metadata: registered.append(
            (name, value, required, metadata)
        )
    )
    try:
        assert intention_loop.get_intention_loop() is instance
    finally:
        install_service_registration_sink(None)
        monkeypatch.setattr(intention_loop, "_instance", None)

    assert registered
    assert registered[0][0] == "intention_loop"
    assert registered[0][1] is instance

    source = inspect.getsource(original_class._get_ledger)
    source += inspect.getsource(original_class._get_belief_engine)
    source += inspect.getsource(original_class._calculate_surprise)
    source += inspect.getsource(intention_loop.get_intention_loop)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_consciousness_system_publication_uses_runtime_registry():
    import inspect
    import core.consciousness.system as consciousness_system

    source = inspect.getsource(consciousness_system)
    assert "core.container" not in source
    assert "ServiceContainer" not in source
    assert "register_runtime_service" in source
    assert "get_runtime_service" in source


def test_being_runtime_publish_uses_runtime_registry():
    import inspect
    import core.being.runtime as being_runtime
    from core.runtime.service_registry import install_service_registration_sink

    registered: list[tuple[str, object, bool, dict[str, str | None]]] = []
    runtime = being_runtime.BeingRuntime.__new__(being_runtime.BeingRuntime)
    now = object()
    install_service_registration_sink(
        lambda name, instance, required, metadata: registered.append(
            (name, instance, required, metadata)
        )
    )
    try:
        runtime._publish(now)
    finally:
        install_service_registration_sink(None)

    assert [item[0] for item in registered] == ["aura_now", "being_runtime"]
    assert registered[0][1] is now
    assert registered[1][1] is runtime
    assert registered[0][2] is False
    assert registered[1][2] is False

    source = inspect.getsource(being_runtime.BeingRuntime._publish)
    assert "from core.container import ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_cognitive_engine_service_lookup_uses_runtime_registry_adapter():
    import inspect
    import core.brain.cognitive_engine as cognitive_engine
    from core.runtime.service_registry import install_service_resolver

    class Router:
        last_tier = "primary"

    router = Router()
    install_service_resolver(lambda name, default=None: router if name == "llm_router" else default)
    try:
        engine = cognitive_engine.CognitiveEngine()
        assert engine._current_tier == "primary"
        assert cognitive_engine.get_container().get("llm_router") is router
    finally:
        install_service_resolver(None)

    source = inspect.getsource(cognitive_engine)
    assert "core.container" not in source
    assert "ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_cryptolalia_decoder_uses_runtime_registry():
    import inspect
    import core.brain.cryptolalia_decoder as cryptolalia_decoder
    from core.runtime.service_registry import (
        install_service_registration_sink,
        install_service_resolver,
    )

    class ConceptBridge:
        _concept_cache = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
        }

    install_service_resolver(
        lambda name, default=None: ConceptBridge() if name == "concept_bridge" else default
    )
    try:
        decoder = cryptolalia_decoder.CryptolaliaDecoder()
        assert decoder.approximate_translation([1.0, 0.0], top_n=1) == "[alpha]"
    finally:
        install_service_resolver(None)

    registered: list[tuple[str, object, bool, dict[str, str | None]]] = []
    install_service_registration_sink(
        lambda name, instance, required, metadata: registered.append(
            (name, instance, required, metadata)
        )
    )
    try:
        instance = cryptolalia_decoder.register_cryptolalia_decoder()
    finally:
        install_service_registration_sink(None)

    assert registered
    assert registered[0][0] == "cryptolalia_decoder"
    assert registered[0][1] is instance
    assert registered[0][2] is True

    source = inspect.getsource(cryptolalia_decoder)
    assert "core.container" not in source
    assert "ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_meta_cognition_structural_review_uses_runtime_registry():
    import asyncio
    import inspect
    import core.meta.meta_cognition as meta_cognition
    from core.runtime.service_registry import install_service_resolver

    queued: list[tuple[str, str]] = []

    class MetaEvolution:
        def queue_optimization(self, *, target_area, context):
            queued.append((target_area, context))

    loop = meta_cognition.MetaCognition()
    loop.error_history = [
        {"decision": f"failure-{idx}", "outcome": "failure", "context": {}, "timestamp": 1.0}
        for idx in range(6)
    ]

    install_service_resolver(
        lambda name, default=None: MetaEvolution() if name == "meta_evolution" else default
    )
    try:
        asyncio.run(loop._trigger_structural_review())
    finally:
        install_service_resolver(None)

    assert queued
    assert queued[0][0] == "cognitive_patterns"
    assert "failure-5" in queued[0][1]
    assert len(loop.error_history) == 2

    source = inspect.getsource(meta_cognition)
    assert "core.container" not in source
    assert "ServiceContainer" not in source
    assert "core.runtime.service_registry" in source


def test_startup_boot_validator_uses_runtime_registry_presence():
    import inspect
    import core.startup.boot_validator as boot_validator
    from core.runtime.service_registry import install_service_presence_resolver

    required = {"event_bus", "orchestrator", "llm_interface", "state_repo"}
    install_service_presence_resolver(lambda name: name in required)
    try:
        result = boot_validator.BootValidator.validate_boot()
    finally:
        install_service_presence_resolver(None)

    assert result.passed is True
    assert result.failures == []

    install_service_presence_resolver(lambda name: name == "event_bus")
    try:
        result = boot_validator.BootValidator.validate_boot()
    finally:
        install_service_presence_resolver(None)

    assert result.passed is False
    assert "Core Orchestrator Ready" in result.failures
    assert "LLM Interface Bound" in result.failures
    assert "State Repository (Persistence) Ready" in result.failures

    class ExplicitContainer:
        def has(self, name):
            return name in required

    assert boot_validator.BootValidator.validate_boot(ExplicitContainer()).passed is True

    source = inspect.getsource(boot_validator)
    assert "core.container" not in source
    assert "ServiceContainer" not in source
    assert "core.runtime.service_registry" in source
