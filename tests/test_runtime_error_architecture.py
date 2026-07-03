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


def test_small_runtime_service_batch_uses_registry():
    import asyncio
    import inspect
    from types import SimpleNamespace

    import core.affect.affect_facade as affect_facade
    import core.agency.latent_distiller as latent_distiller
    import core.brain.personality_bridge as personality_bridge
    import core.brain.scratchpad as scratchpad
    import core.identity.identity_anchor as identity_anchor
    import core.ops.singularity_monitor as singularity_monitor
    import core.resilience.hotfix_engine as hotfix_engine
    from core.brain.types import ThinkingMode, Thought
    from core.runtime.service_registry import install_service_resolver
    from core.state.aura_state import AuraState

    class AffectEngine:
        def get_status(self):
            return {
                "mood": "steady",
                "energy": 60,
                "curiosity": 55,
                "frustration": 0,
                "stability": 100,
                "valence": 0.2,
                "arousal": 0.3,
            }

    class CognitiveEngine:
        async def think(self, objective, **_kwargs):
            return Thought(
                id="registry-thought",
                content=f"summary:{objective[:20]}",
                mode=ThinkingMode.FAST,
            )

    class Memory:
        def __init__(self):
            self.records = []

        async def store_memory(self, **kwargs):
            self.records.append(kwargs)

    class Repo:
        def __init__(self):
            self._current = AuraState.default()
            self._current.identity.name = "Aura"
            self._current.state_id = "abcdefgh1234"

        async def get_current(self):
            return self._current

    class Mirror:
        def get_audit_summary(self):
            return {"health_score": 0.95}

    class MetaCognition:
        mirror = Mirror()

    repo = Repo()
    memory = Memory()
    services = {
        "affect_engine": AffectEngine(),
        "cognitive_engine": CognitiveEngine(),
        "state_repo": repo,
        "state_repository": repo,
        "metacognition": MetaCognition(),
    }
    install_service_resolver(lambda name, default=None: services.get(name, default))
    try:
        anchor = identity_anchor.IdentityAnchor()
        assert anchor.get_identity() == "Aura-abcdefgh"

        facade = affect_facade.AffectFacade()
        assert facade.is_ready() is True
        assert facade.get_status()["mood"] == "steady"

        bridge = personality_bridge.PersonalityBridge()
        assert asyncio.run(bridge.sync_embodiment(SimpleNamespace(model=None)))["damping_mult"] > 0

        pad = scratchpad.ScratchpadEngine()
        plan = asyncio.run(pad.think_recursive("plan carefully", {"history": []}, depth=0))
        assert plan.startswith("[Plan] summary:")

        monitor = singularity_monitor.SingularityMonitor(orchestrator=SimpleNamespace(container=True))
        monitor.pulse()
        assert monitor.is_accelerated is True
        assert monitor.acceleration_factor == 1.5

        distiller = latent_distiller.LatentSpaceDistiller(memory_provider=memory)
        long_history = [{"content": "This session produced a useful design insight. " * 8}]
        asyncio.run(distiller.distill_session(long_history))
        assert memory.records
        assert memory.records[0]["metadata"]["type"] == "distilled_wisdom"
    finally:
        install_service_resolver(None)

    for module in (
        identity_anchor,
        affect_facade,
        personality_bridge,
        scratchpad,
        singularity_monitor,
        latent_distiller,
        hotfix_engine,
    ):
        source = inspect.getsource(module)
        assert "core.container" not in source
        assert "ServiceContainer" not in source


def test_runtime_service_registry_supports_lazy_factory_publication():
    from core.runtime.service_registry import (
        install_service_factory_registration_sink,
        register_runtime_factory,
    )

    captured = []
    factory = lambda: "ready"
    install_service_factory_registration_sink(
        lambda name, fn, lifetime, required, metadata: captured.append(
            (name, fn, lifetime, required, metadata)
        )
    )
    try:
        assert register_runtime_factory(
            "demo_factory",
            factory,
            lifetime="singleton",
            required=False,
            owner="tests",
            registered_by="test_runtime_service_registry_supports_lazy_factory_publication",
            metadata={"extra": "value"},
        )
    finally:
        install_service_factory_registration_sink(None)

    assert captured == [
        (
            "demo_factory",
            factory,
            "singleton",
            False,
            {
                "owner": "tests",
                "registered_by": "test_runtime_service_registry_supports_lazy_factory_publication",
                "required_for": None,
                "failure_policy": None,
                "extra": "value",
            },
        )
    ]


def test_runtime_registry_batch_two_service_seams():
    import asyncio
    import inspect
    from types import SimpleNamespace

    import core.capabilities.source_summarizer as source_summarizer
    import core.emotional_coloring as emotional_coloring
    import core.evals.adaptive_test_chamber as adaptive_test_chamber
    import core.memory.provenance as provenance
    import core.orchestrator.coordinators.affect as affect_coordinator
    import core.plasticity_controller as plasticity_controller
    import core.senses.voice_socket_logic as voice_socket_logic
    import core.skill_evolution as skill_evolution
    import core.utils.telemetry_enrichment as telemetry_enrichment
    from core.brain import concept_vector_bridge
    from core.runtime.service_registry import (
        install_service_factory_registration_sink,
        install_service_registration_sink,
        install_service_resolver,
    )

    class EventBus:
        def __init__(self):
            self.events = []

        async def publish(self, event, payload):
            self.events.append((event, payload))

    class Client:
        async def generate_embedding(self, text):
            return [float(len(text)), 1.0]

    class Router:
        async def route(self, **_kwargs):
            return SimpleNamespace(text="synthesized summary")

        def get_health_report(self):
            return {"foreground_tier": "primary"}

    class Browser:
        async def extract_article_text(self, url):
            return SimpleNamespace(
                url=url,
                title="Article",
                body="Useful article body.",
                author="Reporter",
                source_domain="example.com",
                word_count=3,
            )

    class LiquidState:
        def get_status(self):
            return {
                "energy": 0.6,
                "curiosity": 0.7,
                "frustration": 0.1,
                "confidence": 0.8,
            }

        def get_valence(self):
            return 0.25

    class Homeostasis:
        def get_health(self):
            return {"will_to_live": 0.9}

    class Memory:
        def search(self, _topic, limit=5):
            return [{"emotion": "joy", "arousal": 0.4}]

    class BeliefGraph:
        def get_all_beliefs(self):
            return [1, 2, 3]

    class InsightJournal:
        def get_highest_confidence_insights(self, limit=10):
            return list(range(8))

    class Omni:
        _execution_logs = {"web_search": [{"status": "error"}] * 4}

    class Swarm:
        def __init__(self):
            self.objectives = []

        async def spawn_shard(self, **kwargs):
            self.objectives.append(kwargs)

    class Reflex:
        def __init__(self):
            self.commands = []

        async def process_emergency_interrupt(self, command, context):
            self.commands.append((command, context))

    class FakeWhisper:
        def transcribe(self, _audio_np, beam_size=1):
            return [SimpleNamespace(text=" stop ")], None

    event_bus = EventBus()
    swarm = Swarm()
    reflex = Reflex()
    services = {
        "event_bus": event_bus,
        "cognitive_engine": SimpleNamespace(client=Client()),
        "llm_router": Router(),
        "browser_controller": Browser(),
        "liquid_state": LiquidState(),
        "homeostasis": Homeostasis(),
        "memory": Memory(),
        "free_energy_engine": SimpleNamespace(current=SimpleNamespace(free_energy=0.2)),
        "belief_graph": BeliefGraph(),
        "insight_journal": InsightJournal(),
        "omni_tool": Omni(),
        "sovereign_swarm": swarm,
        "orchestrator": SimpleNamespace(reflex_engine=reflex),
        "affect_engine": SimpleNamespace(
            get_mood=lambda: "Curious",
            get_status=lambda: {"mood": "Curious"},
        ),
    }
    registered = []
    factories = []
    install_service_resolver(lambda name, default=None: services.get(name, default))
    install_service_registration_sink(
        lambda name, instance, required, metadata: registered.append(
            (name, instance, required, metadata)
        )
    )
    install_service_factory_registration_sink(
        lambda name, factory, lifetime, required, metadata: factories.append(
            (name, factory, lifetime, required, metadata)
        )
    )
    try:
        bridge = concept_vector_bridge.register_concept_bridge()
        assert ("concept_bridge", bridge, True, registered[-1][3]) == registered[-1]
        assert asyncio.run(bridge.transmit("a", "b", [1.0])) .startswith("latent_")
        assert event_bus.events[0][0] == "cryptolalia_transmission"
        assert asyncio.run(bridge.generate_concept_vector("cat")) == [3.0, 1.0]

        summarizer = source_summarizer.SourceSummarizer()
        asyncio.run(summarizer.start())
        assert any(item[0] == "source_summarizer" for item in registered)
        result = asyncio.run(
            summarizer.summarize_urls(["https://example.com"], objective="brief")
        )
        assert result.summary == "synthesized summary"

        stamped = provenance.wrap({"fact": "sample"})
        assert round(stamped.provenance.confidence, 2) == 0.8

        enriched = telemetry_enrichment.enrich_telemetry({})
        assert enriched["energy"] == 60.0
        assert enriched["llm_tier"] == "primary"

        emotional_coloring.register_emotional_coloring()
        assert any(item[0] == "emotional_coloring" for item in factories)
        texture = asyncio.run(emotional_coloring.EmotionalColoring().get_texture_for_topic("cat"))
        assert texture.tone_hint == "warm/exploratory"

        plasticity_controller.register_plasticity_controller()
        assert any(item[0] == "plasticity_controller" for item in factories)
        plasticity = plasticity_controller.PlasticityController()
        assert asyncio.run(plasticity.update_plasticity()) == 0.8

        skill_evolution.register_skill_evolution()
        assert any(item[0] == "skill_evolution" for item in factories)
        engine = skill_evolution.SkillEvolutionEngine()
        assert asyncio.run(engine.identify_evolution_targets()) == ["web_search"]
        asyncio.run(engine.spawn_evolution_shard("web_search"))
        assert swarm.objectives[0]["context"] == {"target_skill": "web_search"}

        chamber = adaptive_test_chamber.register_test_chamber()
        assert chamber.get_status()["healthy"] is True
        assert [item[0] for item in registered if item[0] in {"glados_test_chamber", "glados"}] == [
            "glados_test_chamber",
            "glados",
        ]

        processor = voice_socket_logic.VoiceStreamProcessor(model_instance=FakeWhisper())
        processor.speech_buffer = [(b"\0\0" * 160)]
        assert asyncio.run(processor.get_transcript()) == "stop"
        assert reflex.commands == [("STOP", "audio_stream")]

        coordinator = affect_coordinator.AffectCoordinator(
            orchestrator=SimpleNamespace(_get_service=lambda _name: None)
        )
        assert coordinator.get_mood() == "Curious"
    finally:
        install_service_resolver(None)
        install_service_registration_sink(None)
        install_service_factory_registration_sink(None)

    for module in (
        concept_vector_bridge,
        source_summarizer,
        provenance,
        telemetry_enrichment,
        emotional_coloring,
        plasticity_controller,
        skill_evolution,
        adaptive_test_chamber,
        voice_socket_logic,
        affect_coordinator,
    ):
        source = inspect.getsource(module)
        assert "core.container" not in source
        assert "ServiceContainer" not in source
        assert "core.runtime.service_registry" in source
