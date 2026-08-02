import sqlite3
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

LEARNED_SERVICE_MODULES = {
    "sentiment_tracker": ("core.cognitive.sentiment_tracker", "get_sentiment_tracker"),
    "anomaly_detector": ("core.cognitive.anomaly_detector", "AnomalyDetector"),
    "strange_loop": ("core.cognitive.strange_loop", "get_strange_loop"),
    "homeostatic_rl": ("core.cognitive.homeostatic_rl", "get_homeostatic_rl"),
    "topology_evolution": ("core.cognitive.topology_evolution", "TopologyEvolution"),
    "autopoiesis": ("core.cognitive.autopoiesis", "get_autopoiesis_engine"),
    "adaptive_immune_system": ("core.adaptation.adaptive_immunity", "get_adaptive_immune_system"),
    "autonomous_resilience_mesh": ("core.adaptation.autonomous_resilience", "get_autonomous_resilience_mesh"),
    "criticality_regulator": ("core.consciousness.criticality_regulator", "get_criticality_regulator"),
    "alife_dynamics": ("core.consciousness.alife_dynamics", "ALifeDynamics"),
    "alife_extensions": ("core.consciousness.alife_extensions", "ALifeExtensions"),
    "endogenous_fitness": ("core.consciousness.endogenous_fitness", "get_endogenous_fitness"),
}


class Service:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.session_id = "test.cognitive-sensory"
        self.started = False
        self.initialized = False

    async def start(self):
        self.started = True

    async def initialize(self):
        self.initialized = True

    def refresh(self):
        self.refreshed = True
        return {}

    def reconcile_service(self, service):
        self.reconciled_service = service
        return ()

    def bind_reality_reach(self, service, coordinator=None):
        self.reality_reach = service
        self.reality_actuation = coordinator

    def bind_observation_router(self, router):
        self.observation_router = router

    def bind_sensory_fabric(self, router, broker):
        self.observation_router = router
        self.attachment_broker = broker

    def register_configured_devices(self):
        self.configured_devices_registered = True
        return ()

    def setup_hooks(self, orchestrator):
        self.hooked = orchestrator


class SelfModelService(Service):
    @classmethod
    async def load(cls):
        return cls()


class DriveControllerService(Service):
    def __init__(self, orchestrator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orchestrator = orchestrator

    def is_alive(self):
        return self.orchestrator is not None

    def get_status(self):
        return {"energy": 100, "curiosity": 50, "focus": 50}


class RealityActuationService(Service):
    def is_alive(self):
        return True


def test_cognitive_sensory_initializer_degradation_audit_is_clean():
    from tools.audit_degradation import analyze_file

    assert analyze_file(Path("core/orchestrator/initializers/cognitive_sensory.py")) == []


def _install_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_success_modules(monkeypatch, *, will_engine_cls=Service):
    _install_module(monkeypatch, "core.self_model", SelfModel=SelfModelService)
    _install_module(monkeypatch, "core.brain.identity", IdentityService=Service)
    _install_module(monkeypatch, "core.soul", Soul=Service)
    _install_module(
        monkeypatch,
        "core.fictional_ai_synthesis",
        register_all_fictional_engines=lambda orchestrator: {"registered": True},
    )
    _install_module(monkeypatch, "core.brain.personality_engine", PersonalityEngine=Service)
    _install_module(monkeypatch, "core.managers.drive_controller", DriveController=DriveControllerService)
    _install_module(monkeypatch, "core.senses.voice_engine", get_voice_engine=lambda: Service())
    _install_module(
        monkeypatch,
        "core.reality_reach.live",
        get_reality_reach_service=lambda: Service(),
    )
    _install_module(
        monkeypatch,
        "core.reality_reach.transactions",
        get_reality_actuation_coordinator=lambda _service: RealityActuationService(),
    )
    _install_module(
        monkeypatch,
        "core.reality_reach.historian",
        RealityHistorian=Service,
    )
    _install_module(
        monkeypatch,
        "core.reality_reach.digital_twin",
        RealityDigitalTwinGraph=Service,
    )
    _install_module(
        monkeypatch,
        "core.reality_reach.observation_router",
        RealityObservationRouter=Service,
    )
    _install_module(
        monkeypatch,
        "core.reality_reach.attachments",
        DeviceAttachmentBroker=Service,
    )
    _install_module(
        monkeypatch,
        "core.reality_reach.trust_custody",
        KeychainAttachmentTrustStore=type(
            "KeychainAttachmentTrustStore",
            (),
            {"provision_system": staticmethod(lambda path: {"path": path})},
        ),
    )
    _install_module(
        monkeypatch,
        "core.embodiment.hardware_manager",
        get_hardware_manager=lambda: Service(),
    )
    _install_module(
        monkeypatch,
        "core.embodiment.iot_bridge",
        get_iot_bridge=lambda: Service(),
    )
    _install_module(monkeypatch, "core.brain.multimodal_orchestrator", MultimodalOrchestrator=Service)
    _install_module(monkeypatch, "core.brain.composer_node", ComposerNode=Service)
    _install_module(monkeypatch, "core.guardians.memory_guard", MemoryGuard=Service)
    _install_module(monkeypatch, "core.soma.resilience_engine", ResilienceEngine=Service)
    _install_module(monkeypatch, "core.identity.drift_monitor", IdentityDriftMonitor=Service)
    _install_module(monkeypatch, "core.identity.spine", SpiritualSpine=Service)
    _install_module(monkeypatch, "core.self_modification.growth_ladder", GrowthLadder=Service)
    _install_module(monkeypatch, "core.memory.sovereign_pruner", SovereignPruner=Service)
    _install_module(monkeypatch, "core.guardians.governor", SystemGovernor=Service)
    _install_module(monkeypatch, "core.self.will_engine", WillEngine=will_engine_cls)
    _install_module(monkeypatch, "core.state.cellular_substrate", CellularSubstrate=Service)

    for _service_name, (module_path, factory_name) in LEARNED_SERVICE_MODULES.items():
        if factory_name[:1].isupper():
            _install_module(monkeypatch, module_path, **{factory_name: Service})
        else:
            _install_module(monkeypatch, module_path, **{factory_name: lambda: Service()})


def _patch_container(monkeypatch):
    import core.orchestrator.initializers.cognitive_sensory as cognitive_sensory

    registered = {}

    def _register_instance(name, instance, *args, **kwargs):
        registered[name] = instance

    def _get(name, default=None):
        return default

    monkeypatch.setattr(cognitive_sensory.ServiceContainer, "register_instance", staticmethod(_register_instance))
    monkeypatch.setattr(cognitive_sensory.ServiceContainer, "get", staticmethod(_get))
    return registered


@pytest.mark.asyncio
async def test_cognitive_sensory_initializer_returns_complete_boot_report(monkeypatch):
    from core.orchestrator.initializers.cognitive_sensory import init_cognitive_sensory_layer

    _install_success_modules(monkeypatch)
    registered = _patch_container(monkeypatch)
    orchestrator = SimpleNamespace(affect=SimpleNamespace(drive_controller=None))

    report = await init_cognitive_sensory_layer(orchestrator)

    assert report["degraded"] == {}
    assert report["learned_services"] == {"registered": len(LEARNED_SERVICE_MODULES), "expected": len(LEARNED_SERVICE_MODULES)}
    assert "identity_personality" in report["completed"]
    assert "reality_reach" in report["completed"]
    assert "cellular_substrate" in report["completed"]
    assert registered["self_model"] is orchestrator.self_model
    assert registered["drive_engine"] is orchestrator.affect.drive_controller
    assert registered["drive_engine"] is orchestrator.drive_controller
    assert registered["drive_engine"].is_alive() is True
    assert registered["reality_reach"] is orchestrator.reality_reach
    assert registered["reality_reach"].refreshed is True
    assert registered["reality_actuation"] is orchestrator.reality_actuation
    assert registered["reality_actuation"].is_alive() is True
    assert registered["reality_historian"] is orchestrator.reality_historian
    assert registered["reality_historian"].args[0].name == "reality_historian.sqlite3"
    assert registered["reality_digital_twin"] is orchestrator.reality_digital_twin
    assert registered["reality_digital_twin"].args[0].name == (
        "reality_digital_twin.sqlite3"
    )
    assert registered["reality_digital_twin"].kwargs["session_id"] == (
        "test.cognitive-sensory"
    )
    assert registered["reality_digital_twin"].reconciled_service is (
        orchestrator.reality_reach
    )
    assert registered["reality_observation_router"] is orchestrator.reality_observation_router
    assert (
        registered["reality_observation_router"].kwargs["historian"]
        is orchestrator.reality_historian
    )
    assert registered["reality_observation_router"].kwargs["digital_twin"] is (
        orchestrator.reality_digital_twin
    )
    assert registered["reality_observation_router"].started is True
    assert registered["reality_attachment_broker"] is orchestrator.reality_attachment_broker
    assert registered["reality_attachment_broker"].started is True
    assert registered["reality_attachment_broker"].kwargs["digital_twin"] is (
        orchestrator.reality_digital_twin
    )
    assert registered["reality_attachment_broker"].kwargs["trust_store"]["path"].name == (
        "reality_attachment_trust.json"
    )
    assert registered["reality_attachment_broker"].kwargs["trust_store_error"] == ""
    assert registered["hardware_manager"] is orchestrator.hardware_manager
    assert registered["hardware_manager"].started is True
    assert registered["hardware_manager"].configured_devices_registered is True
    assert registered["hardware_manager"].observation_router is orchestrator.reality_observation_router
    assert registered["iot_bridge"] is orchestrator.iot_bridge
    assert registered["iot_bridge"].started is True
    assert registered["iot_bridge"].reality_reach is orchestrator.reality_reach
    assert registered["iot_bridge"].reality_actuation is orchestrator.reality_actuation
    assert registered["iot_bridge"].observation_router is orchestrator.reality_observation_router
    assert registered["iot_bridge"].attachment_broker is orchestrator.reality_attachment_broker
    assert registered["cellular_substrate"] is orchestrator.cellular_substrate


@pytest.mark.asyncio
async def test_cognitive_sensory_initializer_replaces_nonlive_drive_placeholder(monkeypatch):
    from core.orchestrator.initializers.cognitive_sensory import init_cognitive_sensory_layer

    _install_success_modules(monkeypatch)
    registered = _patch_container(monkeypatch)
    stale_controller = SimpleNamespace(get_status=lambda: {"energy": 100})
    orchestrator = SimpleNamespace(
        affect=SimpleNamespace(_drive_controller=stale_controller, drive_controller=stale_controller)
    )

    report = await init_cognitive_sensory_layer(orchestrator)

    assert report["degraded"] == {}
    assert registered["drive_engine"] is not stale_controller
    assert registered["drive_engine"].orchestrator is orchestrator
    assert registered["drive_engine"].is_alive() is True


@pytest.mark.asyncio
async def test_cognitive_sensory_initializer_continues_after_will_engine_failure(monkeypatch):
    from core.orchestrator.initializers.cognitive_sensory import init_cognitive_sensory_layer

    class BrokenWillEngine(Service):
        async def initialize(self):
            reason = "will engine unavailable"
            raise RuntimeError(reason)

    _install_success_modules(monkeypatch, will_engine_cls=BrokenWillEngine)
    registered = _patch_container(monkeypatch)
    orchestrator = SimpleNamespace(affect=SimpleNamespace(drive_controller=None))

    report = await init_cognitive_sensory_layer(orchestrator)

    assert "will_engine" in report["degraded"]
    assert report["degraded"]["will_engine"]["severity"] == "critical"
    assert "cellular_substrate" in report["completed"]
    assert "cellular_substrate" in registered
    assert "will_engine" not in registered


@pytest.mark.asyncio
async def test_keychain_failure_keeps_physical_discovery_but_closes_durable_trust(
    monkeypatch,
):
    from core.orchestrator.initializers.cognitive_sensory import init_cognitive_sensory_layer

    class BrokenTrustStore:
        @staticmethod
        def provision_system(_path):
            raise RuntimeError("Keychain locked")

    _install_success_modules(monkeypatch)
    _install_module(
        monkeypatch,
        "core.reality_reach.trust_custody",
        KeychainAttachmentTrustStore=BrokenTrustStore,
    )
    registered = _patch_container(monkeypatch)
    orchestrator = SimpleNamespace(affect=SimpleNamespace(drive_controller=None))

    report = await init_cognitive_sensory_layer(orchestrator)

    assert "reality_attachment_trust_custody" in report["degraded"]
    assert "reality_sensory_fabric" in report["completed"]
    broker = registered["reality_attachment_broker"]
    assert broker.started is True
    assert broker.kwargs["trust_store"] is None
    assert "Keychain locked" in broker.kwargs["trust_store_error"]


@pytest.mark.asyncio
async def test_historian_failure_keeps_live_sensing_and_actuation_registered(
    monkeypatch,
):
    from core.orchestrator.initializers.cognitive_sensory import (
        init_cognitive_sensory_layer,
    )

    class BrokenHistorian:
        def __init__(self, _path):
            raise RuntimeError("synthetic historian corruption")

    _install_success_modules(monkeypatch)
    _install_module(
        monkeypatch,
        "core.reality_reach.historian",
        RealityHistorian=BrokenHistorian,
    )
    registered = _patch_container(monkeypatch)
    orchestrator = SimpleNamespace(affect=SimpleNamespace(drive_controller=None))

    report = await init_cognitive_sensory_layer(orchestrator)

    assert "reality_historian" in report["degraded"]
    assert "reality_reach" in report["completed"]
    assert "reality_sensory_fabric" in report["completed"]
    assert registered["reality_reach"] is orchestrator.reality_reach
    assert registered["reality_actuation"] is orchestrator.reality_actuation
    assert "reality_historian" not in registered
    assert orchestrator.reality_historian is None
    assert registered["reality_observation_router"].kwargs["historian"] is None
    assert registered["reality_observation_router"].started is True


@pytest.mark.asyncio
async def test_native_sqlite_historian_failure_degrades_only_durable_history(
    monkeypatch,
):
    from core.orchestrator.initializers.cognitive_sensory import (
        init_cognitive_sensory_layer,
    )

    class BrokenHistorian:
        def __init__(self, _path):
            raise sqlite3.OperationalError("synthetic locked database")

    _install_success_modules(monkeypatch)
    _install_module(
        monkeypatch,
        "core.reality_reach.historian",
        RealityHistorian=BrokenHistorian,
    )
    registered = _patch_container(monkeypatch)
    orchestrator = SimpleNamespace(affect=SimpleNamespace(drive_controller=None))

    report = await init_cognitive_sensory_layer(orchestrator)

    assert "reality_historian" in report["degraded"]
    assert "reality_reach" in report["completed"]
    assert "reality_sensory_fabric" in report["completed"]
    assert registered["reality_reach"] is orchestrator.reality_reach
    assert registered["reality_actuation"] is orchestrator.reality_actuation
    assert orchestrator.reality_historian is None
    assert registered["reality_observation_router"].kwargs["historian"] is None
