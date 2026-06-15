################################################################################


import pytest
import re
from core.mycelium import MycelialNetwork, HardwiredPathway, Hypha

@pytest.fixture
def network():
    # Reset singleton state for clean test
    MycelialNetwork._instance = None
    MycelialNetwork._initialized = False
    return MycelialNetwork()

def test_singleton_safety():
    net1 = MycelialNetwork()
    net2 = MycelialNetwork()
    assert net1 is net2
    assert net1._initialized is True

def test_pathway_validation(network):
    network.register_pathway(
        pathway_id="img_gen",
        pattern=r"draw\s+(.+)",
        skill_name="generate_image",
        param_map={"prompt": 1}
    )
    
    assert "img_gen" in network.pathways
    pw = network.pathways["img_gen"]
    assert isinstance(pw, HardwiredPathway)
    assert pw.skill_name == "generate_image"
    
    match = network.match_hardwired("draw a neon cat")
    assert match is not None
    pw, params = match
    assert params["prompt"] == "a neon cat"


@pytest.mark.parametrize(
    "message",
    ["who are you", "status", "how are you", "help", "what can you do"],
)
def test_conversation_is_not_intercepted_by_canned_mycelium_reflexes(network, message):
    assert network.match_hardwired(message) is None
    assert all(pathway.direct_response is None for pathway in network.pathways.values())

def test_hypha_pydantic(network):
    network.establish_connection("A", "B", priority=2.0)
    assert "A->B" in network.hyphae
    h = network.hyphae["A->B"]
    assert isinstance(h, Hypha)
    assert h.priority == 2.0
    
    h.pulse(success=True)
    assert h.strength > 1.0


def test_dormant_hyphae_are_not_monitored_until_they_carry_traffic(network):
    network.establish_connection("Dormant", "Edge", priority=2.0)
    h = network.hyphae["Dormant->Edge"]

    assert h.pulse_count == 0
    assert network._should_monitor_hypha(h) is False

    h.refresh_heartbeat()
    assert h.strength == 1.0
    assert network._should_monitor_hypha(h) is False

    h.pulse(success=True)
    assert network._should_monitor_hypha(h) is True

def test_infrastructure_mapping(network):
    # Seed mapped files for deterministic routing.
    network.mapped_files = {"core.logic": {"path": "/path/to/logic.py"}}
    network.infrastructure_mapped = True
    
    # Register a pathway with a source file
    network.register_pathway(
        pathway_id="test_geo",
        pattern=r"where\s+is\s+(.+)",
        skill_name="geo_skill",
    )
    network.pathways["test_geo"].source_file = "/path/to/logic.py"
    
    # Establish a physical hypha
    network.hyphae["phys_1"] = Hypha(name="phys_1", source="core.logic", target="core.utils", is_physical=True)
    
    # Reinforce the pathway and verify physical hypha pulses
    network.reinforce("test_geo", success=True)
    assert network.hyphae["phys_1"].strength > 1.0


def test_foreground_boot_defers_infrastructure_mapping_quiet_window(network, monkeypatch, tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "sample.py").write_text("import os\n", encoding="utf-8")
    monkeypatch.setenv("AURA_FOREGROUND_ONLY", "1")
    monkeypatch.setenv("AURA_FOREGROUND_INFRASTRUCTURE_MAPPING_QUIET_S", "180")

    network.map_infrastructure(str(tmp_path))

    report = network.get_infrastructure_report()
    assert report["mapped"] is False
    assert report["mapping_state"] == "deferred"
    assert report["total_modules"] == 0
    assert str(report["deferred_reason"]).startswith("foreground_quiet_window:")


def test_foreground_infrastructure_mapping_force_override(network, monkeypatch, tmp_path):
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "sample.py").write_text("import os\n", encoding="utf-8")
    monkeypatch.setenv("AURA_FOREGROUND_ONLY", "1")

    network.map_infrastructure(str(tmp_path), force=True)

    report = network.get_infrastructure_report()
    assert report["mapped"] is True
    assert report["mapping_state"] == "ready"
    assert report["total_modules"] == 1


def test_setup_schedules_only_one_owned_mapping_thread(network, monkeypatch):
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    started = []

    class FakeThread:
        def __init__(self, *, target, args, kwargs, daemon, name):
            self.target = target
            self.args = args
            self.kwargs = kwargs
            self.daemon = daemon
            self.name = name
            self._alive = False

        def start(self):
            self._alive = True
            started.append(self)

        def is_alive(self):
            return self._alive

    monkeypatch.setattr("core.mycelium.threading.Thread", FakeThread)

    assert network.setup() is True
    assert network.setup() is False
    assert len(started) == 1
    assert started[0].name == "MyceliumInfrastructureMap"


def test_mapping_worker_clears_running_state_after_failure(network, monkeypatch):
    def fail_mapping(*args, **kwargs):
        network._is_mapping = True
        raise OSError("scan failed")

    monkeypatch.setattr(network, "map_infrastructure", fail_mapping)

    network._mapping_worker("/tmp")

    report = network.get_infrastructure_report()
    assert report["mapping_state"] == "failed"
    assert report["mapped"] is False
    assert network._is_mapping is False
    assert "OSError: scan failed" == report["mapping_last_error"]


##
