################################################################################
import asyncio
import json
import logging
import re
import sqlite3
import threading
import time

import pytest

from core.mycelium import HardwiredPathway, Hypha, MycelialNetwork


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


def test_logical_hyphae_do_not_invent_a_continuous_liveness_contract(network):
    network.establish_connection("Dormant", "Edge", priority=2.0)
    h = network.hyphae["Dormant->Edge"]

    assert h.pulse_count == 0
    assert network._should_monitor_hypha(h) is False

    h.refresh_heartbeat()
    assert h.strength == 1.0
    assert network._should_monitor_hypha(h) is False

    h.pulse(success=True)
    assert network._should_monitor_hypha(h) is False

    h.is_physical = True
    assert network._should_monitor_hypha(h) is False

    root = network.establish_neural_root("monitor-test", hardware_id="test-device")
    assert network._should_monitor_hypha(root) is True


def test_route_signal_deduplicates_unchanged_feed_logs(network, caplog):
    payload = {"event": "threshold_shift", "rms_gate": 0.01, "conf_gate": -0.7}

    with caplog.at_level(logging.INFO, logger="Aura.Mycelium"):
        network.route_signal("voice_engine", "sensory_gate", payload)
        network.route_signal("voice_engine", "sensory_gate", payload)

    matching = [
        record
        for record in caplog.records
        if "Signal Routed: voice_engine -> sensory_gate" in record.getMessage()
    ]
    assert len(matching) == 1

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
        raise OSError("scan failed")

    monkeypatch.setattr(network, "map_infrastructure", fail_mapping)
    admission_token = object()
    network._mapping_admission_token = admission_token
    network._is_mapping = True

    network._mapping_worker("/tmp", _admission_token=admission_token)

    report = network.get_infrastructure_report()
    assert report["mapping_state"] == "failed"
    assert report["mapped"] is False
    assert network._is_mapping is False
    assert "OSError: scan failed" == report["mapping_last_error"]


def test_mapping_worker_that_loses_admission_cannot_open_latch_for_third_mapper(
    network, monkeypatch, tmp_path
):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def blocking_generation(*_args, **_kwargs):
        calls.append("owner")
        entered.set()
        assert release.wait(timeout=3.0)
        return True

    monkeypatch.setattr(network, "_map_infrastructure_generation", blocking_generation)
    owner = threading.Thread(
        target=network.map_infrastructure,
        args=(str(tmp_path),),
        kwargs={"force": True},
    )
    owner.start()
    assert entered.wait(timeout=2.0)

    loser = threading.Thread(
        target=network._mapping_worker,
        args=(str(tmp_path),),
        kwargs={"force": True},
    )
    loser.start()
    loser.join(timeout=2.0)

    assert loser.is_alive() is False
    assert network._is_mapping is True
    assert network._mapping_admission_token is not None
    assert network.map_infrastructure(str(tmp_path), force=True) is False
    assert calls == ["owner"]

    release.set()
    owner.join(timeout=3.0)
    assert owner.is_alive() is False
    assert network._is_mapping is False
    assert network._mapping_admission_token is None


def test_direct_mapping_exception_clears_admission_latch(network, monkeypatch, tmp_path):
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    (core_dir / "alpha.py").write_text("VALUE = 1\n", encoding="utf-8")

    def fail_extract(*_args, **_kwargs):
        raise KeyError("unexpected parser failure")

    monkeypatch.setattr(network, "_extract_imports", fail_extract)

    with pytest.raises(KeyError, match="unexpected parser failure"):
        network.map_infrastructure(str(tmp_path), force=True)

    report = network.get_infrastructure_report()
    assert network._is_mapping is False
    assert report["mapping_state"] == "failed"
    assert report["mapping_last_error"] == "KeyError: 'unexpected parser failure'"


def test_default_mapping_covers_runtime_roots_and_root_modules(network, tmp_path):
    for directory in ("core", "interface", "llm"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "aura_main.py").write_text("VALUE = 1\n", encoding="utf-8")

    assert network.map_infrastructure(str(tmp_path), force=True) is True

    assert set(network.get_mapped_files_snapshot()) == {
        "aura_main",
        "core.module",
        "interface.module",
        "llm.module",
    }


def test_infrastructure_mapping_publishes_one_complete_generation(
    network, monkeypatch, tmp_path
):
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    (core_dir / "alpha.py").write_text("import core.beta\n", encoding="utf-8")
    (core_dir / "beta.py").write_text("VALUE = 1\n", encoding="utf-8")

    entered = threading.Event()
    release = threading.Event()
    original_extract = network._extract_imports

    def blocking_extract(file_path, base_dir):
        entered.set()
        assert release.wait(timeout=2.0)
        return original_extract(file_path, base_dir)

    monkeypatch.setattr(network, "_extract_imports", blocking_extract)
    worker = threading.Thread(
        target=network.map_infrastructure,
        args=(str(tmp_path),),
        kwargs={"force": True},
    )
    worker.start()
    assert entered.wait(timeout=2.0)

    assert network.get_mapped_files_snapshot() == {}

    release.set()
    worker.join(timeout=3.0)
    assert worker.is_alive() is False
    snapshot = network.get_mapped_files_snapshot()
    assert set(snapshot) == {"core.alpha", "core.beta"}
    assert snapshot["core.alpha"]["imports"] == ["core.beta"]


def test_mapped_files_snapshot_is_detached(network, tmp_path):
    alpha_path = str(tmp_path / "alpha.py")
    network.mapped_files["core.alpha"] = {
        "path": alpha_path,
        "imports": ["core.beta"],
    }

    snapshot = network.get_mapped_files_snapshot()
    snapshot["core.alpha"]["path"] = "/changed.py"
    snapshot["core.alpha"]["imports"].append("core.changed")

    assert network.mapped_files["core.alpha"]["path"] == alpha_path
    assert network.mapped_files["core.alpha"]["imports"] == ["core.beta"]


def test_qualia_tension_uses_coherent_topology_counts(monkeypatch):
    from core.consciousness.qualia_synthesizer import QualiaSynthesizer
    from core.container import ServiceContainer

    class SnapshotOnlyMycelium:
        @property
        def hyphae(self):
            raise AssertionError("qualia synthesis must not read mutable topology")

        @staticmethod
        def get_topology_counts():
            return {"hyphae": 75}

    mycelium = SnapshotOnlyMycelium()
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: mycelium if name == "mycelium" else default
        ),
    )

    synthesizer = QualiaSynthesizer()
    synthesizer.synthesize({}, {})

    assert synthesizer.q_vector[5] == pytest.approx(0.75 * 0.15)


def test_force_remap_keeps_previous_generation_until_atomic_replacement(
    network, monkeypatch, tmp_path
):
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    alpha = core_dir / "alpha.py"
    beta = core_dir / "beta.py"
    alpha.write_text("import core.beta\n", encoding="utf-8")
    beta.write_text("VALUE = 1\n", encoding="utf-8")

    assert network.map_infrastructure(str(tmp_path), force=True) is True
    first = network.get_graph_snapshot()
    assert first["mapping_generation"] == 1
    assert set(first["mapped_files"]) == {"core.alpha", "core.beta"}
    assert "import:core.alpha->core.beta" in first["topology"]["hyphae"]

    alpha.unlink()
    gamma = core_dir / "gamma.py"
    gamma.write_text("import core.beta\n", encoding="utf-8")
    entered = threading.Event()
    release = threading.Event()
    original_extract = network._extract_imports

    def blocking_extract(file_path, base_dir):
        entered.set()
        assert release.wait(timeout=2.0)
        return original_extract(file_path, base_dir)

    monkeypatch.setattr(network, "_extract_imports", blocking_extract)
    worker = threading.Thread(
        target=network.map_infrastructure,
        args=(str(tmp_path),),
        kwargs={"force": True},
    )
    worker.start()
    assert entered.wait(timeout=2.0)

    during = network.get_graph_snapshot()
    assert during["mapping_generation"] == 1
    assert during["mapping_state"] == "refreshing"
    assert set(during["mapped_files"]) == {"core.alpha", "core.beta"}

    release.set()
    worker.join(timeout=3.0)
    assert worker.is_alive() is False
    after = network.get_graph_snapshot()
    assert after["mapping_generation"] == 2
    assert after["mapping_state"] == "ready"
    assert set(after["mapped_files"]) == {"core.beta", "core.gamma"}
    assert "import:core.alpha->core.beta" not in after["topology"]["hyphae"]
    assert "import:core.gamma->core.beta" in after["topology"]["hyphae"]


def test_force_remap_preserves_import_learning_and_cannot_collide_with_logical_edge(
    network, tmp_path
):
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    (core_dir / "alpha.py").write_text("import core.beta\n", encoding="utf-8")
    (core_dir / "beta.py").write_text("VALUE = 1\n", encoding="utf-8")
    network.establish_connection("core.alpha", "core.beta", priority=2.0)

    assert network.map_infrastructure(str(tmp_path), force=True) is True
    edge_id = "import:core.alpha->core.beta"
    assert set(network.get_graph_snapshot()["topology"]["hyphae"]) >= {
        "core.alpha->core.beta",
        edge_id,
    }
    assert network.pulse_hypha(edge_id, success=True) is True
    assert network.log_hypha(edge_id, None, "learned import route") is True
    before = network.get_graph_snapshot()["topology"]["hyphae"][edge_id]

    assert network.map_infrastructure(str(tmp_path), force=True) is True
    after = network.get_graph_snapshot()["topology"]["hyphae"][edge_id]

    for field in ("strength", "created_at", "last_pulse", "pulse_count", "trace"):
        assert after[field] == before[field]


def test_force_remap_clears_annotation_when_backing_file_disappears(network, tmp_path):
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    alpha = core_dir / "alpha.py"
    alpha.write_text("VALUE = 1\n", encoding="utf-8")
    (core_dir / "beta.py").write_text("VALUE = 1\n", encoding="utf-8")
    network.register_pathway("alpha_path", r"alpha", "alpha")

    assert network.map_infrastructure(str(tmp_path), force=True) is True
    assert network.pathways["alpha_path"].source_file == str(alpha)

    alpha.unlink()
    assert network.map_infrastructure(str(tmp_path), force=True) is True
    assert network.pathways["alpha_path"].source_file is None
    assert network.pathways["alpha_path"].dependencies == []


def test_hypha_reads_are_detached_and_mutations_use_owner_api(network):
    network.establish_connection("source", "target")
    detached = network.get_hypha("source", "target")
    assert detached is not None

    detached.pulse(success=True)
    assert network.get_hypha("source", "target").pulse_count == 0

    assert network.pulse_hypha("source", "target", success=True) is True
    assert network.get_hypha("source", "target").pulse_count == 1


def test_topology_summary_read_model_does_not_wait_for_graph_lock(network):
    network.establish_connection("source", "target")
    entered = threading.Event()
    release = threading.Event()

    def hold_topology_lock():
        with MycelialNetwork._lock:
            entered.set()
            assert release.wait(timeout=2.0)

    holder = threading.Thread(target=hold_topology_lock)
    holder.start()
    assert entered.wait(timeout=1.0)
    started = time.monotonic()
    summary = network.get_topology_summary()
    elapsed = time.monotonic() - started
    release.set()
    holder.join(timeout=2.0)

    assert elapsed < 0.05
    assert summary["links"] >= 1
    assert holder.is_alive() is False


def test_graph_snapshot_is_detached_across_every_published_surface(network, tmp_path):
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    (core_dir / "alpha.py").write_text("import core.beta\n", encoding="utf-8")
    (core_dir / "beta.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert network.map_infrastructure(str(tmp_path), force=True) is True

    snapshot = network.get_graph_snapshot()
    snapshot["mapped_files"]["core.alpha"]["imports"].append("core.changed")
    snapshot["centrality"]["core.beta"] = 999
    snapshot["topology"]["hyphae"]["import:core.alpha->core.beta"]["strength"] = 999

    current = network.get_graph_snapshot()
    assert current["mapped_files"]["core.alpha"]["imports"] == ["core.beta"]
    assert current["centrality"]["core.beta"] == 1
    assert (
        current["topology"]["hyphae"]["import:core.alpha->core.beta"]["strength"]
        != 999
    )


def test_owned_refresh_failure_retains_generation_and_clears_running_state(
    network, monkeypatch, tmp_path
):
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    (core_dir / "alpha.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert network.map_infrastructure(str(tmp_path), force=True) is True

    def fail_refresh(*_args, **_kwargs):
        with network._mapping_lock:
            network._is_mapping = True
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(network, "map_infrastructure", fail_refresh)
    assert network.setup(force=True) is True
    worker = network._mapping_thread
    assert worker is not None
    worker.join(timeout=3.0)
    assert worker.is_alive() is False

    report = network.get_infrastructure_report()
    assert report["mapped"] is True
    assert report["mapping_generation"] == 1
    assert report["mapping_state"] == "ready_with_refresh_error"
    assert report["mapping_last_error"] == "RuntimeError: refresh failed"
    assert report["total_modules"] == 1
    assert network._mapping_thread is None


def test_shutdown_during_mapping_cannot_publish_into_retired_network(
    network, monkeypatch, tmp_path
):
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    (core_dir / "alpha.py").write_text("VALUE = 1\n", encoding="utf-8")
    entered = threading.Event()
    release = threading.Event()

    def blocking_extract(_file_path, _base_dir):
        entered.set()
        assert release.wait(timeout=2.0)
        return []

    monkeypatch.setattr(network, "_extract_imports", blocking_extract)
    worker = threading.Thread(
        target=network.map_infrastructure,
        args=(str(tmp_path),),
        kwargs={"force": True},
    )
    worker.start()
    assert entered.wait(timeout=2.0)

    network._stop_event.set()
    release.set()
    worker.join(timeout=3.0)

    assert worker.is_alive() is False
    assert network.get_mapped_files_snapshot() == {}
    assert network.infrastructure_mapped is False


def test_shutdown_drains_owned_mapping_worker_without_lock_inversion(
    network, monkeypatch
):
    entered = threading.Event()

    def wait_for_shutdown(*_args, **_kwargs):
        with network._mapping_lock:
            network._is_mapping = True
        entered.set()
        assert network._stop_event.wait(timeout=2.0)
        with network._mapping_lock:
            network._is_mapping = False
        return False

    monkeypatch.setattr(network, "map_infrastructure", wait_for_shutdown)
    assert network.setup(force=True) is True
    assert entered.wait(timeout=2.0)

    network.shutdown()

    assert network._mapping_thread is None
    assert MycelialNetwork._instance is None
    assert MycelialNetwork._initialized is False


def test_retired_instance_shutdown_cannot_clear_replacement_singleton(network):
    retired = network
    retired.shutdown()
    replacement = MycelialNetwork()

    retired.shutdown()

    assert MycelialNetwork._instance is replacement
    assert MycelialNetwork._initialized is True
    assert replacement._stop_event.is_set() is False


def test_discovery_updates_and_topology_snapshots_share_one_lock(network):
    errors = []

    def write_discovery_state():
        try:
            for index in range(1_000):
                network.record_execution(
                    f"request {index}",
                    f"skill_{index}",
                    {"index": index},
                    True,
                )
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            errors.append(exc)

    writer = threading.Thread(target=write_discovery_state)
    writer.start()
    while writer.is_alive():
        snapshot = network.get_network_topology()
        assert isinstance(snapshot["discovery_candidates"], dict)
    writer.join(timeout=2.0)

    assert writer.is_alive() is False
    assert errors == []
    assert len(network.get_network_topology()["discovery_candidates"]) == 1_000


@pytest.mark.asyncio
async def test_versioned_vault_roundtrip_restores_one_coherent_generation(
    network,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("AURA_ROOT", str(tmp_path))
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    (core_dir / "alpha.py").write_text("import core.beta\n", encoding="utf-8")
    (core_dir / "beta.py").write_text("VALUE = 1\n", encoding="utf-8")
    network.register_pathway(
        pathway_id="vault_search",
        pattern=r"vault\s+(.+)",
        skill_name="search_vault",
        param_map={"query": 1},
    )
    assert network.map_infrastructure(str(tmp_path), force=True) is True
    before = network.get_graph_snapshot()
    assert await network.vault_sync() is True

    network.register_pathway(
        pathway_id="post_vault_change",
        pattern=r"changed\s+(.+)",
        skill_name="changed_skill",
    )
    network.establish_connection("changed", "edge")

    assert await MycelialNetwork.restore_from_vault() is True
    after = network.get_graph_snapshot()
    match = network.match_hardwired("vault retained query")

    assert set(after["mapped_files"]) == set(before["mapped_files"])
    assert set(after["topology"]["hyphae"]) == set(before["topology"]["hyphae"])
    assert "post_vault_change" not in after["topology"]["pathways"]
    assert match is not None and match[1] == {"query": "retained query"}
    assert network.direct_roots["vault_search"] == "search_vault"
    assert network._aegis_locked is True
    assert after["mapping_generation"] > before["mapping_generation"]


@pytest.mark.asyncio
async def test_invalid_vault_generation_cannot_partially_replace_topology(
    network,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("AURA_ROOT", str(tmp_path))
    assert await network.vault_sync() is True
    vault_path = tmp_path / "data" / "mycelium_vault.db"
    with sqlite3.connect(vault_path) as connection:
        row = connection.execute(
            "SELECT data FROM aegis_vault WHERE key = ?",
            ("topology_v3",),
        ).fetchone()
        assert row is not None
        payload = json.loads(row[0])
        payload["pathways"]["corrupt"] = {
            "pathway_id": "different_identity",
            "pattern": "corrupt",
            "skill_name": "corrupt",
        }
        connection.execute(
            "UPDATE aegis_vault SET data = ? WHERE key = ?",
            (json.dumps(payload), "topology_v3"),
        )
        connection.commit()

    before = network.get_graph_snapshot()
    assert await MycelialNetwork.restore_from_vault() is False
    after = network.get_graph_snapshot()

    assert after == before
    assert network._aegis_locked is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    ("invalid_regex", "missing_path", "malformed_imports", "external_pathway_source"),
)
async def test_vault_rejects_malformed_nested_surfaces_without_partial_restore(
    network,
    monkeypatch,
    tmp_path,
    corruption,
):
    monkeypatch.setenv("AURA_ROOT", str(tmp_path))
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    (core_dir / "alpha.py").write_text("import core.beta\n", encoding="utf-8")
    (core_dir / "beta.py").write_text("VALUE = 1\n", encoding="utf-8")
    network.register_pathway("vault_search", r"vault\s+(.+)", "search_vault")
    assert network.map_infrastructure(str(tmp_path), force=True) is True
    assert await network.vault_sync() is True

    vault_path = tmp_path / "data" / "mycelium_vault.db"
    with sqlite3.connect(vault_path) as connection:
        row = connection.execute(
            "SELECT data FROM aegis_vault WHERE key = ?",
            ("topology_v3",),
        ).fetchone()
        assert row is not None
        payload = json.loads(row[0])
        if corruption == "invalid_regex":
            payload["pathways"]["vault_search"]["pattern"] = "["
        elif corruption == "missing_path":
            payload["mapped_files"]["core.alpha"].pop("path")
        elif corruption == "malformed_imports":
            payload["mapped_files"]["core.alpha"]["imports"] = None
        else:
            payload["pathways"]["vault_search"]["source_file"] = "/outside/map.py"
        connection.execute(
            "UPDATE aegis_vault SET data = ? WHERE key = ?",
            (json.dumps(payload), "topology_v3"),
        )
        connection.commit()

    before = network.get_graph_snapshot()
    assert await MycelialNetwork.restore_from_vault() is False
    assert network.get_graph_snapshot() == before


@pytest.mark.asyncio
async def test_vault_rebases_monotonic_ages_instead_of_persisting_process_clock(
    network,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("AURA_ROOT", str(tmp_path))
    assert await network.vault_sync() is True
    vault_path = tmp_path / "data" / "mycelium_vault.db"

    with sqlite3.connect(vault_path) as connection:
        row = connection.execute(
            "SELECT data FROM aegis_vault WHERE key = ?",
            ("topology_v3",),
        ).fetchone()
        assert row is not None
        payload = json.loads(row[0])
        pathway = payload["pathways"]["direct_web_search"]
        root = payload["hyphae"]["voice_presence->hardware:macos_say"]
        assert "last_matched" not in pathway
        assert "created_at" not in root
        assert "last_pulse" not in root
        pathway["last_matched_age_s"] = 30.0
        pathway["created_at"] = min(
            float(pathway["created_at"]),
            float(payload["captured_at_unix"]) - 31.0,
        )
        root["created_age_s"] = 90.0
        root["last_pulse_age_s"] = 45.0
        connection.execute(
            "UPDATE aegis_vault SET data = ? WHERE key = ?",
            (json.dumps(payload), "topology_v3"),
        )
        connection.commit()

    assert await MycelialNetwork.restore_from_vault() is True
    restored_at = time.monotonic()
    restored_root = network.get_hypha("voice_presence", "hardware:macos_say")
    assert restored_root is not None
    assert restored_at - restored_root.created_at == pytest.approx(90.0, abs=1.0)
    assert restored_at - restored_root.last_pulse == pytest.approx(45.0, abs=1.0)
    assert (
        restored_at - network.pathways["direct_web_search"].last_matched
        == pytest.approx(30.0, abs=1.0)
    )


@pytest.mark.asyncio
async def test_retired_instance_cannot_overwrite_replacement_vault(
    network,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("AURA_ROOT", str(tmp_path))
    assert await network.vault_sync() is True
    network.shutdown()
    replacement = MycelialNetwork()

    assert await network.vault_sync() is False
    assert MycelialNetwork._instance is replacement


@pytest.mark.asyncio
async def test_mycelial_graph_route_uses_canonical_mapped_file_snapshot(monkeypatch):
    from interface.routes import subsystems as subsystem_routes

    caller_thread = threading.get_ident()
    snapshot_threads = []
    transform_threads = []
    response_threads = []

    class ThreadObservedDict(dict):
        def items(self):
            transform_threads.append(threading.get_ident())
            return super().items()

    class MutationSensitiveMapping(dict):
        def __iter__(self):
            raise RuntimeError("raw mapped_files iteration is unsafe")

        def items(self):
            raise RuntimeError("raw mapped_files iteration is unsafe")

    class DummyMycelium:
        mapped_files = MutationSensitiveMapping({"unsafe": {"path": "/unsafe.py"}})
        _centrality = MutationSensitiveMapping({"unsafe": 999})
        _critical_modules = []

        def get_graph_snapshot(self):
            snapshot_threads.append(threading.get_ident())
            return {
                "topology": {
                    "hyphae": ThreadObservedDict(),
                    "pathways": {},
                    "system_cohesion": 1.0,
                    "pathway_count": 0,
                    "critical_modules": [],
                },
                "mapped_files": {
                    "core.module_000": {"path": "/core/module_000.py"}
                },
                "centrality": {"core.module_000": 7},
                "mapping_generation": 3,
                "mapping_state": "ready",
            }

        def get_network_topology(self):
            raise AssertionError("route must use the combined canonical snapshot")

    monkeypatch.setattr(
        subsystem_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: DummyMycelium()
            if name == "mycelium"
            else default
        ),
    )
    real_json_response = subsystem_routes.JSONResponse

    def observed_json_response(*args, **kwargs):
        response_threads.append(threading.get_ident())
        return real_json_response(*args, **kwargs)

    monkeypatch.setattr(subsystem_routes, "JSONResponse", observed_json_response)

    response = await subsystem_routes.api_mycelial_graph()
    payload = json.loads(response.body)
    node_ids = {node["id"] for node in payload["nodes"]}

    assert node_ids == {"core.module_000"}
    assert payload["nodes"][0]["centrality"] == 7
    assert payload["mapping_generation"] == 3
    assert payload["mapping_state"] == "ready"
    assert snapshot_threads and snapshot_threads[0] != caller_thread
    assert transform_threads and all(thread != caller_thread for thread in transform_threads)
    assert response_threads and response_threads[0] != caller_thread


@pytest.mark.asyncio
async def test_mycelial_graph_route_singleflights_serialization_and_invalidates_on_structure(
    monkeypatch,
):
    from interface.routes import subsystems as subsystem_routes

    class DummyMycelium:
        structure_revision = 1

        def get_route_cache_token(self):
            return id(self), self.structure_revision

    mycelium = DummyMycelium()
    build_calls = []

    def build(owner):
        build_calls.append(owner.structure_revision)
        time.sleep(0.05)
        return subsystem_routes.JSONResponse(
            {
                "nodes": [{"id": f"revision-{owner.structure_revision}"}],
                "links": [],
            }
        )

    monkeypatch.setattr(
        subsystem_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: mycelium if name == "mycelium" else default
        ),
    )
    monkeypatch.setattr(subsystem_routes, "_build_mycelial_graph_response", build)
    subsystem_routes._MYCELIAL_GRAPH_RESPONSE_CACHE.clear()

    responses = await asyncio.gather(
        *(subsystem_routes.api_mycelial_graph() for _ in range(12))
    )

    assert build_calls == [1]
    assert len({bytes(response.body) for response in responses}) == 1
    cache_states = [response.headers["x-aura-snapshot-cache"] for response in responses]
    assert cache_states.count("miss") == 1
    assert cache_states.count("hit") == 11
    assert len({id(response) for response in responses}) == 12

    mycelium.structure_revision = 2
    refreshed = await subsystem_routes.api_mycelial_graph()

    assert build_calls == [1, 2]
    assert refreshed.headers["x-aura-snapshot-cache"] == "miss"
    assert json.loads(refreshed.body)["nodes"] == [{"id": "revision-2"}]


@pytest.mark.asyncio
async def test_mycelium_route_uses_one_runtime_snapshot(monkeypatch):
    from interface.routes import subsystems as subsystem_routes

    caller_thread = threading.get_ident()
    snapshot_threads = []
    response_threads = []

    class DummyMycelium:
        @staticmethod
        def get_runtime_snapshot():
            snapshot_threads.append(threading.get_ident())
            return {
                "topology": {
                    "pathway_count": 2,
                    "hyphae": {"alpha->beta": {}},
                },
                "infrastructure": {
                    "mapping_generation": 9,
                    "mapping_state": "ready",
                },
            }

        @staticmethod
        def get_network_topology():
            raise AssertionError("route must not split the runtime snapshot")

        @staticmethod
        def get_infrastructure_report():
            raise AssertionError("route must not split the runtime snapshot")

    monkeypatch.setattr(
        subsystem_routes.ServiceContainer,
        "get",
        staticmethod(
            lambda name, default=None: DummyMycelium()
            if name == "mycelium"
            else default
        ),
    )
    real_json_response = subsystem_routes.JSONResponse

    def observed_json_response(*args, **kwargs):
        response_threads.append(threading.get_ident())
        return real_json_response(*args, **kwargs)

    monkeypatch.setattr(subsystem_routes, "JSONResponse", observed_json_response)

    response = await subsystem_routes.api_mycelium()
    payload = json.loads(response.body)

    assert payload["pathway_count"] == 2
    assert payload["infrastructure"]["mapping_generation"] == 9
    assert payload["infrastructure"]["mapping_state"] == "ready"
    assert snapshot_threads and snapshot_threads[0] != caller_thread
    assert response_threads and response_threads[0] != caller_thread


def test_hardwired_match_returns_detached_pathway(network):
    network.register_pathway(
        "detached_match",
        r"detached\s+(.+)",
        "detached_skill",
        param_map={"value": 1},
        priority=10.0,
    )

    match = network.match_hardwired("detached value")
    assert match is not None
    pathway, params = match
    pathway.confidence = 0.0
    pathway.param_map["injected"] = "alias"
    pathway.dependencies.append("mutated")

    owned = network.pathways["detached_match"]
    assert params == {"value": "value"}
    # Newly-registered pathways start at the honest untested default (0.5),
    # not a fabricated perfect 1.0. The point of this test is isolation: the
    # owned pathway must be UNCHANGED by mutations to the detached copy.
    assert owned.confidence == 0.5
    assert "injected" not in owned.param_map
    assert owned.dependencies == []


def test_hardwired_match_discards_pathway_replaced_during_regex_search(network):
    entered = threading.Event()
    release = threading.Event()
    outcome = {}

    class BlockingPattern:
        def __deepcopy__(self, _memo):
            return self

        def search(self, text):
            entered.set()
            assert release.wait(timeout=2.0)
            return re.search(r"race", text)

    network.register_pathway(
        "replacement_race", r"race", "stale_skill", priority=10.0
    )
    network.pathways["replacement_race"].pattern = BlockingPattern()

    matcher = threading.Thread(
        target=lambda: outcome.setdefault("match", network.match_hardwired("race"))
    )
    matcher.start()
    assert entered.wait(timeout=2.0)
    network.register_pathway(
        "replacement_race", r"does-not-match", "replacement_skill", priority=10.0
    )
    release.set()
    matcher.join(timeout=3.0)

    assert matcher.is_alive() is False
    assert outcome["match"] is None
    assert network.pathways["replacement_race"].skill_name == "replacement_skill"


def test_retired_cached_reference_routes_only_to_live_replacement(network):
    retired = network
    retired.shutdown()
    replacement = MycelialNetwork()

    retired.register_pathway("cached_route", r"cached", "live_skill")
    retired.establish_connection("cached", "live")
    assert retired.pulse_hypha("cached", "live", success=True) is True

    assert retired.pathways == {}
    assert retired.hyphae == {}
    assert "cached_route" in replacement.pathways
    assert replacement.get_hypha("cached", "live").pulse_count == 1
    assert retired.get_topology_summary() == replacement.get_topology_summary()


def test_system_cohesion_read_routes_to_live_replacement(network):
    retired = network
    retired.shutdown()
    replacement = MycelialNetwork()

    with MycelialNetwork._lock:
        for hypha in replacement.hyphae.values():
            hypha.strength = 0.2
        for pathway in replacement.pathways.values():
            pathway.confidence = 0.4

    assert retired.get_system_cohesion() == replacement.get_system_cohesion()
    assert retired.get_system_cohesion() < 0.7


def test_migrated_cached_hypha_callers_follow_live_replacement(network):
    from core.consciousness.homeostatic_coupling import HomeostaticCoupling
    from core.senses.voice_engine import SovereignVoiceEngine

    retired = network
    retired.shutdown()
    replacement = MycelialNetwork()

    coupling = HomeostaticCoupling.__new__(HomeostaticCoupling)
    coupling._mycelium = retired
    voice = SovereignVoiceEngine.__new__(SovereignVoiceEngine)
    voice._mycelium = retired

    coupling._pulse_root("cognition", success=True)
    voice._pulse_hypha("voice_engine", "cognition", success=True)

    homeostasis_edge = replacement.get_hypha("homeostasis", "cognition")
    voice_edge = replacement.get_hypha("voice_engine", "cognition")
    assert homeostasis_edge is not None and homeostasis_edge.pulse_count == 1
    assert voice_edge is not None and voice_edge.pulse_count == 1
    assert retired.hyphae == {}


def test_fully_retired_instance_rejects_writes_without_replacement(network):
    retired = network
    retired.shutdown()

    with pytest.raises(RuntimeError, match="no active owner"):
        retired.establish_connection("retired", "edge")
    with pytest.raises(RuntimeError, match="no active owner"):
        retired.register_pathway("retired", r"retired", "retired_skill")
    assert retired.pulse_hypha("retired", "edge") is False
    assert retired.route_signal("retired", "edge", {}) is False
    assert retired.match_hardwired("retired") is None
    assert retired.pathways == {}
    assert retired.hyphae == {}


@pytest.mark.asyncio
async def test_rooted_flow_handle_persists_logs_and_exposes_absorbed_failure(network):
    network.reflex = None

    async with network.rooted_flow(
        "owner_backed", "flow", activity="successful owner-backed flow"
    ) as flow:
        flow.log("custom trace persisted")

    successful = network.get_hypha("owner_backed", "flow")
    assert successful is not None
    assert any("custom trace persisted" in entry for entry in successful.trace)
    assert flow.failed is False

    async with network.rooted_flow(
        "owner_backed", "failed", activity="absorbed failure"
    ) as failed_flow:
        raise RuntimeError("optimization failed")

    assert failed_flow.failed is True
    assert isinstance(failed_flow.error, RuntimeError)
    failed = network.get_hypha("owner_backed", "failed")
    assert failed is not None
    assert any("STALL/FAILURE" in entry for entry in failed.trace)


@pytest.mark.asyncio
async def test_rooted_flow_enforces_timeout_and_exposes_timeout_failure(network):
    network.reflex = None

    started = time.monotonic()
    async with network.rooted_flow(
        "owner_backed",
        "timed_out",
        activity="bounded owner-backed flow",
        timeout=0.01,
    ) as flow:
        await asyncio.sleep(0.2)
    elapsed = time.monotonic() - started

    assert elapsed < 0.15
    assert flow.failed is True
    assert isinstance(flow.error, TimeoutError)
    timed_out = network.get_hypha("owner_backed", "timed_out")
    assert timed_out is not None
    assert timed_out.pulse_count == 1
    assert any("STALL/FAILURE" in entry for entry in timed_out.trace)


@pytest.mark.asyncio
async def test_rooted_flow_bounds_emergency_override_without_masking_original(network):
    class HangingReflex:
        @staticmethod
        async def trigger_reflex(_signal, _metadata):
            await asyncio.sleep(1.0)

    network.reflex = HangingReflex()
    started = time.monotonic()
    async with network.rooted_flow(
        "owner_backed",
        "override_timeout",
        activity="bounded recovery",
        timeout=0.01,
    ) as flow:
        raise RuntimeError("original flow failure")
    elapsed = time.monotonic() - started

    assert elapsed < 0.25
    assert isinstance(flow.error, RuntimeError)
    assert str(flow.error) == "original flow failure"
    with pytest.raises(RuntimeError, match="original flow failure"):
        flow.raise_for_status()


@pytest.mark.asyncio
async def test_rooted_flow_completion_rebinds_to_replacement_owner(network):
    async with network.rooted_flow(
        "owner_replacement",
        "success",
        activity="flow spanning owner replacement",
    ) as flow:
        network.shutdown()
        replacement = MycelialNetwork()

    edge = replacement.get_hypha("owner_replacement", "success")
    assert flow.failed is False
    assert edge is not None
    assert edge.pulse_count == 1
    assert any("SUCCESS: flow spanning owner replacement" in item for item in edge.trace)


@pytest.mark.asyncio
async def test_rooted_flow_failure_rebinds_to_replacement_owner(network):
    network.reflex = None

    async with network.rooted_flow(
        "owner_replacement",
        "failure",
        activity="failed flow spanning owner replacement",
    ) as flow:
        network.shutdown()
        replacement = MycelialNetwork()
        replacement.reflex = None
        raise RuntimeError("failure after owner replacement")

    edge = replacement.get_hypha("owner_replacement", "failure")
    assert flow.failed is True
    assert str(flow.error) == "failure after owner replacement"
    assert edge is not None
    assert edge.pulse_count == 1
    assert any("STALL/FAILURE" in item for item in edge.trace)


@pytest.mark.asyncio
async def test_maintenance_heartbeat_advances_topology_revision(network):
    network.establish_neural_root("heartbeat", hardware_id="revision")
    network.pulse_hypha("heartbeat", "hardware:revision", success=True)
    with MycelialNetwork._lock:
        network.hyphae["heartbeat->hardware:revision"].last_pulse = time.monotonic() - 301.0
        before_revision = network._topology_revision
        before_structure_revision = network._topology_structure_revision

    await network._pulse_once()

    refreshed = network.get_hypha("heartbeat", "hardware:revision")
    assert refreshed is not None
    assert refreshed.last_pulse > time.monotonic() - 5.0
    assert network._topology_revision == before_revision + 1
    assert network._topology_structure_revision == before_structure_revision


def test_route_cache_token_changes_only_for_structural_topology_mutation(network):
    initial_token = network.get_route_cache_token()
    network.establish_connection("cache", "structure", priority=1.0)
    structural_token = network.get_route_cache_token()
    network.pulse_hypha("cache", "structure", success=True)

    assert structural_token != initial_token
    assert network.get_route_cache_token() == structural_token


@pytest.mark.asyncio
async def test_learning_meta_evolution_does_not_read_unassigned_suppressed_result(
    network, monkeypatch, caplog
):
    from core.orchestrator.mixins import learning_evolution

    network.reflex = None

    class FailingMetaEvolution:
        @staticmethod
        async def run_optimization_cycle():
            raise RuntimeError("cycle failed inside flow")

    class DummyLearning(learning_evolution.LearningEvolutionMixin):
        @staticmethod
        def _emit_telemetry(*_args, **_kwargs):
            return None

    services = {
        "mycelial_network": network,
        "meta_evolution": FailingMetaEvolution(),
    }
    monkeypatch.setattr(
        learning_evolution.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
    )

    with caplog.at_level(logging.ERROR, logger=learning_evolution.__name__):
        await DummyLearning()._run_meta_evolution()

    assert "failed inside rooted flow" in caplog.text
    assert "UnboundLocalError" not in caplog.text


@pytest.mark.asyncio
async def test_meta_evolution_false_result_records_failure_without_success_pulse(
    network, monkeypatch
):
    from core.orchestrator.mixins import learning_evolution

    network.reflex = None

    class RefusedMetaEvolution:
        @staticmethod
        async def run_optimization_cycle():
            return {"ok": False, "error": "optimization refused"}

    class DummyLearning(learning_evolution.LearningEvolutionMixin):
        @staticmethod
        def _emit_telemetry(*_args, **_kwargs):
            return None

    services = {
        "mycelial_network": network,
        "meta_evolution": RefusedMetaEvolution(),
    }
    monkeypatch.setattr(
        learning_evolution.ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: services.get(name, default)),
    )

    await DummyLearning()._run_meta_evolution()

    edge = network.get_hypha("meta_evolution", "cognition")
    assert edge is not None
    assert any("STALL/FAILURE" in entry for entry in edge.trace)
    assert not any("SUCCESS: meta-evolution optimization cycle" in entry for entry in edge.trace)


@pytest.mark.asyncio
async def test_meta_evolution_reports_absorbed_rooted_flow_failure(
    network, monkeypatch
):
    from core.cognition import meta_cognition

    network.reflex = None

    class FailingScratchpad:
        @staticmethod
        async def think_recursive(**_kwargs):
            raise RuntimeError("audit failed inside rooted flow")

    services = {
        "mycelial_network": network,
        "scratchpad_engine": FailingScratchpad(),
        "self_modification_engine": object(),
        "hephaestus_engine": object(),
    }
    monkeypatch.setattr(
        meta_cognition,
        "get_runtime_service",
        lambda name, default=None: services.get(name, default),
    )
    engine = meta_cognition.MetaEvolutionEngine()

    result = await engine.run_optimization_cycle()

    assert result["ok"] is False
    assert result["error"] == "audit failed inside rooted flow"
    assert engine._is_optimizing is False


def test_mapper_retries_when_pathway_is_replaced_during_annotation(
    network, monkeypatch, tmp_path
):
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    alpha = core_dir / "alpha_skill.py"
    beta = core_dir / "beta_skill.py"
    alpha.write_text("VALUE = 'alpha'\n", encoding="utf-8")
    beta.write_text("VALUE = 'beta'\n", encoding="utf-8")
    network.register_pathway("annotation_race", r"alpha", "alpha_skill")

    entered = threading.Event()
    release = threading.Event()
    calls = []
    outcome = {}
    original_builder = network._build_pathway_annotations

    def blocking_builder(pathway_skills, all_files, dependency_graph):
        calls.append(dict(pathway_skills))
        if len(calls) == 1:
            entered.set()
            assert release.wait(timeout=2.0)
        return original_builder(pathway_skills, all_files, dependency_graph)

    monkeypatch.setattr(network, "_build_pathway_annotations", blocking_builder)

    def run_mapping():
        outcome["mapped"] = network.map_infrastructure(str(tmp_path), force=True)

    mapper = threading.Thread(target=run_mapping)
    mapper.start()
    assert entered.wait(timeout=2.0)
    network.register_pathway("annotation_race", r"beta", "beta_skill")
    release.set()
    mapper.join(timeout=3.0)

    assert mapper.is_alive() is False
    assert outcome["mapped"] is True
    assert len(calls) >= 2
    pathway = network.pathways["annotation_race"]
    assert pathway.skill_name == "beta_skill"
    assert pathway.source_file == str(beta)
    assert pathway.source_file != str(alpha)


def test_mapper_cannot_publish_after_singleton_owner_replacement(
    network, monkeypatch, tmp_path
):
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    (core_dir / "alpha.py").write_text("VALUE = 1\n", encoding="utf-8")
    entered = threading.Event()
    release = threading.Event()
    outcome = {}

    def blocking_extract(_file_path, _base_dir):
        entered.set()
        assert release.wait(timeout=2.0)
        return []

    monkeypatch.setattr(network, "_extract_imports", blocking_extract)
    mapper = threading.Thread(
        target=lambda: outcome.setdefault(
            "mapped", network.map_infrastructure(str(tmp_path), force=True)
        )
    )
    mapper.start()
    assert entered.wait(timeout=2.0)

    with MycelialNetwork._lock:
        MycelialNetwork._instance = None
        MycelialNetwork._initialized = False
    replacement = MycelialNetwork()
    release.set()
    mapper.join(timeout=3.0)

    assert mapper.is_alive() is False
    assert outcome["mapped"] is False
    assert network.get_mapped_files_snapshot() == replacement.get_mapped_files_snapshot() == {}
    assert network.infrastructure_mapped is False
    assert replacement.infrastructure_mapped is False


@pytest.mark.asyncio
async def test_restore_and_sync_are_linearized_against_newer_memory_revision(
    network, monkeypatch, tmp_path
):
    monkeypatch.setenv("AURA_ROOT", str(tmp_path))
    assert await network.vault_sync() is True

    entered = threading.Event()
    release = threading.Event()
    original_decoder = MycelialNetwork._decode_vault_topology

    def blocking_decoder(cls, payload):
        entered.set()
        assert release.wait(timeout=3.0)
        return original_decoder(payload)

    monkeypatch.setattr(
        MycelialNetwork,
        "_decode_vault_topology",
        classmethod(blocking_decoder),
    )

    restore_task = asyncio.create_task(MycelialNetwork.restore_from_vault())
    assert await asyncio.to_thread(entered.wait, 2.0)
    network.register_pathway("newer_memory", r"newer", "newer_skill")
    sync_task = asyncio.create_task(network.vault_sync())
    await asyncio.sleep(0.05)
    assert sync_task.done() is False
    release.set()

    assert await restore_task is False
    assert await sync_task is True
    assert "newer_memory" in network.pathways

    vault_path = tmp_path / "data" / "mycelium_vault.db"
    with sqlite3.connect(vault_path) as connection:
        row = connection.execute(
            "SELECT data FROM aegis_vault WHERE key = ?", ("topology_v3",)
        ).fetchone()
    assert row is not None
    assert "newer_memory" in json.loads(row[0])["pathways"]


@pytest.mark.asyncio
async def test_vault_sync_commits_coherent_snapshot_while_live_topology_advances(
    network, monkeypatch, tmp_path
):
    monkeypatch.setenv("AURA_ROOT", str(tmp_path))
    assert await network.vault_sync() is True

    entered = threading.Event()
    release = threading.Event()
    original_decoder = MycelialNetwork._decode_vault_topology

    def blocking_decoder(cls, payload):
        entered.set()
        assert release.wait(timeout=3.0)
        return original_decoder(payload)

    monkeypatch.setattr(
        MycelialNetwork,
        "_decode_vault_topology",
        classmethod(blocking_decoder),
    )

    sync_task = asyncio.create_task(network.vault_sync())
    assert await asyncio.to_thread(entered.wait, 2.0)
    captured_revision = network._topology_revision
    network.register_pathway("raced_memory", r"raced", "raced_skill")
    release.set()

    assert await sync_task is True
    vault_path = tmp_path / "data" / "mycelium_vault.db"
    with sqlite3.connect(vault_path) as connection:
        row = connection.execute(
            "SELECT data FROM aegis_vault WHERE key = ?", ("topology_v3",)
        ).fetchone()
    assert row is not None
    committed = json.loads(row[0])
    assert "raced_memory" not in committed["pathways"]
    assert committed["topology_revision"] == captured_revision
    assert network._last_vault_sync_revision == captured_revision
    assert network._last_vault_sync_lag_revisions >= 1

    assert await network.vault_sync() is True
    with sqlite3.connect(vault_path) as connection:
        row = connection.execute(
            "SELECT data FROM aegis_vault WHERE key = ?", ("topology_v3",)
        ).fetchone()
    assert row is not None
    assert "raced_memory" in json.loads(row[0])["pathways"]
    assert network._last_vault_sync_lag_revisions == 0


@pytest.mark.asyncio
async def test_vault_commit_holds_topology_barrier_and_enables_full_sync(
    network, monkeypatch, tmp_path
):
    monkeypatch.setenv("AURA_ROOT", str(tmp_path))
    original_connect = sqlite3.connect
    commit_entered = threading.Event()
    release_commit = threading.Event()
    synchronous_levels = []

    class ProbedConnection:
        def __init__(self, connection):
            self._connection = connection

        def __enter__(self):
            self._connection.__enter__()
            return self

        def __exit__(self, *args):
            return self._connection.__exit__(*args)

        def execute(self, statement, *args):
            result = self._connection.execute(statement, *args)
            if str(statement).strip().upper().startswith("PRAGMA SYNCHRONOUS=FULL"):
                synchronous_levels.append(
                    self._connection.execute("PRAGMA synchronous").fetchone()[0]
                )
            return result

        def commit(self):
            commit_entered.set()
            assert release_commit.wait(timeout=3.0)
            return self._connection.commit()

        def __getattr__(self, name):
            return getattr(self._connection, name)

    monkeypatch.setattr(
        sqlite3,
        "connect",
        lambda *args, **kwargs: ProbedConnection(
            original_connect(*args, **kwargs)
        ),
    )

    sync_task = asyncio.create_task(network.vault_sync())
    assert await asyncio.to_thread(commit_entered.wait, 2.0)
    mutation_task = asyncio.create_task(
        asyncio.to_thread(
            network.register_pathway,
            "after_commit_barrier",
            r"after commit",
            "after_commit_skill",
        )
    )
    await asyncio.sleep(0.05)
    assert mutation_task.done() is False
    release_commit.set()

    assert await sync_task is True
    await mutation_task
    assert synchronous_levels == [2]

    vault_path = tmp_path / "data" / "mycelium_vault.db"
    with original_connect(vault_path) as connection:
        row = connection.execute(
            "SELECT data FROM aegis_vault WHERE key = ?", ("topology_v3",)
        ).fetchone()
    assert row is not None
    assert "after_commit_barrier" not in json.loads(row[0])["pathways"]

    assert await network.vault_sync() is True
    with original_connect(vault_path) as connection:
        row = connection.execute(
            "SELECT data FROM aegis_vault WHERE key = ?", ("topology_v3",)
        ).fetchone()
    assert row is not None
    assert "after_commit_barrier" in json.loads(row[0])["pathways"]


@pytest.mark.asyncio
async def test_vault_serialization_rejects_nonfinite_if_decoder_is_bypassed(
    network, monkeypatch, tmp_path
):
    monkeypatch.setenv("AURA_ROOT", str(tmp_path))
    with MycelialNetwork._lock:
        next(iter(network.hyphae.values())).strength = float("nan")
        network._mark_topology_mutated_locked()
    monkeypatch.setattr(
        MycelialNetwork,
        "_decode_vault_topology",
        classmethod(lambda _cls, _payload: {}),
    )

    assert await network.vault_sync() is False
    assert not (tmp_path / "data" / "mycelium_vault.db").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("surface", "value_kind"),
    (
        ("hypha_last_pulse", "nan"),
        ("hypha_last_pulse", "infinity"),
        ("hypha_last_pulse", "future"),
        ("hypha_chronology", "reversed"),
        ("pathway_last_matched", "nan"),
        ("pathway_last_matched", "infinity"),
        ("pathway_last_matched", "future"),
        ("pathway_chronology", "reversed"),
    ),
)
async def test_vault_sync_rejects_invalid_live_event_timestamps(
    network, monkeypatch, tmp_path, surface, value_kind
):
    monkeypatch.setenv("AURA_ROOT", str(tmp_path))
    with MycelialNetwork._lock:
        root = network.hyphae["voice_presence->hardware:macos_say"]
        pathway = network.pathways["direct_web_search"]
        if value_kind == "nan":
            value = float("nan")
        elif value_kind == "infinity":
            value = float("inf")
        elif value_kind == "future":
            value = time.monotonic() + 2.0
        else:
            value = None

        if surface == "hypha_last_pulse":
            root.last_pulse = value
        elif surface == "hypha_chronology":
            root.created_at = time.monotonic()
            root.last_pulse = root.created_at - 5.0
        elif surface == "pathway_last_matched":
            pathway.last_matched = value
        else:
            pathway.created_at = time.time()
            pathway.last_matched = time.monotonic() - 5.0
        network._mark_topology_mutated_locked()

    assert await network.vault_sync() is False
    assert not (tmp_path / "data" / "mycelium_vault.db").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    (
        "nan_strength",
        "infinite_priority",
        "negative_pulse_count",
        "centrality_mismatch",
        "critical_mismatch",
        "physical_file_mismatch",
        "unknown_top_level",
        "unknown_pathway_field",
        "future_capture",
        "future_pathway_creation",
        "nan_last_pulse_age",
        "hypha_reversed_chronology",
        "pathway_reversed_chronology",
    ),
)
async def test_vault_rejects_nonfinite_negative_and_cross_surface_corruption(
    network, monkeypatch, tmp_path, corruption
):
    monkeypatch.setenv("AURA_ROOT", str(tmp_path))
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    (core_dir / "alpha.py").write_text("import core.beta\n", encoding="utf-8")
    (core_dir / "beta.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert network.map_infrastructure(str(tmp_path), force=True) is True
    assert await network.vault_sync() is True

    vault_path = tmp_path / "data" / "mycelium_vault.db"
    with sqlite3.connect(vault_path) as connection:
        row = connection.execute(
            "SELECT data FROM aegis_vault WHERE key = ?", ("topology_v3",)
        ).fetchone()
        assert row is not None
        payload = json.loads(row[0])
        if corruption == "nan_strength":
            payload["hyphae"]["voice_presence->hardware:macos_say"]["strength"] = float("nan")
        elif corruption == "infinite_priority":
            payload["pathways"]["direct_web_search"]["priority"] = float("inf")
        elif corruption == "negative_pulse_count":
            payload["hyphae"]["voice_presence->hardware:macos_say"]["pulse_count"] = -1
        elif corruption == "centrality_mismatch":
            payload["centrality"]["core.beta"] = 99
            payload["mapped_files"]["core.beta"]["centrality"] = 99
        elif corruption == "critical_mismatch":
            payload["critical_modules"] = []
            for module in payload["mapped_files"].values():
                module["is_critical"] = False
        elif corruption == "physical_file_mismatch":
            payload["hyphae"]["import:core.alpha->core.beta"]["source_file"] = str(
                tmp_path / "wrong.py"
            )
        elif corruption == "unknown_top_level":
            payload["unexpected"] = True
        elif corruption == "unknown_pathway_field":
            payload["pathways"]["direct_web_search"]["unexpected"] = True
        elif corruption == "future_capture":
            payload["captured_at_unix"] = time.time() + 2.0
        elif corruption == "future_pathway_creation":
            payload["pathways"]["direct_web_search"]["created_at"] = (
                payload["captured_at_unix"] + 2.0
            )
        elif corruption == "nan_last_pulse_age":
            payload["hyphae"]["voice_presence->hardware:macos_say"][
                "last_pulse_age_s"
            ] = float("nan")
        elif corruption == "hypha_reversed_chronology":
            root = payload["hyphae"]["voice_presence->hardware:macos_say"]
            root["created_age_s"] = 1.0
            root["last_pulse_age_s"] = 5.0
        elif corruption == "pathway_reversed_chronology":
            pathway = payload["pathways"]["direct_web_search"]
            pathway["created_at"] = payload["captured_at_unix"] - 1.0
            pathway["last_matched_age_s"] = 5.0
        else:
            raise AssertionError(f"unhandled corruption fixture: {corruption}")
        connection.execute(
            "UPDATE aegis_vault SET data = ? WHERE key = ?",
            (json.dumps(payload), "topology_v3"),
        )
        connection.commit()

    before = network.get_graph_snapshot()
    assert await MycelialNetwork.restore_from_vault() is False
    assert network.get_graph_snapshot() == before


@pytest.mark.asyncio
async def test_vault_restored_ages_include_time_elapsed_since_capture(
    network, monkeypatch, tmp_path
):
    monkeypatch.setenv("AURA_ROOT", str(tmp_path))
    assert await network.vault_sync() is True
    vault_path = tmp_path / "data" / "mycelium_vault.db"
    elapsed_offline = 86_400.0

    with sqlite3.connect(vault_path) as connection:
        row = connection.execute(
            "SELECT data FROM aegis_vault WHERE key = ?", ("topology_v3",)
        ).fetchone()
        assert row is not None
        payload = json.loads(row[0])
        payload["captured_at_unix"] = time.time() - elapsed_offline
        for pathway in payload["pathways"].values():
            pathway["created_at"] = min(
                float(pathway["created_at"]),
                payload["captured_at_unix"] - 1.0,
            )
        payload["pathways"]["direct_web_search"]["last_matched_age_s"] = 30.0
        payload["pathways"]["direct_web_search"]["created_at"] = min(
            payload["pathways"]["direct_web_search"]["created_at"],
            payload["captured_at_unix"] - 31.0,
        )
        root = payload["hyphae"]["voice_presence->hardware:macos_say"]
        root["created_age_s"] = max(float(root["created_age_s"]), 46.0)
        root["last_pulse_age_s"] = 45.0
        connection.execute(
            "UPDATE aegis_vault SET data = ? WHERE key = ?",
            (json.dumps(payload), "topology_v3"),
        )
        connection.commit()

    assert await MycelialNetwork.restore_from_vault() is True
    restored_at = time.monotonic()
    pathway_age = restored_at - network.pathways["direct_web_search"].last_matched
    root = network.get_hypha("voice_presence", "hardware:macos_say")
    assert root is not None
    root_age = restored_at - root.last_pulse
    assert pathway_age == pytest.approx(elapsed_offline + 30.0, abs=2.0)
    assert root_age == pytest.approx(elapsed_offline + 45.0, abs=2.0)


def test_graph_uses_full_critical_snapshot_and_exact_pathway_annotation():
    from interface.routes import subsystems as subsystem_routes

    class DummyMycelium:
        @staticmethod
        def get_graph_snapshot():
            return {
                "topology": {
                    "hyphae": {},
                    "pathways": {
                        "opaque_pathway": {
                            "confidence": 1.0,
                            "skill_name": "name_that_does_not_match_module",
                            "source_file": "/core/opaque.py",
                        }
                    },
                    "system_cohesion": 1.0,
                    "pathway_count": 1,
                    "critical_modules": [],
                },
                "mapped_files": {
                    "core.opaque_module": {"path": "/core/opaque.py"}
                },
                "centrality": {"core.opaque_module": 11},
                "critical_modules": ["core.opaque_module"],
                "mapping_generation": 4,
                "mapping_state": "ready",
            }

    response = subsystem_routes._build_mycelial_graph_response(DummyMycelium())
    payload = json.loads(response.body)
    node = next(node for node in payload["nodes"] if node["id"] == "core.opaque_module")
    links = {(link["source"], link["target"]) for link in payload["links"]}

    assert node["is_critical"] is True
    assert node["type"] == "critical"
    assert ("pw:opaque_pathway", "core.opaque_module") in links


##
