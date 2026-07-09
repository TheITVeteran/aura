from types import SimpleNamespace

import pytest

from core.agency_core import AgencyCore
from core.self_modification.code_refiner import CodeRefinerService
from core.container import ServiceContainer
from core.learning.skill_evolution import SkillEvolutionEngine
from core.system_monitor import SystemStateMonitor


class _ShardRecorder:
    def __init__(self):
        self.shards = []

    async def spawn_shard(self, **kwargs):
        self.shards.append(kwargs)
        return True


class _CompletedTask:
    def add_done_callback(self, callback):
        callback(self)

    def cancelled(self):
        return False

    def exception(self):
        return None


class _ClosingTracker:
    def create_task(self, awaitable, name=None):
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        return _CompletedTask()


class _RegistryRecorder:
    def __init__(self):
        self.updates = []

    def get_state(self):
        return SimpleNamespace(reasoning_queue_size=0)

    async def update(self, **kwargs):
        self.updates.append(kwargs)


@pytest.mark.asyncio
async def test_code_refiner_audits_core_tree_and_records_proposals(tmp_path):
    source = tmp_path / "dense_module.py"
    source.write_text("\n\n".join(f"def function_{index}():\n    return {index}" for index in range(31)))

    refiner = CodeRefinerService()
    refiner.root_dir = tmp_path

    proposals = await refiner.audit_core()

    assert refiner.proposals == proposals
    assert any(proposal.category == "complexity" for proposal in proposals)
    assert any(proposal.file_path == str(source) for proposal in proposals)


@pytest.mark.asyncio
async def test_skill_evolution_selects_error_targets_and_spawns_research(monkeypatch):
    swarm = _ShardRecorder()
    services = {
        "omni_tool": SimpleNamespace(
            _execution_logs={
                "filesystem_repair": [{"status": "error"} for _ in range(4)] + [{"status": "ok"}]
            }
        ),
        "sovereign_swarm": swarm,
    }
    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda cls, name, default=None: services.get(name, default)))

    evolver = SkillEvolutionEngine()

    targets = await evolver.identify_evolution_targets()
    await evolver.spawn_evolution_shard(targets[0])

    assert targets == ["filesystem_repair"]
    assert len(swarm.shards) == 1
    assert swarm.shards[0]["context"] == {"target_skill": "filesystem_repair"}
    assert "filesystem_repair" in swarm.shards[0]["objective"]


@pytest.mark.asyncio
async def test_skill_evolution_falls_back_when_capability_registry_is_empty(monkeypatch):
    services = {
        "omni_tool": SimpleNamespace(_execution_logs={}),
        "capability_engine": SimpleNamespace(),
    }
    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda cls, name, default=None: services.get(name, default)))

    targets = await SkillEvolutionEngine().identify_evolution_targets()

    assert targets == ["web_search", "memory_query"]


@pytest.mark.asyncio
async def test_system_monitor_audits_stability_from_registered_services(monkeypatch):
    registry = _RegistryRecorder()
    services = {
        "sovereign_swarm": SimpleNamespace(shards=[object(), object()]),
        "code_refiner": SimpleNamespace(proposals=[object(), object()]),
    }

    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda cls, name, default=None: services.get(name, default)))
    monkeypatch.setattr("core.state.state_registry.get_registry", lambda: registry)
    monkeypatch.setattr("core.system_monitor.get_task_tracker", lambda: _ClosingTracker())

    health = await SystemStateMonitor().audit_stability()

    assert health.active_shards == 2
    assert health.unresolved_refinements == 2
    assert health.cognitive_stability == pytest.approx(0.86)


@pytest.mark.asyncio
async def test_agency_self_architect_routes_to_skill_evolution(monkeypatch):
    class _Evolver:
        def __init__(self):
            self.spawned = []

        async def identify_evolution_targets(self):
            return ["adaptive_browser_use"]

        async def spawn_evolution_shard(self, skill_name):
            self.spawned.append(skill_name)

    evolver = _Evolver()
    services = {
        "code_refiner": object(),
        "skill_evolution": evolver,
        "system_monitor": SimpleNamespace(audit_stability=lambda: None),
    }

    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda cls, name, default=None: services.get(name, default)))
    monkeypatch.setattr("core.agency.agency_core.random.random", lambda: 0.5)

    agency = AgencyCore(orchestrator=SimpleNamespace())
    agency.state.initiative_energy = 0.9
    agency.state.frustration_level = 0.8

    action = await agency._pathway_self_architect(now=1000.0, idle_seconds=300.0)

    assert action["type"] == "skill_evolution"
    assert action["skill"] == "adaptive_browser_use"
    assert evolver.spawned == ["adaptive_browser_use"]
