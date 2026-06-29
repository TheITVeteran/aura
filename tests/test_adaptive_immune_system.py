import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.adaptation.adaptive_immunity import (
    AdaptiveImmuneConfig,
    AdaptiveImmuneSystem,
    Antigen,
    CellKind,
    EffectorArtifact,
    EffectorKind,
    ImmuneCell,
    TissueField,
)


def run(coro):
    return asyncio.run(coro)


@dataclass
class _RepairResult:
    success: bool = True
    governance_approved: bool = True
    health_before: float = 0.0
    health_after: float = 0.0


class _AutopoiesisStub:
    def __init__(self, *, initial_health=None, health_delta=0.45, success=True):
        self.calls = []
        self.health = dict(initial_health or {})
        self.health_delta = health_delta
        self.success = success
        self._health_fns = {
            name: (lambda name=name: self.health.get(name, 0.0)) for name in self.health
        }

    def get_component_health(self, component):
        return float(self.health.get(component, 0.0))

    async def request_repair(self, component, strategy):
        self.calls.append((component, getattr(strategy, "value", str(strategy))))
        before = self.get_component_health(component)
        after = before if not self.success else min(1.0, before + self.health_delta)
        self.health[component] = after
        self._health_fns[component] = lambda component=component: self.health.get(component, 0.0)
        return _RepairResult(
            success=self.success,
            governance_approved=True,
            health_before=before,
            health_after=after,
        )


class _PatchMeshStub:
    def __init__(self, *, applied=True):
        self.calls = []
        self.applied = applied

    async def attempt_patch_for_antigen(self, artifact, antigen):
        self.calls.append((artifact.kind.value, artifact.component, antigen.error_signature))
        return {
            "attempted": True,
            "applied": self.applied,
            "status": "applied" if self.applied else "validated_unapplied",
            "notes": "sandbox + verification pipeline completed",
        }


def _test_antigen(subsystem="runtime_engine"):
    return Antigen(
        antigen_id="ag_test",
        subsystem=subsystem,
        vector=np.zeros(16, dtype=np.float32),
        danger=0.8,
        subsystem_need=0.7,
        threat_probability=0.8,
        resource_pressure=0.2,
        error_load=0.5,
        health_pressure=0.3,
        temporal_pressure=0.1,
        recurrence_pressure=0.2,
        error_signature="RuntimeError",
    )


def _test_artifact(kind=EffectorKind.PATCH_PROPOSAL, component="runtime_engine"):
    return EffectorArtifact(
        artifact_id="eff_test",
        kind=kind,
        component=component,
        confidence=0.95,
        source_cell_id="cell",
        lineage_id="lineage",
        bounded_payload={},
    )


def test_tissue_field_diffuses_local_damage():
    field = TissueField(diffusion=0.35, decay=0.0)
    field.register_edge("state_repository", "continuity", 1.0)

    antigen = Antigen(
        antigen_id="ag_tissue",
        subsystem="state_repository",
        vector=np.zeros(16, dtype=np.float32),
        danger=0.9,
        subsystem_need=0.8,
        threat_probability=0.9,
        resource_pressure=0.7,
        error_load=0.6,
        health_pressure=0.4,
        temporal_pressure=0.2,
        recurrence_pressure=0.1,
        protected=False,
    )
    field.ingest_antigen(antigen)

    snapshot = field.to_dict()
    assert snapshot["danger"]["state_repository"] > 0.0
    assert snapshot["danger"]["continuity"] > 0.0
    assert field.get_need("continuity") > 0.0


def test_regulatory_cells_suppress_autoimmune_effectors(tmp_path):
    immune = AdaptiveImmuneSystem(state_dir=tmp_path, rng_seed=1)
    antigen = immune.present_antigen(
        {
            "type": "exception",
            "text": "Identity coherence failure",
            "subsystem": "identity_core",
            "source": "identity_guard",
            "danger": 0.95,
            "error_count": 6,
            "resource_pressure": 0.4,
            "stack_trace": "X" * 800,
        }
    )

    killer = ImmuneCell(
        cell_id="killer",
        lineage_id="killer_lineage",
        kind=CellKind.CYTOTOXIC,
        receptor=antigen.vector.copy(),
        subsystem_scope="identity_core",
        persistence=0.6,
    )
    regulator = ImmuneCell(
        cell_id="regulator",
        lineage_id="reg_lineage",
        kind=CellKind.REGULATORY,
        receptor=antigen.vector.copy(),
        subsystem_scope="identity_core",
        persistence=0.9,
        regulatory_strength=1.4,
    )
    immune._cells = [killer, regulator]

    response, _ = immune._observe_core(antigen)
    assert response.suppression_applied > 0.0
    assert response.artifacts
    assert all(artifact.suppressed for artifact in response.artifacts)


def test_successful_lineage_clones_and_persists_as_memory(tmp_path):
    cfg = AdaptiveImmuneConfig(
        population_size=1,
        max_population=8,
        dream_every_observations=100,
    )
    immune = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=2)
    autopoiesis = _AutopoiesisStub(initial_health={"state_repository": 0.2})
    immune._get_service = lambda name: autopoiesis if name == "autopoiesis" else None

    antigen = immune.present_antigen(
        {
            "type": "error_signature",
            "text": "database is locked",
            "subsystem": "state_repository",
            "source": "signature",
            "danger": 0.9,
            "error_count": 5,
            "resource_pressure": 0.2,
        }
    )
    seed = ImmuneCell(
        cell_id="seed_b",
        lineage_id="seed_lineage",
        kind=CellKind.B,
        receptor=antigen.vector.copy(),
        subsystem_scope="state_repository",
        persistence=0.7,
        fitness=0.5,
    )
    immune._cells = [seed]

    event = {
        "type": "error_signature",
        "text": "database is locked",
        "subsystem": "state_repository",
        "source": "signature",
        "danger": 0.9,
        "error_count": 5,
        "resource_pressure": 0.2,
    }
    first_response = run(immune.observe_event(event))
    second_response = run(immune.observe_event(event))

    assert autopoiesis.calls
    assert first_response.verification_report["verified_success"] is True
    assert second_response.diagnostic_verdict["status"] == "verified_recovery"
    assert len(immune._cells) > 1

    immune.dream_consolidate()
    assert any(cell.kind == CellKind.MEMORY for cell in immune._cells)

    reloaded = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=2)
    assert any(cell.kind == CellKind.MEMORY for cell in reloaded._cells)


def test_auto_dream_consolidation_defers_under_background_policy(tmp_path, monkeypatch):
    import core.adaptation.adaptive_immunity as adaptive_immunity

    cfg = AdaptiveImmuneConfig(
        population_size=1,
        max_population=8,
        dream_every_observations=1,
    )
    immune = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=11)
    monkeypatch.setattr(
        adaptive_immunity,
        "_maintenance_background_deferral_reason",
        lambda: f"boot_grace_{42 + immune._observation_count}s",
    )
    antigen = immune.present_antigen(
        {
            "type": "error_signature",
            "text": "transient startup noise",
            "subsystem": "runtime",
            "source": "startup",
            "danger": 0.5,
        }
    )

    response, _ = immune._observe_core(antigen)

    assert response.dream_consolidated is False
    assert immune._last_dream_at == 0


def test_auto_dream_deferral_log_is_rate_limited(tmp_path, monkeypatch, caplog):
    import core.adaptation.adaptive_immunity as adaptive_immunity

    cfg = AdaptiveImmuneConfig(
        population_size=1,
        max_population=8,
        dream_every_observations=1,
    )
    immune = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=12)
    monkeypatch.setattr(
        adaptive_immunity,
        "_maintenance_background_deferral_reason",
        lambda: "boot_grace_42s",
    )
    antigen = immune.present_antigen(
        {
            "type": "error_signature",
            "text": "transient startup noise",
            "subsystem": "runtime",
            "source": "startup",
            "danger": 0.5,
        }
    )

    with caplog.at_level(logging.INFO, logger="Aura.AdaptiveImmunity"):
        immune._observe_core(antigen)
        immune._observe_core(antigen)

    messages = [
        record.message
        for record in caplog.records
        if "dream consolidation deferred" in record.message
    ]
    assert len(messages) == 1


def test_species_assignment_preserves_multiple_niches(tmp_path):
    cfg = AdaptiveImmuneConfig(population_size=6, max_population=12)
    immune = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=3)
    immune._cells = []

    low_cluster = np.full(16, 0.1, dtype=np.float32)
    high_cluster = np.full(16, 0.9, dtype=np.float32)

    for idx in range(3):
        immune._cells.append(
            ImmuneCell(
                cell_id=f"low_{idx}",
                lineage_id=f"low_lineage_{idx}",
                kind=CellKind.B,
                receptor=low_cluster.copy(),
                subsystem_scope="state_repository",
            )
        )
    for idx in range(3):
        immune._cells.append(
            ImmuneCell(
                cell_id=f"high_{idx}",
                lineage_id=f"high_lineage_{idx}",
                kind=CellKind.CYTOTOXIC,
                receptor=high_cluster.copy(),
                subsystem_scope="llm_router",
            )
        )

    immune._assign_species()
    assert immune._species_count >= 2
    assert len({cell.species_id for cell in immune._cells}) >= 2


def test_truthful_reporting_avoids_false_all_clear_under_thin_coverage(tmp_path):
    cfg = AdaptiveImmuneConfig(population_size=1, max_population=6, dream_every_observations=100)
    immune = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=4)
    immune._get_service = lambda name: None
    immune._cells = []

    response = run(
        immune.observe_event(
            {
                "type": "tick",
                "text": "",
                "subsystem": "unknown",
                "source": "thin_probe",
                "danger": 0.02,
                "timestamp": 123.0,
            }
        )
    )

    assert response.coverage_report["coverage_label"] == "thin"
    assert response.diagnostic_verdict["status"] == "no_confirmed_issue_under_limited_visibility"
    assert response.diagnostic_verdict["all_clear"] is False


def test_recurrence_memory_persists_across_reload(tmp_path):
    cfg = AdaptiveImmuneConfig(population_size=1, max_population=6, dream_every_observations=100)
    immune = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=5)
    immune.observe_signature("state_repository", "RuntimeError", error_count=3)
    immune.observe_signature("state_repository", "RuntimeError", error_count=3)

    assert immune._estimate_recurrence_pressure("state_repository", "RuntimeError") > 0.0

    reloaded = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=5)
    assert reloaded._estimate_recurrence_pressure("state_repository", "RuntimeError") > 0.0
    hotspots = reloaded.get_status()["recurrence_hotspots"]
    assert hotspots
    assert hotspots[0]["subsystem"] == "state_repository"


def test_patch_proposal_artifacts_route_into_patch_pipeline(tmp_path):
    cfg = AdaptiveImmuneConfig(population_size=1, max_population=6, dream_every_observations=100)
    immune = AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=6)
    patch_mesh = _PatchMeshStub(applied=True)

    antigen = immune.present_antigen(
        {
            "type": "exception",
            "text": "ZeroDivisionError in runtime",
            "subsystem": "runtime_engine",
            "source": "exception",
            "danger": 0.86,
            "error_count": 3,
            "stack_trace": f'Traceback\n  File "{Path(tempfile.gettempdir()) / "demo.py"}", line 12, in run\n',
            "exception_type": "ZeroDivisionError",
        }
    )
    immune._cells = [
        ImmuneCell(
            cell_id="patcher",
            lineage_id="patch_lineage",
            kind=CellKind.B,
            receptor=antigen.vector.copy(),
            subsystem_scope="runtime_engine",
            persistence=0.8,
            fitness=0.4,
        )
    ]
    immune._get_service = lambda name: patch_mesh if name == "autonomous_resilience_mesh" else None
    artifact = immune._emit_artifact(
        immune._cells[0],
        antigen,
        affinity=1.0,
        activation=0.92,
    )

    assert artifact is not None
    assert artifact.kind.value == "patch_proposal"

    verification = run(
        immune._maybe_execute_artifact(
            artifact,
            antigen,
            coverage_report={"coverage_ratio": 0.9},
        )
    )

    assert patch_mesh.calls
    assert artifact.executed is True
    assert artifact.success is True
    assert verification["status"] == "applied"


def test_patch_artifact_failure_returns_execution_report(tmp_path, monkeypatch):
    recorded = []
    monkeypatch.setattr(
        "core.adaptation.adaptive_immunity._record_adaptive_immunity_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    class FailingPatchMesh:
        def __init__(self):
            self.attempts = 0

        async def attempt_patch_for_antigen(self, _artifact, _antigen):
            self.attempts += 1
            raise TimeoutError("patch mesh timeout")

    immune = AdaptiveImmuneSystem(state_dir=tmp_path, rng_seed=7)
    patch_mesh = FailingPatchMesh()
    immune._get_service = lambda name: (
        patch_mesh if name == "autonomous_resilience_mesh" else None
    )
    artifact = _test_artifact()

    report = run(
        immune._maybe_execute_artifact(
            artifact,
            _test_antigen(),
            coverage_report={"coverage_ratio": 0.9},
        )
    )

    assert report["status"] == "execution_error"
    assert patch_mesh.attempts == 1
    assert artifact.executed is True
    assert artifact.success is False
    assert "patch execution failed" in artifact.notes
    assert recorded[0][1]["action"] == (
        "Marked patch artifact execution failed and returned an execution_error verification report"
    )


def test_behavioral_rule_artifact_defers_during_foreground_turn(tmp_path, monkeypatch):
    from core.runtime import foreground_guard

    # Pin the maintenance environment so desktop-safe-boot (set via setdefault at import by
    # core/architect/safe_boot_harness.py, which leaks across a broad run) can't mask the
    # foreground-deferral reason this test asserts.
    monkeypatch.setenv("AURA_SAFE_BOOT_DESKTOP", "0")
    monkeypatch.setenv("AURA_ENABLE_BACKGROUND_COGNITION", "1")

    immune = AdaptiveImmuneSystem(state_dir=tmp_path, rng_seed=21)
    antigen = _test_antigen(subsystem="runtime_engine")
    behavioral_cell = ImmuneCell(
        cell_id="behavioral",
        lineage_id="behavioral_lineage",
        kind=CellKind.B,
        receptor=antigen.vector.copy(),
        subsystem_scope="runtime_engine",
        persistence=0.8,
        behavioral_rule={
            "conditions": [{"sensor": "port_east_load", "operator": ">", "value": 1.0}],
            "actions": [
                {
                    "actuator": "reallocate_flow",
                    "params": {
                        "source_id": "Port_East",
                        "target_id": "Port_West",
                        "amount": 100.0,
                    },
                }
            ],
        },
    )
    immune._cells = [behavioral_cell]
    artifact = EffectorArtifact(
        artifact_id="eff_behavioral",
        kind=EffectorKind.REDUCE_LOAD,
        component="runtime_engine",
        confidence=0.95,
        source_cell_id="behavioral",
        lineage_id="behavioral_lineage",
        bounded_payload={},
    )

    foreground_guard._reset_for_tests()
    lease = foreground_guard.begin_foreground_turn(owner="test", source="chat_api")
    try:
        report = run(
            immune._maybe_execute_artifact(
                artifact,
                antigen,
                coverage_report={"coverage_ratio": 0.9},
            )
        )
    finally:
        lease.close()
        foreground_guard._reset_for_tests()

    assert report["status"] == "deferred"
    assert report["verified_success"] is False
    assert artifact.executed is False
    assert artifact.success is False
    assert "foreground_chat_active" in artifact.notes


def test_autopoiesis_failure_returns_execution_report(tmp_path, monkeypatch):
    recorded = []
    monkeypatch.setattr(
        "core.adaptation.adaptive_immunity._record_adaptive_immunity_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    class FailingAutopoiesis:
        def __init__(self):
            self.repair_requests = 0

        async def request_repair(self, _component, _strategy):
            self.repair_requests += 1
            raise RuntimeError("repair loop unavailable")

    immune = AdaptiveImmuneSystem(state_dir=tmp_path, rng_seed=8)
    autopoiesis = FailingAutopoiesis()
    immune._get_service = lambda name: autopoiesis if name == "autopoiesis" else None
    artifact = _test_artifact(kind=EffectorKind.RESTART_COMPONENT)

    report = run(
        immune._maybe_execute_artifact(
            artifact,
            _test_antigen(),
            coverage_report={"coverage_ratio": 0.9},
        )
    )

    assert report["status"] == "execution_error"
    assert autopoiesis.repair_requests == 1
    assert artifact.executed is True
    assert artifact.success is False
    assert "execution failed" in artifact.notes
    assert recorded[0][1]["action"] == (
        "Marked autopoiesis artifact execution failed and returned an execution_error verification report"
    )
