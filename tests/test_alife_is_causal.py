"""The ALife layer must change the mesh, not just describe it.

A production-wiring review found that the standalone ALife classes implement
Lenia coupling, compute-credit allocation, replication, speciation and
thermodynamic accounting — and their unit tests pass — while the canonical
cognitive phase never applied any of it to the mesh. The advertised "living
neural ecology" was telemetry.

Reading the code showed the disconnection was more complete than that. The phase
fetched ``mesh.column_activations`` and ``mesh.inter_column_weights``; NeuralMesh
defined neither. Every consumer guarded with ``if activations is None: return``,
so ALife dynamics, topology evolution and the criticality regulator all returned
early on every tick — the mathematics never executed outside its unit tests at
all. And the extension layer was handed field names it does not read, so its
replicator received no column weights and everything else fell back to synthetic
constants.

These tests pin the wiring, not the mathematics (which has its own suite).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.neural_mesh import NeuralMesh


@pytest.fixture
def mesh():
    return NeuralMesh()


# ---------------------------------------------------------------------------
# The state surface the consumers actually read
# ---------------------------------------------------------------------------


def test_mesh_publishes_the_state_its_consumers_read(mesh):
    """Absent attributes silently disabled three whole subsystems."""
    for name in ("column_activations", "inter_column_weights", "projection_weights"):
        value = getattr(mesh, name, None)
        assert value is not None, (
            f"NeuralMesh.{name} is missing — every consumer guards with "
            f"`if {name} is None: return`, so the layer silently never runs"
        )
        assert isinstance(value, np.ndarray)


def test_published_activations_track_the_live_mesh(mesh):
    """The published vector must be the mesh's real state, not a stub."""
    assert mesh.column_activations.shape == (mesh.cfg.columns,)
    before = mesh.column_activations.copy()

    mesh.inject_sensory(np.ones(mesh.cfg.sensory_end * 64, dtype=np.float32) * 0.6)
    for _ in range(6):
        mesh._tick()

    assert not np.allclose(before, mesh.column_activations), (
        "column_activations did not respond to real input — it is not wired to "
        "the mesh's dynamics"
    )


def test_published_weights_are_the_causal_matrix(mesh):
    """`inter_column_weights` must be the matrix that actually drives the mesh."""
    assert mesh.inter_column_weights is mesh._inter_W


# ---------------------------------------------------------------------------
# Lenia coupling is applied, and it is causal
# ---------------------------------------------------------------------------


def test_applying_coupling_changes_the_mesh(mesh):
    """The write-back path exists and moves the real matrix."""
    before = mesh.inter_column_weights.copy()
    target = np.full((mesh.cfg.columns, mesh.cfg.columns), 0.5, dtype=np.float32)

    assert mesh.apply_inter_column_coupling(target, blend=0.5, source="test") is True
    assert not np.allclose(before, mesh.inter_column_weights)


def test_applied_coupling_changes_the_mesh_dynamics(mesh):
    """Causal, not cosmetic: different coupling ⇒ different evolution.

    This is the claim under test. If the mesh evolves identically with and
    without the ALife coupling, the ecology is decorative however real its
    mathematics is.
    """
    def _run(apply_coupling: bool) -> np.ndarray:
        m = NeuralMesh()
        m._rng = np.random.default_rng(11)
        if apply_coupling:
            rng = np.random.default_rng(3)
            coupling = rng.standard_normal(
                (m.cfg.columns, m.cfg.columns)
            ).astype(np.float32) * 0.4
            m.apply_inter_column_coupling(coupling, blend=0.9, source="test")
        m.inject_sensory(np.ones(m.cfg.sensory_end * 64, dtype=np.float32) * 0.5)
        for _ in range(8):
            m._tick()
        return m.column_activations.copy()

    plain = _run(False)
    coupled = _run(True)
    assert not np.allclose(plain, coupled, atol=1e-6), (
        "the mesh evolved identically with and without ALife coupling — the "
        "coupling is not causal"
    )


def test_coupling_is_blended_not_overwritten(mesh):
    """One layer must not be able to erase STDP-learned structure in a tick."""
    before = mesh.inter_column_weights.copy()
    target = np.full((mesh.cfg.columns, mesh.cfg.columns), 1.0, dtype=np.float32)
    mesh.apply_inter_column_coupling(target, blend=0.1, source="test")

    after = mesh.inter_column_weights
    assert not np.allclose(after, target), "coupling overwrote the mesh wholesale"
    assert not np.allclose(after, before), "coupling had no effect"


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 4), dtype=np.float32),                       # wrong shape
        np.full((64, 64), np.nan, dtype=np.float32),              # non-finite
        "not an array",
    ],
)
def test_bad_coupling_is_refused(mesh, bad):
    """A malformed matrix must be refused, not silently poison the mesh."""
    before = mesh.inter_column_weights.copy()
    assert mesh.apply_inter_column_coupling(bad, source="test") is False
    assert np.allclose(before, mesh.inter_column_weights)


def test_coupling_stays_bounded(mesh):
    """The mesh's own clip invariant must survive external writes."""
    huge = np.full((mesh.cfg.columns, mesh.cfg.columns), 1000.0, dtype=np.float32)
    mesh.apply_inter_column_coupling(huge, blend=1.0, source="test")
    W = mesh.inter_column_weights
    assert np.all(np.isfinite(W))
    assert W.max() <= 1.0 and W.min() >= -1.0


# ---------------------------------------------------------------------------
# The extension layer receives real state
# ---------------------------------------------------------------------------


def test_extension_state_carries_the_fields_the_replicator_reads(mesh):
    """columns_W resolved to [] — the replicator had nothing to replicate."""
    state = mesh.alife_mesh_state()

    for key in ("columns_W", "contributions", "stabilities", "error_rates",
                "specialization_profiles"):
        assert key in state, f"extension layer reads {key!r}; it is not supplied"

    assert len(state["columns_W"]) == mesh.cfg.columns, (
        "columns_W is empty — the replication path receives no column weights"
    )
    assert state["contributions"].shape == (mesh.cfg.columns,)
    assert state["specialization_profiles"].shape == (mesh.cfg.columns, 8)


def test_columns_W_is_passed_by_reference_so_replication_is_causal(mesh):
    """The replicator modifies intra-column weights in place."""
    state = mesh.alife_mesh_state()
    state["columns_W"][0][0, 0] = 0.4242
    assert mesh.columns[0].W[0, 0] == pytest.approx(0.4242), (
        "columns_W was copied — in-place replication would not reach the mesh"
    )


def test_extension_state_is_not_synthetic_constants(mesh):
    """The old fallback fed the layer np.full(n, 0.5) — measure Aura instead."""
    mesh.inject_sensory(np.ones(mesh.cfg.sensory_end * 64, dtype=np.float32) * 0.7)
    for _ in range(6):
        mesh._tick()

    state = mesh.alife_mesh_state()
    contributions = state["contributions"]
    profiles = state["specialization_profiles"]

    assert not np.allclose(contributions, 0.5), (
        "contributions are the synthetic default — the extension layer is not "
        "seeing the real mesh"
    )
    assert float(contributions.std()) > 1e-6, "contributions carry no information"
    assert float(profiles.std()) > 1e-6, (
        "specialization profiles are constant — speciation would cluster noise"
    )


def test_extension_state_is_finite(mesh):
    state = mesh.alife_mesh_state()
    for key in ("contributions", "stabilities", "error_rates",
                "specialization_profiles"):
        assert np.all(np.isfinite(state[key])), f"{key} contains non-finite values"


# ---------------------------------------------------------------------------
# The phase applies it
# ---------------------------------------------------------------------------


def test_cognitive_phase_applies_lenia_coupling_to_the_mesh():
    """The phase read `entropy` off the ALife state and dropped kernel_weights."""
    import inspect

    from core.phases.cognitive_integration_phase import CognitiveIntegrationPhase

    src = inspect.getsource(CognitiveIntegrationPhase._run_alife_dynamics)
    assert "apply_inter_column_coupling" in src, (
        "the cognitive phase computes Lenia kernel weights every tick and never "
        "applies them to the mesh — the ecology is telemetry"
    )
