"""Tests for Grassmann-geometry residual phi encoding."""
from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.grassmann_phi import (
    GrassmannResidualComplex,
    grassmann_distance,
    principal_angles,
    subspace_basis,
)


def _basis(cols):
    return np.asarray(cols, dtype=np.float64).T  # columns = subspace vectors


# ── Grassmann distance correctness ────────────────────────────────────────

def test_identical_subspace_distance_zero():
    q = _basis([[1, 0, 0, 0], [0, 1, 0, 0]])
    assert grassmann_distance(q, q) == pytest.approx(0.0, abs=1e-9)


def test_orthogonal_subspaces_max_distance():
    qa = _basis([[1, 0, 0, 0], [0, 1, 0, 0]])
    qb = _basis([[0, 0, 1, 0], [0, 0, 0, 1]])
    # all principal angles = π/2 → distance = sqrt(k)·π/2
    assert grassmann_distance(qa, qb) == pytest.approx(np.sqrt(2) * np.pi / 2, abs=1e-9)
    assert np.allclose(principal_angles(qa, qb), np.pi / 2)


def test_rotation_within_subspace_is_invariant():
    qa = _basis([[1, 0, 0, 0], [0, 1, 0, 0]])
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    qb = qa @ rot                          # same plane, rotated basis
    assert grassmann_distance(qa, qb) == pytest.approx(0.0, abs=1e-7)


def test_partial_overlap_between_zero_and_max():
    qa = _basis([[1, 0, 0, 0], [0, 1, 0, 0]])
    qb = _basis([[1, 0, 0, 0], [0, 0, 1, 0]])   # shares one direction
    d = grassmann_distance(qa, qb)
    assert 0.0 < d < np.sqrt(2) * np.pi / 2


# ── subspace extraction ───────────────────────────────────────────────────

def test_subspace_basis_is_orthonormal():
    rng = np.random.default_rng(0)
    window = rng.standard_normal((20, 32))
    q = subspace_basis(window, k=4)
    assert q.shape[1] <= 4
    assert np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-9)


def test_subspace_recovers_dominant_directions():
    rng = np.random.default_rng(1)
    # variation concentrated in dims 0,1; tiny noise elsewhere
    t = rng.standard_normal((40, 2))
    d = 16
    data = np.zeros((40, d))
    data[:, 0] = t[:, 0] * 5
    data[:, 1] = t[:, 1] * 5
    data += rng.standard_normal((40, d)) * 1e-3
    q = subspace_basis(data, k=2)
    plane = _basis([[1] + [0] * (d - 1), [0, 1] + [0] * (d - 2)])
    assert grassmann_distance(q, plane) < 0.05    # recovered the (e0,e1) plane


# ── geometric complex ─────────────────────────────────────────────────────

def test_complex_emits_valid_states_and_learns_anchors():
    rng = np.random.default_rng(2)
    cx = GrassmannResidualComplex(n_anchors=8, subspace_dim=4, window=12, max_dims=64)
    states = []
    # three distinct geometric regimes
    for regime in range(3):
        base = rng.standard_normal((64,))
        for _ in range(60):
            v = base * (regime + 1) + rng.standard_normal(64) * 0.1
            s = cx.observe(v)
            if s is not None:
                states.append(s)
    assert states
    assert all(0 <= s < 256 for s in states)
    assert cx.status()["anchors"] >= 2           # distinct regimes seeded anchors


def test_phi_core_grassmann_residual_pipeline():
    """End-to-end: feeding residual vectors yields a computable geometric φ."""
    from core.consciousness.phi_core import PhiCore

    pc = PhiCore()
    rng = np.random.default_rng(7)
    # alternate between two geometric regimes so the TPM has real transitions
    bases = [rng.standard_normal(128), rng.standard_normal(128)]
    for i in range(400):
        b = bases[(i // 7) % 2]
        pc.record_residual_stream(b + rng.standard_normal(128) * 0.2, layer_idx=12)
    assert len(pc._grassmann_state_history) >= 50
    result = pc.compute_grassmann_residual_phi()
    assert result is not None
    assert result.phi_s >= 0.0
    status = pc.get_status()
    assert status["grassmann_phi_s"] is not None
    assert status["grassmann_status"]["anchors"] >= 2


def test_same_regime_gives_stable_state():
    rng = np.random.default_rng(3)
    cx = GrassmannResidualComplex(n_anchors=8, subspace_dim=3, window=10, max_dims=48)
    base = rng.standard_normal((48,))
    seen = []
    for _ in range(80):
        v = base + rng.standard_normal(48) * 0.01    # one tight regime
        s = cx.observe(v)
        if s is not None:
            seen.append(s)
    # a single stable regime should not explode into many distinct states
    assert len(set(seen[-20:])) <= 3
