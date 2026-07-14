"""Whole-system Φ estimation: known-answer contracts.

Every test here is a synthetic system whose integration structure is known
by construction — the estimator must recover it: exact MIP versus brute
force, zero-Φ on decomposable systems, detection on coupled ones, grain
discovery on systems with designed macro structure, and bounded claims.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.consciousness.integrated_information import (
    LaggedGaussian,
    MAX_EXACT_ELEMENTS,
    PhiEstimate,
    coarse_grain,
    estimate_whole_system_phi,
    exact_state_phi,
    minimum_information_bipartition,
    queyranne_min_bipartition,
    stochastic_interaction,
)

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic system builders
# ─────────────────────────────────────────────────────────────────────────────

def var1(A: np.ndarray, T: int, *, noise: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    X = np.zeros((T, n))
    for t in range(1, T):
        X[t] = A @ X[t - 1] + noise * rng.standard_normal(n)
    return X[50:]  # burn-in


def coupled_pairs_A() -> np.ndarray:
    """(0,1) cross-coupled, (2,3) cross-coupled, pairs independent."""
    A = np.zeros((4, 4))
    A[0, 1] = A[1, 0] = 0.7
    A[2, 3] = A[3, 2] = 0.7
    np.fill_diagonal(A, 0.2)
    return A


def ring_A(n: int, w: float = 0.55) -> np.ndarray:
    A = 0.2 * np.eye(n)
    for i in range(n):
        A[i, (i + 1) % n] = w
    return A


# ─────────────────────────────────────────────────────────────────────────────
# Queyranne: exact against brute force
# ─────────────────────────────────────────────────────────────────────────────

def _brute_force_mip(model: LaggedGaussian) -> float:
    n = model.n_channels
    h_whole = model.h_whole()
    best = math.inf
    for mask in range(1, 2 ** (n - 1)):
        S = tuple(i for i in range(n) if (mask >> i) & 1)
        C = tuple(i for i in range(n) if not (mask >> i) & 1)
        best = min(best, model.h_part(S) + model.h_part(C) - h_whole)
    return max(0.0, best)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_queyranne_matches_brute_force_on_random_systems(seed):
    rng = np.random.default_rng(seed)
    n = 6
    A = 0.5 * rng.standard_normal((n, n)) / math.sqrt(n)
    X = var1(A, 600, seed=seed)
    model = LaggedGaussian.fit(X)
    _, _, phi_q = minimum_information_bipartition(model)
    phi_b = _brute_force_mip(model)
    assert phi_q == pytest.approx(phi_b, abs=1e-9)


def test_queyranne_requires_two_elements():
    with pytest.raises(ValueError):
        queyranne_min_bipartition(1, lambda s: 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth integration structure
# ─────────────────────────────────────────────────────────────────────────────

def test_decomposable_system_has_near_zero_phi_and_correct_cut():
    X = var1(coupled_pairs_A(), 3000, seed=7)
    model = LaggedGaussian.fit(X)
    part, comp, phi = minimum_information_bipartition(model)
    # the MIP must separate the two independent pairs
    assert set(part) in ({0, 1}, {2, 3})
    assert set(comp) == {0, 1, 2, 3} - set(part)
    assert phi < 0.02  # decomposable → Φ at the true cut ≈ 0


def test_independent_channels_do_not_beat_the_null():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((800, 6))
    est = estimate_whole_system_phi(X, n_surrogates=16, n_boot=8, seed=3)
    assert not est.integration_established()
    assert est.z < 3.0


def test_ring_system_is_detected_as_integrated():
    X = var1(ring_A(6), 1500, seed=9)
    est = estimate_whole_system_phi(X, n_surrogates=16, n_boot=12, seed=5)
    assert est.phi_raw > 0.05
    assert est.z >= 3.0
    assert est.ci_5 > 0.0
    assert est.integration_established()


def test_stochastic_interaction_nonnegative_and_atomic_ge_bipartition():
    X = var1(ring_A(5), 1200, seed=13)
    model = LaggedGaussian.fit(X)
    atoms = [(i,) for i in range(5)]
    si_atoms = stochastic_interaction(model, atoms)
    _, _, phi_bi = minimum_information_bipartition(model)
    assert si_atoms >= -1e-9
    assert si_atoms >= phi_bi - 1e-9  # finest partition loses the most


# ─────────────────────────────────────────────────────────────────────────────
# Rail B: grain discovery / causal emergence
# ─────────────────────────────────────────────────────────────────────────────

def _macro_with_noisy_replicas(T: int = 2200, copies: int = 4, seed: int = 21):
    """4 strongly-coupled macro latents, each observed as `copies` noisy
    replicas: the real integration lives at k=4, drowned at the micro grain."""
    rng = np.random.default_rng(seed)
    L = var1(ring_A(4, 0.6), T, seed=seed)
    cols, names = [], []
    for i in range(4):
        for c in range(copies):
            cols.append(L[:, i] + 1.5 * rng.standard_normal(L.shape[0]))
            names.append(f"m{i}_rep{c}")
    return np.stack(cols, axis=1), tuple(names)


def test_grain_search_recovers_designed_macro_structure():
    X, names = _macro_with_noisy_replicas()
    est = estimate_whole_system_phi(
        X, channel_names=names, n_surrogates=12, n_boot=6,
        grains=[16, 8, 4], seed=17,
    )
    g4 = next(g for g in est.grains if g.k == 4)
    # the k=4 grouping must reunite the replica blocks
    for group in g4.groups:
        latents = {names[i].split("_")[0] for i in group}
        assert len(latents) == 1, f"grain mixed latents: {group}"
    # and the emergent grain is the designed one, beating micro
    assert est.emergent_grain_k == 4
    assert est.emergence_delta_z > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Rail C: exact discrete Φ at small k
# ─────────────────────────────────────────────────────────────────────────────

def test_exact_phi_zero_for_independent_chains_and_splits_them():
    rng = np.random.default_rng(31)
    T = 4000
    a = np.zeros(T)
    b = np.zeros(T)
    for t in range(1, T):  # two independent sticky binary chains
        a[t] = a[t - 1] if rng.random() < 0.9 else 1 - a[t - 1]
        b[t] = b[t - 1] if rng.random() < 0.9 else 1 - b[t - 1]
    Xk = np.stack([a + 0.01 * rng.standard_normal(T),
                   b + 0.01 * rng.standard_normal(T)], axis=1)
    out = exact_state_phi(Xk)
    assert out["phi_s"] < 0.02
    assert set(out["mip"][0]) in ({0}, {1})


def test_exact_phi_detects_crosswired_coupling():
    rng = np.random.default_rng(37)
    T = 4000
    s = np.zeros((T, 2), dtype=int)
    s[0] = (0, 1)
    for t in range(1, T):
        # crosswired swap: each unit's future is the OTHER unit's past —
        # each part alone predicts itself poorly, the whole predicts
        # near-perfectly.  Integration by construction.
        s[t, 0] = s[t - 1, 1] if rng.random() < 0.95 else 1 - s[t - 1, 1]
        s[t, 1] = s[t - 1, 0] if rng.random() < 0.95 else 1 - s[t - 1, 0]
    Xk = s + 0.01 * rng.standard_normal((T, 2))
    out = exact_state_phi(Xk)
    assert out["phi_s"] > 0.1
    assert out["cuts_searched"] == 1


def test_exact_phi_accepts_interventional_transitions():
    rng = np.random.default_rng(41)
    Xk = rng.standard_normal((200, 3))
    extra = [((0, 0, 0), (1, 1, 1))] * 5
    out = exact_state_phi(Xk, extra_transitions=extra)
    assert out["n_interventional_transitions"] == 5


def test_exact_phi_caps_at_twelve_elements():
    rng = np.random.default_rng(43)
    with pytest.raises(ValueError):
        exact_state_phi(rng.standard_normal((100, MAX_EXACT_ELEMENTS + 1)))


# ─────────────────────────────────────────────────────────────────────────────
# Honesty layer
# ─────────────────────────────────────────────────────────────────────────────

def test_report_carries_provenance_and_bounded_claim():
    X = var1(ring_A(5), 900, seed=51)
    est = estimate_whole_system_phi(X, n_surrogates=10, n_boot=6, seed=7)
    d = est.to_dict()
    for key in ("estimator", "null_mean", "ci_5", "ci_95", "diagnostics",
                "claim", "grains", "integration_established"):
        assert key in d
    assert "not a consciousness meter" in est.claim
    assert d["estimator"].startswith("gaussian_stochastic_interaction")
    assert d["diagnostics"]["n_samples"] > 0


def test_dead_channels_are_dropped_and_reported():
    X = var1(ring_A(4), 700, seed=53)
    X = np.hstack([X, np.zeros((X.shape[0], 1))])  # a flatlined channel
    est = estimate_whole_system_phi(X, n_surrogates=8, n_boot=4, seed=9)
    assert est.n_channels == 4
    assert est.diagnostics["dropped_dead_channels"] == 1.0


def test_estimates_are_deterministic_under_a_seed():
    X = var1(ring_A(5), 800, seed=57)
    a = estimate_whole_system_phi(X, n_surrogates=8, n_boot=6, seed=42)
    b = estimate_whole_system_phi(X, n_surrogates=8, n_boot=6, seed=42)
    assert a.phi_raw == b.phi_raw
    assert a.z == b.z
    assert a.ci_5 == b.ci_5


def test_coarse_grain_shapes():
    X = np.random.default_rng(3).standard_normal((100, 6))
    Xk = coarse_grain(X, [[0, 1], [2, 3], [4, 5]])
    assert Xk.shape == (100, 3)


def test_integration_established_logic():
    est = PhiEstimate(
        schema_version=1, estimator="x", computed_at=0.0, n_channels=4,
        n_samples=100, channel_names=("a", "b", "c", "d"), phi_raw=0.5,
        mip=((0,), (1, 2, 3)), null_mean=0.1, null_std=0.05, z=8.0,
        ci_5=0.2, ci_95=0.7,
    )
    assert est.integration_established()
    est.z = 2.0
    assert not est.integration_established()
