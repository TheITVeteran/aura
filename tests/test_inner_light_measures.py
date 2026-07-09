"""Known-answer tests for the consciousness-discriminator measures.

The whole instrument's credibility rests here: each measure must land in the
right regime on synthetic inputs whose answer we already know from theory —
noise is maximally differentiated but not integrated, a synchronised system is
integrated but not differentiated, brown noise is more long-range-correlated than
white noise, a bimodal signal ignites and a Gaussian one does not.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.inner_light.measures import (
    avalanche_criticality,
    bimodality_ignition,
    binarize,
    dfa,
    gaussian_integration,
    lz76,
    normalized_lz,
    tse_complexity,
)


def _rng(seed=0):
    return np.random.default_rng(seed)


# ── Lempel-Ziv / differentiation ─────────────────────────────────────────────

def test_lz76_orders_constant_periodic_random():
    n = 400
    const = [0] * n
    periodic = [0, 1] * (n // 2)
    rand = _rng(1).integers(0, 2, n).tolist()
    assert lz76(const) < lz76(periodic) < lz76(rand)


def test_normalized_lz_regimes():
    T = 300
    const = np.zeros((5, T))
    periodic = np.tile(np.array([0.0, 1.0]), (5, T // 2))
    rand = _rng(2).random((5, T))  # continuous → median split gives a fair coin
    nl_c = normalized_lz(const)
    nl_p = normalized_lz(periodic)
    nl_r = normalized_lz(rand)
    assert nl_c < 0.25
    assert nl_r > 0.6
    assert nl_c < nl_p < nl_r


def test_binarize_above_median():
    M = np.array([[0.0, 1.0, 2.0, 3.0]])
    b = binarize(M)
    assert b.shape == M.shape
    assert b.tolist() == [[0, 0, 1, 1]]  # above the median (1.5)


# ── integration ──────────────────────────────────────────────────────────────

def test_gaussian_integration_independent_vs_correlated():
    rng = _rng(3)
    indep = np.cov(rng.normal(size=(4, 2000)))
    assert gaussian_integration(indep) < 0.2  # ~0 for independent channels

    latent = rng.normal(size=2000)
    corr_data = np.stack([latent + 0.05 * rng.normal(size=2000) for _ in range(4)])
    assert gaussian_integration(np.cov(corr_data)) > 2.0  # strong redundancy


# ── TSE neural complexity ────────────────────────────────────────────────────

def test_tse_peaks_between_noise_and_synchrony():
    rng = _rng(4)
    T = 1500
    independent = rng.normal(size=(4, T))

    base = rng.normal(size=T)
    synchronised = np.stack([base + 0.02 * rng.normal(size=T) for _ in range(4)])

    a, b = rng.normal(size=T), rng.normal(size=T)
    modular = np.stack([
        a + 0.1 * rng.normal(size=T),
        a + 0.1 * rng.normal(size=T),
        b + 0.1 * rng.normal(size=T),
        b + 0.1 * rng.normal(size=T),
    ])

    c_ind = tse_complexity(independent)
    c_sync = tse_complexity(synchronised)
    c_mod = tse_complexity(modular)

    # Complexity is high only when integrated AND differentiated: the modular
    # system beats both the pure-noise and the fully-synchronised extremes, with
    # clear separation from noise.
    assert c_mod > c_ind
    assert c_mod > c_sync
    assert c_mod > 1.5 * c_ind


# ── criticality ──────────────────────────────────────────────────────────────

def test_dfa_white_vs_brown():
    rng = _rng(5)
    white = rng.normal(size=4000)
    brown = np.cumsum(rng.normal(size=4000))
    a_white = dfa(white)
    a_brown = dfa(brown)
    assert abs(a_white - 0.5) < 0.15   # uncorrelated
    assert a_brown > 1.2               # random walk ≈ 1.5
    assert a_white < a_brown


def test_avalanche_structure_on_bursty_signal():
    rng = _rng(6)
    T = 600
    M = 0.01 * rng.random((5, T))
    # inject synchronized bursts of varying size
    for start, width, amp in [(20, 3, 3), (80, 6, 5), (160, 2, 2), (250, 8, 7),
                              (330, 4, 4), (400, 5, 6), (470, 3, 3), (520, 7, 5),
                              (560, 2, 2)]:
        M[:, start:start + width] += amp
    out = avalanche_criticality(M)
    assert set(out) == {"exponent", "fit_r2", "n_avalanches", "score"}
    assert out["n_avalanches"] >= 8
    assert 0.0 <= out["score"] <= 1.0


# ── ignition ─────────────────────────────────────────────────────────────────

def test_bimodality_ignition_gaussian_vs_bimodal():
    rng = _rng(7)
    # Unimodal Gaussian global activation → low BC. (Single channel so the global
    # signal is exactly the Gaussian, not a sum that would re-Gaussianise.)
    gaussian = rng.normal(size=(1, 2000))
    bc_uni = bimodality_ignition(gaussian)

    # All-or-none: long quiet baseline punctuated by ignited high states.
    ig = np.where(rng.random(2000) < 0.25, 8.0, 0.0) + 0.1 * rng.normal(size=2000)
    bimodal = ig.reshape(1, -1)
    bc_bi = bimodality_ignition(bimodal)

    assert bc_uni < 0.55
    assert bc_bi > 0.6
    assert bc_bi > bc_uni


def test_flat_signal_has_no_ignition():
    assert bimodality_ignition(np.ones((3, 100))) == 0.0
