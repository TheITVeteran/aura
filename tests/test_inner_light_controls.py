"""Tests that each negative control destroys the axis it is supposed to.

A "rich" reference matrix is built to be high on every axis — modular integration,
long-range temporal correlations, and all-or-none ignition bursts. Each control
must then knock down its target axis (and the synthetic references must sit in
their known regimes), which is exactly what lets the battery make a conjunctive
claim.
"""
from __future__ import annotations

import numpy as np

from core.consciousness.inner_light import controls as ctl
from core.consciousness.inner_light.measures import (
    bimodality_ignition,
    dfa,
    normalized_lz,
    tse_complexity,
)


def _pink(T, rng, beta=1.4):
    """Stationary 1/f^beta (pink) noise — brain-like long-range temporal
    correlations WITHOUT the spurious trend-correlation of a random walk."""
    w = rng.standard_normal(T)
    F = np.fft.rfft(w)
    f = np.arange(len(F), dtype=float)
    f[0] = 1.0
    x = np.fft.irfft(F / f ** (beta / 2.0), n=T)
    return (x - x.mean()) / x.std()


def _rich(seed=0, nmod=3, per=2, T=2000, amp=4.0, dur=2, rate=22, noise=0.35, frac=0.7):
    """High on every axis at once: MANY small differentiated modules (integration
    + differentiation), pink temporal structure (criticality), and PARTIAL
    all-or-none bursts on a channel subset (ignition without collapsing to one
    synchronised common mode — the real reconciliation brains use)."""
    rng = np.random.default_rng(seed)
    n = nmod * per
    lats = [_pink(T, rng) for _ in range(nmod)]
    chans = [lats[m] + noise * rng.standard_normal(T) for m in range(nmod) for _ in range(per)]
    M = np.stack(chans)
    k = max(1, int(round(frac * n)))
    for t in rng.choice(T - dur, size=T // rate, replace=False):
        who = rng.choice(n, size=k, replace=False)
        M[np.ix_(who, np.arange(t, t + dur))] += amp
    return M


def _global(M):
    return np.asarray(M).sum(axis=0)


# ── the reference really is high on every axis ───────────────────────────────

def test_reference_is_conscious_like_on_all_axes():
    M = _rich()
    assert normalized_lz(M) > 0.3            # differentiated
    assert tse_complexity(M) > 0.005         # integrated + differentiated
    assert dfa(_global(M)) > 0.7             # long-range temporal correlations
    assert bimodality_ignition(M) > 0.5      # all-or-none ignition


# ── each transform destroys its target axis ──────────────────────────────────

def test_time_shuffle_destroys_criticality():
    M = _rich()
    before = dfa(_global(M))
    after = dfa(_global(ctl.time_shuffle(M)))
    assert after < before - 0.3
    assert abs(after - 0.5) < 0.2  # collapses toward uncorrelated


def test_lesion_decouple_destroys_integration():
    M = _rich()
    before = tse_complexity(M)
    after = tse_complexity(ctl.lesion_decouple(M))
    assert after < before                     # between-channel binding gone
    assert after < 0.5 * before + 1e-6


def test_phase_randomize_destroys_ignition():
    M = _rich()
    before = bimodality_ignition(M)
    after = bimodality_ignition(ctl.phase_randomize(M))
    assert after < before                     # non-linear all-or-none smeared out


def test_phase_randomize_preserves_power_spectrum():
    M = _rich()
    surrogate = ctl.phase_randomize(M)
    p_before = np.abs(np.fft.rfft(M, axis=1)) ** 2
    p_after = np.abs(np.fft.rfft(surrogate, axis=1)) ** 2
    # the spectrum is what a linear-Gaussian twin must keep
    assert np.allclose(p_before, p_after, rtol=1e-6, atol=1e-6)


# ── synthetic references sit in their known regimes ──────────────────────────

def test_white_noise_is_differentiated_not_integrated():
    M = ctl.white_noise((6, 2000))
    assert normalized_lz(M) > 0.6            # high differentiation
    assert tse_complexity(M) < 0.02          # ~zero integration → not complex


def test_ordered_is_not_differentiated():
    M = ctl.ordered((6, 2000), period=8)
    assert normalized_lz(M) < 0.3            # repeating pattern → low complexity


def test_feedforward_chain_carries_info_forward_but_does_not_bind():
    M = ctl.feedforward_chain((5, 1500))
    assert M.shape == (5, 1500)
    assert np.all(np.isfinite(M))
    # Information flows forward with a lag (stage i-1 predicts stage i one step later)…
    lagged = np.corrcoef(M[0, :-1], M[1, 1:])[0, 1]
    assert abs(lagged) > 0.2
    # …but there is no instantaneous binding and no recurrent loop, so it lacks the
    # rich reference's integrated complexity.
    instantaneous = np.corrcoef(M[0], M[1])[0, 1]
    assert abs(instantaneous) < 0.1
    assert tse_complexity(M) < tse_complexity(_rich())
