"""Tests for the voltage-dependent plasticity engine (Clopath 2010 / Pantheon board).

These assert the *defining* properties the whiteboard circles in red — the
things plain spike-timing STDP lacks:

  * voltage-gating (no sub-threshold plasticity; LTP needs high voltage),
  * a homeostatic fixed point and hard runaway/"epilepsy" prevention,
  * BCM-like sliding-threshold LTD scaling,
  * synaptic competition (winner amplification),
  * an STDP window that *emerges from voltage* rather than hand-tuned timing.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.voltage_plasticity import (
    VoltageDependentPlasticityEngine,
    VoltagePlasticityConfig,
    stable_exp,
)


def _engine(**kw) -> VoltageDependentPlasticityEngine:
    return VoltageDependentPlasticityEngine(VoltagePlasticityConfig(**kw))


# ── numerical armour ──────────────────────────────────────────────────────

def test_stable_exp_never_overflows():
    assert np.isfinite(stable_exp(1e9))
    assert np.isfinite(stable_exp(-1e9))
    assert stable_exp(0.0) == pytest.approx(1.0)


# ── board equations ───────────────────────────────────────────────────────

def test_firing_rate_is_exponential_escape_rate():
    eng = _engine(n_nodes=4, rho0=1.0, theta=1.0, delta_beta=0.5)
    # b = ρ₀·exp((V−θ)/Δβ): equals ρ₀ at threshold, monotone increasing.
    assert eng.firing_rate(1.0) == pytest.approx(1.0)
    assert eng.firing_rate(1.5) > eng.firing_rate(1.0) > eng.firing_rate(0.5)


def test_homeostatic_pressure_monotone_and_crosses_unity_at_theta():
    eng = _engine(n_nodes=4, theta=1.0, delta_u=1.0)
    low = eng.homeostatic_pressure(np.array([0.1, 0.1, 0.0, 0.0]))   # Σb=0.2 < θ
    high = eng.homeostatic_pressure(np.array([1.0, 1.0, 1.0, 0.0]))  # Σb=3.0 > θ
    at = eng.homeostatic_pressure(np.array([0.5, 0.5, 0.0, 0.0]))    # Σb=1.0 = θ
    assert low < 1.0 < high
    assert at == pytest.approx(1.0)


def test_activity_fixed_point_matches_closed_form():
    cfg = VoltagePlasticityConfig(n_nodes=3, lam=0.30, delta=0.45, rho0=1.0,
                                  kappa=0.15, theta=1.0, delta_u=1.0)
    eng = VoltageDependentPlasticityEngine(cfg)
    b = np.zeros(3)
    pressure = eng.homeostatic_pressure(b)
    expected = (cfg.delta / cfg.lam) * cfg.rho0 * pressure - cfg.kappa
    assert eng.activity_fixed_point(b)[0] == pytest.approx(max(0.0, expected))


# ── mean-field weight expression + competition difference + input gate ────

def test_mean_field_weights_match_closed_form():
    cfg = VoltagePlasticityConfig(n_nodes=4, p=0.7, w_bar_plus=1.0, w_bar_minus=0.8,
                                  kappa=0.15, rho0=1.0, theta=1.0, delta_u=1.0)
    eng = VoltageDependentPlasticityEngine(cfg)
    b = np.array([0.2, 0.5, 0.1, 0.4])
    pressure = eng.homeostatic_pressure(b)
    expected = cfg.p ** 2 * cfg.w_bar_plus * (b + cfg.kappa) - cfg.p * cfg.w_bar_minus * cfg.rho0 * pressure
    assert np.allclose(eng.mean_field_weights(b), expected)


def test_mean_field_weight_difference_cancels_population_term():
    # w_k − w_j = p² W̄₊ (b_k − b_j): the homeostatic depression cancels in diffs.
    eng = _engine(n_nodes=3, p=0.7, w_bar_plus=1.0)
    b = np.array([0.6, 0.2, 0.4])
    diff = eng.mean_field_weight_difference(b)
    w = eng.mean_field_weights(b)
    assert diff[0, 1] == pytest.approx(w[0] - w[1])
    assert diff[0, 1] == pytest.approx(0.7 ** 2 * 1.0 * (b[0] - b[1]))


def test_competition_difference_saturates_and_is_zero_at_no_edge():
    eng = _engine(n_nodes=2, p=0.7, w_bar_plus=1.0, delta_beta=0.5)
    b = np.array([1.0, 0.3])
    comp = eng.competition_difference(b)
    # no initial edge (k==j) → zero; larger edge → larger (but bounded) amplification
    assert comp[0, 0] == pytest.approx(0.0)
    small = eng.competition_difference(b, delta0=0.1)[0, 0]
    big = eng.competition_difference(b, delta0=5.0)[0, 0]
    assert 0.0 < small < big
    # saturates: 1−exp(−Δ/Δβ) < 1, so bounded by p²W̄₊·b_k
    assert big < 0.7 ** 2 * 1.0 * b[0]


def test_input_gate_transfer_function():
    eng = _engine(n_nodes=2, beta_vrp=2.0, theta0=0.5, delta_b_in=1.0)
    assert eng.input_gate(0.5) == pytest.approx(0.0)          # at threshold → 0
    assert eng.input_gate(1.5) == pytest.approx(2.0)          # β·((1.5−0.5)/1)
    assert float(np.mean(eng.input_gate(np.array([0.5, 1.0])))) == pytest.approx(0.5)


# ── stability: the whole reason the board adds homeostasis ────────────────

def test_no_runaway_under_extreme_constant_drive():
    """Drive every node with a massive constant input → must stay bounded.

    This is the anti-epilepsy guarantee: exponential homeostatic depression
    cannot be out-grown by polynomial self-excitation.
    """
    eng = _engine(n_nodes=64)
    inputs = np.full((400, 64), 100.0)
    states = eng.run(inputs, learn=True)
    assert np.isfinite(states).all()
    assert states.max() <= eng.cfg.state_clip + 1e-6
    assert eng.is_stable()
    assert np.isfinite(eng.W).all()
    assert np.abs(eng.W).max() <= eng.cfg.weight_clip + 1e-6


def test_field_converges_to_bounded_steady_state_under_input():
    eng = _engine(n_nodes=8, seed=3)
    inputs = np.full((500, 8), 0.30)
    states = eng.run(inputs, learn=False)
    tail = states[-50:].sum(axis=1)            # total activity over the tail
    assert np.std(tail) < 1e-2                  # converged
    assert 0.0 <= tail.mean() <= 8 * eng.cfg.state_clip


def test_energy_stays_finite_over_long_run():
    eng = _engine(n_nodes=32)
    rng = np.random.default_rng(0)
    for _ in range(300):
        eng.step(external_input=rng.random(32) * 2.0)
        assert np.isfinite(eng.energy())
    assert eng.is_stable()


# ── voltage gating ────────────────────────────────────────────────────────

def test_no_plasticity_below_low_threshold():
    eng = _engine(n_nodes=6, theta_minus=0.15, theta_plus=0.55)
    pre = np.ones(6)
    sub = np.full(6, 0.05)                      # voltage well below θ₋
    dw = np.zeros((6, 6))
    for _ in range(5):
        dw = eng.voltage_plasticity_delta(pre, sub)
    assert np.abs(dw).max() < 1e-9


def test_ltp_requires_voltage_above_high_threshold():
    # Between θ₋ and θ₊ with a pre spike → depression only (no LTP).
    eng = _engine(n_nodes=2, theta_minus=0.15, theta_plus=0.55)
    pre = np.array([0.0, 1.0])                  # node1 is presynaptic
    mid = np.array([0.40, 0.0])                 # node0 voltage between thresholds
    dw = np.zeros((2, 2))
    for _ in range(6):
        dw = eng.voltage_plasticity_delta(pre, mid)
    assert dw[0, 1] <= 1e-9                      # synapse pre(1)→post(0): no potentiation


# ── BCM sliding threshold ─────────────────────────────────────────────────

def test_bcm_homeostatic_scaling_increases_ltd_with_average_voltage():
    pre = np.array([0.0, 1.0])
    volt = np.array([0.5, 0.0])                 # elevated low-pass voltage on node0

    low = _engine(n_nodes=2)
    low.u_avg = np.full(2, 0.20)                # low historical activity
    high = _engine(n_nodes=2)
    high.u_avg = np.full(2, 0.90)               # high historical activity → stronger LTD

    dw_low = np.zeros((2, 2))
    dw_high = np.zeros((2, 2))
    for _ in range(8):
        dw_low = low.voltage_plasticity_delta(pre, volt)
        dw_high = high.voltage_plasticity_delta(pre, volt)
    # More depression (more negative dw on the active synapse) when ⟨ū⟩ is high.
    assert dw_high[0, 1] < dw_low[0, 1]


# ── competition / winner amplification ────────────────────────────────────

def test_competition_drive_amplifies_above_average_nodes():
    eng = _engine(n_nodes=3, competition=0.1)
    eng.W = np.array([[0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5],
                      [0.5, 0.5, 0.0]], dtype=float)
    b = np.array([1.0, 0.1, 0.1])               # node0 well above the mean
    drive = eng.competition_drive(b)
    # Above-average row (0) is potentiated (same sign as W → |W| grows);
    # below-average rows (1,2) are depressed (opposite sign → |W| shrinks).
    assert (drive[0] * np.sign(eng.W[0]))[1] > 0
    assert (drive[1] * np.sign(eng.W[1]))[0] < 0


def test_competition_grows_winner_weight_under_differential_input():
    eng = _engine(n_nodes=4, seed=5, competition=0.08)
    drive = np.zeros((300, 4))
    drive[:, 0] = 0.8                           # node0 persistently driven
    drive[:, 1] = 0.05
    eng.run(drive, learn=True)
    # The persistently-active node's incoming-weight row ends stronger than a
    # weakly-driven node's row.
    assert np.linalg.norm(eng.W[0]) > np.linalg.norm(eng.W[1])


# ── STDP window emerges from voltage (the bottom-right graph) ─────────────

def _accumulate_dw(eng, sequence, post_idx, pre_idx):
    total = 0.0
    for pre, volt in sequence:
        dw = eng.voltage_plasticity_delta(np.asarray(pre, float), np.asarray(volt, float))
        total += dw[post_idx, pre_idx]
    return total


def test_causal_pairing_potentiates_anticausal_depresses():
    # Synapse pre(node1) → post(node0).
    low = [0.05, 0.05]
    # Causal: pre spikes first (build x̄), then post depolarizes high → LTP.
    causal = (
        [([0.0, 1.0], low)] * 3
        + [([0.0, 0.0], [1.0, 0.05])]
    )
    # Anti-causal: post depolarizes first (build ū₋), then pre spikes low → LTD.
    anticausal = (
        [([0.0, 0.0], [1.0, 0.05])] * 3
        + [([0.0, 1.0], low)]
    )
    net_causal = _accumulate_dw(_engine(n_nodes=2), causal, 0, 1)
    net_anti = _accumulate_dw(_engine(n_nodes=2), anticausal, 0, 1)
    assert net_causal > 0.0      # pre-before-post → potentiation
    assert net_anti < 0.0        # post-before-pre → depression


# ── determinism ───────────────────────────────────────────────────────────

def test_determinism_with_seed():
    a = _engine(n_nodes=16, seed=42)
    b = _engine(n_nodes=16, seed=42)
    rng_inputs = np.random.default_rng(1).random((50, 16))
    sa = a.run(rng_inputs, learn=True)
    sb = b.run(rng_inputs, learn=True)
    assert np.allclose(sa, sb)
    assert np.allclose(a.W, b.W)
