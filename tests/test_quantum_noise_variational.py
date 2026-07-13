"""Quantum depth pass: density-matrix noise channels + parameter-shift
gradients — every assertion against a closed-form law.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.quantum import Statevector
from core.quantum.density import MAX_DENSITY_QUBITS, DensityMatrix
from core.quantum.statevector import QuantumCircuitError
from core.quantum.variational import (
    gradient_descent,
    parameter_shift_gradient,
)


# ── Density matrix fundamentals ──────────────────────────────────

def test_pure_state_roundtrip_and_purity():
    bell = Statevector(2, seed=1).h(0).cx(0, 1)
    rho = DensityMatrix.from_statevector(bell)
    assert rho.purity() == pytest.approx(1.0, abs=1e-12)
    assert rho.trace() == pytest.approx(1.0, abs=1e-12)
    populations = rho.populations()
    assert populations[0b00] == pytest.approx(0.5, abs=1e-12)
    assert populations[0b11] == pytest.approx(0.5, abs=1e-12)


def test_gate_application_matches_statevector():
    rho = DensityMatrix(1).apply_gate("X", 0)
    assert rho.probability_of_one(0) == pytest.approx(1.0, abs=1e-12)


def test_malformed_kraus_channel_refused():
    with pytest.raises(QuantumCircuitError, match="K†K"):
        DensityMatrix(1).apply_kraus([np.eye(2) * 0.5], 0)
    with pytest.raises(QuantumCircuitError):
        DensityMatrix(MAX_DENSITY_QUBITS + 1)


# ── T1 relaxation (amplitude damping) ────────────────────────────

def test_amplitude_damping_follows_exponential_decay():
    """P(|1⟩) after n damping steps of strength γ is exactly (1-γ)ⁿ —
    the discrete form of e^{-t/T1}."""
    gamma = 0.05
    rho = DensityMatrix(1).apply_gate("X", 0)  # start in |1⟩
    for n in range(1, 40):
        rho.amplitude_damping(gamma, 0)
        assert rho.probability_of_one(0) == pytest.approx(
            (1.0 - gamma) ** n, abs=1e-12), n
    assert rho.trace() == pytest.approx(1.0, abs=1e-12)


def test_ground_state_is_amplitude_damping_fixed_point():
    rho = DensityMatrix(1)
    for _ in range(20):
        rho.amplitude_damping(0.3, 0)
    assert rho.probability_of_one(0) == pytest.approx(0.0, abs=1e-12)
    assert rho.purity() == pytest.approx(1.0, abs=1e-12)


# ── T2 dephasing (phase damping) ─────────────────────────────────

def test_phase_damping_kills_coherence_not_population():
    lam = 0.1
    rho = DensityMatrix(1).apply_gate("H", 0)  # |+⟩: coherence 0.5
    for n in range(1, 30):
        rho.phase_damping(lam, 0)
        assert abs(rho.coherence(0, 1)) == pytest.approx(
            0.5 * (1.0 - lam) ** (n / 2.0), rel=1e-9), n
        assert rho.probability_of_one(0) == pytest.approx(0.5, abs=1e-12)


# ── Depolarizing ─────────────────────────────────────────────────

def test_depolarizing_contracts_to_maximally_mixed():
    p = 0.2
    rho = DensityMatrix(1).apply_gate("H", 0)
    for n in range(1, 30):
        rho.depolarizing(p, 0)
        # Bloch vector contracts by (1 - 4p/3) per application.
        assert abs(rho.coherence(0, 1)) == pytest.approx(
            0.5 * (1.0 - 4.0 * p / 3.0) ** n, rel=1e-9), n
    assert rho.purity() == pytest.approx(0.5, abs=1e-3)  # → I/2


def test_noisy_bell_state_fidelity_decays():
    bell = Statevector(2, seed=1).h(0).cx(0, 1)
    rho = DensityMatrix.from_statevector(bell)
    reference = bell.state.copy()
    last = 1.0
    for _ in range(5):
        rho.depolarizing(0.05, 0).depolarizing(0.05, 1)
        fidelity = rho.fidelity_to_pure(reference)
        assert fidelity < last
        last = fidelity
    assert 0.5 < last < 1.0
    assert rho.trace() == pytest.approx(1.0, abs=1e-12)


# ── Parameter-shift gradients ────────────────────────────────────

def _ry_circuit(params):
    return Statevector(1, seed=0).ry(params[0], 0)


def test_parameter_shift_matches_closed_form():
    """⟨Z⟩ of RY(θ)|0⟩ is cos θ; the gradient must be −sin θ exactly."""
    for theta in (0.0, 0.3, 1.1, math.pi / 2, 2.7, -0.8):
        gradient = parameter_shift_gradient(_ry_circuit, [theta], "Z")
        assert gradient[0] == pytest.approx(-math.sin(theta), abs=1e-10), theta


def _entangled_ansatz(params):
    state = Statevector(2, seed=0)
    state.ry(params[0], 0).ry(params[1], 1).cx(0, 1).rz(params[2], 1)
    return state


def test_parameter_shift_agrees_with_finite_difference():
    params = [0.4, -0.9, 1.3]
    analytic = parameter_shift_gradient(_entangled_ansatz, params, "ZZ")
    epsilon = 1e-6
    for index in range(3):
        plus, minus = list(params), list(params)
        plus[index] += epsilon
        minus[index] -= epsilon
        numeric = (
            _entangled_ansatz(plus).expectation_pauli("ZZ")
            - _entangled_ansatz(minus).expectation_pauli("ZZ")
        ) / (2 * epsilon)
        assert analytic[index] == pytest.approx(numeric, abs=1e-6), index


def test_vqe_descent_finds_the_ground_state():
    """Minimizing ⟨Z⟩ over RY(θ)|0⟩ must land at θ=π (⟨Z⟩=−1)."""
    result = gradient_descent(_ry_circuit, [0.5], "Z", learning_rate=0.3, steps=200)
    assert result["value"] == pytest.approx(-1.0, abs=1e-6)
    assert result["initial_value"] > result["value"]
    assert result["converged"]
