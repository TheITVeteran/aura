"""Quantum computational module: analytic verification battery.

Every assertion here has a closed-form quantum-mechanical answer; the
simulator must reproduce it to numerical precision, not approximately.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.quantum import (
    MAX_QUBITS,
    QuantumCircuitError,
    Statevector,
    bell_pair,
    ghz_state,
    grover_search,
    qft_circuit,
    teleport,
)
from core.quantum.algorithms import qft_matrix


# ── Elementary gate identities ───────────────────────────────────

def test_h_twice_is_identity():
    sv = Statevector(1, seed=7).h(0).h(0)
    assert abs(sv.state[0] - 1.0) < 1e-12
    assert abs(sv.state[1]) < 1e-12


def test_x_flips_basis_state():
    sv = Statevector(2, seed=7).x(0)
    assert sv.probability_of("10") == pytest.approx(1.0)


def test_gates_preserve_norm():
    sv = Statevector(3, seed=7)
    sv.h(0).cx(0, 1).t(2).rz(0.7, 1).ry(1.1, 0).ccx(0, 1, 2).swap(0, 2)
    assert float(np.sum(sv.probabilities())) == pytest.approx(1.0, abs=1e-12)


def test_expectation_values_on_known_states():
    zero = Statevector(1, seed=1)
    assert zero.expectation_pauli("Z") == pytest.approx(1.0)
    plus = Statevector(1, seed=1).h(0)
    assert plus.expectation_pauli("X") == pytest.approx(1.0)
    assert plus.expectation_pauli("Z") == pytest.approx(0.0, abs=1e-12)


# ── Entanglement ─────────────────────────────────────────────────

def test_bell_pair_amplitudes_and_correlations():
    sv = bell_pair(seed=3)
    assert sv.probability_of("00") == pytest.approx(0.5)
    assert sv.probability_of("11") == pytest.approx(0.5)
    assert sv.probability_of("01") == pytest.approx(0.0, abs=1e-12)
    # Perfect correlation in both Z and X bases — the entanglement witness.
    assert sv.expectation_pauli("ZZ") == pytest.approx(1.0)
    assert sv.expectation_pauli("XX") == pytest.approx(1.0)
    # No signaling: each local marginal stays maximally mixed.
    assert sv.expectation_pauli("ZI") == pytest.approx(0.0, abs=1e-12)
    assert sv.expectation_pauli("IZ") == pytest.approx(0.0, abs=1e-12)


def test_bell_measurements_always_agree():
    for seed in range(20):
        sv = bell_pair(seed=seed)
        assert sv.measure(0) == sv.measure(1)


def test_ghz_state_structure():
    sv = ghz_state(4, seed=5)
    assert sv.probability_of("0000") == pytest.approx(0.5)
    assert sv.probability_of("1111") == pytest.approx(0.5)


# ── Measurement semantics ────────────────────────────────────────

def test_measurement_collapse_is_stable():
    sv = Statevector(1, seed=11).h(0)
    first = sv.measure(0)
    for _ in range(5):
        assert sv.measure(0) == first
    assert float(np.sum(sv.probabilities())) == pytest.approx(1.0)


def test_entropy_source_drives_collapse():
    # entropy → 0.99: r < p(1) is false for a fair qubit → outcome 0.
    sv = Statevector(1, entropy_source=lambda: 0.99).h(0)
    assert sv.measure(0) == 0
    # entropy → 0.01 forces outcome 1.
    sv = Statevector(1, entropy_source=lambda: 0.01).h(0)
    assert sv.measure(0) == 1


def test_sampling_is_deterministic_with_seed():
    counts_a = bell_pair(seed=42).sample_counts(1000)
    counts_b = bell_pair(seed=42).sample_counts(1000)
    assert counts_a == counts_b
    assert set(counts_a) <= {"00", "11"}
    assert sum(counts_a.values()) == 1000


# ── Canonical algorithms ─────────────────────────────────────────

@pytest.mark.parametrize("alpha,beta", [
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
    (0.6, 0.8j),
    (0.5 + 0.5j, 0.5 - 0.5j),
])
def test_teleportation_has_unit_fidelity(alpha, beta):
    for seed in range(8):
        result = teleport(alpha, beta, seed=seed)
        assert result["fidelity"] == pytest.approx(1.0, abs=1e-10)


def test_teleportation_exercises_all_correction_branches():
    branches = {
        (teleport(0.6, 0.8, seed=seed)["m0"], teleport(0.6, 0.8, seed=seed)["m1"])
        for seed in range(64)
    }
    assert branches == {(0, 0), (0, 1), (1, 0), (1, 1)}


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 6, 8])
def test_grover_matches_analytic_success_probability(num_qubits):
    marked = (1 << num_qubits) - 2
    result = grover_search(num_qubits, marked, seed=0)
    assert result["success_probability"] == pytest.approx(
        result["analytic_prediction"], abs=1e-9
    )
    if num_qubits >= 3:
        assert result["success_probability"] > 0.9


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4, 5])
def test_qft_circuit_reproduces_analytic_matrix(num_qubits):
    analytic = qft_matrix(num_qubits)
    for basis in range(1 << num_qubits):
        sv = Statevector(num_qubits, seed=0)
        sv.state[:] = 0.0
        sv.state[basis] = 1.0
        qft_circuit(sv)
        np.testing.assert_allclose(sv.state, analytic[:, basis], atol=1e-10)


# ── Bounds and error handling ────────────────────────────────────

def test_qubit_cap_enforced():
    with pytest.raises(QuantumCircuitError):
        Statevector(MAX_QUBITS + 1)


def test_invalid_operations_raise():
    sv = Statevector(2, seed=0)
    with pytest.raises(QuantumCircuitError):
        sv.h(5)
    with pytest.raises(QuantumCircuitError):
        sv.cx(1, 1)
    with pytest.raises(QuantumCircuitError):
        sv.probability_of("0")
    with pytest.raises(QuantumCircuitError):
        sv.expectation_pauli("QQ")


# ── Skill facade (the causal wiring into cognition) ─────────────

async def test_quantum_lab_skill_actions():
    from core.skills.quantum_lab import QuantumLabSkill

    skill = QuantumLabSkill()
    bell = await skill.execute({"action": "bell", "seed": 1, "shots": 200}, {})
    assert bell["ok"] and set(bell["counts"]) <= {"00", "11"}
    assert bell["entropy_mode"] == "seeded_prng"

    grover = await skill.execute(
        {"action": "grover", "num_qubits": 5, "marked": 17, "seed": 1}, {}
    )
    assert grover["ok"] and grover["matches_theory"]

    tp = await skill.execute(
        {"action": "teleport", "alpha_real": 0.6, "beta_real": 0.8, "seed": 2}, {}
    )
    assert tp["ok"] and tp["fidelity"] == pytest.approx(1.0, abs=1e-10)

    qft = await skill.execute({"action": "qft_verify", "num_qubits": 3}, {})
    assert qft["ok"] and qft["verified"]

    circuit = await skill.execute({
        "action": "circuit", "num_qubits": 3, "seed": 3,
        "gates": [["h", 0], ["cx", 0, 1], ["ccx", 0, 1, 2]],
    }, {})
    assert circuit["ok"] and circuit["norm_preserved"]
    assert set(circuit["counts"]) <= {"000", "111"}


def test_quantum_lab_skill_matches_quantum_goals():
    from core.skills.quantum_lab import QuantumLabSkill

    skill = QuantumLabSkill()
    assert skill.match({"objective": "simulate a quantum teleportation protocol"})
    assert skill.match({"objective": "show me Grover search"})
    assert not skill.match({"objective": "what time is it"})


def test_quantum_lab_is_discovered_by_the_skill_catalog():
    from core.skills.discovery import build_skill_catalog

    catalog = build_skill_catalog(try_rust=False)
    names = {declaration.name for declaration in catalog.accepted}
    assert "quantum_lab" in names
    excluded = {declaration.name for declaration in catalog.excluded}
    assert "quantum_lab" not in excluded
