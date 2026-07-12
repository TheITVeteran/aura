"""core/quantum/algorithms.py
───────────────────────────
Canonical quantum algorithms with analytically verifiable outcomes.

Each function returns both the result and the analytic expectation so
callers (tests, the quantum_lab skill, the discovery engine) can check
the simulation against ground truth rather than trusting it.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from core.quantum.statevector import (
    QuantumCircuitError,
    Statevector,
)


def bell_pair(
    *, seed: int | None = None, entropy_source: Callable[[], float] | None = None
) -> Statevector:
    """(|00⟩ + |11⟩)/√2 — maximal two-qubit entanglement."""
    return Statevector(2, seed=seed, entropy_source=entropy_source).h(0).cx(0, 1)


def ghz_state(
    num_qubits: int,
    *,
    seed: int | None = None,
    entropy_source: Callable[[], float] | None = None,
) -> Statevector:
    """(|0…0⟩ + |1…1⟩)/√2 on ``num_qubits`` qubits."""
    if num_qubits < 2:
        raise QuantumCircuitError("GHZ needs at least 2 qubits")
    sv = Statevector(num_qubits, seed=seed, entropy_source=entropy_source).h(0)
    for q in range(1, num_qubits):
        sv.cx(0, q)
    return sv


def teleport(
    alpha: complex,
    beta: complex,
    *,
    seed: int | None = None,
    entropy_source: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Teleport the qubit state α|0⟩+β|1⟩ from qubit 0 to qubit 2.

    Returns the classical correction bits and the fidelity between the
    received qubit and the input state (analytically always 1.0)."""
    components = (alpha.real, alpha.imag, beta.real, beta.imag)
    if not all(math.isfinite(value) for value in components):
        raise QuantumCircuitError("input state amplitudes must be finite")
    norm = math.sqrt(abs(alpha) ** 2 + abs(beta) ** 2)
    if norm <= 0.0:
        raise QuantumCircuitError("input state must have positive norm")
    alpha, beta = alpha / norm, beta / norm

    sv = Statevector(3, seed=seed, entropy_source=entropy_source)
    # Prepare the payload on qubit 0: α|0⟩+β|1⟩ via a manual amplitude load
    # (equivalent to RY/RZ prep; exact loading keeps the check analytic).
    sv.state[:] = 0.0
    sv.state[0b000] = alpha
    sv.state[0b100] = beta

    # Entangle qubits 1 and 2 as the shared Bell channel.
    sv.h(1).cx(1, 2)
    # Bell measurement of qubits 0 and 1.
    sv.cx(0, 1).h(0)
    m0 = sv.measure(0)
    m1 = sv.measure(1)
    # Classical corrections on the receiving qubit.
    if m1:
        sv.x(2)
    if m0:
        sv.z(2)

    # Extract the received single-qubit state (qubits 0,1 are collapsed
    # to |m0 m1⟩, so the remaining amplitudes factor exactly).
    base = (m0 << 2) | (m1 << 1)
    received = np.array([sv.state[base], sv.state[base | 1]], dtype=np.complex128)
    expected = np.array([alpha, beta], dtype=np.complex128)
    fidelity = float(np.abs(np.vdot(expected, received)) ** 2)
    return {"m0": m0, "m1": m1, "fidelity": fidelity}


def grover_search(
    num_qubits: int,
    marked: int,
    *,
    seed: int | None = None,
    entropy_source: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Grover's search for one marked item in 2**num_qubits entries.

    Runs the optimal ⌊π/4·√N⌋ iterations and reports the success
    probability alongside the analytic prediction sin²((2k+1)θ)."""
    if not 1 <= num_qubits <= 12:
        raise QuantumCircuitError("grover_search supports 1..12 qubits")
    size = 1 << num_qubits
    if not 0 <= marked < size:
        raise QuantumCircuitError("marked index out of range")

    sv = Statevector(num_qubits, seed=seed, entropy_source=entropy_source)
    for q in range(num_qubits):
        sv.h(q)

    theta = math.asin(1.0 / math.sqrt(size))
    iterations = max(1, int(math.floor(math.pi / (4.0 * theta)))) if size > 2 else 1

    for _ in range(iterations):
        # Oracle: phase-flip the marked basis state.
        sv.state[marked] = -sv.state[marked]
        # Diffusion: reflect about the uniform superposition.
        mean = np.mean(sv.state)
        sv.state[:] = 2.0 * mean - sv.state

    success = float(np.abs(sv.state[marked]) ** 2)
    predicted = math.sin((2 * iterations + 1) * theta) ** 2
    return {
        "iterations": iterations,
        "success_probability": success,
        "analytic_prediction": predicted,
        "num_candidates": size,
        "statevector": sv,
    }


def qft_circuit(sv: Statevector) -> Statevector:
    """In-place quantum Fourier transform over the full register.

    Matches the unitary F[j,k] = exp(2πi·jk/N)/√N (with a final qubit-
    order reversal via SWAPs, per the standard circuit)."""
    n = sv.num_qubits
    for q in range(n):
        sv.h(q)
        for offset in range(1, n - q):
            sv.cphase(math.pi / (1 << offset), q + offset, q)
    for q in range(n // 2):
        sv.swap(q, n - 1 - q)
    return sv


def qft_matrix(num_qubits: int) -> NDArray[np.complex128]:
    """The analytic DFT matrix the circuit must reproduce."""
    size = 1 << num_qubits
    grids = cast(
        list[NDArray[np.int64]],
        np.meshgrid(np.arange(size), np.arange(size), indexing="ij"),
    )
    j: NDArray[np.int64] = grids[0]
    k: NDArray[np.int64] = grids[1]
    result: NDArray[np.complex128] = np.exp(2j * np.pi * j * k / size) / math.sqrt(size)
    return result
