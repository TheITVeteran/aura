"""core/quantum — Aura's quantum computational module.

An honest statevector simulator: it executes real quantum circuits
(unitary evolution + Born-rule measurement) on classical hardware.
It is *simulation of quantum computation*, not quantum hardware —
capped at a qubit count where exact simulation is tractable.

Measurement collapse can draw its randomness from the live
QuantumEntropyBridge (physically quantum entropy), giving Aura
measurements whose outcomes are genuinely non-deterministic while the
unitary dynamics stay exactly reproducible.
"""
from core.quantum.statevector import (
    MAX_QUBITS,
    QuantumCircuitError,
    Statevector,
)
from core.quantum.algorithms import (
    bell_pair,
    ghz_state,
    grover_search,
    qft_circuit,
    teleport,
)

__all__ = [
    "MAX_QUBITS",
    "QuantumCircuitError",
    "Statevector",
    "bell_pair",
    "ghz_state",
    "grover_search",
    "qft_circuit",
    "teleport",
]
