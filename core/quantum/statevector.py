"""core/quantum/statevector.py
─────────────────────────────
Exact statevector simulation of quantum circuits.

Design:
- Amplitudes live in a dense complex128 vector of length 2**n.
- Gates are applied by vectorized index arithmetic (no per-amplitude
  Python loops): for a single-qubit unitary on qubit t, the basis
  splits into (bit=0, bit=1) partner pairs and the 2×2 matrix acts on
  each pair simultaneously. Controls restrict the pair set to indices
  whose control bits are all 1 — this one primitive implements every
  controlled gate (CNOT, CZ, Toffoli, controlled-phase for QFT, …).
- Qubit 0 is the most significant bit of the basis label, matching the
  textbook |q0 q1 … q(n-1)⟩ convention.
- Measurement uses the Born rule. The randomness source is injectable:
  a seeded PRNG for reproducibility, or the live QuantumEntropyBridge
  for physically quantum collapse outcomes.

Bounds: MAX_QUBITS caps memory at 2**20 amplitudes (16 MiB per state).
This is an honest exact simulator, not a claim of scalable QC.
"""
from __future__ import annotations

import cmath
import math
from typing import Callable, Sequence

import numpy as np

MAX_QUBITS = 20

_SQRT2_INV = 1.0 / math.sqrt(2.0)

# Canonical single-qubit unitaries.
_GATES_1Q: dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "H": np.array([[_SQRT2_INV, _SQRT2_INV], [_SQRT2_INV, -_SQRT2_INV]], dtype=np.complex128),
    "S": np.array([[1, 0], [0, 1j]], dtype=np.complex128),
    "SDG": np.array([[1, 0], [0, -1j]], dtype=np.complex128),
    "T": np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=np.complex128),
    "TDG": np.array([[1, 0], [0, cmath.exp(-1j * math.pi / 4)]], dtype=np.complex128),
}


class QuantumCircuitError(ValueError):
    """Invalid circuit construction or execution request."""


def rx_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def ry_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz_matrix(theta: float) -> np.ndarray:
    return np.array(
        [[cmath.exp(-1j * theta / 2.0), 0], [0, cmath.exp(1j * theta / 2.0)]],
        dtype=np.complex128,
    )


def phase_matrix(theta: float) -> np.ndarray:
    return np.array([[1, 0], [0, cmath.exp(1j * theta)]], dtype=np.complex128)


class Statevector:
    """A pure quantum state of ``num_qubits`` qubits with gate application,
    Born-rule measurement, sampling, and Pauli expectation values."""

    def __init__(
        self,
        num_qubits: int,
        *,
        seed: int | None = None,
        entropy_source: Callable[[], float] | None = None,
    ):
        if not isinstance(num_qubits, int) or num_qubits < 1:
            raise QuantumCircuitError("num_qubits must be a positive integer")
        if num_qubits > MAX_QUBITS:
            raise QuantumCircuitError(
                f"{num_qubits} qubits exceeds the exact-simulation cap of {MAX_QUBITS}"
            )
        self.num_qubits = num_qubits
        self.state = np.zeros(1 << num_qubits, dtype=np.complex128)
        self.state[0] = 1.0
        self._rng = np.random.default_rng(seed)
        self._entropy_source = entropy_source
        self._gate_count = 0

    # ── randomness ─────────────────────────────────────────────

    def _random_unit(self) -> float:
        """Collapse randomness: injected entropy source first (may be the
        live quantum entropy bridge), seeded PRNG otherwise."""
        if self._entropy_source is not None:
            try:
                value = float(self._entropy_source())
                if 0.0 <= value < 1.0:
                    return value
            except (TypeError, ValueError, RuntimeError, OSError):
                pass  # fall through to the deterministic generator
        return float(self._rng.random())

    # ── gate primitives ────────────────────────────────────────

    def _bit(self, qubit: int) -> int:
        if not 0 <= qubit < self.num_qubits:
            raise QuantumCircuitError(
                f"qubit {qubit} out of range for {self.num_qubits}-qubit register"
            )
        return 1 << (self.num_qubits - 1 - qubit)

    @staticmethod
    def _transform_vector(
        vector: np.ndarray, matrix: np.ndarray, target_bit: int, control_mask: int
    ) -> None:
        """Apply a 2×2 unitary in place to ``vector`` on the target bit,
        restricted to basis states whose control bits are all 1."""
        indices = np.arange(vector.size, dtype=np.intp)
        lower = indices[
            ((indices & target_bit) == 0)
            & ((indices & control_mask) == control_mask)
        ]
        upper = lower | target_bit
        a, b = vector[lower], vector[upper]
        vector[lower] = matrix[0, 0] * a + matrix[0, 1] * b
        vector[upper] = matrix[1, 0] * a + matrix[1, 1] * b

    def apply_unitary(
        self,
        matrix: np.ndarray,
        target: int,
        controls: Sequence[int] = (),
    ) -> "Statevector":
        """Apply a 2×2 unitary to ``target``, optionally conditioned on
        every qubit in ``controls`` being |1⟩."""
        matrix = np.asarray(matrix, dtype=np.complex128)
        if matrix.shape != (2, 2):
            raise QuantumCircuitError("apply_unitary expects a 2x2 matrix")
        target_bit = self._bit(target)
        control_mask = 0
        for control in controls:
            if control == target:
                raise QuantumCircuitError("control and target qubits must differ")
            control_mask |= self._bit(control)
        self._transform_vector(self.state, matrix, target_bit, control_mask)
        self._gate_count += 1
        return self

    # ── named gates (chainable) ────────────────────────────────

    def h(self, q: int) -> "Statevector":
        return self.apply_unitary(_GATES_1Q["H"], q)

    def x(self, q: int) -> "Statevector":
        return self.apply_unitary(_GATES_1Q["X"], q)

    def y(self, q: int) -> "Statevector":
        return self.apply_unitary(_GATES_1Q["Y"], q)

    def z(self, q: int) -> "Statevector":
        return self.apply_unitary(_GATES_1Q["Z"], q)

    def s(self, q: int) -> "Statevector":
        return self.apply_unitary(_GATES_1Q["S"], q)

    def sdg(self, q: int) -> "Statevector":
        return self.apply_unitary(_GATES_1Q["SDG"], q)

    def t(self, q: int) -> "Statevector":
        return self.apply_unitary(_GATES_1Q["T"], q)

    def tdg(self, q: int) -> "Statevector":
        return self.apply_unitary(_GATES_1Q["TDG"], q)

    def rx(self, theta: float, q: int) -> "Statevector":
        return self.apply_unitary(rx_matrix(theta), q)

    def ry(self, theta: float, q: int) -> "Statevector":
        return self.apply_unitary(ry_matrix(theta), q)

    def rz(self, theta: float, q: int) -> "Statevector":
        return self.apply_unitary(rz_matrix(theta), q)

    def phase(self, theta: float, q: int) -> "Statevector":
        return self.apply_unitary(phase_matrix(theta), q)

    def cx(self, control: int, target: int) -> "Statevector":
        return self.apply_unitary(_GATES_1Q["X"], target, controls=(control,))

    def cz(self, control: int, target: int) -> "Statevector":
        return self.apply_unitary(_GATES_1Q["Z"], target, controls=(control,))

    def cphase(self, theta: float, control: int, target: int) -> "Statevector":
        return self.apply_unitary(phase_matrix(theta), target, controls=(control,))

    def ccx(self, c1: int, c2: int, target: int) -> "Statevector":
        return self.apply_unitary(_GATES_1Q["X"], target, controls=(c1, c2))

    def swap(self, q1: int, q2: int) -> "Statevector":
        self.cx(q1, q2)
        self.cx(q2, q1)
        self.cx(q1, q2)
        return self

    # ── readout ────────────────────────────────────────────────

    def probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2

    def probability_of(self, bitstring: str) -> float:
        cleaned = bitstring.replace(" ", "")
        if len(cleaned) != self.num_qubits or set(cleaned) - {"0", "1"}:
            raise QuantumCircuitError(
                f"bitstring must be {self.num_qubits} chars of 0/1"
            )
        return float(self.probabilities()[int(cleaned, 2)])

    def measure(self, qubit: int) -> int:
        """Born-rule projective measurement of one qubit; collapses state."""
        bit = self._bit(qubit)
        indices = np.arange(self.state.size, dtype=np.intp)
        ones = (indices & bit) != 0
        p_one = float(np.sum(np.abs(self.state[ones]) ** 2))
        outcome = 1 if self._random_unit() < p_one else 0
        keep = ones if outcome == 1 else ~ones
        self.state[~keep] = 0.0
        norm = math.sqrt(float(np.sum(np.abs(self.state) ** 2)))
        if norm <= 0.0:
            raise QuantumCircuitError("measurement collapsed to zero norm")
        self.state /= norm
        return outcome

    def measure_all(self) -> str:
        return "".join(str(self.measure(q)) for q in range(self.num_qubits))

    def sample_counts(self, shots: int) -> dict[str, int]:
        """Sample bitstrings from the current distribution without collapse."""
        if shots < 1:
            raise QuantumCircuitError("shots must be >= 1")
        probs = self.probabilities()
        probs = probs / probs.sum()
        draws = self._rng.multinomial(shots, probs)
        width = self.num_qubits
        return {
            format(i, f"0{width}b"): int(count)
            for i, count in enumerate(draws)
            if count
        }

    def expectation_pauli(self, pauli: str) -> float:
        """⟨ψ|P|ψ⟩ for a Pauli string like "ZZI" (one letter per qubit)."""
        cleaned = pauli.replace(" ", "").upper()
        if len(cleaned) != self.num_qubits or set(cleaned) - {"I", "X", "Y", "Z"}:
            raise QuantumCircuitError(
                f"Pauli string must be {self.num_qubits} chars of I/X/Y/Z"
            )
        transformed = self.state.copy()
        for q, letter in enumerate(cleaned):
            if letter == "I":
                continue
            self._transform_vector(transformed, _GATES_1Q[letter], self._bit(q), 0)
        return float(np.real(np.vdot(self.state, transformed)))

    @property
    def gate_count(self) -> int:
        return self._gate_count

    def fidelity(self, other: "Statevector | np.ndarray") -> float:
        other_state = other.state if isinstance(other, Statevector) else np.asarray(other)
        return float(np.abs(np.vdot(self.state, other_state)) ** 2)
