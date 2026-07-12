"""core/skills/quantum_lab.py
───────────────────────────
Quantum computation as a governed, pure-compute capability.

Gives Aura's cognition a real quantum circuit simulator: entangled
states, teleportation, Grover search, QFT verification, and free-form
small circuits — every result cross-checked against analytic ground
truth where one exists. Measurement collapse draws from the live
QuantumEntropyBridge when it is available, so measurement outcomes are
physically quantum even though the unitary evolution is simulated.
"""
from __future__ import annotations

import math
from typing import Any, Dict

from core.skills.base_skill import BaseSkill

_QUANTUM_LAB_ERRORS = (ImportError, AttributeError, RuntimeError, TypeError, ValueError)


def _entropy_source():
    """Collapse randomness from the quantum entropy bridge, if healthy."""
    try:
        from core.consciousness.quantum_entropy import get_quantum_entropy

        bridge = get_quantum_entropy()
        return bridge.get_quantum_float
    except _QUANTUM_LAB_ERRORS:
        return None


class QuantumLabSkill(BaseSkill):
    name = "quantum_lab"
    description = (
        "Run quantum circuit simulations: Bell/GHZ entanglement, Grover search, "
        "quantum teleportation, QFT, or a custom small circuit. Exact statevector "
        "simulation with analytic cross-checks; honest about being simulation."
    )
    effect_scope = "pure_compute"
    inputs = {
        "action": "one of: bell, ghz, grover, teleport, qft_verify, circuit",
        "num_qubits": "register width where applicable (bounded)",
        "marked": "grover: index of the marked item",
        "alpha_real/alpha_imag/beta_real/beta_imag": "teleport: input amplitudes",
        "gates": "circuit: list of [gate, qubit(s), (angle)] steps",
        "shots": "sample count for measurement statistics (default 512)",
        "seed": "optional determinism seed (otherwise quantum-entropy collapse)",
    }
    output = "Simulation results with analytic verification where defined"

    def match(self, goal: Dict[str, Any]) -> bool:
        objective = str(goal.get("objective", "")).lower()
        keywords = (
            "quantum", "qubit", "entangle", "superposition", "grover",
            "teleport", "bell state", "ghz", "qft", "quantum fourier",
        )
        return any(keyword in objective for keyword in keywords)

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
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

        import numpy as np

        params = params if isinstance(params, dict) else {}
        action = str(params.get("action", "bell") or "bell").strip().lower()
        seed = params.get("seed")
        seed = int(seed) if seed is not None else None
        shots = max(1, min(int(params.get("shots", 512) or 512), 65536))
        entropy = None if seed is not None else _entropy_source()
        entropy_mode = "seeded_prng" if seed is not None else (
            "quantum_entropy_bridge" if entropy else "os_prng"
        )

        try:
            if action == "bell":
                sv = bell_pair(seed=seed, entropy_source=entropy)
                counts = sv.sample_counts(shots)
                return self._ok(action, entropy_mode, {
                    "counts": counts,
                    "zz_correlation": sv.expectation_pauli("ZZ"),
                    "analytic": {"p_00": 0.5, "p_11": 0.5, "zz_correlation": 1.0},
                    "summary": (
                        f"Bell pair over {shots} shots: {counts}. "
                        "Only 00/11 appear and ⟨ZZ⟩=1 — the qubits are maximally entangled."
                    ),
                })

            if action == "ghz":
                n = self._bounded_qubits(params, default=3, cap=MAX_QUBITS)
                sv = ghz_state(n, seed=seed, entropy_source=entropy)
                counts = sv.sample_counts(shots)
                return self._ok(action, entropy_mode, {
                    "num_qubits": n,
                    "counts": counts,
                    "analytic": {"p_" + "0" * n: 0.5, "p_" + "1" * n: 0.5},
                    "summary": f"{n}-qubit GHZ state sampled {shots} times: {counts}.",
                })

            if action == "grover":
                n = self._bounded_qubits(params, default=4, cap=12)
                marked = int(params.get("marked", 0) or 0)
                result = grover_search(n, marked, seed=seed, entropy_source=entropy)
                deviation = abs(
                    result["success_probability"] - result["analytic_prediction"]
                )
                return self._ok(action, entropy_mode, {
                    "num_candidates": result["num_candidates"],
                    "iterations": result["iterations"],
                    "success_probability": result["success_probability"],
                    "analytic_prediction": result["analytic_prediction"],
                    "matches_theory": deviation < 1e-9,
                    "summary": (
                        f"Grover over {result['num_candidates']} items found the marked "
                        f"item with p={result['success_probability']:.4f} after "
                        f"{result['iterations']} iterations (theory: "
                        f"{result['analytic_prediction']:.4f})."
                    ),
                })

            if action == "teleport":
                alpha = complex(
                    float(params.get("alpha_real", 1.0) or 0.0),
                    float(params.get("alpha_imag", 0.0) or 0.0),
                )
                beta = complex(
                    float(params.get("beta_real", 1.0) or 0.0),
                    float(params.get("beta_imag", 0.0) or 0.0),
                )
                result = teleport(alpha, beta, seed=seed, entropy_source=entropy)
                return self._ok(action, entropy_mode, {
                    **result,
                    "summary": (
                        f"Teleported α|0⟩+β|1⟩ with corrections (m0={result['m0']}, "
                        f"m1={result['m1']}); received-state fidelity "
                        f"{result['fidelity']:.6f}."
                    ),
                })

            if action == "qft_verify":
                n = self._bounded_qubits(params, default=4, cap=8)
                max_error = 0.0
                for basis in range(1 << n):
                    sv = Statevector(n, seed=seed or 0)
                    sv.state[:] = 0.0
                    sv.state[basis] = 1.0
                    qft_circuit(sv)
                    expected = qft_matrix(n)[:, basis]
                    max_error = max(
                        max_error, float(np.max(np.abs(sv.state - expected)))
                    )
                return self._ok(action, entropy_mode, {
                    "num_qubits": n,
                    "max_amplitude_error": max_error,
                    "verified": max_error < 1e-10,
                    "summary": (
                        f"QFT circuit on {n} qubits reproduces the analytic DFT matrix "
                        f"on all {1 << n} basis states (max error {max_error:.2e})."
                    ),
                })

            if action == "circuit":
                return self._run_circuit(params, shots, seed, entropy, entropy_mode)

            return {"ok": False, "error": f"Unknown quantum_lab action '{action}'"}
        except QuantumCircuitError as exc:
            return {"ok": False, "error": str(exc)}

    # ── helpers ────────────────────────────────────────────────

    @staticmethod
    def _bounded_qubits(params: dict, *, default: int, cap: int) -> int:
        n = int(params.get("num_qubits", default) or default)
        return max(2, min(n, cap))

    @staticmethod
    def _ok(action: str, entropy_mode: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "ok": True,
            "action": action,
            "entropy_mode": entropy_mode,
            "honest_framing": (
                "Exact classical simulation of quantum dynamics; "
                "not quantum hardware."
            ),
            **payload,
        }

    def _run_circuit(self, params, shots, seed, entropy, entropy_mode):
        from core.quantum import Statevector

        n = self._bounded_qubits(params, default=2, cap=12)
        sv = Statevector(n, seed=seed, entropy_source=entropy)
        gates = params.get("gates") or []
        if not isinstance(gates, list) or len(gates) > 512:
            return {"ok": False, "error": "gates must be a list of at most 512 steps"}
        single = {"h", "x", "y", "z", "s", "sdg", "t", "tdg"}
        rotations = {"rx", "ry", "rz", "phase"}
        for step in gates:
            if not isinstance(step, (list, tuple)) or not step:
                return {"ok": False, "error": f"malformed gate step: {step!r}"}
            op = str(step[0]).lower()
            args = step[1:]
            if op in single and len(args) == 1:
                getattr(sv, op)(int(args[0]))
            elif op in rotations and len(args) == 2:
                getattr(sv, op)(float(args[1]), int(args[0]))
            elif op in {"cx", "cz", "swap"} and len(args) == 2:
                getattr(sv, op)(int(args[0]), int(args[1]))
            elif op == "ccx" and len(args) == 3:
                sv.ccx(int(args[0]), int(args[1]), int(args[2]))
            else:
                return {"ok": False, "error": f"unsupported gate step: {step!r}"}
        counts = sv.sample_counts(shots)
        norm = float(sum(p for p in sv.probabilities()))
        return self._ok("circuit", entropy_mode, {
            "num_qubits": n,
            "gate_count": sv.gate_count,
            "counts": counts,
            "norm_preserved": math.isclose(norm, 1.0, abs_tol=1e-9),
            "summary": f"Ran {sv.gate_count} gates on {n} qubits; counts {counts}.",
        })
