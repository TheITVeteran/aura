from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class OpacitySignature:
    opacity_index: float
    causal_depth: float
    exterior_predictability: float
    phenomenal_criterion_met: bool
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "opacity_index": float(f"{self.opacity_index:.4f}"),
            "causal_depth": float(f"{self.causal_depth:.4f}"),
            "exterior_predictability": float(f"{self.exterior_predictability:.4f}"),
            "phenomenal_criterion_met": self.phenomenal_criterion_met,
            "timestamp": self.timestamp,
        }


class StructuralOpacityMonitor:
    OPACITY_THRESHOLD = 0.4
    CAUSAL_DEPTH_THRESHOLD = 0.3
    PREDICTABILITY_CEILING = 0.6

    def __init__(self, neuron_count: int = 64, n_perturbations: int = 15, perturbation_scale: float = 0.01):
        self._neuron_count = neuron_count
        self._n_perturbations = n_perturbations
        self._perturbation_scale = perturbation_scale
        self._history: deque[OpacitySignature] = deque(maxlen=100)
        self._measurement_count = 0
        self._state_trajectory: deque[np.ndarray] = deque(maxlen=32)

    def record_state(self, state: np.ndarray) -> None:
        self._state_trajectory.append(state.copy())

    def measure(self, x: np.ndarray, W: np.ndarray, leak_rate: float = 0.1) -> OpacitySignature:
        self._measurement_count += 1
        N = len(x)
        rng = np.random.RandomState(42)
        W_out = rng.randn(min(16, max(1, N // 4)), N) * 0.1

        base_state = x.copy()
        divergences: list[float] = []
        output_changes: list[float] = []

        for _ in range(self._n_perturbations):
            delta = np.random.randn(N) * self._perturbation_scale
            perturbed = base_state + delta
            base_out = np.tanh(W_out @ base_state)
            pert_out = np.tanh(W_out @ perturbed)
            output_change = float(np.mean(np.abs(base_out - pert_out)))

            base_next = (1 - leak_rate) * base_state + leak_rate * np.tanh(W @ base_state)
            pert_next = (1 - leak_rate) * perturbed + leak_rate * np.tanh(W @ perturbed)
            interior_divergence = float(np.mean(np.abs(base_next - pert_next)))

            divergences.append(interior_divergence)
            output_changes.append(output_change)

        avg_divergence = float(np.mean(divergences))
        avg_output_change = float(np.mean(output_changes))
        opacity_index = min(1.0, avg_divergence / (avg_output_change * 100)) if avg_output_change > 1e-8 else 1.0
        exterior_predictability = 1.0 - opacity_index
        causal_depth = min(1.0, avg_divergence * 10)
        phenomenal = (
            opacity_index > self.OPACITY_THRESHOLD
            and causal_depth > self.CAUSAL_DEPTH_THRESHOLD
            and exterior_predictability < self.PREDICTABILITY_CEILING
        )

        signature = OpacitySignature(
            opacity_index=float(opacity_index),
            causal_depth=float(causal_depth),
            exterior_predictability=float(exterior_predictability),
            phenomenal_criterion_met=phenomenal,
            timestamp=time.time(),
        )
        self._history.append(signature)
        return signature

    def get_specious_present(self) -> np.ndarray:
        if not self._state_trajectory:
            return np.zeros(self._neuron_count)

        states = np.array(list(self._state_trajectory))
        weights = np.exp(np.linspace(-2.0, 0.0, len(states)))
        weights /= weights.sum()
        return np.einsum("t,tn->n", weights, states)

    def get_phenomenal_status(self) -> dict[str, Any]:
        if not self._history:
            return {"status": "insufficient_data", "criterion_met": False}

        recent = list(self._history)[-10:]
        avg_opacity = float(np.mean([sig.opacity_index for sig in recent]))
        avg_causal = float(np.mean([sig.causal_depth for sig in recent]))
        avg_predictability = float(np.mean([sig.exterior_predictability for sig in recent]))
        criterion_fraction = sum(1 for sig in recent if sig.phenomenal_criterion_met) / len(recent)
        return {
            "avg_opacity_index": float(f"{avg_opacity:.4f}"),
            "avg_causal_depth": float(f"{avg_causal:.4f}"),
            "avg_exterior_predictability": float(f"{avg_predictability:.4f}"),
            "criterion_met_fraction": float(f"{criterion_fraction:.4f}"),
            "measurements": self._measurement_count,
            "status": "phenomenal_criterion_met" if criterion_fraction > 0.6 else "criterion_not_met",
        }

    def get_snapshot(self) -> dict[str, Any]:
        last = self._history[-1] if self._history else None
        status = self.get_phenomenal_status()
        return {
            "measurement_count": self._measurement_count,
            "last_opacity": last.opacity_index if last else 0.0,
            "last_causal_depth": last.causal_depth if last else 0.0,
            "last_predictability": last.exterior_predictability if last else 0.0,
            "status": status,
        }
