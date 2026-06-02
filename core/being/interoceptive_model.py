from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .aura_now import BodyState, PredictionState


@dataclass
class InteroceptiveModel:
    """Predicts and compares Aura's internal body/resource state."""

    learning_rate: float = 0.15
    _last_observed: dict[str, float] = field(default_factory=dict)

    def predict_body(self, body: BodyState, candidate_action: str = "") -> dict[str, float]:
        observed = body.pressure_vector()
        if not self._last_observed:
            self._last_observed = dict(observed)
        predicted: dict[str, float] = {}
        action = str(candidate_action or "").lower()
        action_load = 0.12 if any(token in action for token in ("test", "build", "browser", "model", "search")) else 0.03
        for key, value in observed.items():
            previous = float(self._last_observed.get(key, value))
            predicted[key] = max(0.0, min(1.0, previous + (value - previous) * self.learning_rate))
        predicted["cpu_pressure"] = max(0.0, min(1.0, predicted.get("cpu_pressure", 0.0) + action_load))
        predicted["latency_pressure"] = max(0.0, min(1.0, predicted.get("latency_pressure", 0.0) + action_load * 0.7))
        return predicted

    def compare(self, body: BodyState, *, candidate_action: str = "") -> PredictionState:
        observed = body.pressure_vector()
        predicted = self.predict_body(body, candidate_action)
        errors = {
            key: abs(float(observed.get(key, 0.0)) - float(predicted.get(key, 0.0)))
            for key in sorted(set(observed) | set(predicted))
        }
        free_energy = sum(errors.values()) / max(1, len(errors))
        controllability = max(0.0, min(1.0, 1.0 - (body.total_pressure * 0.65) - (free_energy * 0.35)))
        information_gain = max(0.0, min(1.0, free_energy + body.context_pressure * 0.25))
        self._last_observed = dict(observed)
        return PredictionState(
            predicted={key: round(value, 4) for key, value in predicted.items()},
            observed={key: round(value, 4) for key, value in observed.items()},
            errors={key: round(value, 4) for key, value in errors.items()},
            free_energy=round(free_energy, 4),
            controllability=round(controllability, 4),
            expected_information_gain=round(information_gain, 4),
        )

    def perturb(self, **pressures: Any) -> BodyState:
        return BodyState(**{key: float(value) for key, value in pressures.items()})
