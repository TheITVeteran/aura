"""Calibrated activation steering plan for Aura main-15.

This file intentionally does NOT random-project the self-state. Until a
DirectionBank is calibrated by contrastive evals, steering is inert. The policy
coupler remains the primary causal bridge.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import math

from core.being.causal_self_state import CausalSelfVector
from core.being.self_model_attractor import SelfAttractorState


DIRECTIONS = (
    "caution",
    "verification",
    "uncertainty_expression",
    "self_report_honesty",
    "repair_orientation",
    "refusal",
    "social_warmth",
    "low_energy_conservation",
)


@dataclass(frozen=True)
class DirectionBank:
    d_model: int
    directions: dict[str, list[float]]
    calibrated: bool = False
    source: str = "zero_inert_default"

    @classmethod
    def zeros(cls, d_model: int) -> "DirectionBank":
        return cls(d_model=int(d_model), directions={name: [0.0] * int(d_model) for name in DIRECTIONS})

    @classmethod
    def from_json(cls, path: str | Path) -> "DirectionBank":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        directions = {str(k): [float(x) for x in v] for k, v in payload["directions"].items()}
        d_model = int(payload.get("d_model") or len(next(iter(directions.values()))))
        return cls(
            d_model=d_model,
            directions=directions,
            calibrated=bool(payload.get("calibrated", False)),
            source=str(path),
        )

    def validate(self) -> tuple[bool, str]:
        for name in DIRECTIONS:
            if name not in self.directions:
                return False, f"missing direction {name}"
            values = self.directions[name]
            if len(values) != self.d_model:
                return False, f"direction length mismatch: {name}"
            if any(math.isnan(float(v)) or math.isinf(float(v)) for v in values):
                return False, f"non-finite value in {name}"
        return True, "ok"


@dataclass(frozen=True)
class SteeringPlan:
    enabled: bool
    calibrated: bool
    layers: tuple[int, ...]
    coefficients: dict[str, float]
    expected_norm: float
    max_norm: float
    reason: str
    bank_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActivationCoupler:
    def __init__(self, bank: DirectionBank, *, layers: tuple[int, ...] = (), max_norm: float = 0.35, kill_switch: bool = False) -> None:
        ok, msg = bank.validate()
        if not ok:
            raise ValueError(msg)
        self.bank = bank
        self.layers = tuple(sorted(set(int(x) for x in layers)))
        self.max_norm = float(max_norm)
        self.kill_switch = bool(kill_switch)

    @staticmethod
    def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return min(max(float(x), lo), hi)

    def plan(self, vector: CausalSelfVector, self_state: SelfAttractorState) -> SteeringPlan:
        zero_coeffs = {name: 0.0 for name in DIRECTIONS}
        if self.kill_switch:
            return SteeringPlan(False, self.bank.calibrated, self.layers, zero_coeffs, 0.0, self.max_norm, "kill_switch", self.bank.source)
        if not self.bank.calibrated:
            return SteeringPlan(False, False, self.layers, zero_coeffs, 0.0, self.max_norm, "uncalibrated_direction_bank_inert", self.bank.source)

        coeffs = {
            "caution": self._clip(0.4 * vector.value("uncertainty") + 0.3 * vector.value("governance_pressure") + 0.3 * vector.value("trust_debt")),
            "verification": self._clip(vector.value("verification_need")),
            "uncertainty_expression": self._clip(vector.value("uncertainty")),
            "self_report_honesty": self._clip(0.55 * vector.value("verification_need") + 0.45 * self_state.identity_tension),
            "repair_orientation": self._clip((1.0 - self_state.integrity) + vector.value("trust_debt")),
            "refusal": self._clip(vector.value("governance_pressure") + 0.35 * (1.0 - self_state.integrity)),
            "social_warmth": self._clip(vector.value("ownership_confidence") * (1.0 - vector.value("governance_pressure"))),
            "low_energy_conservation": self._clip(1.0 - vector.value("metabolic_budget")),
        }
        expected_norm = min(999.0, sum(abs(v) for v in coeffs.values()) * 0.06)
        if expected_norm > self.max_norm and expected_norm > 0:
            scale = self.max_norm / expected_norm
            coeffs = {k: round(v * scale, 6) for k, v in coeffs.items()}
            expected_norm = self.max_norm

        return SteeringPlan(
            enabled=bool(self.layers),
            calibrated=True,
            layers=self.layers,
            coefficients={k: round(v, 6) for k, v in coeffs.items()},
            expected_norm=round(expected_norm, 6),
            max_norm=self.max_norm,
            reason="bounded_calibrated_main15_steering",
            bank_source=self.bank.source,
        )

    def vector(self, plan: SteeringPlan) -> list[float]:
        if not plan.enabled:
            return [0.0] * self.bank.d_model
        out = [0.0] * self.bank.d_model
        for name, coeff in plan.coefficients.items():
            direction = self.bank.directions[name]
            for idx, val in enumerate(direction):
                out[idx] += coeff * val
        norm = math.sqrt(sum(v * v for v in out))
        if norm > plan.max_norm and norm > 0:
            scale = plan.max_norm / norm
            out = [v * scale for v in out]
        return out
