from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .aura_now import AuraNow


@dataclass(frozen=True)
class HigherOrderObservation:
    observed_state: str
    confidence: float
    source: tuple[str, ...]
    reportable: bool
    risk: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HigherOrderMonitor:
    """Observes first-order AuraNow states through a limited reportable bottleneck."""

    def observe(self, now: AuraNow) -> tuple[HigherOrderObservation, ...]:
        observations: list[HigherOrderObservation] = []
        if now.affect.distress > 0.55:
            observations.append(
                HigherOrderObservation(
                    "distress_control_state",
                    min(0.95, now.affect.distress),
                    ("affect", "interoception", "prediction"),
                    True,
                    "avoid_dramatic_overclaim",
                )
            )
        if now.prediction.free_energy > 0.35:
            observations.append(
                HigherOrderObservation(
                    "uncertainty_about_next_state",
                    min(0.95, now.prediction.free_energy + 0.25),
                    ("prediction", "workspace"),
                    True,
                    "state_uncertainty",
                )
            )
        if now.ownership.attribution != "self_authored":
            observations.append(
                HigherOrderObservation(
                    "ownership_mismatch",
                    max(0.55, 1.0 - now.ownership.predicted_action_match),
                    ("ownership", "tool_result"),
                    True,
                    "do_not_claim_full_agency",
                )
            )
        if not observations:
            observations.append(
                HigherOrderObservation(
                    "stable_low_salience_state",
                    0.72,
                    ("affect", "workspace", "prediction"),
                    True,
                    "",
                )
            )
        return tuple(obs for obs in observations if obs.reportable)
