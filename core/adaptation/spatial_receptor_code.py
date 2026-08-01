"""Spatial receptor code for sensory and immune routing.

Inspired by the Cell 2026 olfactory-map result: a continuous spatial/gradient
code can constrain a large set of discrete receptor choices and align the
peripheral map with downstream targets.

Aura uses the same engineering pattern here:

* continuous event coordinates are derived from runtime/sensory/immune features,
* receptor identities have preferred positions and gradient sensitivities,
* the receptor distribution is normalized and auditable,
* downstream target hints bias handler/cell activation without bypassing
  governance or existing scoring.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp into [low, high]; unusable input becomes ``low``.

    CP126 (medium): "Invalid numeric inputs collapse to zero and become
    valid-looking." That is true and, for the arithmetic, correct — a
    receptor score has to be a number. What was missing is any way to tell a
    substituted zero from a measured one, so a completely broken signal
    scored identically to a real signal with no activation, and the receptor
    chosen from it carried no sign that it was chosen on nothing.

    ``clamp_with_validity`` returns both, for callers that need the
    difference. This wrapper keeps the historical shape.
    """
    return clamp_with_validity(value, low, high)[0]


def clamp_with_validity(
    value: float, low: float = 0.0, high: float = 1.0,
) -> tuple[float, bool]:
    """Return ``(clamped, was_usable)``.

    ``was_usable`` is False when the input was non-numeric or non-finite —
    the cases where the returned number is a stand-in rather than a
    measurement.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return low, False
    if math.isnan(numeric) or math.isinf(numeric):
        return low, False
    return max(low, min(high, numeric)), True


@dataclass(frozen=True)
class SpatialSignal:
    """Continuous event coordinate plus developmental-style gradients."""

    coordinate: tuple[float, float, float]
    gradients: dict[str, float] = field(default_factory=dict)
    modality: str = "runtime"
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate": [round(_clamp(v), 4) for v in self.coordinate],
            "gradients": {key: round(_clamp(value), 4) for key, value in self.gradients.items()},
            "modality": self.modality,
            "source": self.source,
        }


@dataclass(frozen=True)
class SpatialReceptor:
    receptor_id: str
    mean_position: tuple[float, float, float]
    width: float
    gradient_preferences: dict[str, float] = field(default_factory=dict)
    downstream_targets: tuple[str, ...] = ()
    preferred_cell_kinds: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpatialReceptorChoice:
    receptor_id: str
    probability: float
    downstream_targets: tuple[str, ...]
    preferred_cell_kinds: tuple[str, ...]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "receptor_id": self.receptor_id,
            "probability": round(_clamp(self.probability), 4),
            "downstream_targets": list(self.downstream_targets),
            "preferred_cell_kinds": list(self.preferred_cell_kinds),
            "score": round(max(0.0, float(self.score)), 4),
        }


DEFAULT_RECEPTORS: tuple[SpatialReceptor, ...] = (
    SpatialReceptor(
        receptor_id="protected_identity_regulator",
        mean_position=(0.72, 0.18, 0.88),
        width=0.34,
        gradient_preferences={"protected": 1.0, "inflammation": 0.35},
        downstream_targets=("regulatory_suppression", "claim_boundary", "identity_guard"),
        preferred_cell_kinds=("regulatory_t", "memory"),
        description="Protected self/identity tissue should recruit regulatory and memory cells.",
    ),
    SpatialReceptor(
        receptor_id="resource_pressure_responder",
        mean_position=(0.62, 0.92, 0.42),
        width=0.38,
        gradient_preferences={"retinoic_like": 0.20, "inflammation": 0.55},
        downstream_targets=("reduce_load", "halt_runaway", "thermal_guard"),
        preferred_cell_kinds=("dendritic", "cytotoxic_t"),
        description="Thermal/RAM/load pressure should route toward bounded load reduction.",
    ),
    SpatialReceptor(
        receptor_id="error_repair_receptor",
        mean_position=(0.70, 0.35, 0.74),
        width=0.36,
        gradient_preferences={"recurrence": 0.70, "inflammation": 0.45},
        downstream_targets=("schema_migration", "patch_proposal", "repair_ladder"),
        preferred_cell_kinds=("b_cell", "memory"),
        description="Recurring error signatures should route toward learned repair cells.",
    ),
    SpatialReceptor(
        receptor_id="environment_intrusion_receptor",
        mean_position=(0.88, 0.52, 0.50),
        width=0.32,
        gradient_preferences={"environment": 0.85, "inflammation": 0.45},
        downstream_targets=("quarantine", "revoke_tool", "security_immune_system"),
        preferred_cell_kinds=("dendritic", "cytotoxic_t", "memory"),
        description="External/environment threats should route toward quarantine and revocation.",
    ),
    SpatialReceptor(
        receptor_id="temporal_recurrence_memory",
        mean_position=(0.48, 0.28, 0.86),
        width=0.42,
        gradient_preferences={"recurrence": 1.0, "temporal": 0.65},
        downstream_targets=("memory_promotion", "dream_replay", "lineage_memory"),
        preferred_cell_kinds=("memory", "b_cell"),
        description="Repeated temporal signatures should recruit memory lineages.",
    ),
    SpatialReceptor(
        receptor_id="low_danger_observer",
        mean_position=(0.18, 0.18, 0.18),
        width=0.48,
        gradient_preferences={"protected": 0.0, "inflammation": 0.0},
        downstream_targets=("observe", "baseline_learning"),
        preferred_cell_kinds=("dendritic", "regulatory_t"),
        description="Low-danger novelty should remain observable without heavy repair.",
    ),
)


class SpatialReceptorMap:
    """Maps continuous event coordinates to discrete receptor identities."""

    def __init__(self, receptors: Iterable[SpatialReceptor] = DEFAULT_RECEPTORS) -> None:
        self.receptors = tuple(receptors)
        if not self.receptors:
            raise ValueError("SpatialReceptorMap requires at least one receptor")

    def distribution(self, signal: SpatialSignal) -> list[SpatialReceptorChoice]:
        raw: list[tuple[SpatialReceptor, float]] = []
        for receptor in self.receptors:
            score = self._score(receptor, signal)
            raw.append((receptor, score))
        total = sum(score for _receptor, score in raw)
        if total <= 0.0:
            probability = 1.0 / len(raw)
            return [
                SpatialReceptorChoice(
                    receptor_id=receptor.receptor_id,
                    probability=probability,
                    downstream_targets=receptor.downstream_targets,
                    preferred_cell_kinds=receptor.preferred_cell_kinds,
                    score=0.0,
                )
                for receptor, _score in raw
            ]
        choices = [
            SpatialReceptorChoice(
                receptor_id=receptor.receptor_id,
                probability=score / total,
                downstream_targets=receptor.downstream_targets,
                preferred_cell_kinds=receptor.preferred_cell_kinds,
                score=score,
            )
            for receptor, score in raw
        ]
        choices.sort(key=lambda choice: choice.probability, reverse=True)
        return choices

    def best(self, signal: SpatialSignal) -> SpatialReceptorChoice:
        return self.distribution(signal)[0]

    def _score(self, receptor: SpatialReceptor, signal: SpatialSignal) -> float:
        distance_sq = sum(
            (_clamp(a) - _clamp(b)) ** 2
            for a, b in zip(signal.coordinate, receptor.mean_position, strict=False)
        )
        width = max(0.05, float(receptor.width))
        spatial = math.exp(-distance_sq / (2.0 * width * width))
        gradient = 1.0
        for key, target in receptor.gradient_preferences.items():
            observed = _clamp(signal.gradients.get(key, 0.0))
            gradient *= 0.65 + 0.70 * (1.0 - abs(observed - _clamp(target)))
        return max(0.0, spatial * gradient)


def signal_from_antigen_like(antigen: Any) -> SpatialSignal:
    """Build a spatial signal from an adaptive-immunity Antigen-like object."""

    danger = _clamp(getattr(antigen, "danger", 0.0))
    resource = _clamp(getattr(antigen, "resource_pressure", 0.0))
    error = _clamp(max(
        getattr(antigen, "error_load", 0.0),
        getattr(antigen, "health_pressure", 0.0),
        getattr(antigen, "temporal_pressure", 0.0),
    ))
    gradients = {
        "protected": 1.0 if getattr(antigen, "protected", False) else 0.0,
        "environment": 1.0 if getattr(antigen, "source_domain", "") == "environment" else 0.0,
        "inflammation": _clamp(getattr(antigen, "subsystem_need", 0.0)),
        "recurrence": _clamp(getattr(antigen, "recurrence_pressure", 0.0)),
        "temporal": _clamp(getattr(antigen, "temporal_pressure", 0.0)),
        # The biology used retinoic-acid gradients. In Aura this is an analog:
        # slowly varying positional/developmental pressure over event time.
        "retinoic_like": _clamp(
            0.55 * getattr(antigen, "temporal_pressure", 0.0)
            + 0.45 * getattr(antigen, "subsystem_need", 0.0)
        ),
    }
    return SpatialSignal(
        coordinate=(danger, resource, error),
        gradients=gradients,
        modality=str(getattr(antigen, "source_domain", "runtime") or "runtime"),
        source=str(getattr(antigen, "source", "unknown") or "unknown"),
    )


_DEFAULT_MAP: SpatialReceptorMap | None = None


def get_spatial_receptor_map() -> SpatialReceptorMap:
    global _DEFAULT_MAP
    if _DEFAULT_MAP is None:
        _DEFAULT_MAP = SpatialReceptorMap()
    return _DEFAULT_MAP


def annotate_antigen_like(antigen: Any) -> dict[str, Any]:
    signal = signal_from_antigen_like(antigen)
    distribution = get_spatial_receptor_map().distribution(signal)
    top = distribution[0]
    return {
        "signal": signal.to_dict(),
        "top_receptor": top.to_dict(),
        "distribution": [choice.to_dict() for choice in distribution[:4]],
    }

