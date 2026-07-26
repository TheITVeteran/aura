"""L3a — features that know where they came from, and say when they are absent.

Two failure modes are being designed out here, both of which are silent.

**Absence is not zero.** ``coherence = 0.0`` means the binding engine reported
total incoherence. A missing coherence reading means the binding engine was not
answering. A learner given zero for both concludes that a dead subsystem is a
crisis, and acts on it. Every feature therefore travels with a presence bit,
and an absent value contributes nothing but its own absence.

**Aura's code changes underneath her features.** A parallel agent renames a
subsystem, a threshold moves, a signal starts meaning something slightly
different — and rows recorded before the change keep occupying the same slot in
the vector, now lying. Features are declared as a versioned schema whose id is
derived from the names *and* the version tag, so a change invalidates the old
rows rather than corrupting the new model with them. Losing history is cheap.
Learning confidently from history that no longer means what it says is not.

Standardisation is kept here too, as running moments rather than a fitted
scaler: the live path sees each episode once, and a scaler fitted on a frozen
snapshot goes stale exactly as fast as the system it measures.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.Ontogeny.Features")

#: Guard against a feature whose scale explodes (a counter that started
#: counting). Standardised values are clipped to this many deviations.
_CLIP_SIGMA = 6.0


@dataclass(frozen=True)
class FeatureSchema:
    """The declared, ordered, versioned feature set for one control point.

    ``version`` is bumped by hand when a feature's *meaning* changes without
    its name changing — the one case a name-derived hash cannot catch.
    """

    control_point: str
    names: tuple[str, ...]
    version: int = 1
    #: Where each feature comes from, for forensics and for the invalidation
    #: sweep when a subsystem is retired.
    sources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"duplicate feature names in schema for {self.control_point}")

    @property
    def schema_id(self) -> str:
        payload = f"v{self.version}|" + "|".join(sorted(self.names))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @property
    def width(self) -> int:
        """Values plus presence bits."""
        return len(self.names) * 2

    def vector(self, values: Mapping[str, float]) -> FeatureVector:
        return FeatureVector.build(self, values)

    def describe(self) -> dict[str, Any]:
        return {
            "control_point": self.control_point,
            "schema_id": self.schema_id,
            "version": self.version,
            "names": list(self.names),
            "width": self.width,
            "sources": dict(self.sources),
        }


@dataclass(frozen=True)
class FeatureVector:
    """One episode's features: values, presence bits, and the schema they honour."""

    schema_id: str
    names: tuple[str, ...]
    values: np.ndarray
    present: np.ndarray

    @classmethod
    def build(cls, schema: FeatureSchema, values: Mapping[str, float]) -> FeatureVector:
        raw = np.zeros(len(schema.names), dtype=np.float64)
        present = np.zeros(len(schema.names), dtype=np.float64)
        for i, name in enumerate(schema.names):
            value = values.get(name)
            if value is None:
                continue
            try:
                as_float = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(as_float):
                continue
            raw[i] = as_float
            present[i] = 1.0
        return cls(schema_id=schema.schema_id, names=tuple(schema.names), values=raw, present=present)

    def as_mapping(self) -> dict[str, float]:
        """Only the features that were actually observed."""
        return {n: float(v) for n, v, p in zip(self.names, self.values, self.present, strict=True) if p}


class RunningMoments:
    """Welford moments per feature, over observed values only.

    Absent values do not move the mean. A subsystem that goes quiet for an
    hour must not drag every statistic toward zero on its way out.
    """

    def __init__(self, width: int) -> None:
        self.count = np.zeros(width, dtype=np.float64)
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, values: np.ndarray, present: np.ndarray) -> None:
        mask = present > 0
        if not mask.any():
            return
        self.count[mask] += 1.0
        delta = np.zeros_like(values)
        delta[mask] = values[mask] - self.mean[mask]
        self.mean[mask] += delta[mask] / self.count[mask]
        delta2 = np.zeros_like(values)
        delta2[mask] = values[mask] - self.mean[mask]
        self.m2[mask] += delta[mask] * delta2[mask]

    @property
    def std(self) -> np.ndarray:
        variance = np.divide(
            self.m2, np.maximum(self.count - 1.0, 1.0),
            out=np.zeros_like(self.m2), where=self.count > 1,
        )
        return np.sqrt(np.maximum(variance, 1e-12))

    def standardise(self, values: np.ndarray, present: np.ndarray) -> np.ndarray:
        """Centre and scale observed values; absent values become exactly zero.

        Zero after standardisation means "at the mean", which for an absent
        feature is the honest neutral: it contributes nothing to the score,
        and the paired presence bit carries the fact that it was missing.
        """
        scaled = np.zeros_like(values)
        seen = (present > 0) & (self.count > 1)
        scaled[seen] = (values[seen] - self.mean[seen]) / self.std[seen]
        return np.clip(scaled, -_CLIP_SIGMA, _CLIP_SIGMA)

    def state_dict(self) -> dict[str, list[float]]:
        return {
            "count": self.count.tolist(),
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
        }

    def load_state(self, state: Mapping[str, Sequence[float]]) -> None:
        width = len(self.count)
        for key in ("count", "mean", "m2"):
            values = np.asarray(state.get(key, []), dtype=np.float64)
            if values.shape == (width,):
                setattr(self, key, values)


def design_row(
    vector: FeatureVector, moments: RunningMoments, *, update: bool = False
) -> np.ndarray:
    """The row a head actually sees: standardised values then presence bits."""
    if update:
        moments.update(vector.values, vector.present)
    standardised = moments.standardise(vector.values, vector.present)
    return np.concatenate([standardised, vector.present])


def row_names(schema: FeatureSchema) -> tuple[str, ...]:
    """Column names for the design row — used for attribution receipts."""
    return (*schema.names, *(f"{n}:present" for n in schema.names))


# ── Declared schemas ────────────────────────────────────────────────────────
#
# A control point's schema lives next to nothing else: it is a contract
# between the subsystem that emits the features and every head that has ever
# been fitted on them. Adding a name changes the id, which retires the old
# corpus for training and keeps it for forensics.

EXECUTIVE_ADMISSION = FeatureSchema(
    control_point="executive.admission",
    version=1,
    names=(
        "priority",
        "confidence",
        "coherence",
        "failure_pressure",
        "active_goals",
        "beliefs_contested",
        "pending_initiatives",
        "blocking",
        "requires_tool",
        "requires_memory_commit",
        "identity_check",
        "self_model_available",
        "source_user",
        "source_autonomous",
        "action_mutates_state",
        "action_external",
        "hour_of_day_sin",
        "hour_of_day_cos",
    ),
    sources={
        "priority": "core/executive/executive_core.py:Intent",
        "confidence": "core/executive/executive_core.py:Intent",
        "coherence": "core/binding/binding_engine.py",
        "failure_pressure": "core/executive/executive_core.py:_get_failure_state",
        "active_goals": "core/executive/executive_core.py:_get_temporal_identity_context",
        "beliefs_contested": "core/executive/executive_core.py:_get_epistemic_state",
        "pending_initiatives": "core/executive/executive_core.py:_get_temporal_identity_context",
    },
)


_SCHEMAS: dict[str, FeatureSchema] = {EXECUTIVE_ADMISSION.control_point: EXECUTIVE_ADMISSION}


def register_schema(schema: FeatureSchema) -> FeatureSchema:
    existing = _SCHEMAS.get(schema.control_point)
    if existing is not None and existing.schema_id != schema.schema_id:
        logger.info(
            "ontogeny: schema for %s changed %s -> %s; prior episodes retire from training",
            schema.control_point, existing.schema_id, schema.schema_id,
        )
    _SCHEMAS[schema.control_point] = schema
    return schema


def get_schema(control_point: str) -> FeatureSchema | None:
    return _SCHEMAS.get(control_point)


def known_schemas() -> tuple[FeatureSchema, ...]:
    return tuple(_SCHEMAS.values())


def schema_ids() -> dict[str, str]:
    return {cp: s.schema_id for cp, s in _SCHEMAS.items()}


def iter_names(schemas: Iterable[FeatureSchema]) -> tuple[str, ...]:
    seen: list[str] = []
    for schema in schemas:
        for name in schema.names:
            if name not in seen:
                seen.append(name)
    return tuple(seen)


__all__ = [
    "EXECUTIVE_ADMISSION",
    "FeatureSchema",
    "FeatureVector",
    "RunningMoments",
    "design_row",
    "get_schema",
    "known_schemas",
    "register_schema",
    "row_names",
    "schema_ids",
]
