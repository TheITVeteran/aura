"""What may drive the substrate, and what may be added to the model's residuals.

Two unguarded surfaces, both on the same causal chain — input becomes substrate
state, substrate state becomes a steering vector, and the steering vector is
added to the hidden states of all 64 transformer blocks on every token:

1. ``LiquidSubstrate.inject_stimulus`` took any vector and any weight::

       self.x = np.clip(self.x + vector * weight * 0.1, -1.0, 1.0)

   ``np.clip`` of ``NaN`` is ``NaN``, so one non-finite element — or a weight
   of ``inf``, which nothing checked either — puts the whole activation vector
   outside the regime the ODE is defined on. Callers include the perceptual
   frame path, the closed loop, the latent bridge and the embodied simulator,
   which means values derived from screen contents, audio and model output all
   arrive here.

2. ``AffectiveSteeringHook.update_substrate_vector`` cached the composite with
   no finiteness check. If any loaded steering vector contains ``NaN`` then
   ``norm`` is ``NaN``, dividing by it makes every element ``NaN``, and the
   guard that follows is ``if current_norm < 1e-4`` — which is False for
   ``NaN``. A ``NaN`` composite is therefore *accepted* and added to every
   hidden state, destroying generation. The vectors are loaded from
   ``data/steering_vectors/``, so this is reachable from a file on disk.

Both are gated here. The design principle is the one the substrate recovery
module already follows: REJECT, do not coerce. Silently substituting zeros for
a malformed stimulus is indistinguishable from having received nothing, and
silently renormalising a hostile vector applies it anyway at a safe magnitude.
A rejected input is not applied, and it is recorded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SteeringAdmission")

#: Activations live in [-1, 1] and a stimulus is a delta on them. Anything an
#: order of magnitude past full scale is not a strong signal, it is a bad one.
MAX_STIMULUS_MAGNITUDE = 10.0

#: A stimulus weight scales the delta. Unbounded weight is unbounded state.
MAX_STIMULUS_WEIGHT = 10.0

#: The composite is normalised to unit length before smoothing, so a norm past
#: this means the normalisation did not happen or was defeated.
MAX_COMPOSITE_NORM = 1.5

#: A unit vector with all its mass in one dimension is still a unit vector, and
#: adding it to every token saturates a single feature of the residual stream.
#: Steering is meant to be a direction in representation space, not a spike.
MAX_SINGLE_DIMENSION_SHARE = 0.9


@dataclass(frozen=True)
class Admission:
    """Whether an input may be applied, and why not if it may not."""

    admitted: bool
    reasons: tuple[str, ...] = ()
    detail: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.detail is None:
            object.__setattr__(self, "detail", {})

    @property
    def rejected(self) -> bool:
        return not self.admitted

    def as_metrics(self) -> dict[str, Any]:
        return {"admitted": self.admitted, "reasons": list(self.reasons), **dict(self.detail)}


def _finite_report(array: np.ndarray) -> tuple[int, int]:
    finite = np.isfinite(array)
    return int((~finite).sum()), int(finite.sum())


def admit_stimulus(
    vector: Any,
    weight: Any,
    *,
    max_magnitude: float = MAX_STIMULUS_MAGNITUDE,
    max_weight: float = MAX_STIMULUS_WEIGHT,
) -> Admission:
    """Decide whether a stimulus may drive the ODE.

    Checks the vector AND the weight: an ordinary vector at ``inf`` weight is
    the same defect as an ``inf`` vector at weight 1.
    """

    reasons: list[str] = []
    detail: dict[str, Any] = {}

    try:
        array = np.asarray(vector, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        return Admission(False, (f"unreadable_stimulus:{type(exc).__name__}",), {})

    if array.size == 0:
        return Admission(False, ("empty_stimulus",), {})

    non_finite, _ = _finite_report(array)
    if non_finite:
        reasons.append("non_finite_stimulus")
        detail["non_finite_elements"] = non_finite

    finite_values = array[np.isfinite(array)]
    magnitude = float(np.max(np.abs(finite_values))) if finite_values.size else float("inf")
    detail["max_magnitude"] = round(magnitude, 6)
    if magnitude > max_magnitude:
        reasons.append("stimulus_out_of_scale")

    try:
        weight_value = float(weight)
    except (TypeError, ValueError):
        reasons.append("unreadable_weight")
        weight_value = float("nan")
    detail["weight"] = weight_value if np.isfinite(weight_value) else None
    if not np.isfinite(weight_value):
        reasons.append("non_finite_weight")
    elif abs(weight_value) > max_weight:
        reasons.append("weight_out_of_scale")

    return Admission(not reasons, tuple(reasons), detail)


def admit_steering_vector(
    composite: Any,
    *,
    max_norm: float = MAX_COMPOSITE_NORM,
    max_single_dimension_share: float = MAX_SINGLE_DIMENSION_SHARE,
) -> Admission:
    """Decide whether a composite may be added to the model's hidden states.

    This is the last gate before the vector reaches all 64 blocks of every
    token, so it is checked for the two shapes that do damage: values the
    arithmetic cannot survive, and a direction that is really a spike.
    """

    reasons: list[str] = []
    detail: dict[str, Any] = {}

    try:
        array = np.asarray(composite, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        return Admission(False, (f"unreadable_vector:{type(exc).__name__}",), {})

    if array.size == 0:
        return Admission(False, ("empty_vector",), {})

    non_finite, _ = _finite_report(array)
    if non_finite:
        # This is the live hazard: `if current_norm < 1e-4` is False for NaN,
        # so a NaN composite passed the only guard there was.
        reasons.append("non_finite_vector")
        detail["non_finite_elements"] = non_finite
        return Admission(False, tuple(reasons), detail)

    norm = float(np.linalg.norm(array))
    detail["norm"] = round(norm, 6)
    if norm > max_norm:
        reasons.append("vector_norm_out_of_envelope")

    if norm > 0.0:
        share = float(np.max(np.abs(array)) / norm)
        detail["max_dimension_share"] = round(share, 6)
        if share > max_single_dimension_share and array.size > 1:
            # A unit vector with all its mass in one coordinate saturates one
            # feature of the residual stream on every token.
            reasons.append("vector_is_a_spike")

    return Admission(not reasons, tuple(reasons), detail)


def refuse(admission: Admission, *, subsystem: str, action: str) -> None:
    """Record a refusal. Never raises; the caller has already declined to act."""
    if admission.admitted:
        return
    record_degradation(
        subsystem,
        ValueError(f"refused input: {','.join(admission.reasons)}"),
        severity="warning",
        action=action,
        extra=admission.as_metrics(),
    )


__all__ = [
    "Admission",
    "MAX_COMPOSITE_NORM",
    "MAX_SINGLE_DIMENSION_SHARE",
    "MAX_STIMULUS_MAGNITUDE",
    "MAX_STIMULUS_WEIGHT",
    "admit_steering_vector",
    "admit_stimulus",
    "refuse",
]
