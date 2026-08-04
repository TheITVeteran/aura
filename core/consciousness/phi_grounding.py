"""Which φ measured what, and why one of them may not be compared to another.

``PhiCore.compute_full_kernel`` ran four estimators and took the largest::

    winner = affective_res
    if mesh_res and (not winner or mesh_res.phi_s > winner.phi_s):   winner = mesh_res
    if residual_res and (not winner or residual_res.phi_s > ...):    winner = residual_res
    if grassmann_res and (not winner or grassmann_res.phi_s > ...):  winner = grassmann_res

Two things are wrong with that, and they compound.

FIRST, these are not four candidate subsets of one system, which is what the
exclusion postulate is about. They are four estimators over four different
substrates: the Grassmann complex encodes the transformer's residual-stream
GEOMETRY, the mesh complex reads real computational units, the residual complex
uses eight chunk-means of a ~5000-dimensional vector, and the affective complex
reads summary statistics of mood. Taking a maximum across them is a category
error — a bigger number from a weaker measurement is not a stronger complex.

SECOND, and this is what the review found: ``compute_grassmann_residual_phi``
returns ``None`` below ``MIN_HISTORY_FOR_TPM`` transitions. An unavailable
measurement loses a maximisation to an available one silently, so during every
warm-up the reported integration score is the summary-statistic path — and it
is reported as the system's φ, with nothing saying the activation-grounded
measurement had simply not accumulated enough history to speak.

So selection is by GROUNDING first and magnitude second. The best-grounded tier
with a live measurement wins; within a tier, the largest φ wins, which is the
comparison exclusion actually licenses. Every candidate that could not run is
carried in the result with the reason it could not, so a summary-statistic
number is never mistaken for an activation-level one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PhiGrounding(str, Enum):
    """What the φ estimate was measured ON, ordered by evidential strength."""

    #: The transformer's residual-stream geometry — Grassmann subspace modes.
    #: Nodes are real directions the representation occupies.
    ACTIVATION_GEOMETRY = "activation_geometry"

    #: Real computational units sampled from the neural mesh executive tier.
    COMPUTATIONAL_UNITS = "computational_units"

    #: Activations, but reduced to eight contiguous chunk-means of a ~5000-d
    #: vector. Derived from the real thing and lossy about it.
    ACTIVATION_SUMMARY = "activation_summary"

    #: Summary statistics of affect and cognition — valence, arousal, agency
    #: score. Real state, but a description of the system rather than a sample
    #: of its units.
    STATE_SUMMARY = "state_summary"


#: Strongest first. Selection walks this order and takes the first tier that
#: has a live measurement.
GROUNDING_ORDER: tuple[PhiGrounding, ...] = (
    PhiGrounding.ACTIVATION_GEOMETRY,
    PhiGrounding.COMPUTATIONAL_UNITS,
    PhiGrounding.ACTIVATION_SUMMARY,
    PhiGrounding.STATE_SUMMARY,
)


def grounding_rank(grounding: PhiGrounding | str) -> int:
    """Lower is better grounded."""
    try:
        value = PhiGrounding(str(getattr(grounding, "value", grounding)))
    except ValueError:
        return len(GROUNDING_ORDER)
    return GROUNDING_ORDER.index(value)


@dataclass(frozen=True)
class PhiCandidate:
    """One estimator's contribution to the selection, available or not."""

    name: str
    grounding: PhiGrounding
    result: Any = None
    unavailable_reason: str = ""

    @property
    def available(self) -> bool:
        return self.result is not None

    @property
    def phi_s(self) -> float:
        return float(getattr(self.result, "phi_s", 0.0)) if self.result is not None else 0.0

    def as_metrics(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "grounding": self.grounding.value,
            "available": self.available,
        }
        if self.available:
            payload["phi_s"] = round(self.phi_s, 6)
            payload["tpm_n_samples"] = int(getattr(self.result, "tpm_n_samples", 0))
        else:
            payload["unavailable_reason"] = self.unavailable_reason
        return payload


@dataclass(frozen=True)
class PhiSelection:
    """The chosen φ, what it measured, and everything that did not run."""

    winner: Any = None
    winner_name: str = ""
    grounding: PhiGrounding | None = None
    candidates: tuple[PhiCandidate, ...] = field(default_factory=tuple)

    @property
    def phi_s(self) -> float:
        return float(getattr(self.winner, "phi_s", 0.0)) if self.winner is not None else 0.0

    @property
    def better_grounded_unavailable(self) -> tuple[PhiCandidate, ...]:
        """Estimators that would have outranked the winner but could not run.

        This is the tuple that was missing. Its emptiness is the difference
        between "the activation-level measurement says this" and "the
        activation-level measurement did not happen, so you are reading the
        summary".
        """
        if self.grounding is None:
            return tuple(c for c in self.candidates if not c.available)
        limit = grounding_rank(self.grounding)
        return tuple(
            candidate
            for candidate in self.candidates
            if not candidate.available and grounding_rank(candidate.grounding) < limit
        )

    @property
    def is_best_grounded(self) -> bool:
        """True when nothing better grounded was merely missing."""
        return not self.better_grounded_unavailable

    def as_metrics(self) -> dict[str, Any]:
        return {
            "winner": self.winner_name,
            "phi_s": round(self.phi_s, 6) if self.winner is not None else None,
            "grounding": self.grounding.value if self.grounding else None,
            "is_best_grounded": self.is_best_grounded,
            "better_grounded_unavailable": [
                {"name": c.name, "reason": c.unavailable_reason}
                for c in self.better_grounded_unavailable
            ],
            "candidates": [c.as_metrics() for c in self.candidates],
        }


def select_phi(candidates: list[PhiCandidate] | tuple[PhiCandidate, ...]) -> PhiSelection:
    """Choose by grounding first, magnitude second.

    Magnitude only decides between measurements of the same kind — that is the
    comparison the exclusion postulate licenses. A summary-statistic estimate
    cannot outrank an activation-level one by being larger.
    """

    ordered = tuple(candidates)
    for tier in GROUNDING_ORDER:
        live = [c for c in ordered if c.available and c.grounding is tier]
        if not live:
            continue
        best = max(live, key=lambda candidate: candidate.phi_s)
        return PhiSelection(
            winner=best.result,
            winner_name=best.name,
            grounding=tier,
            candidates=ordered,
        )
    return PhiSelection(candidates=ordered)


__all__ = [
    "GROUNDING_ORDER",
    "PhiCandidate",
    "PhiGrounding",
    "PhiSelection",
    "grounding_rank",
    "select_phi",
]
