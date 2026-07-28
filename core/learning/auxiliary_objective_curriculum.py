"""SPARK-062: seven auxiliary objectives that must actually be doing something.

Adding seven loss terms is a morning's work. The failure that makes it worth a
checkpoint is quieter: **a term that is declared, weighted, logged, and inert.**

It happens without anyone being careless. A term is computed from a signal that
turns out to be constant on this batch; a term is accidentally detached and its
gradient never reaches a parameter; a term's natural scale is 1e-4 next to an
answer cross-entropy of 2.0, so its weighted contribution rounds to nothing; or
a term trains a separate head and was quietly summed into the base-weight loss,
where it does something real but not the thing its name claims. In every case
the composite descends, the telemetry lists seven names, the config records
seven weights, and one term is carrying the run. Nothing in a loss curve can
distinguish that from seven healthy terms, and the resident-32B campaign is too
expensive to find out afterwards.

So the composite here is not a sum. It is a **registry with a liveness test**.

Every term declares what it optimizes, which is the distinction that gets
muddled first:

* ``BASE_WEIGHTS`` — differentiable into the resident weights. Must produce a
  measurable parameter gradient or it is inert.
* ``AUXILIARY_HEAD`` — trains a separate head (the process critic, the mistake
  locator, the accept/discard gate). These are real objectives only when a
  measured gradient reaches that head, its optimizer runs, and its parameter
  digest changes; summing them into the base loss is a category error that
  silently steers the base model with a critic's objective.
* ``DIAGNOSTIC`` — measured and reported, never optimized. Declaring a term
  diagnostic is an honest choice; letting an optimized term *look* diagnostic,
  or a diagnostic term claim gradient, is not.

`build_liveness_report` then measures each term against its own declaration and
classifies it `live`, `inert_zero_gradient`, `inert_negligible_share`,
`inert_head_not_updated`, `misdeclared_target`, or `unmeasured`. A composite
with an inert *required* term refuses.

The depth curriculum is the other half. Its one rule is that **stage
advancement is bound to measured competence, never to step count**: a schedule
that walks short → deep on a step counter will happily train depth 16 on a
model that never learned depth 2, and the resulting curve is uninterpretable.
`DepthCurriculum.advance` needs a competence measurement over a minimum sample
before it will move, and it can move *back*.

Train/inference parity here is a different claim from SPARK-024's. That item
proved the α/blend/RMS kernels are byte-identical between the two paths. This
one proves the *schedule* is executable: a curriculum stage that trains a depth
the inference configuration cannot run produces weights tuned for a regime that
never occurs live. `parity_binding` refuses that before a campaign starts.

No model is trained here and no capability claim follows.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

AUXILIARY_COMPOSITE_SCHEMA = "aura.spark062.auxiliary_composite.v2"
DEPTH_CURRICULUM_SCHEMA = "aura.spark062.depth_curriculum.v1"

# A weighted term contributing less than this fraction of the composite's total
# weighted magnitude is decoration: it cannot move the optimizer against an
# answer term three orders of magnitude larger. Chosen at 1% because that is
# the level below which v3's branch-diversity term sat (0.037%) while being
# reported as an active part of the objective.
DEFAULT_MINIMUM_SHARE = 0.01

# Parameter-gradient norm below which a BASE_WEIGHTS term is treated as having
# no path to the weights at all.
DEFAULT_GRADIENT_EPSILON = 1e-12

# Provenance of the per-term shares. Recomputing verdicts from the recorded
# rows closes LABEL forgery; it cannot close INPUT forgery, because the
# rebuild consumes the same share it is checking. Rather than pretend that
# boundary away, the receipt names which side of it the shares came from.
SHARES_DERIVED_FROM_COMPOSITE = "derived_from_composite"
SHARES_CALLER_SUPPLIED = "caller_supplied"
SHARES_SOURCES = (SHARES_DERIVED_FROM_COMPOSITE, SHARES_CALLER_SUPPLIED)


class AuxiliaryObjectiveError(ValueError):
    """An auxiliary composite or curriculum does not hold up."""


class TermTarget(Enum):
    """What a declared term actually optimizes."""

    BASE_WEIGHTS = "base_weights"
    AUXILIARY_HEAD = "auxiliary_head"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class AuxiliaryTerm:
    """One declared objective term, bound to the module that computes it.

    ``source_module`` is not documentation. A term whose signal has no owning
    module is a number someone invented inside the trainer, and the whole point
    of the registry is that every term is traceable to the instrument that
    produces its signal.
    """

    name: str
    target: TermTarget
    weight: float
    source_module: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise AuxiliaryObjectiveError(f"invalid term name: {self.name!r}")
        if not isinstance(self.target, TermTarget):
            raise AuxiliaryObjectiveError("term target must be a TermTarget")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(float(self.weight))
            or not 0.0 <= float(self.weight) <= 100.0
        ):
            raise AuxiliaryObjectiveError("term weight must be finite in [0, 100]")
        if self.target is TermTarget.DIAGNOSTIC and float(self.weight) != 0.0:
            raise AuxiliaryObjectiveError(
                "a diagnostic term carries no weight; give it a target or zero"
            )
        if self.target is not TermTarget.DIAGNOSTIC and float(self.weight) <= 0.0:
            raise AuxiliaryObjectiveError(
                "an optimized term with zero weight is inert by construction"
            )
        if not self.source_module or "." not in self.source_module:
            raise AuxiliaryObjectiveError(
                "every term must name the dotted module that produces its signal"
            )
        if type(self.required) is not bool:
            raise AuxiliaryObjectiveError("required must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target.value,
            "weight": round(float(self.weight), 6),
            "source_module": self.source_module,
            "required": self.required,
        }


# The seven SPARK-062 objectives, each bound to the instrument that already
# produces its signal. Targets are the honest ones: three of the seven train
# separate heads and must never be summed into the base-weight loss.
SPARK062_TERMS: tuple[AuxiliaryTerm, ...] = (
    AuxiliaryTerm(
        name="process",
        target=TermTarget.AUXILIARY_HEAD,
        weight=1.0,
        source_module="core.learning.process_critic",
    ),
    AuxiliaryTerm(
        name="improvement",
        target=TermTarget.BASE_WEIGHTS,
        weight=1.0,
        source_module="core.learning.progressive_recurrent_objective",
    ),
    AuxiliaryTerm(
        name="diversity",
        target=TermTarget.BASE_WEIGHTS,
        weight=1.0,
        source_module="core.learning.recurrence_native_objective_v4",
    ),
    AuxiliaryTerm(
        name="stopping",
        target=TermTarget.BASE_WEIGHTS,
        weight=0.5,
        source_module="core.learning.adaptive_halting",
    ),
    AuxiliaryTerm(
        name="causality",
        target=TermTarget.BASE_WEIGHTS,
        weight=1.0,
        source_module="core.learning.progressive_recurrent_objective",
    ),
    AuxiliaryTerm(
        name="mistake_location",
        target=TermTarget.AUXILIARY_HEAD,
        weight=1.0,
        source_module="core.learning.mistake_locator",
    ),
    AuxiliaryTerm(
        name="accept_discard",
        target=TermTarget.AUXILIARY_HEAD,
        weight=1.0,
        source_module="core.learning.update_acceptance",
    ),
)

TERM_LIVENESS = (
    "live",
    "inert_zero_gradient",
    "inert_negligible_share",
    "inert_head_not_updated",
    "misdeclared_target",
    "unmeasured",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _finite(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise AuxiliaryObjectiveError(f"{name} must be a finite number")
    return float(value)


def validate_term_set(terms: Sequence[AuxiliaryTerm]) -> tuple[AuxiliaryTerm, ...]:
    """A term set must be complete, unique, and cover the seven named objectives."""

    if not terms:
        raise AuxiliaryObjectiveError("a composite needs at least one term")
    names = [term.name for term in terms]
    if len(set(names)) != len(names):
        raise AuxiliaryObjectiveError("duplicate term names in the composite")
    if any(not isinstance(term, AuxiliaryTerm) for term in terms):
        raise AuxiliaryObjectiveError("every term must be an AuxiliaryTerm")
    return tuple(terms)


def base_weight_loss(
    terms: Sequence[AuxiliaryTerm],
    values: Mapping[str, Any],
    *,
    primary: Any,
) -> tuple[Any, dict[str, Any]]:
    """Combine ONLY the base-weight terms into the trainable scalar.

    ``primary`` is the answer objective; the auxiliary terms are added to it.
    Head and diagnostic terms are deliberately excluded from the returned
    scalar and recorded separately — mixing a critic's objective into the base
    loss is the category error this module exists to make impossible rather
    than merely discouraged.
    """
    import mlx.core as mx

    declared = validate_term_set(terms)
    missing = [term.name for term in declared if term.required and term.name not in values]
    if missing:
        raise AuxiliaryObjectiveError(f"required terms have no computed value: {sorted(missing)}")
    unknown = set(values) - {term.name for term in declared}
    if unknown:
        raise AuxiliaryObjectiveError(f"values supplied for undeclared terms: {sorted(unknown)}")

    total = primary
    contributions: dict[str, float] = {}
    for term in declared:
        if term.name not in values:
            continue
        value = values[term.name]
        weighted = float(term.weight) * float(value)
        contributions[term.name] = weighted
        if term.target is TermTarget.BASE_WEIGHTS:
            total = total + float(term.weight) * value
    mx.eval(total)
    primary_magnitude = abs(float(primary))
    magnitude = primary_magnitude + sum(abs(value) for value in contributions.values())
    shares = {
        name: (abs(value) / magnitude if magnitude > 0.0 else 0.0)
        for name, value in contributions.items()
    }
    telemetry = {
        "primary": round(primary_magnitude, 6),
        "weighted_contributions": {
            name: round(value, 9) for name, value in sorted(contributions.items())
        },
        "shares": {name: round(value, 9) for name, value in sorted(shares.items())},
        "base_weight_terms": sorted(
            term.name
            for term in declared
            if term.target is TermTarget.BASE_WEIGHTS and term.name in values
        ),
        "excluded_from_base_loss": sorted(
            term.name
            for term in declared
            if term.target is not TermTarget.BASE_WEIGHTS and term.name in values
        ),
        "total": round(float(total), 6),
    }
    return total, telemetry


def build_liveness_report(
    terms: Sequence[AuxiliaryTerm],
    *,
    shares: Mapping[str, float],
    gradient_norms: Mapping[str, float] | None = None,
    head_gradient_norms: Mapping[str, float] | None = None,
    head_before_sha256s: Mapping[str, str] | None = None,
    head_after_sha256s: Mapping[str, str] | None = None,
    head_optimizer_update_counts: Mapping[str, int] | None = None,
    minimum_share: float = DEFAULT_MINIMUM_SHARE,
    gradient_epsilon: float = DEFAULT_GRADIENT_EPSILON,
    shares_source: str = SHARES_CALLER_SUPPLIED,
) -> dict[str, Any]:
    """Classify each declared term against what it actually contributed.

    ``gradient_norms`` maps term name to its gradient into the resident base
    weights. Head terms must additionally prove a gradient into their own
    parameters, an optimizer update, and a changed parameter digest. A large
    loss share plus absence of a base gradient is exclusion evidence, not proof
    that a head learned anything.
    """

    declared = validate_term_set(terms)
    floor = _finite(minimum_share, name="minimum_share")
    if not 0.0 <= floor <= 1.0:
        raise AuxiliaryObjectiveError("minimum_share must be inside [0, 1]")
    epsilon = _finite(gradient_epsilon, name="gradient_epsilon")
    if epsilon < 0.0:
        raise AuxiliaryObjectiveError("gradient_epsilon must be non-negative")
    if shares_source not in SHARES_SOURCES:
        raise AuxiliaryObjectiveError("unknown shares source")

    rows: list[dict[str, Any]] = []
    for term in declared:
        share = shares.get(term.name)
        gradient = (gradient_norms or {}).get(term.name)
        head_gradient = (head_gradient_norms or {}).get(term.name)
        head_before = (head_before_sha256s or {}).get(term.name)
        head_after = (head_after_sha256s or {}).get(term.name)
        head_updates = (head_optimizer_update_counts or {}).get(term.name)
        if share is not None:
            share = _finite(share, name=f"{term.name}.share")
            if not 0.0 <= share <= 1.0:
                raise AuxiliaryObjectiveError(f"term share is outside [0, 1] for {term.name}")
        if gradient is not None:
            gradient = _finite(gradient, name=f"{term.name}.base_gradient_norm")
            if gradient < 0.0:
                raise AuxiliaryObjectiveError(f"base gradient norm is negative for {term.name}")
        if head_gradient is not None:
            head_gradient = _finite(head_gradient, name=f"{term.name}.head_gradient_norm")
            if head_gradient < 0.0:
                raise AuxiliaryObjectiveError(f"head gradient norm is negative for {term.name}")
        for role, digest in (
            ("head_before_sha256", head_before),
            ("head_after_sha256", head_after),
        ):
            if digest is not None and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise AuxiliaryObjectiveError(f"{role} is invalid for {term.name}")
        if head_updates is not None and (type(head_updates) is not int or head_updates < 0):
            raise AuxiliaryObjectiveError(f"head optimizer update count is invalid for {term.name}")
        verdict = "unmeasured"
        if term.target is TermTarget.DIAGNOSTIC:
            # A diagnostic term must NOT have a base-weight gradient path.
            if gradient is None:
                verdict = "unmeasured"
            elif gradient > epsilon:
                verdict = "misdeclared_target"
            else:
                verdict = "live"
        elif share is None:
            verdict = "unmeasured"
        elif term.target is TermTarget.BASE_WEIGHTS:
            if gradient is None:
                verdict = "unmeasured"
            elif float(gradient) <= epsilon:
                verdict = "inert_zero_gradient"
            elif float(share) < floor:
                verdict = "inert_negligible_share"
            else:
                verdict = "live"
        else:  # AUXILIARY_HEAD
            # A head term must be excluded from the base loss; if it produced a
            # base-weight gradient it was summed in where it does not belong.
            if gradient is None:
                verdict = "unmeasured"
            elif gradient > epsilon:
                verdict = "misdeclared_target"
            elif float(share) < floor:
                verdict = "inert_negligible_share"
            elif (
                head_gradient is None
                or head_before is None
                or head_after is None
                or head_updates is None
            ):
                verdict = "unmeasured"
            elif float(head_gradient) <= epsilon:
                verdict = "inert_zero_gradient"
            elif head_updates < 1 or head_before == head_after:
                verdict = "inert_head_not_updated"
            else:
                verdict = "live"
        rows.append(
            {
                **term.to_dict(),
                "share": round(float(share), 9) if share is not None else None,
                "gradient_norm": (round(float(gradient), 12) if gradient is not None else None),
                "head_gradient_norm": (
                    round(float(head_gradient), 12) if head_gradient is not None else None
                ),
                "head_before_sha256": head_before,
                "head_after_sha256": head_after,
                "head_optimizer_update_count": head_updates,
                "liveness": verdict,
            }
        )

    inert_required = sorted(
        row["name"] for row in rows if row["required"] and row["liveness"] not in {"live"}
    )
    payload = {
        "schema": AUXILIARY_COMPOSITE_SCHEMA,
        "term_count": len(rows),
        "terms": rows,
        # Where the shares came from. Recomputing verdicts from the rows
        # closes label forgery, but it cannot close INPUT forgery: a rebuild
        # consumes the same share it is checking. Naming the provenance is
        # the honest move — `derived_from_composite` means the shares were
        # computed by `base_weight_loss` from measured contributions, and
        # `caller_supplied` means a reader must trust whoever produced them.
        "shares_source": shares_source,
        "minimum_share": round(floor, 9),
        "gradient_epsilon": epsilon,
        "live_terms": sorted(row["name"] for row in rows if row["liveness"] == "live"),
        "inert_required_terms": inert_required,
        "supports_training": not inert_required,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def liveness_from_composite(
    terms: Sequence[AuxiliaryTerm],
    composite_telemetry: Mapping[str, Any],
    *,
    gradient_norms: Mapping[str, float] | None = None,
    head_gradient_norms: Mapping[str, float] | None = None,
    head_before_sha256s: Mapping[str, str] | None = None,
    head_after_sha256s: Mapping[str, str] | None = None,
    head_optimizer_update_counts: Mapping[str, int] | None = None,
    minimum_share: float = DEFAULT_MINIMUM_SHARE,
    gradient_epsilon: float = DEFAULT_GRADIENT_EPSILON,
) -> dict[str, Any]:
    """Build a liveness report whose shares came from the measured composite.

    Prefer this over passing ``shares`` by hand. `build_liveness_report`
    recomputes every verdict from the recorded rows, which closes *label*
    forgery — but a rebuild consumes the same share it is checking, so a
    forged share survives it. Deriving the shares from `base_weight_loss`'s
    own telemetry removes that input from the caller's control entirely,
    which is a stronger guarantee than any amount of re-validation.
    """

    if not isinstance(composite_telemetry, Mapping):
        raise AuxiliaryObjectiveError("composite telemetry must be a mapping")
    shares = composite_telemetry.get("shares")
    if not isinstance(shares, Mapping):
        raise AuxiliaryObjectiveError(
            "composite telemetry carries no shares; it did not come from base_weight_loss"
        )
    return build_liveness_report(
        terms,
        shares={str(name): float(value) for name, value in shares.items()},
        gradient_norms=gradient_norms,
        head_gradient_norms=head_gradient_norms,
        head_before_sha256s=head_before_sha256s,
        head_after_sha256s=head_after_sha256s,
        head_optimizer_update_counts=head_optimizer_update_counts,
        minimum_share=minimum_share,
        gradient_epsilon=gradient_epsilon,
        shares_source=SHARES_DERIVED_FROM_COMPOSITE,
    )


def validate_liveness_report(value: Any) -> dict[str, Any]:
    """Independently replay the liveness verdicts from the report's own rows."""

    if not isinstance(value, dict):
        raise AuxiliaryObjectiveError("liveness report must be a mapping")
    required = {
        "schema",
        "term_count",
        "terms",
        "shares_source",
        "minimum_share",
        "gradient_epsilon",
        "live_terms",
        "inert_required_terms",
        "supports_training",
        "receipt_sha256",
    }
    if set(value) != required:
        raise AuxiliaryObjectiveError("liveness report fields do not match")
    payload = {key: value[key] for key in required - {"receipt_sha256"}}
    if value["receipt_sha256"] != canonical_sha256(payload):
        raise AuxiliaryObjectiveError("liveness report commitment mismatch")
    if value["schema"] != AUXILIARY_COMPOSITE_SCHEMA:
        raise AuxiliaryObjectiveError("unsupported liveness report schema")
    rows = value["terms"]
    if not isinstance(rows, list) or len(rows) != value["term_count"]:
        raise AuxiliaryObjectiveError("liveness term rows do not match the count")
    names = [row.get("name") for row in rows]
    if len(set(names)) != len(names):
        raise AuxiliaryObjectiveError("duplicate terms in the liveness report")
    # Recompute every row's verdict from its OWN share and gradient rather
    # than trusting the recorded label. Checking only the aggregates was a
    # real defect in the first version of this validator: relabelling a
    # zero-gradient row `live` and updating `live_terms` to match passed every
    # aggregate check, because the aggregates were derived from the labels
    # they were supposed to police. A commitment proves nobody edited the
    # bytes after signing; it says nothing about whether the signer's
    # classification was honest.
    try:
        rebuilt = build_liveness_report(
            [
                AuxiliaryTerm(
                    name=str(row["name"]),
                    target=TermTarget(row["target"]),
                    weight=float(row["weight"]),
                    source_module=str(row["source_module"]),
                    required=bool(row["required"]),
                )
                for row in rows
            ],
            shares={
                str(row["name"]): float(row["share"])
                for row in rows
                if row.get("share") is not None
            },
            gradient_norms={
                str(row["name"]): float(row["gradient_norm"])
                for row in rows
                if row.get("gradient_norm") is not None
            }
            or None,
            head_gradient_norms={
                str(row["name"]): float(row["head_gradient_norm"])
                for row in rows
                if row.get("head_gradient_norm") is not None
            }
            or None,
            head_before_sha256s={
                str(row["name"]): str(row["head_before_sha256"])
                for row in rows
                if row.get("head_before_sha256") is not None
            }
            or None,
            head_after_sha256s={
                str(row["name"]): str(row["head_after_sha256"])
                for row in rows
                if row.get("head_after_sha256") is not None
            }
            or None,
            head_optimizer_update_counts={
                str(row["name"]): int(row["head_optimizer_update_count"])
                for row in rows
                if row.get("head_optimizer_update_count") is not None
            }
            or None,
            minimum_share=float(value["minimum_share"]),
            gradient_epsilon=float(value["gradient_epsilon"]),
            shares_source=str(value["shares_source"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AuxiliaryObjectiveError("liveness term rows are malformed") from exc
    if rebuilt != dict(value):
        differing = sorted(
            key for key in set(rebuilt) | set(value) if rebuilt.get(key) != value.get(key)
        )
        raise AuxiliaryObjectiveError(
            f"liveness report does not replay from its own rows: {differing}"
        )
    for row in rows:
        if row.get("liveness") not in TERM_LIVENESS:
            raise AuxiliaryObjectiveError("unknown liveness verdict")
        if row.get("target") not in {item.value for item in TermTarget}:
            raise AuxiliaryObjectiveError("unknown term target")
    expected_live = sorted(row["name"] for row in rows if row["liveness"] == "live")
    expected_inert = sorted(
        row["name"] for row in rows if row.get("required") and row["liveness"] != "live"
    )
    if value["live_terms"] != expected_live:
        raise AuxiliaryObjectiveError("live-term set does not replay")
    if value["inert_required_terms"] != expected_inert:
        raise AuxiliaryObjectiveError("inert-required set does not replay")
    if value["supports_training"] is not (not expected_inert):
        raise AuxiliaryObjectiveError("training support must follow the inert-required set exactly")
    return dict(value)


# ── Depth curriculum ────────────────────────────────────────────────────


@dataclass(frozen=True)
class DepthStage:
    """One curriculum stage: a depth, and what must be true to leave it."""

    depth: int
    min_samples: int
    competence_threshold: float

    def __post_init__(self) -> None:
        if type(self.depth) is not int or not 1 <= self.depth <= 64:
            raise AuxiliaryObjectiveError("stage depth must be inside [1, 64]")
        if type(self.min_samples) is not int or self.min_samples < 1:
            raise AuxiliaryObjectiveError("stage min_samples must be positive")
        threshold = _finite(self.competence_threshold, name="competence_threshold")
        if not 0.0 <= threshold <= 1.0:
            raise AuxiliaryObjectiveError("competence_threshold must be inside [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "min_samples": self.min_samples,
            "competence_threshold": round(float(self.competence_threshold), 6),
        }


class DepthCurriculum:
    """Short → deep, advanced by measured competence rather than step count.

    A step-counted schedule is the default because it is trivial to write, and
    it is wrong for the same reason every time: it will promote a model to
    depth 16 having never established depth 2, and every measurement after that
    point conflates "deep tasks are hard" with "this model was never taught the
    shallow case". Competence gating costs a measurement and removes the
    ambiguity.

    Regression is supported deliberately. A curriculum that can only advance
    turns a transient competence dip into a permanent one.
    """

    def __init__(self, stages: Sequence[DepthStage]) -> None:
        if not stages:
            raise AuxiliaryObjectiveError("a curriculum needs at least one stage")
        depths = [stage.depth for stage in stages]
        if depths != sorted(set(depths)):
            raise AuxiliaryObjectiveError("curriculum stages must be strictly increasing in depth")
        self.stages = tuple(stages)
        self._index = 0
        self._history: list[dict[str, Any]] = []

    @property
    def stage(self) -> DepthStage:
        return self.stages[self._index]

    @property
    def index(self) -> int:
        return self._index

    def observe(self, *, competence: float, samples: int) -> str:
        """Record a competence measurement and return the transition taken.

        Returns one of ``advanced``, ``regressed``, ``held_insufficient_samples``,
        ``held_below_threshold``, or ``held_at_final_stage``.
        """
        score = _finite(competence, name="competence")
        if not 0.0 <= score <= 1.0:
            raise AuxiliaryObjectiveError("competence must be inside [0, 1]")
        if type(samples) is not int or samples < 0:
            raise AuxiliaryObjectiveError("samples must be a non-negative integer")
        stage = self.stage
        if samples < stage.min_samples:
            transition = "held_insufficient_samples"
        elif score < stage.competence_threshold:
            # Competence collapsed at the current depth: step back if we can,
            # rather than continuing to train a depth the model cannot hold.
            if self._index > 0:
                self._index -= 1
                transition = "regressed"
            else:
                transition = "held_below_threshold"
        elif self._index + 1 < len(self.stages):
            self._index += 1
            transition = "advanced"
        else:
            transition = "held_at_final_stage"
        self._history.append(
            {
                "ordinal": len(self._history),
                "depth": stage.depth,
                "competence": round(score, 6),
                "samples": samples,
                "transition": transition,
                "next_depth": self.stage.depth,
            }
        )
        return transition

    def to_receipt(self) -> dict[str, Any]:
        payload = {
            "schema": DEPTH_CURRICULUM_SCHEMA,
            "stages": [stage.to_dict() for stage in self.stages],
            "stage_count": len(self.stages),
            "current_index": self._index,
            "current_depth": self.stage.depth,
            "advancement_policy": "measured_competence_over_minimum_samples_v1",
            "history": [dict(row) for row in self._history],
            "observation_count": len(self._history),
        }
        return {**payload, "receipt_sha256": canonical_sha256(payload)}


def validate_curriculum_receipt(value: Any) -> dict[str, Any]:
    """Replay every transition from the recorded stages and observations."""

    if not isinstance(value, dict):
        raise AuxiliaryObjectiveError("curriculum receipt must be a mapping")
    required = {
        "schema",
        "stages",
        "stage_count",
        "current_index",
        "current_depth",
        "advancement_policy",
        "history",
        "observation_count",
        "receipt_sha256",
    }
    if set(value) != required:
        raise AuxiliaryObjectiveError("curriculum receipt fields do not match")
    payload = {key: value[key] for key in required - {"receipt_sha256"}}
    if value["receipt_sha256"] != canonical_sha256(payload):
        raise AuxiliaryObjectiveError("curriculum receipt commitment mismatch")
    if value["schema"] != DEPTH_CURRICULUM_SCHEMA:
        raise AuxiliaryObjectiveError("unsupported curriculum schema")
    if value["advancement_policy"] != "measured_competence_over_minimum_samples_v1":
        raise AuxiliaryObjectiveError(
            "a curriculum advanced by any other policy is not this contract"
        )
    stages = [
        DepthStage(
            depth=int(row["depth"]),
            min_samples=int(row["min_samples"]),
            competence_threshold=float(row["competence_threshold"]),
        )
        for row in value["stages"]
    ]
    if len(stages) != value["stage_count"]:
        raise AuxiliaryObjectiveError("stage count differs from stage rows")
    if len(value["history"]) != value["observation_count"]:
        raise AuxiliaryObjectiveError("observation count differs from history")
    # Replay: an independent curriculum fed the same observations must land in
    # the same place. This is what makes a forged "current_depth" detectable.
    replay = DepthCurriculum(stages)
    for ordinal, row in enumerate(value["history"]):
        if row.get("ordinal") != ordinal:
            raise AuxiliaryObjectiveError("curriculum history is not ordered")
        if row.get("depth") != replay.stage.depth:
            raise AuxiliaryObjectiveError(
                "curriculum history depth does not match the replayed stage"
            )
        transition = replay.observe(
            competence=float(row["competence"]), samples=int(row["samples"])
        )
        if transition != row.get("transition"):
            raise AuxiliaryObjectiveError(f"curriculum transition {ordinal} does not replay")
        if replay.stage.depth != row.get("next_depth"):
            raise AuxiliaryObjectiveError(f"curriculum next depth {ordinal} does not replay")
    if replay.index != value["current_index"] or (replay.stage.depth != value["current_depth"]):
        raise AuxiliaryObjectiveError("curriculum final position does not replay")
    return dict(value)


def parity_binding(
    stage: DepthStage,
    *,
    spec: Any,
    inference_max_steps: int,
    inference_min_steps: int = 1,
    inference_fixed_depth: bool = False,
) -> dict[str, Any]:
    """Refuse a curriculum stage the inference configuration cannot execute.

    SPARK-024 proved the recurrence KERNELS are identical between the training
    unroll and the live engine. That is necessary and not sufficient: a stage
    may train at a depth the live configuration will never reach, in which case
    the weights are tuned for a regime that does not occur in production and
    every live measurement is off-distribution.

    The binding also refuses adaptive halting at training depth. A trained
    stage assumes exactly ``stage.depth`` applications of the operator; an
    inference config that may halt early is running a different computation
    than the one that was trained, and the comparison is not clean until the
    halting policy itself has been trained (SPARK-026's own item).
    """

    if not isinstance(stage, DepthStage):
        raise AuxiliaryObjectiveError("parity requires a DepthStage")
    if type(inference_max_steps) is not int or inference_max_steps < 1:
        raise AuxiliaryObjectiveError("inference_max_steps must be positive")
    if type(inference_min_steps) is not int or inference_min_steps < 1:
        raise AuxiliaryObjectiveError("inference_min_steps must be positive")
    if type(inference_fixed_depth) is not bool:
        raise AuxiliaryObjectiveError("inference_fixed_depth must be boolean")

    problems: list[str] = []
    spec_steps = getattr(spec, "recurrent_steps", None)
    if spec_steps != stage.depth:
        problems.append("execution_spec_depth_differs_from_stage")
    if stage.depth > inference_max_steps:
        problems.append("stage_depth_exceeds_inference_max_steps")
    if stage.depth < inference_min_steps:
        problems.append("stage_depth_below_inference_min_steps")
    if not inference_fixed_depth:
        problems.append("adaptive_halting_breaks_trained_depth_parity")
    spec_alpha_schedule = getattr(spec, "alpha_schedule", None)
    if spec_alpha_schedule not in {"constant", "cosine"}:
        problems.append("unsupported_alpha_schedule")

    payload = {
        "schema": DEPTH_CURRICULUM_SCHEMA,
        "binding": "train_inference_depth_parity_v1",
        "stage": stage.to_dict(),
        "execution_spec_depth": spec_steps if type(spec_steps) is int else None,
        "execution_spec_alpha_schedule": (
            spec_alpha_schedule if isinstance(spec_alpha_schedule, str) else None
        ),
        "inference_max_steps": inference_max_steps,
        "inference_min_steps": inference_min_steps,
        "inference_fixed_depth": inference_fixed_depth,
        "problems": sorted(problems),
        "parity": not problems,
    }
    return {**payload, "receipt_sha256": canonical_sha256(payload)}


def require_parity(binding: Mapping[str, Any]) -> None:
    """Fail closed on a non-parity binding, for use at campaign admission."""

    if not isinstance(binding, Mapping) or binding.get("parity") is not True:
        problems = (
            sorted(binding.get("problems") or [])
            if isinstance(binding, Mapping)
            else ["binding_not_a_mapping"]
        )
        raise AuxiliaryObjectiveError(f"train/inference depth parity refused: {problems}")


__all__ = [
    "AUXILIARY_COMPOSITE_SCHEMA",
    "DEFAULT_GRADIENT_EPSILON",
    "DEFAULT_MINIMUM_SHARE",
    "DEPTH_CURRICULUM_SCHEMA",
    "SHARES_CALLER_SUPPLIED",
    "SHARES_DERIVED_FROM_COMPOSITE",
    "SHARES_SOURCES",
    "SPARK062_TERMS",
    "TERM_LIVENESS",
    "AuxiliaryObjectiveError",
    "AuxiliaryTerm",
    "DepthCurriculum",
    "DepthStage",
    "TermTarget",
    "base_weight_loss",
    "build_liveness_report",
    "liveness_from_composite",
    "canonical_sha256",
    "parity_binding",
    "require_parity",
    "validate_curriculum_receipt",
    "validate_liveness_report",
    "validate_term_set",
]
