"""Exact, deterministic grading for resident paired RLC campaigns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Never

from core.brain.llm.latent_cortex.exact_paired_statistics import (
    DEFAULT_FAMILY_ALPHA,
    EFFECT_BOUND_CERTIFICATE_VERSION,
    EFFECT_BOUND_METHOD,
    MAX_CERTIFIED_BOUND_TRIALS,
    CertifiedEffectBounds,
    ExactStatisticsError,
    Rational,
    certified_rational_effect_bounds,
    exact_compute_tolerance_decision,
    exact_holm_adjustment,
    exact_paired_binomial_tail,
    exact_sign_flip_tail,
)
from core.brain.llm.latent_cortex.experiments import (
    CONJECTURE,
    PROVEN,
    REFUTED,
    SUPPORTED,
)

EXACT_PAIRED_COMPARISON_SCHEMA: Final = (
    "aura.latent_cortex.exact_paired_comparison.v1"
)
EXACT_INTERACTION_SCHEMA: Final = (
    "aura.latent_cortex.exact_paired_interaction.v1"
)
EXACT_NONINFERIORITY_POWER_SCHEMA: Final = (
    "aura.latent_cortex.exact_noninferiority_power.v1"
)
EXACT_GROUP_SEQUENTIAL_POWER_SCHEMA: Final = (
    "aura.latent_cortex.exact_group_sequential_power.v1"
)
EXACT_GRADE_METHOD: Final = (
    "exact paired binomial + exact Holm + simultaneous rational "
    "Clopper-Pearson effect bounds"
)
EXACT_INTERACTION_METHOD: Final = (
    "exact task-paired 2x2 difference-in-differences sign flip + "
    "simultaneous contrast-composed Clopper-Pearson bounds"
)
MINIMUM_EFFECT: Final = Rational(1, 50)
ALPHA: Final = DEFAULT_FAMILY_ALPHA
MIN_OBSERVATIONS_FOR_VERDICT: Final = 20
BOUND_PRECISION_BITS: Final = 40
SIGN_FLIP_ASSUMPTION: Final = (
    "task draws are independent and arm labels are exchangeable under the "
    "sharp no-interaction null"
)


class ExactPairedGradeError(ValueError):
    """Stable fail-closed exact paired-grade error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise ExactPairedGradeError(code)


def _rational(value: Rational) -> dict[str, int]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _less(left: Rational, right: Rational) -> bool:
    return left.numerator * right.denominator < (
        right.numerator * left.denominator
    )


def _less_or_equal(left: Rational, right: Rational) -> bool:
    return left.numerator * right.denominator <= (
        right.numerator * left.denominator
    )


def _subtract(left: Rational, right: Rational) -> Rational:
    return Rational(
        left.numerator * right.denominator
        - right.numerator * left.denominator,
        left.denominator * right.denominator,
    )


def _add(left: Rational, right: Rational) -> Rational:
    return Rational(
        left.numerator * right.denominator
        + right.numerator * left.denominator,
        left.denominator * right.denominator,
    )


def _multiply(left: Rational, right: Rational) -> Rational:
    return Rational(
        left.numerator * right.numerator,
        left.denominator * right.denominator,
    )


def _bound_payload(bounds: CertifiedEffectBounds) -> dict[str, Any]:
    return {
        "certificate_version": bounds.certificate_version,
        "method": bounds.method,
        "certified": bounds.certified,
        "wins": bounds.wins,
        "losses": bounds.losses,
        "ties": bounds.ties,
        "observations": bounds.observations,
        "lower": _rational(bounds.lower),
        "upper": _rational(bounds.upper),
        "family_count": bounds.family_count,
        "family_alpha": _rational(bounds.family_alpha),
        "component_alpha": _rational(bounds.component_alpha),
        "simultaneous_coverage_lower": _rational(
            bounds.simultaneous_coverage_lower
        ),
        "precision_bits": bounds.precision_bits,
        "grid_step": _rational(bounds.grid_step),
        "endpoint_max_outward_rounding": _rational(
            bounds.endpoint_max_outward_rounding
        ),
        "components": [
            {
                "component": component.component,
                "bound_kind": component.bound_kind,
                "tail_kind": component.tail_kind,
                "successes": component.successes,
                "trials": component.trials,
                "bound": _rational(component.bound),
                "tail_probability": (
                    _rational(component.tail_probability)
                    if component.tail_probability is not None
                    else None
                ),
                "component_alpha": _rational(component.component_alpha),
                "adjacent_bound": (
                    _rational(component.adjacent_bound)
                    if component.adjacent_bound is not None
                    else None
                ),
                "adjacent_tail_probability": (
                    _rational(component.adjacent_tail_probability)
                    if component.adjacent_tail_probability is not None
                    else None
                ),
                "precision_bits": component.precision_bits,
                "certified": component.certified,
            }
            for component in bounds.components
        ],
    }


@dataclass(frozen=True, slots=True)
class ExactPairedObservation:
    """One task-level treatment/control outcome with measured compute."""

    task_id: str
    family: str
    treatment_success: bool
    control_success: bool
    treatment_compute: int | None
    control_compute: int | None


def _validated_observations(
    observations_by_family: Mapping[str, Sequence[ExactPairedObservation]],
) -> tuple[
    tuple[str, tuple[ExactPairedObservation, ...]],
    ...,
]:
    if not isinstance(observations_by_family, Mapping) or not observations_by_family:
        _fail("exact_grade_families_invalid")
    seen_tasks: set[str] = set()
    normalized: list[tuple[str, tuple[ExactPairedObservation, ...]]] = []
    for family in sorted(observations_by_family):
        observations = observations_by_family[family]
        if (
            not isinstance(family, str)
            or not family
            or family != family.strip()
            or isinstance(observations, (str, bytes))
            or not isinstance(observations, Sequence)
            or not observations
        ):
            _fail("exact_grade_family_invalid")
        materialized = tuple(observations)
        if any(
            not isinstance(observation, ExactPairedObservation)
            for observation in materialized
        ):
            _fail("exact_grade_observation_invalid")
        ordered = tuple(
            sorted(materialized, key=lambda item: item.task_id)
        )
        for observation in ordered:
            if (
                not observation.task_id
                or observation.task_id != observation.task_id.strip()
                or observation.task_id in seen_tasks
                or observation.family != family
                or type(observation.treatment_success) is not bool
                or type(observation.control_success) is not bool
            ):
                _fail("exact_grade_observation_invalid")
            seen_tasks.add(observation.task_id)
            for compute in (
                observation.treatment_compute,
                observation.control_compute,
            ):
                if compute is not None and (
                    type(compute) is not int or compute < 0
                ):
                    _fail("exact_grade_compute_invalid")
        normalized.append((family, ordered))
    return tuple(normalized)


def _compute_state(
    observations: Sequence[ExactPairedObservation],
    *,
    tolerance: Rational,
) -> tuple[bool, bool, list[str]]:
    missing = any(
        observation.treatment_compute is None
        or observation.control_compute is None
        for observation in observations
    )
    nonpositive = any(
        observation.treatment_compute is not None
        and observation.control_compute is not None
        and (
            observation.treatment_compute <= 0
            or observation.control_compute <= 0
        )
        for observation in observations
    )
    mismatches: list[str] = []
    for observation in observations:
        treatment = observation.treatment_compute
        control = observation.control_compute
        if (
            treatment is None
            or control is None
            or treatment <= 0
            or control <= 0
        ):
            continue
        decision = exact_compute_tolerance_decision(
            treatment,
            control,
            tolerance_numerator=tolerance.numerator,
            tolerance_denominator=tolerance.denominator,
        )
        if not decision.within_tolerance:
            mismatches.append(observation.task_id)
    return missing, nonpositive, mismatches


def campaign_global_bound_family_count(
    *,
    domain_count: int,
    comparison_count: int,
) -> int:
    """Return the preregistered campaign-wide bound multiplicity budget."""

    if (
        type(domain_count) is not int
        or domain_count <= 0
        or type(comparison_count) is not int
        or comparison_count <= 0
    ):
        _fail("exact_grade_global_bound_shape_invalid")
    return comparison_count * (domain_count + 1) + 2


def grade_exact_paired_comparison(
    *,
    experiment: str,
    statement: str,
    treatment: str,
    control: str,
    observations_by_family: Mapping[
        str, Sequence[ExactPairedObservation]
    ],
    require_compute: bool,
    compute_tolerance: Rational,
    global_bound_family_count: int,
    family_alpha: Rational = ALPHA,
) -> dict[str, Any]:
    """Grade one comparison and emit a complete exact certificate tree."""

    if (
        not isinstance(experiment, str)
        or not experiment
        or not isinstance(statement, str)
        or not statement
        or not isinstance(treatment, str)
        or not treatment
        or not isinstance(control, str)
        or not control
        or treatment == control
        or type(require_compute) is not bool
        or type(compute_tolerance) is not Rational
        or type(global_bound_family_count) is not int
        or global_bound_family_count <= 0
        or type(family_alpha) is not Rational
        or not _less(Rational(0, 1), family_alpha)
        or _less(Rational(1, 2), family_alpha)
    ):
        _fail("exact_grade_contract_invalid")
    families = _validated_observations(observations_by_family)
    if global_bound_family_count < len(families) + 1:
        _fail("exact_grade_global_bound_budget_insufficient")
    family_material: dict[str, dict[str, Any]] = {}
    family_bounds: dict[str, CertifiedEffectBounds] = {}
    family_pvalues: dict[str, Rational] = {}
    invalid_compute: list[str] = []
    underpowered: list[str] = []
    pooled_differences: list[int] = []

    try:
        for family, observations in families:
            differences = [
                int(observation.treatment_success)
                - int(observation.control_success)
                for observation in observations
            ]
            wins = differences.count(1)
            losses = differences.count(-1)
            ties = len(differences) - wins - losses
            effect = Rational(sum(differences), len(differences))
            bounds = certified_rational_effect_bounds(
                wins,
                losses,
                ties,
                family_count=global_bound_family_count,
                family_alpha=family_alpha,
                precision_bits=BOUND_PRECISION_BITS,
            )
            tail = exact_paired_binomial_tail(wins, losses)
            pvalue = Rational(tail.numerator, tail.denominator)
            missing, nonpositive, mismatches = _compute_state(
                observations,
                tolerance=compute_tolerance,
            )
            if require_compute and (missing or nonpositive or mismatches):
                invalid_compute.append(family)
            if len(observations) < MIN_OBSERVATIONS_FOR_VERDICT:
                underpowered.append(family)
            else:
                family_pvalues[family] = pvalue
            family_bounds[family] = bounds
            family_material[family] = {
                "n": len(observations),
                "treatment_wins": wins,
                "control_wins": losses,
                "ties": ties,
                "paired_effect": _rational(effect),
                "effect_bounds": _bound_payload(bounds),
                "one_sided_exact_p": _rational(pvalue),
                "holm_adjusted_p": None,
                "missing_compute": missing,
                "nonpositive_compute": nonpositive,
                "compute_mismatch_task_ids": mismatches,
            }
            pooled_differences.extend(differences)
    except ExactStatisticsError as exc:
        raise ExactPairedGradeError("exact_grade_statistics_failed") from exc

    adjusted: dict[str, Rational] = {}
    holm_entries: list[dict[str, Any]] = []
    if family_pvalues:
        try:
            holm = exact_holm_adjustment(family_pvalues)
        except ExactStatisticsError as exc:
            raise ExactPairedGradeError(
                "exact_grade_statistics_failed"
            ) from exc
        holm_method = holm.method
        holm_family_size = holm.family_size
        for entry in holm.ordered:
            adjusted[entry.hypothesis] = entry.adjusted
            family_material[entry.hypothesis]["holm_adjusted_p"] = _rational(
                entry.adjusted
            )
            holm_entries.append(
                {
                    "hypothesis": entry.hypothesis,
                    "rank": entry.rank,
                    "raw": _rational(entry.raw),
                    "adjusted": _rational(entry.adjusted),
                }
            )
    else:
        holm_method = "Holm step-down, exact rational"
        holm_family_size = 0

    positive = [
        family
        for family, _observations in families
        if family in adjusted
        and _less(adjusted[family], family_alpha)
        and _less(MINIMUM_EFFECT, family_bounds[family].lower)
        and family not in invalid_compute
    ]
    negative_minimum = Rational(
        -MINIMUM_EFFECT.numerator,
        MINIMUM_EFFECT.denominator,
    )
    regressed = [
        family
        for family, _observations in families
        if _less(family_bounds[family].upper, negative_minimum)
    ]
    noninferior = [
        family
        for family, _observations in families
        if _less(negative_minimum, family_bounds[family].lower)
    ]

    pooled_wins = pooled_differences.count(1)
    pooled_losses = pooled_differences.count(-1)
    pooled_ties = len(pooled_differences) - pooled_wins - pooled_losses
    pooled_effect = Rational(sum(pooled_differences), len(pooled_differences))
    try:
        pooled_bounds = certified_rational_effect_bounds(
            pooled_wins,
            pooled_losses,
            pooled_ties,
            family_count=global_bound_family_count,
            family_alpha=family_alpha,
            precision_bits=BOUND_PRECISION_BITS,
        )
        pooled_tail = exact_paired_binomial_tail(
            pooled_wins,
            pooled_losses,
        )
    except ExactStatisticsError as exc:
        raise ExactPairedGradeError("exact_grade_statistics_failed") from exc
    pooled_p = Rational(pooled_tail.numerator, pooled_tail.denominator)
    pooled_positive = (
        len(pooled_differences) >= MIN_OBSERVATIONS_FOR_VERDICT
        and _less(pooled_p, family_alpha)
        and _less(MINIMUM_EFFECT, pooled_bounds.lower)
    )
    required_positive = max(2, (2 * len(families) + 2) // 3)
    if (
        len(positive) >= required_positive
        and pooled_positive
        and not regressed
    ):
        tier = PROVEN
    elif positive and pooled_positive and not regressed:
        tier = SUPPORTED
    elif regressed or _less_or_equal(pooled_bounds.upper, Rational(0, 1)):
        tier = REFUTED
    elif invalid_compute or underpowered:
        tier = CONJECTURE
    else:
        tier = CONJECTURE

    evidence = {
        "schema": EXACT_PAIRED_COMPARISON_SCHEMA,
        "method": EXACT_GRADE_METHOD,
        "treatment": treatment,
        "control": control,
        "alpha": _rational(family_alpha),
        "minimum_effect": _rational(MINIMUM_EFFECT),
        "require_compute": require_compute,
        "compute_tolerance": _rational(compute_tolerance),
        "global_bound_family_count": global_bound_family_count,
        "bound_precision_bits": BOUND_PRECISION_BITS,
        "families": family_material,
        "holm": {
            "method": holm_method,
            "family_size": holm_family_size,
            "ordered": holm_entries,
        },
        "positive_families": positive,
        "noninferior_families": noninferior,
        "all_families_noninferior": len(noninferior) == len(families),
        "regressed_families": regressed,
        "underpowered_families": underpowered,
        "invalid_compute_families": invalid_compute,
        "required_positive_families": required_positive,
        "pooled": {
            "n": len(pooled_differences),
            "treatment_wins": pooled_wins,
            "control_wins": pooled_losses,
            "ties": pooled_ties,
            "paired_effect": _rational(pooled_effect),
            "effect_bounds": _bound_payload(pooled_bounds),
            "one_sided_exact_p": _rational(pooled_p),
        },
    }
    return {
        "experiment": experiment,
        "statement": statement,
        "tier": tier,
        "evidence": evidence,
    }


def grade_exact_interaction(
    *,
    adapter_differences: Sequence[int],
    base_differences: Sequence[int],
    global_bound_family_count: int,
    family_alpha: Rational = ALPHA,
) -> dict[str, Any]:
    """Certify the paired 2x2 interaction and its exact one-sided test."""

    if (
        isinstance(adapter_differences, (str, bytes))
        or isinstance(base_differences, (str, bytes))
        or not isinstance(adapter_differences, Sequence)
        or not isinstance(base_differences, Sequence)
        or not adapter_differences
        or len(adapter_differences) != len(base_differences)
        or type(global_bound_family_count) is not int
        or global_bound_family_count < 2
        or type(family_alpha) is not Rational
        or not _less(Rational(0, 1), family_alpha)
        or _less(Rational(1, 2), family_alpha)
    ):
        _fail("exact_interaction_values_invalid")
    adapter = tuple(adapter_differences)
    base = tuple(base_differences)
    if any(type(value) is not int or value not in {-1, 0, 1} for value in adapter):
        _fail("exact_interaction_values_invalid")
    if any(type(value) is not int or value not in {-1, 0, 1} for value in base):
        _fail("exact_interaction_values_invalid")
    interaction = tuple(
        adapter_value - base_value
        for adapter_value, base_value in zip(adapter, base, strict=True)
    )
    try:
        adapter_bounds = certified_rational_effect_bounds(
            adapter.count(1),
            adapter.count(-1),
            adapter.count(0),
            family_count=global_bound_family_count,
            family_alpha=family_alpha,
            precision_bits=BOUND_PRECISION_BITS,
        )
        base_bounds = certified_rational_effect_bounds(
            base.count(1),
            base.count(-1),
            base.count(0),
            family_count=global_bound_family_count,
            family_alpha=family_alpha,
            precision_bits=BOUND_PRECISION_BITS,
        )
        sign_flip = exact_sign_flip_tail(interaction)
    except ExactStatisticsError as exc:
        raise ExactPairedGradeError(
            "exact_interaction_statistics_failed"
        ) from exc
    lower = _subtract(adapter_bounds.lower, base_bounds.upper)
    upper = _subtract(adapter_bounds.upper, base_bounds.lower)
    if _less(upper, lower):
        _fail("exact_interaction_bounds_inverted")
    effect = Rational(sum(interaction), len(interaction))
    pvalue = Rational(sign_flip.numerator, sign_flip.denominator)
    return {
        "schema": EXACT_INTERACTION_SCHEMA,
        "method": EXACT_INTERACTION_METHOD,
        "n": len(interaction),
        "mean": _rational(effect),
        "lower": _rational(lower),
        "upper": _rational(upper),
        "alpha": _rational(family_alpha),
        "minimum_effect": _rational(MINIMUM_EFFECT),
        "global_bound_family_count": global_bound_family_count,
        "simultaneous_coverage_lower": _rational(
            Rational(
                family_alpha.denominator - family_alpha.numerator,
                family_alpha.denominator,
            )
        ),
        "one_sided_exact_sign_flip_p": _rational(pvalue),
        "sign_flip_assumption": SIGN_FLIP_ASSUMPTION,
        "sign_flip_assumption_preregistered": True,
        "sign_flip": {
            "threshold": sign_flip.threshold,
            "observations": sign_flip.observations,
            "observed_sum": sign_flip.observed_sum,
            "total_assignments": sign_flip.total_assignments,
        },
        "adapter_contrast_bounds": _bound_payload(adapter_bounds),
        "base_contrast_bounds": _bound_payload(base_bounds),
        "interaction_values": list(interaction),
    }


def minimum_zero_loss_noninferiority_observations(
    *,
    global_bound_family_count: int,
    margin: Rational = MINIMUM_EFFECT,
    family_alpha: Rational = ALPHA,
) -> dict[str, Any]:
    """Certify the minimum all-tie sample that clears non-inferiority."""

    if (
        type(global_bound_family_count) is not int
        or global_bound_family_count <= 0
        or type(margin) is not Rational
        or margin.numerator <= 0
        or margin.numerator >= margin.denominator
        or type(family_alpha) is not Rational
        or not _less(Rational(0, 1), family_alpha)
        or _less(Rational(1, 2), family_alpha)
    ):
        _fail("exact_noninferiority_power_contract_invalid")
    negative_margin = Rational(-margin.numerator, margin.denominator)

    def powered(observations: int) -> tuple[bool, CertifiedEffectBounds]:
        try:
            bounds = certified_rational_effect_bounds(
                0,
                0,
                observations,
                family_count=global_bound_family_count,
                family_alpha=family_alpha,
                precision_bits=BOUND_PRECISION_BITS,
            )
        except ExactStatisticsError as exc:
            raise ExactPairedGradeError(
                "exact_noninferiority_power_failed"
            ) from exc
        return _less(negative_margin, bounds.lower), bounds

    low = 0
    high = 1
    while high < MAX_CERTIFIED_BOUND_TRIALS:
        attainable, _bounds = powered(high)
        if attainable:
            break
        low = high
        high = min(high * 2, MAX_CERTIFIED_BOUND_TRIALS)
    attainable, _bounds = powered(high)
    if not attainable:
        _fail("exact_noninferiority_power_unattainable_within_resource_bound")
    while low + 1 < high:
        middle = (low + high) // 2
        passes, _bounds = powered(middle)
        if passes:
            high = middle
        else:
            low = middle
    passes, selected_bounds = powered(high)
    if not passes:
        _fail("exact_noninferiority_power_failed")
    if high > 1:
        prior_passes, prior_bounds = powered(high - 1)
        if prior_passes:
            _fail("exact_noninferiority_power_minimum_not_proven")
        prior_upper = prior_bounds.upper
        prior_lower = prior_bounds.lower
    else:
        prior_upper = None
        prior_lower = None
    return {
        "schema": EXACT_NONINFERIORITY_POWER_SCHEMA,
        "certified": True,
        "global_bound_family_count": global_bound_family_count,
        "margin": _rational(margin),
        "minimum_observations": high,
        "selected_lower": _rational(selected_bounds.lower),
        "selected_upper": _rational(selected_bounds.upper),
        "prior_observations": high - 1 if high > 1 else None,
        "prior_lower": (
            _rational(prior_lower) if prior_lower is not None else None
        ),
        "prior_upper": (
            _rational(prior_upper) if prior_upper is not None else None
        ),
        "precision_bits": BOUND_PRECISION_BITS,
        "resource_max_observations": MAX_CERTIFIED_BOUND_TRIALS,
    }


def exact_campaign_power_plan(
    *,
    domain_count: int,
    comparison_count: int,
    arm_count: int,
    planned_observations_per_domain: int,
) -> dict[str, Any]:
    """Return the complete exact-power receipt bound into a campaign plan."""

    if (
        type(arm_count) is not int
        or arm_count <= 0
        or type(planned_observations_per_domain) is not int
        or planned_observations_per_domain <= 0
    ):
        _fail("exact_campaign_power_contract_invalid")
    global_count = campaign_global_bound_family_count(
        domain_count=domain_count,
        comparison_count=comparison_count,
    )
    receipt = minimum_zero_loss_noninferiority_observations(
        global_bound_family_count=global_count,
    )
    planned_tasks = planned_observations_per_domain * domain_count
    return {
        **receipt,
        "domain_count": domain_count,
        "comparison_count": comparison_count,
        "planned_observations_per_domain": planned_observations_per_domain,
        "planned_total_tasks": planned_tasks,
        "planned_total_cells": planned_tasks * arm_count,
        "powered_for_zero_loss_noninferiority": (
            planned_observations_per_domain >= receipt["minimum_observations"]
        ),
    }


def exact_group_sequential_power_plan(
    *,
    domain_count: int,
    comparison_count: int,
    arm_count: int,
    look_observations_per_domain: Sequence[int],
    alpha_weights: Sequence[Rational],
) -> dict[str, Any]:
    """Freeze exact Bonferroni spending across preregistered campaign looks.

    Each look receives a disjoint rational share of the campaign familywise
    alpha. Reaching a look does not transfer or recycle unused alpha. This
    makes every look independently replayable and keeps optional stopping from
    inflating the complete certificate's error budget.
    """

    if (
        type(arm_count) is not int
        or arm_count <= 0
        or isinstance(look_observations_per_domain, (str, bytes))
        or not isinstance(look_observations_per_domain, Sequence)
        or isinstance(alpha_weights, (str, bytes))
        or not isinstance(alpha_weights, Sequence)
    ):
        _fail("exact_group_sequential_power_contract_invalid")
    observations = tuple(look_observations_per_domain)
    weights = tuple(alpha_weights)
    if (
        not observations
        or len(observations) != len(weights)
        or any(type(value) is not int or value <= 0 for value in observations)
        or any(
            current <= previous
            for previous, current in zip(
                observations,
                observations[1:],
                strict=False,
            )
        )
        or any(
            type(weight) is not Rational
            or not _less(Rational(0, 1), weight)
            for weight in weights
        )
    ):
        _fail("exact_group_sequential_power_contract_invalid")
    total_weight = Rational(0, 1)
    for weight in weights:
        total_weight = _add(total_weight, weight)
    if total_weight != Rational(1, 1):
        _fail("exact_group_sequential_alpha_not_conserved")

    global_count = campaign_global_bound_family_count(
        domain_count=domain_count,
        comparison_count=comparison_count,
    )
    fixed_terminal = exact_campaign_power_plan(
        domain_count=domain_count,
        comparison_count=comparison_count,
        arm_count=arm_count,
        planned_observations_per_domain=observations[-1],
    )
    looks: list[dict[str, Any]] = []
    for index, (planned, weight) in enumerate(
        zip(observations, weights, strict=True),
        start=1,
    ):
        look_alpha = _multiply(ALPHA, weight)
        receipt = minimum_zero_loss_noninferiority_observations(
            global_bound_family_count=global_count,
            family_alpha=look_alpha,
        )
        planned_tasks = planned * domain_count
        looks.append(
            {
                "look": index,
                "observations_per_domain": planned,
                "alpha_weight": _rational(weight),
                "family_alpha": _rational(look_alpha),
                "minimum_observations": receipt["minimum_observations"],
                "powered_for_zero_loss_noninferiority": (
                    planned >= receipt["minimum_observations"]
                ),
                "planned_total_tasks": planned_tasks,
                "planned_total_cells": planned_tasks * arm_count,
                "power_receipt": receipt,
            }
        )
    return {
        "schema": EXACT_GROUP_SEQUENTIAL_POWER_SCHEMA,
        "certified": True,
        "method": "preregistered disjoint-look Bonferroni alpha spending",
        "stopping_rule": "evaluate only declared cumulative looks; no alpha recycling",
        "domain_count": domain_count,
        "comparison_count": comparison_count,
        "arm_count": arm_count,
        "familywise_alpha": _rational(ALPHA),
        "alpha_weight_sum": _rational(total_weight),
        "look_count": len(looks),
        "looks": looks,
        "terminal_fixed_design": fixed_terminal,
        "all_looks_powered_for_zero_loss_noninferiority": all(
            look["powered_for_zero_loss_noninferiority"] for look in looks
        ),
        "terminal_look_powered_for_zero_loss_noninferiority": looks[-1][
            "powered_for_zero_loss_noninferiority"
        ],
    }


def exact_interaction_proven(interaction: Mapping[str, Any]) -> bool:
    """Apply the exact preregistered positive interaction gate."""

    try:
        lower = Rational(**interaction["lower"])
        pvalue = Rational(**interaction["one_sided_exact_sign_flip_p"])
        alpha = Rational(**interaction["alpha"])
    except (KeyError, TypeError, ExactStatisticsError):
        _fail("exact_interaction_payload_invalid")
    return _less(MINIMUM_EFFECT, lower) and _less(pvalue, alpha)


def exact_interaction_refuted(interaction: Mapping[str, Any]) -> bool:
    """Return whether the certified interaction upper bound is non-positive."""

    try:
        upper = Rational(**interaction["upper"])
    except (KeyError, TypeError, ExactStatisticsError):
        _fail("exact_interaction_payload_invalid")
    return _less_or_equal(upper, Rational(0, 1))


__all__ = [
    "ALPHA",
    "BOUND_PRECISION_BITS",
    "EFFECT_BOUND_CERTIFICATE_VERSION",
    "EFFECT_BOUND_METHOD",
    "EXACT_GRADE_METHOD",
    "EXACT_GROUP_SEQUENTIAL_POWER_SCHEMA",
    "EXACT_INTERACTION_METHOD",
    "EXACT_INTERACTION_SCHEMA",
    "EXACT_NONINFERIORITY_POWER_SCHEMA",
    "EXACT_PAIRED_COMPARISON_SCHEMA",
    "ExactPairedGradeError",
    "ExactPairedObservation",
    "MINIMUM_EFFECT",
    "SIGN_FLIP_ASSUMPTION",
    "campaign_global_bound_family_count",
    "exact_campaign_power_plan",
    "exact_group_sequential_power_plan",
    "exact_interaction_proven",
    "exact_interaction_refuted",
    "grade_exact_interaction",
    "grade_exact_paired_comparison",
    "minimum_zero_loss_noninferiority_observations",
]
