"""Exact, dependency-free statistics for paired RLC campaign certificates.

The APIs in this module deliberately expose reduced numerator/denominator
values.  No floating-point arithmetic, pseudo-randomness, resampling, or
third-party numerical code is used.

Effect bounds use four one-sided Clopper-Pearson inversions.  A paired outcome
is represented by D in {-1, 0, 1}; the marginal win and loss counts are each
binomial.  Lower/upper bounds for P(D=1) and P(D=-1) are inverted separately,
then combined as P(D=1) - P(D=-1).  Bonferroni allocation over four component
bounds and every declared family gives simultaneous coverage of at least
1 - family_alpha.  Irrational binomial roots are rounded outward on a dyadic
grid, and exact integer tail arithmetic certifies both the selected endpoint
and its adjacent grid point.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Final, Literal, final

MAX_RATIONAL_BITS: Final = 262_144
MAX_PAIRED_OBSERVATIONS: Final = 1_000_000
MAX_EXACT_BINOMIAL_TRIALS: Final = 4_096
MAX_CERTIFIED_BOUND_TRIALS: Final = 4_096
MAX_HOLM_HYPOTHESES: Final = 1_024
MAX_HYPOTHESIS_NAME_LENGTH: Final = 256
MAX_COMPUTE_UNITS: Final = (1 << 63) - 1
MAX_SIGN_FLIP_OBSERVATIONS: Final = 4_096
MAX_SIGN_FLIP_ABSOLUTE_VALUE: Final = 100_000
MAX_SIGN_FLIP_TOTAL_MAGNITUDE: Final = 100_000
MAX_SIGN_FLIP_STATES: Final = 200_001
MAX_SIGN_FLIP_TRANSITIONS: Final = 100_000_000
MIN_BOUND_PRECISION_BITS: Final = 4
MAX_BOUND_PRECISION_BITS: Final = 64

EFFECT_BOUND_CERTIFICATE_VERSION: Final = (
    "aura.latent_cortex.exact_paired_effect_bounds.v1"
)
EFFECT_BOUND_METHOD: Final = (
    "four one-sided Clopper-Pearson marginal bounds; Bonferroni over "
    "win/loss x lower/upper x declared families; dyadic outward rounding "
    "with exact binomial-tail witnesses"
)


class ExactStatisticsError(ValueError):
    """Base class for exact-statistics validation failures."""


class StatisticsResourceError(ExactStatisticsError):
    """The requested exact computation exceeds a declared resource bound."""


class CertificationError(ExactStatisticsError):
    """An exact bound could not produce all required proof witnesses."""


def _require_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not bool or another numeric type")
    return value


def _require_nonnegative_count(name: str, value: object) -> int:
    result = _require_int(name, value)
    if result < 0:
        raise ExactStatisticsError(f"{name} must be non-negative")
    if result > MAX_PAIRED_OBSERVATIONS:
        raise StatisticsResourceError(
            f"{name} exceeds MAX_PAIRED_OBSERVATIONS={MAX_PAIRED_OBSERVATIONS}"
        )
    return result


def _validate_total(total: int, *, maximum: int, operation: str) -> None:
    if total > maximum:
        raise StatisticsResourceError(
            f"{operation} supports at most {maximum} observations, got {total}"
        )


@final
@dataclass(frozen=True, slots=True)
class Rational:
    """A canonical reduced rational with a strictly positive denominator."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        numerator = _require_int("numerator", self.numerator)
        denominator = _require_int("denominator", self.denominator)
        if denominator <= 0:
            raise ExactStatisticsError("denominator must be positive")
        if (
            numerator.bit_length() > MAX_RATIONAL_BITS
            or denominator.bit_length() > MAX_RATIONAL_BITS
        ):
            raise StatisticsResourceError(
                f"rational inputs may not exceed {MAX_RATIONAL_BITS} bits"
            )
        if numerator == 0:
            object.__setattr__(self, "denominator", 1)
            return
        divisor = math.gcd(abs(numerator), denominator)
        object.__setattr__(self, "numerator", numerator // divisor)
        object.__setattr__(self, "denominator", denominator // divisor)

    def as_tuple(self) -> tuple[int, int]:
        return self.numerator, self.denominator


ZERO: Final = Rational(0, 1)
ONE: Final = Rational(1, 1)
NEGATIVE_ONE: Final = Rational(-1, 1)
DEFAULT_FAMILY_ALPHA: Final = Rational(1, 20)


def _fraction(value: Rational) -> Fraction:
    return Fraction(value.numerator, value.denominator)


def _add(left: Rational, right: Rational) -> Rational:
    return Rational(
        left.numerator * right.denominator + right.numerator * left.denominator,
        left.denominator * right.denominator,
    )


def _subtract(left: Rational, right: Rational) -> Rational:
    return Rational(
        left.numerator * right.denominator - right.numerator * left.denominator,
        left.denominator * right.denominator,
    )


def _multiply_by_int(value: Rational, multiplier: int) -> Rational:
    return Rational(value.numerator * multiplier, value.denominator)


def _divide_by_int(value: Rational, divisor: int) -> Rational:
    return Rational(value.numerator, value.denominator * divisor)


def _less_than(left: Rational, right: Rational) -> bool:
    return left.numerator * right.denominator < right.numerator * left.denominator


def _less_than_or_equal(left: Rational, right: Rational) -> bool:
    return left.numerator * right.denominator <= right.numerator * left.denominator


def _clamp_unit_effect(value: Rational) -> Rational:
    if _less_than(value, NEGATIVE_ONE):
        return NEGATIVE_ONE
    if _less_than(ONE, value):
        return ONE
    return value


@final
@dataclass(frozen=True, slots=True)
class RationalEffect:
    """Observed paired mean effect (wins - losses) / all observations."""

    numerator: int
    denominator: int
    wins: int
    losses: int
    ties: int
    observations: int


def rational_effect(wins: int, losses: int, ties: int = 0) -> RationalEffect:
    """Return the canonical observed effect, including ties in the denominator."""

    checked_wins = _require_nonnegative_count("wins", wins)
    checked_losses = _require_nonnegative_count("losses", losses)
    checked_ties = _require_nonnegative_count("ties", ties)
    observations = checked_wins + checked_losses + checked_ties
    if observations == 0:
        raise ExactStatisticsError("rational effect requires at least one observation")
    _validate_total(
        observations,
        maximum=MAX_PAIRED_OBSERVATIONS,
        operation="rational effect",
    )
    effect = Rational(checked_wins - checked_losses, observations)
    return RationalEffect(
        numerator=effect.numerator,
        denominator=effect.denominator,
        wins=checked_wins,
        losses=checked_losses,
        ties=checked_ties,
        observations=observations,
    )


@final
@dataclass(frozen=True, slots=True)
class ExactPairedBinomialTail:
    """Exact P(Binomial(wins + losses, 1/2) >= wins)."""

    numerator: int
    denominator: int
    wins: int
    losses: int
    discordant: int


@lru_cache(maxsize=8_192, typed=True)
def exact_paired_binomial_tail(wins: int, losses: int) -> ExactPairedBinomialTail:
    """Return the exact one-sided paired binomial tail for treatment superiority."""

    checked_wins = _require_nonnegative_count("wins", wins)
    checked_losses = _require_nonnegative_count("losses", losses)
    discordant = checked_wins + checked_losses
    _validate_total(
        discordant,
        maximum=MAX_EXACT_BINOMIAL_TRIALS,
        operation="exact paired binomial tail",
    )
    if discordant == 0:
        probability = ONE
    else:
        numerator = sum(
            math.comb(discordant, successes)
            for successes in range(checked_wins, discordant + 1)
        )
        probability = Rational(numerator, 1 << discordant)
    return ExactPairedBinomialTail(
        numerator=probability.numerator,
        denominator=probability.denominator,
        wins=checked_wins,
        losses=checked_losses,
        discordant=discordant,
    )


@final
@dataclass(frozen=True, slots=True)
class HolmEntry:
    hypothesis: str
    rank: int
    raw: Rational
    adjusted: Rational


@final
@dataclass(frozen=True, slots=True)
class HolmAdjustment:
    """Holm step-down results in deterministic (raw p, hypothesis) order."""

    method: Literal["Holm step-down, exact rational"]
    family_size: int
    ordered: tuple[HolmEntry, ...]

    def for_hypothesis(self, hypothesis: str) -> HolmEntry:
        if type(hypothesis) is not str:
            raise TypeError("hypothesis must be a string")
        for entry in self.ordered:
            if entry.hypothesis == hypothesis:
                return entry
        raise KeyError(hypothesis)


def _validate_hypothesis_name(name: object) -> str:
    if type(name) is not str:
        raise TypeError("Holm hypothesis names must be strings")
    if not name or name != name.strip():
        raise ExactStatisticsError(
            "Holm hypothesis names must be non-empty and have no edge whitespace"
        )
    if len(name) > MAX_HYPOTHESIS_NAME_LENGTH:
        raise StatisticsResourceError(
            "Holm hypothesis name exceeds "
            f"MAX_HYPOTHESIS_NAME_LENGTH={MAX_HYPOTHESIS_NAME_LENGTH}"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ExactStatisticsError("Holm hypothesis names may not contain control characters")
    return name


def exact_holm_adjustment(p_values: Mapping[str, Rational]) -> HolmAdjustment:
    """Apply exact Holm adjustment with ties ordered by hypothesis name."""

    if not isinstance(p_values, Mapping):
        raise TypeError("p_values must be a mapping of names to Rational values")
    if not p_values:
        raise ExactStatisticsError("Holm adjustment requires at least one hypothesis")
    if len(p_values) > MAX_HOLM_HYPOTHESES:
        raise StatisticsResourceError(
            f"Holm adjustment supports at most {MAX_HOLM_HYPOTHESES} hypotheses"
        )

    validated: list[tuple[str, Rational]] = []
    for raw_name, raw_probability in p_values.items():
        name = _validate_hypothesis_name(raw_name)
        if type(raw_probability) is not Rational:
            raise TypeError(f"p-value for {name!r} must be a Rational")
        if _less_than(raw_probability, ZERO) or _less_than(ONE, raw_probability):
            raise ExactStatisticsError(f"p-value for {name!r} must be inside [0, 1]")
        validated.append((name, raw_probability))

    ordered_values = sorted(
        validated,
        key=lambda item: (_fraction(item[1]), item[0]),
    )
    running = ZERO
    entries: list[HolmEntry] = []
    family_size = len(ordered_values)
    for index, (name, raw_probability) in enumerate(ordered_values):
        scaled = _multiply_by_int(raw_probability, family_size - index)
        if _less_than(ONE, scaled):
            scaled = ONE
        if _less_than(running, scaled):
            running = scaled
        entries.append(
            HolmEntry(
                hypothesis=name,
                rank=index + 1,
                raw=raw_probability,
                adjusted=running,
            )
        )
    return HolmAdjustment(
        method="Holm step-down, exact rational",
        family_size=family_size,
        ordered=tuple(entries),
    )


@final
@dataclass(frozen=True, slots=True)
class ComputeToleranceDecision:
    """Exact asymmetric compute comparison relative to the control cost."""

    treatment: int
    control: int
    tolerance: Rational
    absolute_difference: int
    comparison_left: int
    comparison_right: int
    within_tolerance: bool


def exact_compute_tolerance_decision(
    treatment: int,
    control: int,
    *,
    tolerance_numerator: int = 1,
    tolerance_denominator: int = 5,
) -> ComputeToleranceDecision:
    """Decide abs(treatment-control)/control <= tolerance without division."""

    checked_treatment = _require_int("treatment", treatment)
    checked_control = _require_int("control", control)
    if checked_treatment <= 0 or checked_control <= 0:
        raise ExactStatisticsError("treatment and control compute must both be positive")
    if checked_treatment > MAX_COMPUTE_UNITS or checked_control > MAX_COMPUTE_UNITS:
        raise StatisticsResourceError(
            f"compute values may not exceed MAX_COMPUTE_UNITS={MAX_COMPUTE_UNITS}"
        )
    tolerance = Rational(tolerance_numerator, tolerance_denominator)
    if _less_than(tolerance, ZERO) or _less_than(ONE, tolerance):
        raise ExactStatisticsError("compute tolerance must be inside [0, 1]")
    difference = abs(checked_treatment - checked_control)
    comparison_left = difference * tolerance.denominator
    comparison_right = checked_control * tolerance.numerator
    return ComputeToleranceDecision(
        treatment=checked_treatment,
        control=checked_control,
        tolerance=tolerance,
        absolute_difference=difference,
        comparison_left=comparison_left,
        comparison_right=comparison_right,
        within_tolerance=comparison_left <= comparison_right,
    )


@final
@dataclass(frozen=True, slots=True)
class SignFlipMass:
    total: int
    multiplicity: int


@final
@dataclass(frozen=True, slots=True)
class ExactSignFlipDistribution:
    observations: int
    observed_sum: int
    total_assignments: int
    masses: tuple[SignFlipMass, ...]


@final
@dataclass(frozen=True, slots=True)
class ExactSignFlipTail:
    numerator: int
    denominator: int
    threshold: int
    observations: int
    observed_sum: int
    total_assignments: int


def _validated_sign_flip_values(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("sign-flip values must be a finite sequence of integers")
    materialized = tuple(values)
    if not materialized:
        raise ExactStatisticsError("sign-flip statistics require at least one value")
    if len(materialized) > MAX_SIGN_FLIP_OBSERVATIONS:
        raise StatisticsResourceError(
            "sign-flip statistics support at most "
            f"{MAX_SIGN_FLIP_OBSERVATIONS} observations"
        )
    magnitude = 0
    for index, value in enumerate(materialized):
        checked = _require_int(f"values[{index}]", value)
        if abs(checked) > MAX_SIGN_FLIP_ABSOLUTE_VALUE:
            raise StatisticsResourceError(
                "sign-flip value exceeds "
                f"MAX_SIGN_FLIP_ABSOLUTE_VALUE={MAX_SIGN_FLIP_ABSOLUTE_VALUE}"
            )
        magnitude += abs(checked)
    if magnitude > MAX_SIGN_FLIP_TOTAL_MAGNITUDE:
        raise StatisticsResourceError(
            "sign-flip total magnitude exceeds "
            f"MAX_SIGN_FLIP_TOTAL_MAGNITUDE={MAX_SIGN_FLIP_TOTAL_MAGNITUDE}"
        )
    return materialized


def exact_sign_flip_distribution(values: Sequence[int]) -> ExactSignFlipDistribution:
    """Enumerate the exact random-sign sum distribution using dynamic programming."""

    materialized = _validated_sign_flip_values(values)
    histogram: dict[int, int] = {0: 1}
    transitions = 0
    for value in materialized:
        transitions += 2 * len(histogram)
        if transitions > MAX_SIGN_FLIP_TRANSITIONS:
            raise StatisticsResourceError(
                "sign-flip dynamic program exceeds "
                f"MAX_SIGN_FLIP_TRANSITIONS={MAX_SIGN_FLIP_TRANSITIONS}"
            )
        updated: dict[int, int] = {}
        for current, multiplicity in histogram.items():
            positive = current + value
            negative = current - value
            updated[positive] = updated.get(positive, 0) + multiplicity
            updated[negative] = updated.get(negative, 0) + multiplicity
        if len(updated) > MAX_SIGN_FLIP_STATES:
            raise StatisticsResourceError(
                "sign-flip dynamic program exceeds "
                f"MAX_SIGN_FLIP_STATES={MAX_SIGN_FLIP_STATES}"
            )
        histogram = updated
    total_assignments = 1 << len(materialized)
    if sum(histogram.values()) != total_assignments:
        raise CertificationError("sign-flip dynamic-program mass is not conserved")
    return ExactSignFlipDistribution(
        observations=len(materialized),
        observed_sum=sum(materialized),
        total_assignments=total_assignments,
        masses=tuple(
            SignFlipMass(total=total, multiplicity=multiplicity)
            for total, multiplicity in sorted(histogram.items())
        ),
    )


def exact_sign_flip_tail(
    values: Sequence[int],
    *,
    threshold: int | None = None,
) -> ExactSignFlipTail:
    """Return exact P(random-sign sum >= threshold); default is the observed sum."""

    distribution = exact_sign_flip_distribution(values)
    if threshold is None:
        checked_threshold = distribution.observed_sum
    else:
        checked_threshold = _require_int("threshold", threshold)
        if abs(checked_threshold) > MAX_SIGN_FLIP_TOTAL_MAGNITUDE + 1:
            raise StatisticsResourceError("sign-flip threshold exceeds the supported range")
    numerator = sum(
        mass.multiplicity
        for mass in distribution.masses
        if mass.total >= checked_threshold
    )
    probability = Rational(numerator, distribution.total_assignments)
    return ExactSignFlipTail(
        numerator=probability.numerator,
        denominator=probability.denominator,
        threshold=checked_threshold,
        observations=distribution.observations,
        observed_sum=distribution.observed_sum,
        total_assignments=distribution.total_assignments,
    )


TailKind = Literal["upper", "lower", "exact-boundary"]
BoundKind = Literal["lower", "upper"]


@final
@dataclass(frozen=True, slots=True)
class BinomialBoundCertificate:
    """Exact witness for one outward-rounded binomial proportion bound."""

    component: Literal["win_lower", "win_upper", "loss_lower", "loss_upper"]
    bound_kind: BoundKind
    tail_kind: TailKind
    successes: int
    trials: int
    bound: Rational
    tail_probability: Rational | None
    component_alpha: Rational
    adjacent_bound: Rational | None
    adjacent_tail_probability: Rational | None
    precision_bits: int
    certified: bool


@final
@dataclass(frozen=True, slots=True)
class CertifiedEffectBounds:
    """Conservative simultaneous rational bounds for E[D], D in {-1,0,1}."""

    certificate_version: str
    method: str
    certified: bool
    wins: int
    losses: int
    ties: int
    observations: int
    lower: Rational
    upper: Rational
    family_count: int
    family_alpha: Rational
    component_alpha: Rational
    simultaneous_coverage_lower: Rational
    precision_bits: int
    grid_step: Rational
    endpoint_max_outward_rounding: Rational
    components: tuple[BinomialBoundCertificate, ...]


def _binomial_probability_counts(
    *,
    trials: int,
    successes: int,
    probability: Rational,
    tail_kind: Literal["upper", "lower"],
) -> tuple[int, int]:
    numerator_p = probability.numerator
    denominator_p = probability.denominator
    numerator_q = denominator_p - numerator_p
    denominator = pow(denominator_p, trials)
    if numerator_p == 0:
        mass_at_zero = denominator
        if tail_kind == "upper":
            return (mass_at_zero if successes == 0 else 0), denominator
        return mass_at_zero, denominator
    if numerator_q == 0:
        mass_at_trials = denominator
        if tail_kind == "upper":
            return mass_at_trials, denominator
        return (
            mass_at_trials if successes == trials else 0
        ), denominator

    first = successes if tail_kind == "upper" else 0
    last = trials if tail_kind == "upper" else successes
    term = pow(numerator_q, trials)
    numerator = 0
    for count in range(trials + 1):
        if first <= count <= last:
            numerator += term
        if count == trials:
            break
        recurrence_numerator = (
            term * (trials - count) * numerator_p
        )
        recurrence_denominator = (count + 1) * numerator_q
        quotient, remainder = divmod(
            recurrence_numerator,
            recurrence_denominator,
        )
        if remainder:
            raise CertificationError(
                "binomial adjacent-term recurrence is not integral"
            )
        term = quotient
    return numerator, denominator


def _counts_leq_rational(
    numerator: int,
    denominator: int,
    threshold: Rational,
) -> bool:
    return numerator * threshold.denominator <= threshold.numerator * denominator


@lru_cache(maxsize=16_384, typed=True)
def _certified_proportion_bound(
    *,
    component: Literal["win_lower", "win_upper", "loss_lower", "loss_upper"],
    successes: int,
    trials: int,
    bound_kind: BoundKind,
    component_alpha: Rational,
    precision_bits: int,
) -> BinomialBoundCertificate:
    if bound_kind == "lower" and successes == 0:
        return BinomialBoundCertificate(
            component=component,
            bound_kind=bound_kind,
            tail_kind="exact-boundary",
            successes=successes,
            trials=trials,
            bound=ZERO,
            tail_probability=None,
            component_alpha=component_alpha,
            adjacent_bound=None,
            adjacent_tail_probability=None,
            precision_bits=precision_bits,
            certified=True,
        )
    if bound_kind == "upper" and successes == trials:
        return BinomialBoundCertificate(
            component=component,
            bound_kind=bound_kind,
            tail_kind="exact-boundary",
            successes=successes,
            trials=trials,
            bound=ONE,
            tail_probability=None,
            component_alpha=component_alpha,
            adjacent_bound=None,
            adjacent_tail_probability=None,
            precision_bits=precision_bits,
            certified=True,
        )

    scale = 1 << precision_bits
    if bound_kind == "lower":
        tail_kind: Literal["upper", "lower"] = "upper"
        low, high = 0, scale
        low_counts = _binomial_probability_counts(
            trials=trials,
            successes=successes,
            probability=ZERO,
            tail_kind=tail_kind,
        )
        high_counts = _binomial_probability_counts(
            trials=trials,
            successes=successes,
            probability=ONE,
            tail_kind=tail_kind,
        )
        if not _counts_leq_rational(*low_counts, component_alpha) or _counts_leq_rational(
            *high_counts, component_alpha
        ):
            raise CertificationError("lower-bound inversion endpoints are not bracketed")
        while low + 1 < high:
            middle = (low + high) // 2
            counts = _binomial_probability_counts(
                trials=trials,
                successes=successes,
                probability=Rational(middle, scale),
                tail_kind=tail_kind,
            )
            if _counts_leq_rational(*counts, component_alpha):
                low = middle
            else:
                high = middle
        selected_index, adjacent_index = low, high
    else:
        tail_kind = "lower"
        low, high = 0, scale
        low_counts = _binomial_probability_counts(
            trials=trials,
            successes=successes,
            probability=ZERO,
            tail_kind=tail_kind,
        )
        high_counts = _binomial_probability_counts(
            trials=trials,
            successes=successes,
            probability=ONE,
            tail_kind=tail_kind,
        )
        if _counts_leq_rational(*low_counts, component_alpha) or not _counts_leq_rational(
            *high_counts, component_alpha
        ):
            raise CertificationError("upper-bound inversion endpoints are not bracketed")
        while low + 1 < high:
            middle = (low + high) // 2
            counts = _binomial_probability_counts(
                trials=trials,
                successes=successes,
                probability=Rational(middle, scale),
                tail_kind=tail_kind,
            )
            if _counts_leq_rational(*counts, component_alpha):
                high = middle
            else:
                low = middle
        selected_index, adjacent_index = high, low

    bound = Rational(selected_index, scale)
    adjacent_bound = Rational(adjacent_index, scale)
    selected_counts = _binomial_probability_counts(
        trials=trials,
        successes=successes,
        probability=bound,
        tail_kind=tail_kind,
    )
    adjacent_counts = _binomial_probability_counts(
        trials=trials,
        successes=successes,
        probability=adjacent_bound,
        tail_kind=tail_kind,
    )
    if not _counts_leq_rational(*selected_counts, component_alpha):
        raise CertificationError("selected binomial endpoint does not satisfy alpha")
    if _counts_leq_rational(*adjacent_counts, component_alpha):
        raise CertificationError("adjacent binomial endpoint does not prove outward rounding")
    return BinomialBoundCertificate(
        component=component,
        bound_kind=bound_kind,
        tail_kind=tail_kind,
        successes=successes,
        trials=trials,
        bound=bound,
        tail_probability=Rational(*selected_counts),
        component_alpha=component_alpha,
        adjacent_bound=adjacent_bound,
        adjacent_tail_probability=Rational(*adjacent_counts),
        precision_bits=precision_bits,
        certified=True,
    )


@lru_cache(maxsize=4_096, typed=True)
def certified_rational_effect_bounds(
    wins: int,
    losses: int,
    ties: int,
    *,
    family_count: int = 1,
    family_alpha: Rational = DEFAULT_FAMILY_ALPHA,
    precision_bits: int = 40,
) -> CertifiedEffectBounds:
    """Certify simultaneous rational bounds for a paired mean effect.

    ``family_count`` must equal the number of families that will share the
    claim.  Four one-sided component errors per family receive
    ``family_alpha / (4 * family_count)``.  The resulting intervals are
    simultaneously valid across all declared families by the union bound.
    """

    checked_wins = _require_nonnegative_count("wins", wins)
    checked_losses = _require_nonnegative_count("losses", losses)
    checked_ties = _require_nonnegative_count("ties", ties)
    observations = checked_wins + checked_losses + checked_ties
    if observations == 0:
        raise ExactStatisticsError("certified effect bounds require observations")
    _validate_total(
        observations,
        maximum=MAX_CERTIFIED_BOUND_TRIALS,
        operation="certified effect bounds",
    )
    checked_family_count = _require_int("family_count", family_count)
    if checked_family_count <= 0:
        raise ExactStatisticsError("family_count must be positive")
    if checked_family_count > MAX_HOLM_HYPOTHESES:
        raise StatisticsResourceError(
            f"family_count may not exceed {MAX_HOLM_HYPOTHESES}"
        )
    if type(family_alpha) is not Rational:
        raise TypeError("family_alpha must be a Rational")
    if not _less_than(ZERO, family_alpha) or _less_than(Rational(1, 2), family_alpha):
        raise ExactStatisticsError("family_alpha must be inside (0, 1/2]")
    checked_precision = _require_int("precision_bits", precision_bits)
    if not MIN_BOUND_PRECISION_BITS <= checked_precision <= MAX_BOUND_PRECISION_BITS:
        raise StatisticsResourceError(
            "precision_bits must be inside "
            f"[{MIN_BOUND_PRECISION_BITS}, {MAX_BOUND_PRECISION_BITS}]"
        )

    component_alpha = _divide_by_int(family_alpha, 4 * checked_family_count)
    components = (
        _certified_proportion_bound(
            component="win_lower",
            successes=checked_wins,
            trials=observations,
            bound_kind="lower",
            component_alpha=component_alpha,
            precision_bits=checked_precision,
        ),
        _certified_proportion_bound(
            component="win_upper",
            successes=checked_wins,
            trials=observations,
            bound_kind="upper",
            component_alpha=component_alpha,
            precision_bits=checked_precision,
        ),
        _certified_proportion_bound(
            component="loss_lower",
            successes=checked_losses,
            trials=observations,
            bound_kind="lower",
            component_alpha=component_alpha,
            precision_bits=checked_precision,
        ),
        _certified_proportion_bound(
            component="loss_upper",
            successes=checked_losses,
            trials=observations,
            bound_kind="upper",
            component_alpha=component_alpha,
            precision_bits=checked_precision,
        ),
    )
    by_component = {component.component: component for component in components}
    if len(by_component) != 4 or not all(component.certified for component in components):
        raise CertificationError("effect-bound component certificate is incomplete")

    lower = _clamp_unit_effect(
        _subtract(
            by_component["win_lower"].bound,
            by_component["loss_upper"].bound,
        )
    )
    upper = _clamp_unit_effect(
        _subtract(
            by_component["win_upper"].bound,
            by_component["loss_lower"].bound,
        )
    )
    if _less_than(upper, lower):
        raise CertificationError("certified effect interval is inverted")

    grid_step = Rational(1, 1 << checked_precision)
    return CertifiedEffectBounds(
        certificate_version=EFFECT_BOUND_CERTIFICATE_VERSION,
        method=EFFECT_BOUND_METHOD,
        certified=True,
        wins=checked_wins,
        losses=checked_losses,
        ties=checked_ties,
        observations=observations,
        lower=lower,
        upper=upper,
        family_count=checked_family_count,
        family_alpha=family_alpha,
        component_alpha=component_alpha,
        simultaneous_coverage_lower=_subtract(ONE, family_alpha),
        precision_bits=checked_precision,
        grid_step=grid_step,
        endpoint_max_outward_rounding=_multiply_by_int(grid_step, 2),
        components=components,
    )


__all__ = [
    "EFFECT_BOUND_CERTIFICATE_VERSION",
    "EFFECT_BOUND_METHOD",
    "DEFAULT_FAMILY_ALPHA",
    "MAX_BOUND_PRECISION_BITS",
    "MAX_CERTIFIED_BOUND_TRIALS",
    "MAX_COMPUTE_UNITS",
    "MAX_EXACT_BINOMIAL_TRIALS",
    "MAX_HOLM_HYPOTHESES",
    "MAX_PAIRED_OBSERVATIONS",
    "MAX_SIGN_FLIP_OBSERVATIONS",
    "BinomialBoundCertificate",
    "CertificationError",
    "CertifiedEffectBounds",
    "ComputeToleranceDecision",
    "ExactPairedBinomialTail",
    "ExactSignFlipDistribution",
    "ExactSignFlipTail",
    "ExactStatisticsError",
    "HolmAdjustment",
    "HolmEntry",
    "Rational",
    "RationalEffect",
    "SignFlipMass",
    "StatisticsResourceError",
    "certified_rational_effect_bounds",
    "exact_compute_tolerance_decision",
    "exact_holm_adjustment",
    "exact_paired_binomial_tail",
    "exact_sign_flip_distribution",
    "exact_sign_flip_tail",
    "rational_effect",
]
