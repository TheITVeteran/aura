from __future__ import annotations

import itertools
import math
from collections import Counter
from collections.abc import Iterator
from fractions import Fraction

import pytest

import core.brain.llm.latent_cortex.exact_paired_statistics as statistics
from core.brain.llm.latent_cortex.exact_paired_statistics import (
    CertificationError,
    ExactStatisticsError,
    Rational,
    StatisticsResourceError,
    certified_rational_effect_bounds,
    exact_compute_tolerance_decision,
    exact_holm_adjustment,
    exact_paired_binomial_tail,
    exact_sign_flip_distribution,
    exact_sign_flip_tail,
    rational_effect,
)


def _fraction(numerator: int, denominator: int) -> Fraction:
    return Fraction(numerator, denominator)


def _weak_compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remainder in _weak_compositions(total - first, parts - 1):
            yield (first, *remainder)


def _brute_sign_flip(values: tuple[int, ...]) -> Counter[int]:
    return Counter(
        sum(sign * value for sign, value in zip(signs, values, strict=True))
        for signs in itertools.product((-1, 1), repeat=len(values))
    )


def test_rational_is_canonical_and_rejects_bool_or_invalid_denominator() -> None:
    assert Rational(6, 8).as_tuple() == (3, 4)
    assert Rational(-6, 8).as_tuple() == (-3, 4)
    assert Rational(0, 99).as_tuple() == (0, 1)
    with pytest.raises(TypeError):
        Rational(True, 1)
    with pytest.raises(TypeError):
        Rational(1, False)
    with pytest.raises(ExactStatisticsError):
        Rational(1, 0)
    with pytest.raises(StatisticsResourceError):
        Rational(1 << statistics.MAX_RATIONAL_BITS, 1)


def test_effect_and_paired_tail_differential_for_all_counts_through_n12() -> None:
    for observations in range(1, 13):
        for wins in range(observations + 1):
            for losses in range(observations - wins + 1):
                ties = observations - wins - losses
                effect = rational_effect(wins, losses, ties)
                assert _fraction(effect.numerator, effect.denominator) == Fraction(
                    wins - losses, observations
                )
                assert effect.observations == observations

                tail = exact_paired_binomial_tail(wins, losses)
                discordant = wins + losses
                expected_numerator = (
                    1
                    if discordant == 0
                    else sum(
                        math.comb(discordant, k)
                        for k in range(wins, discordant + 1)
                    )
                )
                expected_denominator = 1 if discordant == 0 else 1 << discordant
                assert _fraction(tail.numerator, tail.denominator) == Fraction(
                    expected_numerator, expected_denominator
                )


def test_paired_tail_golden_values() -> None:
    assert exact_paired_binomial_tail(0, 0).numerator == 1
    assert exact_paired_binomial_tail(0, 0).denominator == 1
    tail = exact_paired_binomial_tail(3, 1)
    assert (tail.numerator, tail.denominator) == (5, 16)
    assert exact_paired_binomial_tail(4, 0).numerator == 1
    assert exact_paired_binomial_tail(4, 0).denominator == 16
    assert exact_paired_binomial_tail(0, 4).numerator == 1
    assert exact_paired_binomial_tail(0, 4).denominator == 1


def test_exact_holm_adjustment_is_monotone_and_ties_use_name_order() -> None:
    result = exact_holm_adjustment(
        {
            "zeta": Rational(1, 50),
            "alpha": Rational(1, 100),
            "beta": Rational(1, 50),
            "certain": Rational(1, 1),
        }
    )
    assert [entry.hypothesis for entry in result.ordered] == [
        "alpha",
        "beta",
        "zeta",
        "certain",
    ]
    assert [entry.adjusted.as_tuple() for entry in result.ordered] == [
        (1, 25),
        (3, 50),
        (3, 50),
        (1, 1),
    ]
    assert result.for_hypothesis("beta").rank == 2
    with pytest.raises(KeyError):
        result.for_hypothesis("missing")


def test_exact_holm_matches_fraction_reference_for_permuted_inputs() -> None:
    source = {
        "d": Rational(1, 5),
        "b": Rational(1, 20),
        "c": Rational(1, 20),
        "a": Rational(1, 100),
    }
    expected_order = ["a", "b", "c", "d"]
    expected_adjusted = {
        "a": Fraction(1, 25),
        "b": Fraction(3, 20),
        "c": Fraction(3, 20),
        "d": Fraction(1, 5),
    }
    for permutation in itertools.permutations(source):
        result = exact_holm_adjustment({name: source[name] for name in permutation})
        assert [entry.hypothesis for entry in result.ordered] == expected_order
        assert {
            entry.hypothesis: _fraction(
                entry.adjusted.numerator, entry.adjusted.denominator
            )
            for entry in result.ordered
        } == expected_adjusted


def test_compute_tolerance_exact_boundary_and_plus_one() -> None:
    boundary = exact_compute_tolerance_decision(120, 100)
    assert boundary.tolerance == Rational(1, 5)
    assert boundary.comparison_left == boundary.comparison_right
    assert boundary.within_tolerance is True

    plus_one = exact_compute_tolerance_decision(121, 100)
    assert plus_one.comparison_left == boundary.comparison_left + 5
    assert plus_one.within_tolerance is False

    lower_boundary = exact_compute_tolerance_decision(80, 100)
    assert lower_boundary.within_tolerance is True
    assert exact_compute_tolerance_decision(79, 100).within_tolerance is False
    assert exact_compute_tolerance_decision(
        101,
        100,
        tolerance_numerator=1,
        tolerance_denominator=100,
    ).within_tolerance
    assert not exact_compute_tolerance_decision(
        102,
        100,
        tolerance_numerator=1,
        tolerance_denominator=100,
    ).within_tolerance


def test_sign_flip_histograms_and_tails_differential_through_n8() -> None:
    alphabet = (-2, -1, 0, 1, 2)
    for observations in range(1, 9):
        for counts in _weak_compositions(observations, len(alphabet)):
            values = tuple(
                value
                for value, count in zip(alphabet, counts, strict=True)
                for _ in range(count)
            )
            expected = _brute_sign_flip(values)
            distribution = exact_sign_flip_distribution(values)
            assert distribution.total_assignments == 1 << observations
            assert {
                mass.total: mass.multiplicity for mass in distribution.masses
            } == dict(sorted(expected.items()))

            tail = exact_sign_flip_tail(values)
            expected_tail = sum(
                multiplicity
                for total, multiplicity in expected.items()
                if total >= sum(values)
            )
            assert _fraction(tail.numerator, tail.denominator) == Fraction(
                expected_tail, 1 << observations
            )


def test_sign_flip_threshold_extremes_and_zero_mass() -> None:
    distribution = exact_sign_flip_distribution((0, 0, 1))
    assert [(mass.total, mass.multiplicity) for mass in distribution.masses] == [
        (-1, 4),
        (1, 4),
    ]
    assert exact_sign_flip_tail((1, 2), threshold=-4).numerator == 1
    assert exact_sign_flip_tail((1, 2), threshold=-4).denominator == 1
    assert exact_sign_flip_tail((1, 2), threshold=4).numerator == 0
    assert exact_sign_flip_tail((1, 2), threshold=4).denominator == 1


def test_certified_bounds_cover_every_count_shape_through_n12() -> None:
    family_alpha = Rational(1, 20)
    for observations in range(1, 13):
        for wins in range(observations + 1):
            for losses in range(observations - wins + 1):
                ties = observations - wins - losses
                result = certified_rational_effect_bounds(
                    wins,
                    losses,
                    ties,
                    family_count=3,
                    family_alpha=family_alpha,
                    precision_bits=12,
                )
                assert result.certified is True
                assert result.observations == observations
                assert result.component_alpha == Rational(1, 240)
                assert result.simultaneous_coverage_lower == Rational(19, 20)
                assert result.grid_step == Rational(1, 1 << 12)
                assert result.endpoint_max_outward_rounding == Rational(1, 1 << 11)
                assert _fraction(
                    result.lower.numerator, result.lower.denominator
                ) <= _fraction(result.upper.numerator, result.upper.denominator)
                assert {component.component for component in result.components} == {
                    "win_lower",
                    "win_upper",
                    "loss_lower",
                    "loss_upper",
                }
                for component in result.components:
                    assert component.certified is True
                    if component.tail_kind == "exact-boundary":
                        assert component.tail_probability is None
                        assert component.adjacent_bound is None
                        assert component.adjacent_tail_probability is None
                    else:
                        assert component.tail_probability is not None
                        assert component.adjacent_bound is not None
                        assert component.adjacent_tail_probability is not None
                        assert _fraction(
                            component.tail_probability.numerator,
                            component.tail_probability.denominator,
                        ) <= _fraction(
                            component.component_alpha.numerator,
                            component.component_alpha.denominator,
                        )
                        assert _fraction(
                            component.adjacent_tail_probability.numerator,
                            component.adjacent_tail_probability.denominator,
                        ) > _fraction(
                            component.component_alpha.numerator,
                            component.component_alpha.denominator,
                        )


def test_certified_bounds_have_exact_boundary_components() -> None:
    all_wins = certified_rational_effect_bounds(
        8,
        0,
        0,
        precision_bits=16,
    )
    components = {component.component: component for component in all_wins.components}
    assert components["win_upper"].bound == Rational(1, 1)
    assert components["win_upper"].tail_kind == "exact-boundary"
    assert components["loss_lower"].bound == Rational(0, 1)
    assert components["loss_lower"].tail_kind == "exact-boundary"
    assert all_wins.method == statistics.EFFECT_BOUND_METHOD
    assert (
        all_wins.certificate_version
        == statistics.EFFECT_BOUND_CERTIFICATE_VERSION
    )


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: rational_effect(True, 0, 0), TypeError),
        (lambda: rational_effect(0, 0, 0), ExactStatisticsError),
        (lambda: rational_effect(-1, 0, 0), ExactStatisticsError),
        (
            lambda: rational_effect(statistics.MAX_PAIRED_OBSERVATIONS, 1, 0),
            StatisticsResourceError,
        ),
        (lambda: exact_paired_binomial_tail(False, 0), TypeError),
        (
            lambda: exact_paired_binomial_tail(
                statistics.MAX_EXACT_BINOMIAL_TRIALS,
                1,
            ),
            StatisticsResourceError,
        ),
        (lambda: exact_holm_adjustment({}), ExactStatisticsError),
        (lambda: exact_holm_adjustment({"x": 1}), TypeError),
        (lambda: exact_holm_adjustment({" x": Rational(1, 2)}), ExactStatisticsError),
        (lambda: exact_holm_adjustment({"x": Rational(-1, 2)}), ExactStatisticsError),
        (lambda: exact_holm_adjustment({"x": Rational(2, 1)}), ExactStatisticsError),
        (lambda: exact_compute_tolerance_decision(True, 10), TypeError),
        (lambda: exact_compute_tolerance_decision(0, 10), ExactStatisticsError),
        (
            lambda: exact_compute_tolerance_decision(
                10,
                10,
                tolerance_numerator=2,
                tolerance_denominator=1,
            ),
            ExactStatisticsError,
        ),
        (lambda: exact_sign_flip_distribution(()), ExactStatisticsError),
        (lambda: exact_sign_flip_distribution((True,)), TypeError),
        (
            lambda: exact_sign_flip_distribution(
                (0,) * (statistics.MAX_SIGN_FLIP_OBSERVATIONS + 1)
            ),
            StatisticsResourceError,
        ),
        (
            lambda: exact_sign_flip_distribution(
                (statistics.MAX_SIGN_FLIP_ABSOLUTE_VALUE + 1,)
            ),
            StatisticsResourceError,
        ),
        (
            lambda: certified_rational_effect_bounds(0, 0, 0),
            ExactStatisticsError,
        ),
        (
            lambda: certified_rational_effect_bounds(
                statistics.MAX_CERTIFIED_BOUND_TRIALS,
                1,
                0,
            ),
            StatisticsResourceError,
        ),
        (
            lambda: certified_rational_effect_bounds(1, 0, 0, family_count=True),
            TypeError,
        ),
        (
            lambda: certified_rational_effect_bounds(1, 0, 0, family_alpha=1),
            TypeError,
        ),
        (
            lambda: certified_rational_effect_bounds(
                1,
                0,
                0,
                precision_bits=statistics.MIN_BOUND_PRECISION_BITS - 1,
            ),
            StatisticsResourceError,
        ),
    ],
)
def test_malformed_and_resource_abusive_inputs_fail_closed(
    call: object,
    error: type[BaseException],
) -> None:
    assert callable(call)
    with pytest.raises(error):
        call()


def test_certification_fails_closed_when_exact_bracket_cannot_be_proved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_probability_counts(**_kwargs: object) -> tuple[int, int]:
        return 1, 1

    monkeypatch.setattr(
        statistics,
        "_binomial_probability_counts",
        invalid_probability_counts,
    )
    with pytest.raises(CertificationError, match="not bracketed"):
        certified_rational_effect_bounds(1, 0, 0, precision_bits=8)
