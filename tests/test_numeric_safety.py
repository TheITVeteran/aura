"""Contract tests for the shared validated-scalar primitives.

These pin the CP126 numeric-safety defect class: a NaN makes every threshold
comparison False, so unvalidated inputs fall through to the cheapest branch.
"""
from __future__ import annotations

import math

import pytest

from core.runtime.numeric_safety import (
    all_faults,
    clamp,
    is_usable,
    safe_mean,
    safe_ratio,
    validated_int,
    validated_positive,
    validated_probability,
    validated_scalar,
    validated_sequence,
    validated_unit,
)


# --- the defect this module exists to remove ------------------------------


def test_a_nan_defeats_every_raw_threshold_comparison():
    """The premise. Both branches are False, so raw code falls through."""
    nan = float("nan")
    assert (nan >= 0.75) is False
    assert (nan < 0.75) is False


def test_a_validated_nan_reaches_the_cautious_branch():
    risk = validated_unit(float("nan"), name="risk", cautious_high=True)

    assert risk >= 0.75
    assert risk.repaired is True
    assert "NaN" in risk.fault


def test_a_validated_scalar_is_a_real_float():
    value = validated_unit(0.4)

    assert isinstance(value, float)
    assert value + 0.1 == pytest.approx(0.5)
    assert value < 0.5
    assert math.isfinite(value)


# --- validated_scalar ------------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), None, "high", {}])
def test_unusable_inputs_are_replaced_and_reported(bad):
    result = validated_scalar(bad, name="x", low=0.0, high=1.0, default=0.25)

    assert math.isfinite(result)
    assert result.fault
    assert result.original == bad or result.original is bad


def test_range_violations_are_clamped_with_a_fault():
    low = validated_scalar(-5.0, name="x", low=0.0, high=1.0)
    high = validated_scalar(9.0, name="x", low=0.0, high=1.0)

    assert float(low) == 0.0 and "below range" in low.fault
    assert float(high) == 1.0 and "above range" in high.fault


def test_a_good_value_passes_through_unmarked():
    result = validated_scalar(0.5, name="x", low=0.0, high=1.0)

    assert float(result) == 0.5
    assert result.fault == ""
    assert result.repaired is False


def test_on_unusable_chooses_the_cautious_direction():
    risky = validated_scalar(float("nan"), name="risk", low=0.0, high=1.0, on_unusable=1.0)
    confident = validated_scalar(float("nan"), name="confidence", low=0.0, high=1.0, on_unusable=0.0)

    assert float(risky) == 1.0
    assert float(confident) == 0.0


def test_integer_strings_are_accepted():
    assert float(validated_scalar("0.5", low=0.0, high=1.0)) == 0.5


# --- validated_unit / probability -----------------------------------------


def test_unit_defaults_to_the_permissive_zero():
    assert float(validated_unit(float("nan"))) == 0.0


def test_unit_cautious_high_flips_the_default():
    assert float(validated_unit(float("nan"), cautious_high=True)) == 1.0


def test_probability_treats_unusable_as_no_evidence():
    assert float(validated_probability(None)) == 0.0
    assert float(validated_probability(1.5)) == 1.0


# --- validated_positive ----------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, -0.0001, float("nan"), "soon", None])
def test_a_non_positive_budget_is_replaced(bad):
    result = validated_positive(bad, name="timeout", default=30.0)

    assert float(result) == 30.0
    assert result.fault


def test_a_positive_budget_survives_and_can_be_capped():
    assert float(validated_positive(5.0, default=30.0)) == 5.0
    assert float(validated_positive(9999.0, default=30.0, high=120.0)) == 120.0


# --- validated_int ---------------------------------------------------------


def test_int_validation_clamps_and_reports():
    assert validated_int(7, low=0, high=4) == (4, "value was above range (7); clamped to 4")
    assert validated_int(-2, low=0, high=4)[0] == 0
    assert validated_int("3", low=0, high=4) == (3, "")


@pytest.mark.parametrize("bad", ["deep", None, float("nan"), float("inf"), {}])
def test_non_integer_depth_falls_back(bad):
    value, fault = validated_int(bad, default=1)

    assert value == 1
    assert fault


def test_a_float_int_is_truncated_not_rejected():
    assert validated_int(3.7)[0] == 3


# --- safe_ratio ------------------------------------------------------------


def test_division_by_zero_is_a_reported_default_not_a_crash():
    result = safe_ratio(1.0, 0.0, default=0.0)

    assert float(result) == 0.0
    assert "denominator" in result.fault


def test_a_tiny_denominator_is_treated_as_zero():
    assert safe_ratio(1.0, 1e-18).fault


def test_a_normal_ratio_is_exact():
    result = safe_ratio(3.0, 4.0)

    assert float(result) == pytest.approx(0.75)
    assert result.fault == ""


def test_an_unusable_operand_is_reported():
    assert safe_ratio(float("nan"), 2.0).fault


# --- safe_mean: measured-zero vs measured-nothing --------------------------


def test_an_empty_mean_is_faulted_not_silently_zero():
    result = safe_mean([])

    assert float(result) == 0.0
    assert "no usable values" in result.fault


def test_a_wholly_unusable_mean_is_faulted():
    assert safe_mean([float("nan"), None, "x"]).fault


def test_a_partially_usable_mean_reports_what_was_dropped():
    result = safe_mean([1.0, float("nan"), 3.0])

    assert float(result) == pytest.approx(2.0)
    assert "1 of 3" in result.fault


def test_a_fully_usable_mean_has_no_fault():
    result = safe_mean([1.0, 2.0, 3.0])

    assert float(result) == pytest.approx(2.0)
    assert result.fault == ""


def test_a_genuine_zero_mean_is_not_faulted():
    """The distinction the class is about: measured 0.0 is not 'no data'."""
    result = safe_mean([0.0, 0.0])

    assert float(result) == 0.0
    assert result.fault == ""


# --- helpers ---------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(1.0, True), (0, True), ("0.5", True), (float("nan"), False), (float("inf"), False),
     (None, False), ("x", False)],
)
def test_is_usable(value, expected):
    assert is_usable(value) is expected


def test_clamp_is_a_plain_clamp():
    assert clamp(-1.0) == 0.0
    assert clamp(2.0) == 1.0
    assert clamp(0.5) == 0.5


def test_all_faults_collects_only_real_repairs():
    good = validated_unit(0.5)
    bad = validated_unit(float("nan"))

    faults = all_faults(good, bad, (1, ""), (1, "explicit fault"))

    assert len(faults) == 2
    assert "explicit fault" in faults


def test_validated_sequence_repairs_a_whole_vector():
    values, faults = validated_sequence([0.5, float("nan"), 9.0], name="v")

    assert values == [0.5, 0.0, 1.0]
    assert len(faults) == 2


def test_scalar_carries_its_original_for_receipts():
    result = validated_unit("bogus", cautious_high=True)

    assert result.original == "bogus"
    assert float(result) == 1.0
