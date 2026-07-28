"""Live event telemetry is untrusted input too.

CP126 8bd58283: present_antigen coerced event fields with
`float(event.get(...)) or ...` and clipped with min/max. NaN is TRUTHY, so
`or` passed it straight through, and min/max PROPAGATE NaN rather than
clipping it. One malformed telemetry field then flowed into danger and
activation — where every threshold comparison against NaN is False, so the
antigen reads as calm and the immune system stands down.
"""
from __future__ import annotations

import math

import pytest

from core.adaptation.adaptive_immunity import (
    _first_unit,
    _optional_unit,
    get_adaptive_immune_system,
)


@pytest.fixture(scope="module")
def immune():
    return get_adaptive_immune_system()


def _pressures(antigen):
    return (
        antigen.danger,
        antigen.resource_pressure,
        antigen.error_load,
        antigen.threat_probability,
        antigen.subsystem_need,
    )


@pytest.mark.parametrize(
    "event",
    [
        {"resource_pressure": float("nan")},
        {"resource_pressure": float("inf")},
        {"cpu": float("nan")},
        {"ram": float("inf")},
        {"error_rate": float("nan")},
        {"error_count": float("nan")},
        {"threat_probability": float("nan")},
        {"danger": float("nan")},
        {"error_count": -500},
        {"error_rate": -1.0},
        {"error_rate": 1e12},
        {"cpu": 1e9},
        {"resource_pressure": "not a number"},
        {"error_count": None},
    ],
)
def test_adversarial_telemetry_never_produces_a_non_finite_pressure(immune, event):
    antigen = immune.present_antigen({"subsystem": "memory", **event})

    for value in _pressures(antigen):
        assert math.isfinite(value), f"{value} is not finite"
        assert 0.0 <= value <= 1.0


def test_a_nan_never_reads_as_calm_by_propagation(immune):
    """The failure mode: nan > threshold is False, so the system stands down."""
    antigen = immune.present_antigen(
        {"subsystem": "memory", "threat_probability": float("nan")}
    )

    assert antigen.danger == antigen.danger      # not NaN
    assert antigen.danger >= 0.0


def test_ordinary_telemetry_is_unchanged(immune):
    antigen = immune.present_antigen(
        {"subsystem": "memory", "resource_pressure": 0.6, "error_rate": 0.2}
    )

    assert antigen.resource_pressure == pytest.approx(0.6)
    assert antigen.error_load == pytest.approx(0.2)


def test_a_high_but_valid_reading_still_raises_danger(immune):
    """Hardening must not flatten real signal into zero."""
    calm = immune.present_antigen({"subsystem": "memory", "error_rate": 0.0})
    loud = immune.present_antigen({"subsystem": "memory", "error_rate": 1.0})

    assert loud.error_load > calm.error_load
    assert loud.danger > calm.danger


# --- the coercion helpers ------------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "x", None, object()])
def test_optional_unit_rejects_unusable_values(bad):
    assert _optional_unit(bad) is None


@pytest.mark.parametrize("value,expected", [(0.5, 0.5), (2.0, 1.0), (-1.0, 0.0)])
def test_optional_unit_clamps_usable_ones(value, expected):
    assert _optional_unit(value) == pytest.approx(expected)


def test_absent_is_distinct_from_zero():
    """Falling through to the computed default matters: treating a missing
    field as 0.0 would read as 'no pressure' rather than 'not reported'."""
    assert _optional_unit(None) is None
    assert _optional_unit(0.0) == 0.0


def test_first_unit_skips_unusable_candidates():
    assert _first_unit(float("nan"), None, 0.4, default=0.9) == pytest.approx(0.4)


def test_first_unit_falls_back_to_the_default():
    assert _first_unit(float("nan"), None, default=0.3) == pytest.approx(0.3)


def test_first_unit_bounds_even_the_default():
    assert _first_unit(None, default=99.0) == 1.0
