"""The no-go, checked against an independent computation rather than quoted."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lawc05.nogo import (  # noqa: E402
    MBAR_PLANCK_GEV,
    black_hole_energy_gev,
    microscope_style_report,
    multi_observable_cost,
    single_field_cost,
    yukawa_form_factor,
)


def _exterior_energy(radius: float, mass: float, delta_phi: float = 1.0) -> float:
    """Solve the Yukawa exterior directly and integrate its energy.

    phi = A e^{-mr}/r with phi(R) = delta_phi. This shares no code with the
    closed form under test, which is the point.
    """
    amplitude = delta_phi * radius * math.exp(mass * radius)

    def integrand(r: float) -> float:
        phi = amplitude * math.exp(-mass * r) / r
        dphi_dr = -amplitude * math.exp(-mass * r) * (1.0 + mass * r) / r**2
        return 0.5 * (dphi_dr**2 + mass**2 * phi**2) * 4.0 * math.pi * r**2

    value, _ = quad(integrand, radius, np.inf, limit=400)
    return value


def _interior_energy(radius: float, mass: float, delta_phi: float = 1.0) -> float:
    """A ball held uniformly at delta_phi: only the potential term survives."""
    return 0.5 * mass**2 * delta_phi**2 * (4.0 / 3.0) * math.pi * radius**3


class TestFormFactorIsNotAssumed:
    @pytest.mark.parametrize("m_r", [0.0, 0.1, 0.5, 1.0, 3.0, 10.0, 30.0])
    def test_it_matches_a_direct_energy_integral(self, m_r: float) -> None:
        radius = 1.0
        numerical = (
            _exterior_energy(radius, m_r) + _interior_energy(radius, m_r)
        ) / (2.0 * math.pi * radius)
        assert numerical == pytest.approx(yukawa_form_factor(m_r), rel=1e-9)

    def test_the_massless_limit_is_pure_gradient_energy(self) -> None:
        assert yukawa_form_factor(0.0) == 1.0

    def test_it_grows_quadratically_at_large_mass(self) -> None:
        """The interior volume term takes over, which is why big regions lose."""
        assert yukawa_form_factor(100.0) == pytest.approx(1e4 / 3.0, rel=0.05)

    def test_a_negative_or_infinite_argument_is_refused(self) -> None:
        with pytest.raises(ValueError):
            yukawa_form_factor(-1.0)
        with pytest.raises(ValueError):
            yukawa_form_factor(float("inf"))


class TestSingleFieldCost:
    def test_the_microscope_example_reproduces_one_half(self) -> None:
        """B ~ 1e-6, a 1 ppm target: half a black hole, as claimed."""
        report = microscope_style_report(coupling_b=1e-6, delta_log_constant=1e-6)
        assert report["delta_phi_over_mbar"] == pytest.approx(1.0)
        assert report["energy_fraction_of_black_hole"] == pytest.approx(0.5)

    def test_cost_is_quadratic_in_the_offset(self) -> None:
        one = single_field_cost(delta_phi_over_mbar=1e-3)
        two = single_field_cost(delta_phi_over_mbar=2e-3)
        assert two.energy_fraction_of_black_hole == pytest.approx(
            4.0 * one.energy_fraction_of_black_hole
        )

    def test_a_weaker_coupling_costs_the_square_of_the_ratio(self) -> None:
        strong = single_field_cost(delta_log_constant=1e-6, coupling_b=1e-3)
        weak = single_field_cost(delta_log_constant=1e-6, coupling_b=1e-6)
        assert weak.energy_fraction_of_black_hole == pytest.approx(
            1e6 * strong.energy_fraction_of_black_hole
        )

    def test_a_zero_coupling_is_refused_rather_than_infinite(self) -> None:
        """The observable does not respond at any energy; say so."""
        with pytest.raises(ValueError, match="does not respond"):
            single_field_cost(delta_log_constant=1e-6, coupling_b=0.0)

    def test_order_unity_offsets_collapse_the_region(self) -> None:
        assert single_field_cost(delta_phi_over_mbar=2.0).verdict == "COLLAPSES_INSTEAD"

    def test_tiny_offsets_are_not_declared_impossible(self) -> None:
        assert (
            single_field_cost(delta_phi_over_mbar=1e-9).verdict
            == "ENERGETICALLY_PLAUSIBLE"
        )


class TestBlackHoleEnergy:
    def test_it_agrees_with_r_over_2g(self) -> None:
        """4 pi Mbar^2 R and R/2G are the same statement; check the conversion."""
        radius = 3.0
        newton_g = 1.0 / (8.0 * math.pi * MBAR_PLANCK_GEV**2)
        assert black_hole_energy_gev(radius) == pytest.approx(radius / (2.0 * newton_g))

    def test_a_nonpositive_radius_is_refused(self) -> None:
        with pytest.raises(ValueError):
            black_hole_energy_gev(0.0)


class TestReachabilityIsNotHiddenByThePseudoInverse:
    def test_a_one_by_one_system_matches_the_single_field_answer(self) -> None:
        b = np.array([[1e-6]])
        result = multi_observable_cost(b, [1e-6])
        assert result.reachable
        assert result.energy_fraction_of_black_hole == pytest.approx(0.5)

    def test_an_unreachable_direction_is_named_not_projected(self) -> None:
        """The failure this exists for.

        Two observables driven by ONE field can only move together. Asking
        them to move oppositely is impossible at any energy, and C^+ answers
        anyway with the least-squares projection — a finite, plausible,
        entirely fictitious number.
        """
        b = np.array([[1.0], [1.0]])  # one field, two observables, locked together
        result = multi_observable_cost(b, [1.0, -1.0])
        assert not result.reachable
        assert result.verdict == "UNREACHABLE_AT_ANY_ENERGY"
        assert result.residual_norm > 0.1
        assert np.allclose(result.residual, [1.0, -1.0], atol=1e-9)

    def test_the_reachable_part_of_that_request_still_reports_its_cost(self) -> None:
        b = np.array([[1.0], [1.0]])
        result = multi_observable_cost(b, [1.0, 1.0])
        assert result.reachable
        assert result.residual_norm == pytest.approx(0.0, abs=1e-12)

    def test_rank_deficiency_is_reported(self) -> None:
        b = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1
        result = multi_observable_cost(b, [1.0, 2.0])
        assert result.reachable_rank == 1
        assert result.field_count == 2

    def test_a_solution_that_achieves_the_target_is_returned(self) -> None:
        b = np.array([[2.0, 0.0], [0.0, 3.0]])
        target = np.array([4.0, 9.0])
        result = multi_observable_cost(b, target)
        assert result.reachable
        assert np.allclose(b @ result.phi_solution_over_mbar, target, atol=1e-9)

    def test_the_metric_changes_the_cost(self) -> None:
        b = np.array([[1.0, 0.0]])
        cheap = multi_observable_cost(b, [1.0], field_metric=np.diag([4.0, 1.0]))
        dear = multi_observable_cost(b, [1.0], field_metric=np.diag([1.0, 1.0]))
        assert cheap.distance_squared_over_mbar2 > dear.distance_squared_over_mbar2

    def test_a_degenerate_metric_is_refused_not_treated_as_cheap(self) -> None:
        b = np.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="positive definite"):
            multi_observable_cost(b, [1.0], field_metric=np.diag([0.0, 1.0]))

    def test_an_asymmetric_metric_is_refused(self) -> None:
        b = np.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="symmetric"):
            multi_observable_cost(b, [1.0], field_metric=np.array([[1.0, 2.0], [0.0, 1.0]]))

    def test_shape_disagreement_is_refused(self) -> None:
        with pytest.raises(ValueError):
            multi_observable_cost(np.array([[1.0]]), [1.0, 2.0])

    def test_non_finite_input_is_refused(self) -> None:
        with pytest.raises(ValueError):
            multi_observable_cost(np.array([[float("nan")]]), [1.0])
