"""The sourcing bound: the one that actually stops you."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lawc05.frontier import (  # noqa: E402
    DENSITY_TUNGSTEN,
    OPTICAL_CLOCK_RESOLUTION,
    best_reachable_shift,
    coupling_needed_for_shift,
    default_alpha_envelope,
    detectability_scan,
    energy_to_hold_joules,
    evaluate_target,
    inverse_gev_to_metres,
    mass_gev_to_range,
    max_coupling_b,
    metres_to_inverse_gev,
    range_to_mass_gev,
    shift_from_source,
)
from lawc05.nogo import MBAR_PLANCK_GEV  # noqa: E402


class TestUnits:
    def test_length_conversion_round_trips(self) -> None:
        assert inverse_gev_to_metres(metres_to_inverse_gev(2.5)) == pytest.approx(2.5)

    def test_range_and_mass_are_inverses(self) -> None:
        assert mass_gev_to_range(range_to_mass_gev(1e-6)) == pytest.approx(1e-6)

    def test_a_micron_range_is_a_sub_eV_mass(self) -> None:
        """1 um <-> about 0.2 eV. If this drifts, every number below is wrong."""
        assert range_to_mass_gev(1e-6) == pytest.approx(1.973e-10, rel=1e-3)


class TestEnergyBound:
    def test_the_ppm_target_is_the_headline_number(self) -> None:
        """B=1e-6, dl=1e-6, R=1m: half of a 1-metre black hole."""
        energy = energy_to_hold_joules(
            delta_log_constant=1e-6, coupling_b=1e-6, region_radius_m=1.0
        )
        black_hole_j = 4.0 * math.pi * MBAR_PLANCK_GEV**2 * metres_to_inverse_gev(1.0)
        black_hole_j *= 1.602177e-10
        assert energy / black_hole_j == pytest.approx(0.5, rel=1e-6)

    def test_energy_is_linear_in_radius_for_a_light_field(self) -> None:
        one = energy_to_hold_joules(
            delta_log_constant=1e-18, coupling_b=1.0, region_radius_m=1e-3
        )
        ten = energy_to_hold_joules(
            delta_log_constant=1e-18, coupling_b=1.0, region_radius_m=1e-2
        )
        assert ten / one == pytest.approx(10.0, rel=1e-9)

    def test_a_detectable_target_costs_almost_nothing(self) -> None:
        """The result that reframes the whole problem: joules, not black holes."""
        energy = energy_to_hold_joules(
            delta_log_constant=OPTICAL_CLOCK_RESOLUTION,
            coupling_b=1.0,
            region_radius_m=1e-3,
        )
        assert energy < 1e5


class TestSourcingBound:
    def test_the_shift_is_quadratic_in_the_coupling(self) -> None:
        """Once to source the field, once to convert it. That is the punishment."""
        mass = range_to_mass_gev(1e-4)
        one = shift_from_source(
            coupling_b=1.0, density_kg_per_m3=DENSITY_TUNGSTEN, field_mass_gev=mass
        )
        ten = shift_from_source(
            coupling_b=10.0, density_kg_per_m3=DENSITY_TUNGSTEN, field_mass_gev=mass
        )
        assert ten / one == pytest.approx(100.0, rel=1e-9)

    def test_the_shift_is_linear_in_density(self) -> None:
        mass = range_to_mass_gev(1e-4)
        light = shift_from_source(
            coupling_b=1.0, density_kg_per_m3=1000.0, field_mass_gev=mass
        )
        heavy = shift_from_source(
            coupling_b=1.0, density_kg_per_m3=20000.0, field_mass_gev=mass
        )
        assert heavy / light == pytest.approx(20.0, rel=1e-9)

    def test_a_shorter_range_sources_a_smaller_shift(self) -> None:
        """1/m^2: squeezing the range to escape the coupling bound costs signal."""
        short = shift_from_source(
            coupling_b=1.0,
            density_kg_per_m3=DENSITY_TUNGSTEN,
            field_mass_gev=range_to_mass_gev(1e-8),
        )
        long_ = shift_from_source(
            coupling_b=1.0,
            density_kg_per_m3=DENSITY_TUNGSTEN,
            field_mass_gev=range_to_mass_gev(1e-6),
        )
        assert long_ > short

    def test_a_massless_field_is_refused_rather_than_infinite(self) -> None:
        with pytest.raises(ValueError, match="infinite range"):
            shift_from_source(
                coupling_b=1.0, density_kg_per_m3=DENSITY_TUNGSTEN, field_mass_gev=0.0
            )

    def test_the_needed_coupling_inverts_the_shift(self) -> None:
        mass = range_to_mass_gev(1e-4)
        needed = coupling_needed_for_shift(
            delta_log_constant=1e-18,
            density_kg_per_m3=DENSITY_TUNGSTEN,
            field_mass_gev=mass,
        )
        recovered = shift_from_source(
            coupling_b=needed, density_kg_per_m3=DENSITY_TUNGSTEN, field_mass_gev=mass
        )
        assert recovered == pytest.approx(1e-18, rel=1e-9)


class TestCouplingEnvelope:
    def test_short_ranges_are_less_constrained_than_long_ones(self) -> None:
        assert default_alpha_envelope(1e-8) > default_alpha_envelope(1e-2)

    def test_it_is_monotonic_across_the_scan(self) -> None:
        ranges = [10.0**e for e in range(-8, 2)]
        values = [default_alpha_envelope(r) for r in ranges]
        assert all(a >= b for a, b in zip(values, values[1:], strict=False))

    def test_planetary_ranges_are_tightly_bounded(self) -> None:
        assert default_alpha_envelope(1e3) <= 1e-4

    def test_b_is_the_square_root_of_alpha(self) -> None:
        assert max_coupling_b(1e-3) == pytest.approx(
            math.sqrt(default_alpha_envelope(1e-3))
        )

    def test_a_nonpositive_range_is_refused(self) -> None:
        with pytest.raises(ValueError):
            default_alpha_envelope(0.0)


class TestWhichBoundBinds:
    def test_a_detectable_target_is_stopped_by_coupling_not_energy(self) -> None:
        """The central finding. Energy is affordable; the source cannot exist."""
        verdict = evaluate_target(
            delta_log_constant=OPTICAL_CLOCK_RESOLUTION,
            region_radius_m=1e-3,
            range_m=1e-4,
        )
        assert verdict.binding_constraint == "COUPLING"
        assert not verdict.reachable
        assert verdict.energy_joules < verdict_energy_budget()
        assert any("Energy is NOT the obstacle" in n for n in verdict.notes)

    def test_the_shortfall_is_reported_in_decades(self) -> None:
        verdict = evaluate_target(
            delta_log_constant=OPTICAL_CLOCK_RESOLUTION,
            region_radius_m=1e-3,
            range_m=1e-4,
        )
        assert verdict.coupling_headroom_decades < 0.0
        assert verdict.coupling_needed > verdict.coupling_allowed

    def test_a_trivial_target_clears_both_bounds(self) -> None:
        verdict = evaluate_target(
            delta_log_constant=1e-40, region_radius_m=1e-3, range_m=1e-4
        )
        assert verdict.reachable
        assert verdict.binding_constraint == "NONE"

    def test_a_range_larger_than_the_region_is_flagged(self) -> None:
        verdict = evaluate_target(
            delta_log_constant=1e-30, region_radius_m=1e-6, range_m=1e-3
        )
        assert any("range exceeds the region" in n for n in verdict.notes)


def verdict_energy_budget() -> float:
    from lawc05.frontier import LAB_ENERGY_BUDGET_JOULES

    return LAB_ENERGY_BUDGET_JOULES


class TestTheAtlas:
    def test_every_scanned_corner_is_coupling_limited(self) -> None:
        """If this ever fails, the energy bound has become relevant again."""
        rows = detectability_scan(region_radius_m=1e-3)
        assert rows
        assert all(row["limited_by"] == "COUPLING" for row in rows)

    def test_nothing_in_the_scan_reaches_clock_sensitivity(self) -> None:
        rows = detectability_scan(region_radius_m=1e-3)
        assert not any(row["detectable"] for row in rows)

    def test_the_best_corner_is_within_two_decades(self) -> None:
        """How close the frontier actually is — the number worth watching."""
        rows = detectability_scan(region_radius_m=1e-3)
        best = max(rows, key=lambda r: r["best_shift"])
        assert -3.0 < best["decades_short"] < 0.0

    def test_a_denser_source_helps_linearly(self) -> None:
        light = best_reachable_shift(
            region_radius_m=1e-3, range_m=1e-6, density_kg_per_m3=1000.0
        )
        heavy = best_reachable_shift(
            region_radius_m=1e-3, range_m=1e-6, density_kg_per_m3=20000.0
        )
        assert heavy["best_shift"] / light["best_shift"] == pytest.approx(20.0, rel=1e-6)

    def test_a_custom_envelope_is_honoured(self) -> None:
        """The envelope decides every conclusion, so it must be replaceable."""
        rows = detectability_scan(region_radius_m=1e-3, envelope=lambda _r: 1e30)
        assert any(row["detectable"] for row in rows)
