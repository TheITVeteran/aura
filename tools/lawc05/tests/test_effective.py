"""The open channel, calibrated against the regime experiments report."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lawc05.effective import (  # noqa: E402
    DEEP_STRONG_THRESHOLD,
    HBAR,
    ULTRASTRONG_THRESHOLD,
    channel_comparison,
    collective_coupling,
    diffraction_limited_volume,
    dipole_from_length,
    emitters_in_2deg,
    estimate_effective_shift,
    mode_volume_for_target_eta,
    vacuum_field_amplitude,
)

THZ = 2.0 * math.pi * 1e12


class TestVacuumField:
    def test_it_scales_as_one_over_root_volume(self) -> None:
        """The entire enhancement mechanism, in one assertion."""
        big = vacuum_field_amplitude(THZ, 1e-12)
        small = vacuum_field_amplitude(THZ, 1e-16)
        assert small / big == pytest.approx(100.0, rel=1e-9)

    def test_it_grows_as_root_frequency(self) -> None:
        low = vacuum_field_amplitude(THZ, 1e-15)
        high = vacuum_field_amplitude(4.0 * THZ, 1e-15)
        assert high / low == pytest.approx(2.0, rel=1e-9)

    def test_nonpositive_inputs_are_refused(self) -> None:
        with pytest.raises(ValueError):
            vacuum_field_amplitude(0.0, 1e-15)
        with pytest.raises(ValueError):
            vacuum_field_amplitude(THZ, 0.0)


class TestCoupling:
    def test_eta_scales_as_the_square_root_of_compression(self) -> None:
        d = dipole_from_length(10e-9)
        base = estimate_effective_shift(
            omega_rad_s=THZ, mode_volume_m3=1e-14, dipole_cm=d
        )
        squeezed = estimate_effective_shift(
            omega_rad_s=THZ, mode_volume_m3=1e-18, dipole_cm=d
        )
        assert squeezed.normalized_coupling / base.normalized_coupling == pytest.approx(
            100.0, rel=1e-9
        )

    def test_the_target_volume_inverts_the_coupling(self) -> None:
        d = dipole_from_length(10e-9)
        volume = mode_volume_for_target_eta(target_eta=0.05, omega_rad_s=THZ, dipole_cm=d)
        got = estimate_effective_shift(
            omega_rad_s=THZ, mode_volume_m3=volume, dipole_cm=d
        )
        assert got.normalized_coupling == pytest.approx(0.05, rel=1e-9)

    def test_collective_coupling_is_the_root_of_n(self) -> None:
        assert collective_coupling(1.0, 10_000.0) == pytest.approx(100.0)

    def test_a_negative_population_is_refused(self) -> None:
        with pytest.raises(ValueError):
            collective_coupling(1.0, -1.0)

    def test_the_2deg_population_counts_the_overlap(self) -> None:
        full = emitters_in_2deg(sheet_density_per_cm2=3e11, mode_area_m2=1e-10)
        third = emitters_in_2deg(
            sheet_density_per_cm2=3e11, mode_area_m2=1e-10, overlap=1 / 3
        )
        assert third / full == pytest.approx(1 / 3, rel=1e-9)

    def test_an_impossible_overlap_is_refused(self) -> None:
        with pytest.raises(ValueError):
            emitters_in_2deg(sheet_density_per_cm2=3e11, mode_area_m2=1e-10, overlap=2.0)


class TestRegimeHonesty:
    def test_a_weak_coupling_reports_a_perturbative_shift(self) -> None:
        d = dipole_from_length(1e-10)
        est = estimate_effective_shift(omega_rad_s=THZ, mode_volume_m3=1e-12, dipole_cm=d)
        assert est.regime == "PERTURBATIVE"
        assert est.perturbative

    def test_ultrastrong_refuses_to_pretend_the_quadratic_is_a_prediction(self) -> None:
        """No fake precision where the formula has stopped applying."""
        d = dipole_from_length(10e-9)
        volume = mode_volume_for_target_eta(
            target_eta=ULTRASTRONG_THRESHOLD * 2.0, omega_rad_s=THZ, dipole_cm=d
        )
        est = estimate_effective_shift(
            omega_rad_s=THZ, mode_volume_m3=volume, dipole_cm=d
        )
        assert est.regime == "ULTRASTRONG"
        assert not est.perturbative
        assert any("order-of-magnitude" in n for n in est.notes)

    def test_deep_strong_says_diagonalise_instead(self) -> None:
        d = dipole_from_length(10e-9)
        volume = mode_volume_for_target_eta(
            target_eta=DEEP_STRONG_THRESHOLD * 2.0, omega_rad_s=THZ, dipole_cm=d
        )
        est = estimate_effective_shift(
            omega_rad_s=THZ, mode_volume_m3=volume, dipole_cm=d
        )
        assert est.regime == "DEEP_STRONG"
        assert any("Diagonalise" in n for n in est.notes)

    def test_a_design_with_no_sub_wavelength_gain_is_flagged(self) -> None:
        d = dipole_from_length(10e-9)
        est = estimate_effective_shift(
            omega_rad_s=THZ,
            mode_volume_m3=diffraction_limited_volume(THZ) * 10.0,
            dipole_cm=d,
        )
        assert est.volume_compression < 1.0
        assert any("exceeds the diffraction limit" in n for n in est.notes)


class TestCalibrationAgainstMeasuredExperiments:
    def test_a_split_ring_over_a_2deg_lands_in_the_reported_regime(self) -> None:
        """Split-ring resonators on a 2DEG report eta of order 0.1 to 1.

        This is the test that keeps the model tethered. If a change here puts
        the estimate orders away from what those experiments measure, the
        model is wrong regardless of how clean the algebra looks.
        """
        d = dipole_from_length(10e-9)
        volume = diffraction_limited_volume(THZ) / 1e6
        single = estimate_effective_shift(
            omega_rad_s=THZ, mode_volume_m3=volume, dipole_cm=d
        )
        n = emitters_in_2deg(
            sheet_density_per_cm2=3e11, mode_area_m2=(5e-6) ** 2, overlap=0.3
        )
        eta = collective_coupling(single.coupling_rad_s, n) / THZ
        assert 0.05 < eta < 5.0

    def test_the_zero_point_energy_is_the_whole_bill(self) -> None:
        """Why this channel wins: the vacuum already paid for the field."""
        est = estimate_effective_shift(
            omega_rad_s=THZ, mode_volume_m3=1e-18, dipole_cm=dipole_from_length(10e-9)
        )
        assert est.zero_point_energy_joules == pytest.approx(0.5 * HBAR * THZ)
        assert est.zero_point_energy_joules < 1e-20


class TestChannelComparison:
    def test_the_effective_channel_wins_on_both_axes(self) -> None:
        est = estimate_effective_shift(
            omega_rad_s=THZ, mode_volume_m3=1e-18, dipole_cm=dipole_from_length(10e-9)
        )
        comparison = channel_comparison(
            ambient_shift=3.6e-20,
            ambient_energy_joules=1e6,
            ambient_coupling_shortfall_decades=9.4,
            effective=est,
        )
        assert comparison["shift_ratio_effective_over_ambient"] > 1e10
        assert comparison["energy_ratio_ambient_over_effective"] > 1e20
        assert "not energy" in comparison["ambient"]["blocked_by"]
