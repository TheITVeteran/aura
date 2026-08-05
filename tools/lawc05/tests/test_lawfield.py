"""The copy universe: is its law actually law, and is it actually measurable?

The interesting assertions are the ones a pretty animation would fail:
energy that is conserved because the operator is right, a causality bound that
holds by construction, and constants an inhabitant can recover from the field
alone and get back the ones that were painted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lawc05.lawfield import (  # noqa: E402
    BOLTZMANN,
    CopyUniverse,
    LawParameters,
    divergence_of_gradient,
    recover_dispersion,
)


def _uniform_world(q_value: float, size: int = 48) -> CopyUniverse:
    world = CopyUniverse(size=size)
    world._q[:] = q_value
    return world


class TestTheOneLawNobodyCanWrite:
    @pytest.mark.parametrize("q_value", [0.0, 1.0, -3.0, 25.0, -100.0])
    def test_the_light_cone_holds_for_any_painted_value(self, q_value: float) -> None:
        """c(q) = c0/(1+beta q^2) <= c0 for every real q. Structural, not policed."""
        world = _uniform_world(q_value)
        assert world.causality_holds()
        assert world.max_speed() <= world.params.c0 + 1e-12

    def test_painting_can_only_slow_light_down(self) -> None:
        fast = _uniform_world(0.0).max_speed()
        slow = _uniform_world(2.0).max_speed()
        assert slow < fast == pytest.approx(1.0)


class TestTheOperatorIsTheRightOne:
    def test_the_flux_form_is_self_adjoint(self) -> None:
        """<u, L v> == <L u, v>.

        The lazy discretisation c^2 lap(phi) fails this whenever c varies, and
        an operator that is not self-adjoint does not conserve the energy the
        display is reporting.
        """
        rng = np.random.default_rng(0)
        u = rng.standard_normal((16, 16))
        v = rng.standard_normal((16, 16))
        coefficient = 1.0 + 0.5 * rng.random((16, 16))
        left = float(np.sum(u * divergence_of_gradient(v, coefficient, 1.0)))
        right = float(np.sum(v * divergence_of_gradient(u, coefficient, 1.0)))
        assert left == pytest.approx(right, rel=1e-12)

    def test_a_constant_field_has_no_divergence(self) -> None:
        coefficient = np.full((8, 8), 2.0)
        assert np.allclose(
            divergence_of_gradient(np.full((8, 8), 3.0), coefficient, 1.0), 0.0
        )


class TestEnergyIsConservedNotDisplayed:
    @staticmethod
    def _energy_history(world: CopyUniverse, steps: int, dt: float) -> np.ndarray:
        history = []
        for _ in range(steps):
            world.step(dt)
            history.append(world.matter_energy())
        return np.asarray(history)

    def test_drift_stays_bounded_over_a_long_run(self) -> None:
        world = _uniform_world(0.7)
        world.seed_noise(amplitude=1e-2, seed=1)
        energies = self._energy_history(world, 600, world.max_stable_dt(0.25))
        excursion = (energies.max() - energies.min()) / energies.mean()
        assert excursion < 0.05, "symplectic integration should not drift"

    def test_the_energy_error_is_discretisation_not_leakage(self) -> None:
        """Halve the step, quarter the excursion.

        This is the assertion that distinguishes a symplectic integrator from
        one that merely has a small error today. A leaking scheme's error does
        not scale as dt^2 — it scales with how long you ran it, and no choice
        of threshold catches that. The dt^2 law does.
        """
        excursions = []
        for safety in (0.30, 0.15):
            world = _uniform_world(0.7)
            world.seed_noise(amplitude=1e-2, seed=1)
            energies = self._energy_history(world, 600, world.max_stable_dt(safety))
            excursions.append((energies.max() - energies.min()) / energies.mean())
        ratio = excursions[0] / excursions[1]
        assert 3.0 < ratio < 5.5, f"expected ~4x (dt^2), got {ratio:.2f}"

    def test_there_is_no_secular_growth(self) -> None:
        """A leaking integrator looks fine briefly. Compare the two halves."""
        world = _uniform_world(0.0)
        world.seed_noise(amplitude=1e-2, seed=2)
        energies = self._energy_history(world, 800, world.max_stable_dt(0.25))
        first = energies[: len(energies) // 2].mean()
        second = energies[len(energies) // 2 :].mean()
        assert abs(second / first - 1.0) < 0.01

    def test_an_inhomogeneous_law_still_conserves_energy(self) -> None:
        """The case where the wrong operator would show itself."""
        world = CopyUniverse(size=48)
        rows, cols = np.ogrid[:48, :48]
        world._q[:] = np.where(cols < 24, -1.0, 1.5)
        world.seed_noise(amplitude=1e-2, seed=5)
        energies = self._energy_history(world, 500, world.max_stable_dt(0.25))
        assert (energies.max() - energies.min()) / energies.mean() < 0.05


class TestTheInhabitantsCanMeasureTheirOwnLaws:
    @pytest.mark.parametrize("q_value", [0.0, 1.0, -1.5])
    def test_blind_recovery_returns_the_painted_constants(self, q_value: float) -> None:
        """Nothing about the parameters is handed to the recovery — only frames."""
        params = LawParameters()
        world = _uniform_world(q_value)
        world.seed_noise(seed=3)
        dt = world.max_stable_dt()
        frames = world.run(384, dt, record=True)

        recovered = recover_dispersion(frames, dt, world.dx)

        assert recovered.speed == pytest.approx(float(params.speed(q_value)), rel=0.03)
        assert recovered.mass == pytest.approx(float(params.mass_a(q_value)), rel=0.05)
        assert recovered.speed_well_determined
        assert recovered.mass_well_determined

    def test_two_regions_of_law_measure_differently(self) -> None:
        """The claim that makes this a copy universe rather than a picture.

        Same code, same integrator, two painted values — and an inhabitant
        measuring locally gets two different sets of constants.
        """
        params = LawParameters()
        results = {}
        for q_value in (0.0, 1.2):
            world = _uniform_world(q_value)
            world.seed_noise(seed=7)
            dt = world.max_stable_dt()
            results[q_value] = recover_dispersion(
                world.run(384, dt, record=True), dt, world.dx
            )
        assert results[0.0].speed > results[1.2].speed
        assert results[0.0].mass < results[1.2].mass
        assert results[0.0].speed == pytest.approx(float(params.speed(0.0)), rel=0.03)
        assert results[1.2].speed == pytest.approx(float(params.speed(1.2)), rel=0.05)

    def test_a_weakly_constrained_speed_is_flagged_not_asserted(self) -> None:
        """When m >> c k the slope is unmeasurable. Say so instead of fitting.

        This is the honest half of the recovery: it returns a number either
        way, and the flag is what tells you the number means nothing.
        """
        world = _uniform_world(2.0)
        world.seed_noise(seed=3)
        dt = world.max_stable_dt()
        recovered = recover_dispersion(world.run(384, dt, record=True), dt, world.dx)

        assert recovered.dispersion_leverage < 0.5
        assert not recovered.speed_well_determined
        # The error bar is a large fraction of the value: the fit returns a
        # number, and says plainly that the number is not a measurement.
        assert recovered.speed_stderr / recovered.speed > 0.5
        # The mass, which the flat line DOES pin down, is still good.
        assert recovered.mass_well_determined
        assert recovered.mass == pytest.approx(2.2, rel=0.05)

    def test_too_few_frames_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 8"):
            recover_dispersion(np.zeros((4, 8, 8)), 0.1, 1.0)

    def test_a_silent_field_cannot_be_fitted(self) -> None:
        with pytest.raises(ValueError, match="not enough clean modes"):
            recover_dispersion(np.zeros((64, 16, 16)), 0.1, 1.0)


class TestLawIsAStateVariable:
    def test_the_decay_channel_opens_and_closes_with_the_paint(self) -> None:
        """m_A(q) > 2 m_B(q) in one region and not the other.

        Same particle, different painted law, different fate.
        """
        params = LawParameters()
        assert params.decay_allowed(2.0)
        assert not params.decay_allowed(-2.0)

    def test_painting_costs_work_and_the_ledger_records_it(self) -> None:
        world = CopyUniverse(size=32)
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        work = world.paint(mask, -1.0)
        assert work > 0.0, "flipping to the other vacuum builds a domain wall"
        assert world.ledger[-1].kind == "paint"
        assert world.ledger[-1].work_joules == pytest.approx(work)

    def test_bubble_cost_grows_with_radius(self) -> None:
        costs = []
        for radius in (4.0, 8.0, 12.0):
            world = CopyUniverse(size=64)
            costs.append(world.nucleate_bubble((32, 32), radius, -1.0)["measured_work"])
        assert costs[0] < costs[1] < costs[2]

    def test_bubble_cost_grows_with_the_law_contrast(self) -> None:
        small = CopyUniverse(size=64).nucleate_bubble((32, 32), 10.0, 0.5)
        large = CopyUniverse(size=64).nucleate_bubble((32, 32), 10.0, -1.0)
        assert large["measured_work"] > small["measured_work"]
        assert abs(large["delta_q"]) > abs(small["delta_q"])

    def test_the_thin_wall_estimate_is_reported_beside_the_measurement(self) -> None:
        """Two numbers, never conflated: what was measured and what was expected."""
        result = CopyUniverse(size=64).nucleate_bubble((32, 32), 10.0, -1.0)
        assert result["measured_work"] > 0.0
        assert result["thin_wall_expectation"] > 0.0
        assert result["measured_work"] != result["thin_wall_expectation"]


class TestLandauer:
    def test_erasure_costs_kt_ln2_per_bit(self) -> None:
        world = CopyUniverse(size=8, temperature_k=300.0)
        heat = world.erase(1000.0)
        assert heat == pytest.approx(1000.0 * BOLTZMANN * 300.0 * np.log(2.0))

    def test_the_floor_accumulates_across_the_ledger(self) -> None:
        world = CopyUniverse(size=8, temperature_k=300.0)
        world.erase(100.0)
        world.erase(400.0)
        assert world.landauer_floor_joules() == pytest.approx(
            500.0 * BOLTZMANN * 300.0 * np.log(2.0)
        )

    def test_a_hotter_world_pays_more(self) -> None:
        cold = CopyUniverse(size=8, temperature_k=4.0).erase(1000.0)
        warm = CopyUniverse(size=8, temperature_k=300.0).erase(1000.0)
        assert warm / cold == pytest.approx(75.0, rel=1e-9)

    def test_negative_erasure_is_refused(self) -> None:
        with pytest.raises(ValueError):
            CopyUniverse(size=8).erase(-1.0)
