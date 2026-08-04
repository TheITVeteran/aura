"""How much of the steering vector survives 4-bit weights.

The review's limitation: "Running local LLMs in 4-bit precision introduces
activation noise that can degrade float32 CAA steering vector precision." True,
and unmeasured — so nothing in the runtime could tell an α that steers from an
α that is being drowned, and the live log printed ``steering α=0.35`` as if
that settled anything.

MEASURED here on a d=5120 layer with MLX's affine 4-bit scheme, group size 64:
quantisation puts noise worth ~8.8% of the activation norm into the same
residual stream the steering vector is added to. Against a residual norm of
~70 that gives

    α = 0.35  SNR 0.056        α = 2.0  SNR 0.321
    α = 1.0   SNR 0.160        α = 6.0  SNR 0.961

The live surface decodes at 0.35 — roughly eighteen times below the noise it
competes with. The live engine α of ~6 is about where the two become
comparable.

This is reported, NOT enforced. Steering below the floor is weak, not harmful,
and a consistent bias summed over 64 blocks and hundreds of tokens is not a
zero-mean perturbation; the live A/B is what decides whether it works. Pinning
α at the SNR=1 floor would put it above the controller's own base and would be
a behavioural change justified by nothing measured.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.consciousness.caa.quantization_floor import (
    DEFAULT_D_MODEL,
    assess_steering_precision,
    measure_noise_floor,
    quantize_dequantize,
)


class TestTheQuantiserIsTheRealOne:
    def test_round_trip_stays_in_range(self):
        rng = np.random.default_rng(0)
        weights = rng.standard_normal((256, 256)).astype(np.float32)
        restored = quantize_dequantize(weights)
        assert restored.shape == weights.shape
        assert np.all(np.isfinite(restored))
        assert restored.min() >= weights.min() - 1e-4
        assert restored.max() <= weights.max() + 1e-4

    def test_fewer_bits_lose_more(self):
        rng = np.random.default_rng(1)
        weights = rng.standard_normal((256, 256)).astype(np.float32)
        error_4 = np.linalg.norm(quantize_dequantize(weights, bits=4) - weights)
        error_8 = np.linalg.norm(quantize_dequantize(weights, bits=8) - weights)
        assert error_8 < error_4

    def test_a_constant_group_quantises_exactly(self):
        """No spread means no error; a scheme that adds noise here is broken."""
        weights = np.full((4, 64), 0.5, dtype=np.float32)
        assert np.allclose(quantize_dequantize(weights), weights)


class TestTheNoiseFloorIsReal:
    def test_four_bit_weights_cost_several_percent_of_the_activation(self):
        floor = measure_noise_floor()
        assert 0.03 < floor.fraction < 0.20, (
            f"measured {floor.fraction:.4f}; the steering SNR figures in this "
            "module were derived from ~0.088"
        )

    def test_eight_bit_weights_cost_much_less(self):
        four = measure_noise_floor(bits=4).fraction
        eight = measure_noise_floor(bits=8).fraction
        assert eight < four / 4.0

    def test_the_measurement_is_deterministic(self):
        assert measure_noise_floor().fraction == measure_noise_floor().fraction

    def test_it_reports_what_it_measured(self):
        metrics = measure_noise_floor().as_metrics()
        assert metrics["quantization_bits"] == 4
        assert metrics["measured_d_model"] == DEFAULT_D_MODEL
        assert metrics["measurement_trials"] >= 1


class TestSteeringAgainstTheFloor:
    @pytest.mark.parametrize(
        ("alpha", "expect_below"),
        [(0.05, True), (0.35, True), (1.0, True), (6.0, True), (20.0, False)],
    )
    def test_the_live_alphas_are_below_the_floor(self, alpha, expect_below):
        precision = assess_steering_precision(alpha, residual_norm=70.0)
        assert precision.below_floor is expect_below

    def test_the_surface_alpha_is_an_order_of_magnitude_under(self):
        """0.35 is what the live worker logs on every decode."""
        precision = assess_steering_precision(0.35, residual_norm=70.0)
        assert precision.snr < 0.1

    def test_snr_scales_with_alpha(self):
        low = assess_steering_precision(1.0, residual_norm=70.0).snr
        high = assess_steering_precision(4.0, residual_norm=70.0).snr
        assert high == pytest.approx(low * 4.0, rel=1e-6)

    def test_a_silent_residual_has_no_noise_to_beat(self):
        assert math.isinf(assess_steering_precision(1.0, residual_norm=0.0).snr)

    def test_the_minimum_effective_alpha_is_where_snr_reaches_one(self):
        floor = measure_noise_floor()
        alpha = floor.minimum_effective_alpha(70.0)
        assert assess_steering_precision(alpha, 70.0).snr == pytest.approx(1.0, rel=1e-6)


class TestTheControllerReportsItWithoutEnforcingIt:
    def test_every_state_carries_the_ratio(self):
        from core.consciousness.caa.alpha_controller import AlphaController

        controller = AlphaController()
        assert controller.state.quantization_snr is not None
        assert controller.state.to_dict()["below_quantization_floor"] is not None

        updated = controller.update(readiness_level="ready", exact_match_ratio=0.9)
        assert updated.quantization_snr is not None

    def test_the_floor_does_not_clamp_alpha(self):
        """Pinning α at SNR=1 would put it above the controller's own base."""
        from core.consciousness.caa.alpha_controller import AlphaController

        controller = AlphaController()
        assert controller._min_alpha == pytest.approx(0.25)
        assert controller._min_alpha < controller._base_alpha

    def test_an_unmeasurable_floor_reports_none_rather_than_a_number(
        self, monkeypatch
    ):
        import core.consciousness.caa.quantization_floor as floor_module
        from core.consciousness.caa.alpha_controller import quantization_snr

        def boom(*_args, **_kwargs):
            raise ValueError("no measurement available")

        monkeypatch.setattr(floor_module, "assess_steering_precision", boom)
        assert quantization_snr(0.35, 70.0) is None
