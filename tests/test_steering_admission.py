"""A NaN on disk could have reached every token the model produced.

``AffectiveSteeringHook.update_substrate_vector`` guarded the composite with
exactly one test::

    current_norm = np.linalg.norm(composite_np)
    if current_norm < 1e-4:
        ...stand down...
    else:
        self._cached_composite_mx = mx.array(composite_np)

``NaN < 1e-4`` is False. So if any steering vector loaded from
``data/steering_vectors/`` contained a NaN, the norm was NaN, the normalisation
made every element NaN, this guard took the ELSE branch, and the NaN composite
was added to the hidden states of all 64 transformer blocks on every token.

``LiquidSubstrate.inject_stimulus`` had the matching hole on the way in: it
never looked at the values, and ``np.clip`` of NaN is NaN. Its callers carry
values derived from screen contents, audio, the closed loop and the model's own
output, so this is the input surface the reviewer meant by "a malformed input
could drive the ODE into an undefined regime".

Both are gated now, and both REJECT rather than coerce — clamping a hostile
input applies it at a survivable magnitude, and zeroing it is indistinguishable
from having received nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.consciousness.steering_admission import (
    MAX_COMPOSITE_NORM,
    MAX_STIMULUS_MAGNITUDE,
    MAX_STIMULUS_WEIGHT,
    admit_steering_vector,
    admit_stimulus,
)


class TestStimulusAdmission:
    def test_an_ordinary_stimulus_is_admitted(self):
        assert admit_stimulus(np.array([0.2, -0.4, 0.1]), 1.0).admitted is True

    @pytest.mark.parametrize(
        ("vector", "reason"),
        [
            ([np.nan, 0.1], "non_finite_stimulus"),
            ([np.inf, 0.1], "non_finite_stimulus"),
            ([-np.inf, 0.1], "non_finite_stimulus"),
            ([MAX_STIMULUS_MAGNITUDE * 10, 0.1], "stimulus_out_of_scale"),
            ([], "empty_stimulus"),
        ],
    )
    def test_a_malformed_vector_is_refused(self, vector, reason):
        admission = admit_stimulus(np.array(vector, dtype=float), 1.0)
        assert admission.rejected is True
        assert reason in admission.reasons

    @pytest.mark.parametrize(
        ("weight", "reason"),
        [
            (np.nan, "non_finite_weight"),
            (np.inf, "non_finite_weight"),
            (MAX_STIMULUS_WEIGHT * 5, "weight_out_of_scale"),
            ("not a number", "unreadable_weight"),
        ],
    )
    def test_a_malformed_weight_is_refused_too(self, weight, reason):
        """An ordinary vector at infinite weight is the same defect."""
        admission = admit_stimulus(np.array([0.2, 0.1]), weight)
        assert admission.rejected is True
        assert reason in admission.reasons

    def test_the_reason_is_named_not_just_flagged(self):
        admission = admit_stimulus(np.array([np.nan, np.nan, 0.1]), 1.0)
        assert admission.detail["non_finite_elements"] == 2


class TestSteeringVectorAdmission:
    def test_a_normalised_direction_is_admitted(self):
        vector = np.ones(64) / np.sqrt(64)
        assert admit_steering_vector(vector).admitted is True

    def test_the_nan_composite_is_refused(self):
        """The exact value that passed `current_norm < 1e-4`."""
        vector = np.full(64, np.nan)
        assert not (np.linalg.norm(vector) < 1e-4), "the old guard let this through"
        admission = admit_steering_vector(vector)
        assert admission.rejected is True
        assert "non_finite_vector" in admission.reasons

    def test_a_single_nan_element_refuses_the_whole_vector(self):
        vector = np.ones(64) / np.sqrt(64)
        vector[7] = np.nan
        assert admit_steering_vector(vector).rejected is True

    def test_an_oversized_vector_is_refused(self):
        vector = np.ones(64) / np.sqrt(64) * (MAX_COMPOSITE_NORM * 3)
        admission = admit_steering_vector(vector)
        assert admission.rejected is True
        assert "vector_norm_out_of_envelope" in admission.reasons

    def test_a_spike_is_refused_even_at_unit_norm(self):
        """All the mass in one coordinate saturates one feature every token."""
        vector = np.zeros(64)
        vector[3] = 1.0
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0)
        admission = admit_steering_vector(vector)
        assert admission.rejected is True
        assert "vector_is_a_spike" in admission.reasons

    def test_a_broadly_distributed_vector_is_not_a_spike(self):
        rng = np.random.default_rng(7)
        vector = rng.standard_normal(64)
        vector /= np.linalg.norm(vector)
        assert admit_steering_vector(vector).admitted is True

    def test_an_empty_vector_is_refused(self):
        assert admit_steering_vector(np.array([])).rejected is True


class TestTheSubstrateRefusesRatherThanClamps:
    def test_a_nan_stimulus_leaves_state_untouched(self):
        import asyncio

        from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

        substrate = LiquidSubstrate(SubstrateConfig(neuron_count=16))
        before = substrate.x.copy()
        asyncio.run(substrate.inject_stimulus(np.full(16, np.nan), weight=0.1))
        assert np.array_equal(substrate.x, before), (
            "a NaN stimulus reached state; np.clip of NaN is NaN"
        )
        assert np.all(np.isfinite(substrate.x))

    def test_an_infinite_weight_leaves_state_untouched(self):
        import asyncio

        from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

        substrate = LiquidSubstrate(SubstrateConfig(neuron_count=16))
        before = substrate.x.copy()
        asyncio.run(substrate.inject_stimulus(np.full(16, 0.2), weight=float("inf")))
        assert np.array_equal(substrate.x, before)

    def test_a_normal_stimulus_still_moves_state(self):
        """The gate must not have turned stimulus injection off."""
        import asyncio

        from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig

        substrate = LiquidSubstrate(SubstrateConfig(neuron_count=16))
        before = substrate.x.copy()
        asyncio.run(substrate.inject_stimulus(np.full(16, 0.5), weight=1.0))
        assert not np.array_equal(substrate.x, before)
        assert np.all(np.isfinite(substrate.x))
