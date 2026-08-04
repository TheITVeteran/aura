"""The spiking model's only actuator was a sentence in a prompt.

``SpikingActiveInferenceAdvisor._sampling_bias`` computes three numbers from
the neurodynamics on every turn::

    "temperature_delta": ...uncertainty, error_pressure, instability, novelty
    "top_p_delta":       ...uncertainty, error_pressure, instability
    "max_tokens_factor": ...memory pressure, load, clarification

Only ``max_tokens_factor`` was ever read. The other two were computed and
dropped, so everything the neurodynamic model had to say about HOW to sample
reached the model as English:

    "Neurodynamic advisory: Keep the reply compact and stable because runtime
     load pressure is elevated."

That is what "advisory-only, decoupled from the decision pipeline" meant in
practice — a neurodynamic model whose only mechanism was asking the language
model nicely. Asking is not a mechanism.

The deltas now move the decode parameters that
``core/brain/cognitive_engine.py`` hands to the router, inside the same bounds
the affective controls already respect. Change the neurodynamics and the
sampler changes; that is falsifiable, which the sentence never was.
"""

from __future__ import annotations

import pytest

from core.brain.cognitive_engine import _apply_neurodynamic_sampling_bias


BASE = {"temperature": 0.58, "top_p": 0.88, "clean_user_surface_recurrent_loops": 1}


def _advice(**sampling):
    return {"sampling_bias": dict(sampling)}


class TestTheDeltasReachTheDecode:
    def test_uncertainty_narrows_the_distribution(self):
        out = _apply_neurodynamic_sampling_bias(
            dict(BASE), _advice(temperature_delta=-0.12, top_p_delta=-0.08)
        )
        assert out["temperature"] < BASE["temperature"]
        assert out["top_p"] < BASE["top_p"]
        assert out["temperature"] == pytest.approx(0.46)
        assert out["top_p"] == pytest.approx(0.80)

    def test_novelty_widens_it(self):
        out = _apply_neurodynamic_sampling_bias(
            dict(BASE), _advice(temperature_delta=0.06, top_p_delta=0.03)
        )
        assert out["temperature"] > BASE["temperature"]
        assert out["top_p"] > BASE["top_p"]

    def test_what_was_applied_is_recorded(self):
        out = _apply_neurodynamic_sampling_bias(
            dict(BASE), _advice(temperature_delta=-0.10, top_p_delta=-0.05)
        )
        assert out["neurodynamic_sampling_applied"] == {
            "temperature_delta": -0.10,
            "top_p_delta": -0.05,
        }


class TestItStaysInsideTheBoundsThatAlreadyExisted:
    def test_temperature_cannot_leave_its_range(self):
        low = _apply_neurodynamic_sampling_bias(
            {"temperature": 0.24, "top_p": 0.88}, _advice(temperature_delta=-0.20)
        )
        assert low["temperature"] >= 0.22
        high = _apply_neurodynamic_sampling_bias(
            {"temperature": 0.80, "top_p": 0.88}, _advice(temperature_delta=0.12)
        )
        assert high["temperature"] <= 0.82

    def test_top_p_cannot_leave_its_range(self):
        low = _apply_neurodynamic_sampling_bias(
            {"temperature": 0.58, "top_p": 0.74}, _advice(top_p_delta=-0.20)
        )
        assert low["top_p"] >= 0.72

    def test_an_out_of_contract_delta_is_ignored_not_clamped(self):
        """A delta outside the advisor's own declared range is not its output."""
        out = _apply_neurodynamic_sampling_bias(
            dict(BASE), _advice(temperature_delta=-5.0, top_p_delta=99.0)
        )
        assert out["temperature"] == BASE["temperature"]
        assert out["top_p"] == BASE["top_p"]
        assert "neurodynamic_sampling_applied" not in out


class TestItNeverInventsControls:
    @pytest.mark.parametrize(
        "advice", [None, {}, {"sampling_bias": None}, {"sampling_bias": []}, "nonsense"]
    )
    def test_missing_advice_changes_nothing(self, advice):
        assert _apply_neurodynamic_sampling_bias(dict(BASE), advice) == BASE

    def test_no_controls_means_nothing_to_move(self):
        assert _apply_neurodynamic_sampling_bias({}, _advice(temperature_delta=-0.1)) == {}

    def test_a_zero_delta_leaves_the_dict_untouched(self):
        out = _apply_neurodynamic_sampling_bias(
            dict(BASE), _advice(temperature_delta=0.0, top_p_delta=0.0)
        )
        assert out == BASE
        assert "neurodynamic_sampling_applied" not in out

    def test_a_non_numeric_delta_is_ignored(self):
        out = _apply_neurodynamic_sampling_bias(
            dict(BASE), _advice(temperature_delta="hot", top_p_delta=None)
        )
        assert out == BASE


class TestTheAdvisorReallyProducesThese:
    def test_the_real_advisor_emits_both_deltas(self):
        from core.cognitive.spiking_active_inference import (
            get_spiking_active_inference_advisor,
        )

        advisor = get_spiking_active_inference_advisor()
        advice = advisor.advise("something novel and unfamiliar", is_background=True)
        sampling = advice.sampling_bias
        assert "temperature_delta" in sampling
        assert "top_p_delta" in sampling
        assert -0.25 <= float(sampling["temperature_delta"]) <= 0.25
        assert -0.25 <= float(sampling["top_p_delta"]) <= 0.25

    def test_advisor_output_flows_through_the_applier(self):
        from core.cognitive.spiking_active_inference import (
            get_spiking_active_inference_advisor,
        )

        advisor = get_spiking_active_inference_advisor()
        advice = advisor.advise("diagnose the failing deploy", is_background=True)
        out = _apply_neurodynamic_sampling_bias(dict(BASE), advice.to_dict())
        assert 0.22 <= out["temperature"] <= 0.82
        assert 0.72 <= out["top_p"] <= 0.94
