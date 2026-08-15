"""CP126 3f30dcf3 + b4dc41d3: the FLOPs were consistent and unmoored.

The estimator was pinned. The architecture it estimated against was not, and
nothing connected the compute profile to the 32B-class model the claim is
about. A producer could pin the right estimator, hand it a toy decoder, and
every FLOPs comparison between the arms would come out internally consistent
and say nothing about the model that ran.

Underneath that, ``layer_apps`` was a free-standing integer sitting beside a
resource receipt that counted the same thing, and nobody compared them. Its
only ceiling excluded a 64-bit overflow.
"""
from __future__ import annotations

import copy

import pytest

from core.brain.llm.latent_cortex.resource_accounting import ModelComputeProfile
from tests.fixtures.latent_frontier import (
    _RESIDENT_PROFILE,
    _bundle,
    _certify,
    _refresh_task_commitment,
)


class TestStructuralParameterCount:
    def test_the_fixture_profile_reconciles_with_its_declared_size(self):
        declared = 32_000_000_000
        structural = _RESIDENT_PROFILE.structural_parameter_count
        assert abs(structural - declared) / declared < 0.05

    def test_a_toy_decoder_is_nowhere_near_a_32b_model(self):
        toy = ModelComputeProfile(
            model_type="fixture-decoder",
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=64,
            head_dim=4,
        )
        assert toy.structural_parameter_count < 10_000

    def test_the_count_scales_with_depth(self):
        deeper = ModelComputeProfile(
            model_type="qwen2",
            hidden_size=5120,
            intermediate_size=27648,
            num_hidden_layers=128,
            num_attention_heads=40,
            num_key_value_heads=8,
            vocab_size=152064,
            head_dim=128,
        )
        assert deeper.structural_parameter_count > (
            _RESIDENT_PROFILE.structural_parameter_count
        )


class TestProfileReconciliation:
    def test_the_fixture_certifies(self):
        certificate = _certify(_bundle())
        assert certificate["accepted"] is True, certificate["reasons"]

    def test_a_bundle_without_a_compute_profile_is_refused(self):
        bundle = _bundle()
        del bundle["resident_model"]["compute_profile"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "resident_compute_profile_missing" in certificate["reasons"]

    def test_a_toy_profile_cannot_stand_in_for_the_resident_model(self):
        """The exact substitution the estimator pin could not see."""
        bundle = _bundle()
        toy = ModelComputeProfile(
            model_type="fixture-decoder",
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=64,
            head_dim=4,
        ).to_receipt()
        bundle["resident_model"]["compute_profile"] = toy
        bundle["preregistration"]["compute_profile_sha256"] = toy["profile_sha256"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert (
            "resident_compute_profile_parameter_count_mismatch"
            in certificate["reasons"]
        )

    def test_the_profile_must_be_the_preregistered_one(self):
        bundle = _bundle()
        bundle["preregistration"]["compute_profile_sha256"] = "5" * 64
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "resident_compute_profile_not_preregistered" in certificate["reasons"]

    def test_a_missing_preregistered_profile_is_named(self):
        bundle = _bundle()
        del bundle["preregistration"]["compute_profile_sha256"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "missing_compute_profile_sha256" in certificate["reasons"]

    @pytest.mark.parametrize("arm", ["treatment", "control"])
    def test_a_trial_costed_against_another_architecture_is_refused(self, arm):
        bundle = _bundle()
        trial = copy.deepcopy(bundle["trials"][0])
        other = ModelComputeProfile(
            model_type="qwen2",
            hidden_size=5120,
            intermediate_size=27648,
            num_hidden_layers=32,
            num_attention_heads=40,
            num_key_value_heads=8,
            vocab_size=152064,
            head_dim=128,
        )
        trial[f"{arm}_compute"]["resource_accounting"]["model_profile"] = (
            other.to_receipt()
        )
        bundle["trials"][0] = trial
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert any(
            reason.endswith(f":{arm}_resource_accounting_invalid")
            or reason.endswith(f":{arm}_compute_profile_not_preregistered")
            for reason in certificate["reasons"]
        )


class TestLayerApplicationAccounting:
    @pytest.mark.parametrize("arm", ["treatment", "control"])
    def test_layer_apps_must_match_the_receipt_that_counts_them(self, arm):
        """Two counts of the same quantity, never compared."""
        bundle = _bundle()
        trial = copy.deepcopy(bundle["trials"][0])
        layers = _RESIDENT_PROFILE.num_hidden_layers
        trial[f"{arm}_compute"]["layer_apps"] = layers * 8
        bundle["trials"][0] = trial
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert any(
            reason.endswith(f":{arm}_layer_apps_not_accounted")
            for reason in certificate["reasons"]
        )

    @pytest.mark.parametrize("arm", ["treatment", "control"])
    def test_a_count_that_is_not_whole_passes_is_refused(self, arm):
        """A forward pass applies the whole stack."""
        bundle = _bundle()
        trial = copy.deepcopy(bundle["trials"][0])
        trial[f"{arm}_compute"]["layer_apps"] = 1000
        bundle["trials"][0] = trial
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert any(
            reason.endswith(f":{arm}_layer_apps_not_whole_passes")
            for reason in certificate["reasons"]
        )

    def test_a_trial_over_its_forward_pass_budget_is_refused(self):
        bundle = _bundle()
        bundle["preregistration"]["max_forward_passes_per_trial"] = 4
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert any(
            reason.endswith(":treatment_forward_passes_over_budget")
            for reason in certificate["reasons"]
        )

    def test_the_budget_must_be_preregistered(self):
        bundle = _bundle()
        del bundle["preregistration"]["max_forward_passes_per_trial"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "invalid_max_forward_passes_per_trial" in certificate["reasons"]

    @pytest.mark.parametrize("value", [0, -1, 4.5, True])
    def test_the_budget_must_be_a_positive_integer(self, value):
        bundle = _bundle()
        bundle["preregistration"]["max_forward_passes_per_trial"] = value
        certificate = _certify(bundle)
        assert "invalid_max_forward_passes_per_trial" in certificate["reasons"]
