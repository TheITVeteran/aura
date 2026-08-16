"""CP126 d0f2ae6c: `params_unchanged` is a sampled claim wearing an absolute name.

The receipt says the parameters did not change. Behind it is a fixed-stride
canary over the parameter tree, hashed before and after the episode — a sample,
which is what makes it affordable on a 32B model and also what makes it
partial. A mutation living entirely in unsampled tensors leaves both digests
identical and the verdict reads "proven".

The certificate published that verdict without ever asking how much of the tree
the sample touched. It now measures the coverage, refuses a canary below the
preregistered floor, and names the method on the certificate so a reader cannot
mistake a sampled comparison for an exhaustive one.
"""
from __future__ import annotations

import copy

import pytest

from core.brain.llm.latent_cortex.frontier_certification import (
    PARAMETER_INTEGRITY_ATTESTATION,
    _parameter_canary_coverage,
)
from tests.fixtures.latent_frontier import _bundle, _certify, _refresh_task_commitment


def _receipt(before: dict | None = None, after: dict | None = None) -> dict:
    snapshot = {"parameter_leaf_count": 128, "sampled_tensor_count": 19}
    return {
        "runtime_integrity": {
            "parameters": {
                "before": {**snapshot, **(before or {})},
                "after": {**snapshot, **(after or {})},
            }
        }
    }


class TestCoverageMeasurement:
    def test_a_complete_pair_yields_the_sampled_fraction(self):
        assert _parameter_canary_coverage(_receipt()) == pytest.approx(19 / 128)

    def test_the_weaker_side_decides(self):
        """A thorough "before" cannot cover for a thin "after"."""
        coverage = _parameter_canary_coverage(
            _receipt(before={"sampled_tensor_count": 128}, after={"sampled_tensor_count": 4})
        )
        assert coverage == pytest.approx(4 / 128)

    @pytest.mark.parametrize(
        "receipt",
        [
            None,
            {},
            {"runtime_integrity": {}},
            {"runtime_integrity": {"parameters": {"before": {}}}},
        ],
    )
    def test_a_receipt_with_no_measurement_returns_nothing(self, receipt):
        assert _parameter_canary_coverage(receipt) is None

    @pytest.mark.parametrize(
        "snapshot",
        [
            {"parameter_leaf_count": 0},
            {"parameter_leaf_count": -1},
            {"sampled_tensor_count": -1},
            # More sampled tensors than exist is a broken measurement, not a
            # thorough one.
            {"sampled_tensor_count": 500},
        ],
    )
    def test_an_impossible_count_is_not_a_coverage_number(self, snapshot):
        assert _parameter_canary_coverage(_receipt(before=snapshot)) is None

    def test_a_canary_that_sampled_nothing_is_zero_not_none(self):
        """Absent and empty are different failures."""
        assert _parameter_canary_coverage(
            _receipt(before={"sampled_tensor_count": 0}, after={"sampled_tensor_count": 0})
        ) == 0.0


class TestCertificateGate:
    def test_the_certificate_names_what_backs_the_claim(self):
        certificate = _certify(_bundle())
        assert certificate["accepted"] is True, certificate["reasons"]
        assert (
            certificate["parameter_integrity_attestation"]
            == PARAMETER_INTEGRITY_ATTESTATION
        )
        assert certificate["min_parameter_canary_tensor_coverage"] == pytest.approx(
            19 / 128, abs=1e-6
        )

    def test_the_floor_must_be_preregistered(self):
        bundle = _bundle()
        del bundle["preregistration"]["min_parameter_canary_tensor_coverage"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert (
            "invalid_min_parameter_canary_tensor_coverage" in certificate["reasons"]
        )

    @pytest.mark.parametrize("value", [0.0, 1.5, "half"])
    def test_the_floor_is_bounded(self, value):
        bundle = _bundle()
        bundle["preregistration"]["min_parameter_canary_tensor_coverage"] = value
        certificate = _certify(bundle)
        assert (
            "invalid_min_parameter_canary_tensor_coverage" in certificate["reasons"]
        )

    def test_a_canary_thinner_than_the_floor_is_refused(self):
        """Two tensors out of 128 is not evidence the weights held still."""
        bundle = _bundle()
        trial = copy.deepcopy(bundle["trials"][0])
        parameters = trial["treatment_receipt"]["runtime_integrity"]["parameters"]
        for side in ("before", "after"):
            parameters[side] = {**parameters[side], "sampled_tensor_count": 2}
        bundle["trials"][0] = trial
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert any(
            reason.endswith(":treatment_parameter_canary_coverage_below_floor")
            for reason in certificate["reasons"]
        )

    def test_an_external_control_is_not_asked_for_a_canary(self):
        """A frontier control runs on somebody else's hardware.

        Demanding a parameter measurement from it would be demanding evidence
        that cannot exist, so the treatment arm carries the claim alone.
        """
        certificate = _certify(
            _bundle(comparison_kind="resident_32b_vs_external_frontier")
        )
        assert certificate["accepted"] is True, certificate["reasons"]
        assert not any(
            "control_parameter_canary" in reason for reason in certificate["reasons"]
        )

    def test_a_receipt_with_no_canary_at_all_is_refused(self):
        bundle = _bundle()
        trial = copy.deepcopy(bundle["trials"][0])
        del trial["control_receipt"]["runtime_integrity"]["parameters"]
        bundle["trials"][0] = trial
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert any(
            reason.endswith(":control_parameter_canary_unmeasured")
            for reason in certificate["reasons"]
        )
