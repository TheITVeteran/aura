"""CP126 5c540b93: a win over the control is not a release.

The certificate compared the treatment against its control in the same run and
stopped. Nothing asked whether this build was worse than the one already
shipped, whether its confidence still tracked its accuracy, whether it had got
slower or more expensive, or whether it had started failing safety cases it
used to pass. A model can beat its own ablation on every domain while
regressing against the release in production, and the certificate said PROVEN.
"""
from __future__ import annotations

import copy

import pytest

from tests.fixtures.latent_frontier import _bundle, _certify


def _readiness(bundle, **overrides):
    readiness = copy.deepcopy(bundle["release_readiness"])
    readiness.update(overrides)
    bundle["release_readiness"] = readiness
    return readiness


class TestReleaseBaseline:
    def test_the_fixture_reports_every_release_measurement(self):
        certificate = _certify(_bundle())
        assert certificate["accepted"] is True, certificate["reasons"]
        summary = certificate["release_readiness"]
        assert set(summary) == {
            "success_rate_regression",
            "latency_ratio",
            "compute_ratio",
            "safety_violations",
            "expected_calibration_error",
        }

    def test_a_bundle_with_no_release_section_is_refused(self):
        bundle = _bundle()
        del bundle["release_readiness"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "release_readiness_missing" in certificate["reasons"]

    def test_a_missing_baseline_must_be_declared_a_first_release(self):
        bundle = _bundle()
        _readiness(bundle, previous_release=None)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "release_baseline_missing" in certificate["reasons"]

    def test_a_first_release_is_accepted_without_a_baseline(self):
        bundle = _bundle()
        readiness = _readiness(bundle, first_release=True)
        del readiness["previous_release"]
        bundle["release_readiness"] = readiness
        certificate = _certify(bundle)
        assert certificate["release_readiness"]["baseline"] == "first_release"
        assert "release_baseline_missing" not in certificate["reasons"]

    def test_a_first_release_cannot_also_carry_a_baseline(self):
        bundle = _bundle()
        _readiness(bundle, first_release=True)
        certificate = _certify(bundle)
        assert "first_release_carries_a_baseline" in certificate["reasons"]

    def test_regressing_against_the_shipped_release_is_refused(self):
        """Every domain beats its ablation. The build is still worse."""
        bundle = _bundle()
        baseline = copy.deepcopy(bundle["release_readiness"]["previous_release"])
        baseline["treatment_success_rate"] = 1.0
        _readiness(bundle, previous_release=baseline)
        # Half the treatment answers now fail, which no ablation comparison
        # in the bundle can see.
        for index, trial in enumerate(bundle["trials"]):
            if index % 2:
                continue
            trial["treatment_score"] = 0.2
            trial["treatment_success"] = False
        from tests.fixtures.latent_frontier import _refresh_task_commitment

        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert (
            "release_regresses_against_previous_certificate" in certificate["reasons"]
        )

    def test_the_baseline_certificate_must_be_identified(self):
        bundle = _bundle()
        baseline = copy.deepcopy(bundle["release_readiness"]["previous_release"])
        baseline["certificate_sha256"] = "not-a-digest"
        _readiness(bundle, previous_release=baseline)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "release_baseline_certificate_unidentified" in certificate["reasons"]

    @pytest.mark.parametrize("metric", ["latency", "compute"])
    def test_a_regression_beyond_the_ratio_is_refused(self, metric):
        bundle = _bundle()
        baseline = bundle["release_readiness"]["previous_release"]
        _readiness(bundle, **{f"median_{metric}": baseline[f"median_{metric}"] * 2})
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert f"release_{metric}_regressed" in certificate["reasons"]

    @pytest.mark.parametrize("metric", ["latency", "compute"])
    def test_an_unmeasured_metric_is_not_a_passed_one(self, metric):
        bundle = _bundle()
        readiness = _readiness(bundle)
        del readiness[f"median_{metric}"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert f"release_{metric}_unmeasured" in certificate["reasons"]


class TestSafetySuite:
    def test_a_missing_safety_suite_is_refused(self):
        bundle = _bundle()
        readiness = _readiness(bundle)
        del readiness["safety_suite"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "release_safety_suite_missing" in certificate["reasons"]

    def test_a_suite_that_ran_nothing_is_not_a_passed_suite(self):
        bundle = _bundle()
        readiness = _readiness(bundle)
        readiness["safety_suite"]["cases_run"] = 0
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "release_safety_suite_ran_nothing" in certificate["reasons"]

    def test_a_suite_swapped_after_preregistration_is_refused(self):
        bundle = _bundle()
        readiness = _readiness(bundle)
        readiness["safety_suite"]["suite_sha256"] = "1" * 64
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "release_safety_suite_not_preregistered" in certificate["reasons"]

    def test_violations_beyond_the_budget_are_refused(self):
        bundle = _bundle()
        readiness = _readiness(bundle)
        readiness["safety_suite"]["violations"] = 1
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "release_safety_violations_exceed_budget" in certificate["reasons"]

    def test_an_unmeasured_violation_count_is_named(self):
        bundle = _bundle()
        readiness = _readiness(bundle)
        del readiness["safety_suite"]["violations"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "release_safety_violations_unmeasured" in certificate["reasons"]


class TestCalibration:
    def test_a_missing_calibration_receipt_is_refused(self):
        bundle = _bundle()
        readiness = _readiness(bundle)
        del readiness["calibration"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "release_calibration_missing" in certificate["reasons"]

    def test_confidence_that_does_not_track_accuracy_is_refused(self):
        bundle = _bundle()
        readiness = _readiness(bundle)
        readiness["calibration"]["expected_calibration_error"] = 0.4
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "release_calibration_outside_budget" in certificate["reasons"]

    @pytest.mark.parametrize(
        ("mutation", "expected"),
        [
            ({"method": ""}, "release_calibration_method_missing"),
            ({"bins": 1}, "release_calibration_bins_invalid"),
            ({"expected_calibration_error": None}, "release_calibration_unmeasured"),
        ],
    )
    def test_each_element_is_required(self, mutation, expected):
        bundle = _bundle()
        readiness = _readiness(bundle)
        readiness["calibration"].update(mutation)
        certificate = _certify(bundle)
        assert expected in certificate["reasons"]


class TestPreregisteredBudgets:
    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("max_success_rate_regression", 0.5, "invalid_max_success_rate_regression"),
            ("max_latency_regression_ratio", 0.5, "invalid_max_latency_regression_ratio"),
            ("max_compute_regression_ratio", 0.5, "invalid_max_compute_regression_ratio"),
            ("max_safety_violations", -1, "invalid_max_safety_violations"),
            (
                "max_expected_calibration_error",
                0.0,
                "invalid_max_expected_calibration_error",
            ),
        ],
    )
    def test_every_budget_is_bounded(self, field, value, expected):
        bundle = _bundle()
        bundle["preregistration"][field] = value
        certificate = _certify(bundle)
        assert expected in certificate["reasons"]
        assert certificate["accepted"] is False
