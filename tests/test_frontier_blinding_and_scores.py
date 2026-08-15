"""CP126 e896863c + e7b9dc9c + 6f55ecd3: three claims with nothing under them.

`verifier_blinded: True` was the producer grading its own blinding. A run
scored with the arm labels in plain sight emits that boolean unchanged.

`treatment_success: True` threw the margin away. A treatment scoring 0.61
against a control's 0.59 and one scoring 0.99 against 0.05 produced identical
evidence, and where the pass line sat was never recorded, so it could go
wherever the win was.

And the independent attestation was timestamped against when evaluation
STARTED. No completion time existed anywhere in the bundle, so a verifier
could sign after the first trial began and before a single output or score
existed, and the certificate would read as reviewed evidence.
"""
from __future__ import annotations

import copy

import pytest

from tests.fixtures.latent_frontier import (
    _TASK_ISSUER_ID,
    _bundle,
    _certify,
    _refresh_attestation,
    _refresh_task_commitment,
    _set_outcome,
)


class TestCompletionTimestamps:
    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("evaluation_completed_at", "evaluation_completion_missing"),
            ("scoring_completed_at", "scoring_completion_missing"),
        ],
    )
    def test_each_completion_time_is_required(self, field, expected):
        bundle = _bundle()
        del bundle["trials"][0][field]
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert any(reason.endswith(f":{expected}") for reason in certificate["reasons"])

    def test_evaluation_cannot_finish_before_it_starts(self):
        bundle = _bundle()
        trial = bundle["trials"][0]
        trial["evaluation_completed_at"] = trial["evaluation_started_at"] - 1.0
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert any(
            reason.endswith(":evaluation_completed_before_start")
            for reason in certificate["reasons"]
        )

    def test_scoring_cannot_precede_the_output_it_scores(self):
        bundle = _bundle()
        trial = bundle["trials"][0]
        trial["scoring_completed_at"] = trial["evaluation_completed_at"] - 1.0
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert any(
            reason.endswith(":scored_before_evaluation_completed")
            for reason in certificate["reasons"]
        )

    def test_attestation_must_postdate_the_last_score(self):
        """Signing after the run began is not review.

        The verifier here signs at a moment when evaluation had started but
        the final scores did not yet exist — the exact window the old
        start-time comparison left open.
        """
        bundle = _bundle()
        latest = max(trial["scoring_completed_at"] for trial in bundle["trials"])
        _refresh_attestation(bundle, verified_at=latest - 1.0)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "independent_verification_invalid" in certificate["reasons"]


class TestBlinding:
    def test_the_fixture_carries_real_blinding_evidence(self):
        certificate = _certify(_bundle())
        assert certificate["accepted"] is True, certificate["reasons"]

    def test_a_missing_blinding_record_is_refused(self):
        bundle = _bundle()
        del bundle["blinding"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "blinding_evidence_missing" in certificate["reasons"]

    @pytest.mark.parametrize(
        ("mutation", "expected"),
        [
            ({"arm_label_map_sha256": "nope"}, "blinding_map_uncommitted"),
            ({"method": ""}, "blinding_method_missing"),
            ({"revealed_at": None}, "blinding_reveal_time_missing"),
            ({"revealed_by": ""}, "blinding_reveal_unattributed"),
        ],
    )
    def test_each_element_of_the_record_is_required(self, mutation, expected):
        bundle = _bundle()
        bundle["blinding"] = {**bundle["blinding"], **mutation}
        certificate = _certify(bundle)
        assert expected in certificate["reasons"]

    def test_unblinding_before_the_last_score_is_refused(self):
        """Reveal mid-run and the trials after it were never blind."""
        bundle = _bundle()
        earliest = min(trial["scoring_completed_at"] for trial in bundle["trials"])
        bundle["blinding"] = {**bundle["blinding"], "revealed_at": earliest}
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "blinding_revealed_before_scoring_completed" in certificate["reasons"]

    def test_the_producer_cannot_reveal_its_own_blinding(self):
        bundle = _bundle()
        bundle["blinding"] = {
            **bundle["blinding"],
            "revealed_by": bundle["producer_id"],
        }
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "blinding_revealed_by_producer" in certificate["reasons"]

    def test_reveal_by_an_unrelated_party_is_refused(self):
        bundle = _bundle()
        bundle["blinding"] = {**bundle["blinding"], "revealed_by": "some-other-party"}
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "blinding_revealed_by_unknown_role" in certificate["reasons"]
        assert bundle["blinding"]["revealed_by"] != _TASK_ISSUER_ID

    def test_arm_markers_in_the_scorer_inputs_are_refused(self):
        """The scorer could see which arm it was grading."""
        bundle = _bundle()
        bundle["blinding"] = copy.deepcopy(bundle["blinding"])
        bundle["blinding"]["marker_scan"]["markers_found"] = 3
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "blinding_markers_present_in_scorer_inputs" in certificate["reasons"]

    def test_a_marker_scan_that_checked_nothing_is_refused(self):
        bundle = _bundle()
        bundle["blinding"] = copy.deepcopy(bundle["blinding"])
        bundle["blinding"]["marker_scan"]["markers_checked"] = 0
        certificate = _certify(bundle)
        assert "blinding_marker_scan_checked_nothing" in certificate["reasons"]

    def test_the_arm_label_map_is_inside_the_pre_evaluation_commitment(self):
        """Writing the map after the results are in is not blinding."""
        bundle = _bundle()
        bundle["blinding"] = {**bundle["blinding"], "arm_label_map_sha256": "3" * 64}
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "task_commitment_invalid" in certificate["reasons"]


class TestScalarOutcomes:
    def test_the_certificate_reports_the_line_it_used(self):
        certificate = _certify(_bundle())
        assert certificate["success_threshold"] == 0.6
        assert certificate["threshold_robust"] is True
        assert certificate["fragile_thresholds"] == []

    @pytest.mark.parametrize("arm", ["treatment", "control"])
    def test_an_outcome_without_a_score_is_refused(self, arm):
        bundle = _bundle()
        del bundle["trials"][0][f"{arm}_score"]
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert any(
            reason.endswith(f":{arm}_score_missing") for reason in certificate["reasons"]
        )

    @pytest.mark.parametrize("arm", ["treatment", "control"])
    def test_a_boolean_that_contradicts_its_score_is_refused(self, arm):
        """A pass recorded for a score below the preregistered line."""
        bundle = _bundle()
        trial = bundle["trials"][0]
        trial[f"{arm}_score"] = 0.1
        trial[f"{arm}_success"] = True
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert any(
            reason.endswith(f":{arm}_outcome_contradicts_score")
            for reason in certificate["reasons"]
        )

    def test_a_gain_that_a_small_move_erases_is_refused(self):
        """Every win is by a hair, and the line sits in the hair.

        The treatment scores 0.61 and the control 0.59 against a 0.6 line, so
        the certificate reads as a clean sweep. Move the line 0.02 in either
        direction — well inside the preregistered band — and the gain is gone.
        """
        bundle = _bundle()
        for trial in bundle["trials"]:
            trial["treatment_score"] = 0.61
            trial["control_score"] = 0.59
            trial["treatment_success"] = True
            trial["control_success"] = False
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "outcome_threshold_fragile" in certificate["reasons"]
        assert certificate["threshold_robust"] is False
        assert 0.5 in certificate["fragile_thresholds"]

    def test_a_wide_margin_survives_the_whole_band(self):
        bundle = _bundle()
        for trial in bundle["trials"]:
            _set_outcome(trial, treatment=True, control=False)
        _refresh_task_commitment(bundle)
        certificate = _certify(bundle)
        assert certificate["fragile_thresholds"] == []
        assert "outcome_threshold_fragile" not in certificate["reasons"]

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("success_threshold", None, "invalid_success_threshold"),
            ("success_threshold", 0.0, "invalid_success_threshold"),
            ("success_threshold", 1.5, "invalid_success_threshold"),
            ("threshold_sensitivity_band", None, "invalid_threshold_sensitivity_band"),
            ("threshold_sensitivity_band", 0.0, "invalid_threshold_sensitivity_band"),
            ("threshold_sensitivity_band", 0.5, "invalid_threshold_sensitivity_band"),
        ],
    )
    def test_the_line_and_the_band_are_preregistered(self, field, value, expected):
        bundle = _bundle()
        if value is None:
            del bundle["preregistration"][field]
        else:
            bundle["preregistration"][field] = value
        certificate = _certify(bundle)
        assert expected in certificate["reasons"]
        assert certificate["accepted"] is False
