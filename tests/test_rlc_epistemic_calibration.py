"""Measured calibration contracts for RLC claim uncertainty."""

from __future__ import annotations

import copy

import pytest

from core.brain.llm.latent_cortex.epistemic_calibration import (
    CalibrationError,
    CalibrationObservation,
    CalibrationPolicy,
    CalibrationProfile,
)
from core.brain.llm.latent_cortex.epistemic_state import text_sha256


def observations(
    *,
    high_prediction: float = 0.9,
    low_prediction: float = 0.1,
) -> tuple[CalibrationObservation, ...]:
    rows: list[CalibrationObservation] = []
    for index in range(40):
        high = index >= 20
        prediction = high_prediction if high else low_prediction
        outcome = index < 2 if not high else index < 38
        rows.append(
            CalibrationObservation(
                observation_id=f"obs.{index:03d}",
                domain="general",
                predicted_probability=prediction,
                outcome=outcome,
                prediction_receipt_sha256=text_sha256(f"prediction:{index}"),
                outcome_receipt_sha256=text_sha256(f"outcome:{index}"),
                outcome_verifier_id="heldout_exact_grader",
                outcome_verifier_version="v3",
                observed_at=float(index + 1),
            )
        )
    return tuple(rows)


def profile(
    *,
    rows: tuple[CalibrationObservation, ...] | None = None,
) -> CalibrationProfile:
    return CalibrationProfile.fit(
        profile_id="cal.general.v1",
        estimator_id="rlc_claim_head",
        estimator_version="adapter.42",
        domain="general",
        dataset_sha256=text_sha256("heldout-dataset"),
        split_manifest_sha256=text_sha256("heldout-split"),
        trained_at=100.0,
        expires_at=1_000.0,
        observations=observations() if rows is None else rows,
        policy=CalibrationPolicy(
            bins=5,
            min_samples=40,
            min_bin_samples=12,
            max_brier=0.2,
            max_ece=0.1,
            support_lower_bound=0.7,
        ),
    )


def test_profile_reports_proper_scores_reliability_and_roundtrips_exactly():
    fitted = profile()
    assert fitted.passed is True
    assert fitted.failure_reasons == ()
    assert fitted.brier_score == pytest.approx(0.09)
    assert fitted.baseline_brier_score == pytest.approx(0.25)
    assert fitted.expected_calibration_error == pytest.approx(0.0)
    assert fitted.maximum_calibration_error == pytest.approx(0.0)
    assert len(fitted.reliability_bins) == 5
    assert sum(cell.count for cell in fitted.reliability_bins) == 40
    assert len(fitted.profile_sha256) == 64

    restored = CalibrationProfile.from_dict(fitted.to_dict())
    assert restored == fitted
    assert restored.to_dict() == fitted.to_dict()


def test_estimate_uses_empirical_bin_and_wilson_support_bound():
    fitted = profile()
    high = fitted.estimate(0.91, evaluated_at=200.0)
    assert high.supported is True
    assert high.abstention_reason == ""
    assert high.sample_count == 20
    assert high.point == pytest.approx(0.9)
    assert 0.7 < high.lower < high.point < high.upper < 1.0

    low = fitted.estimate(0.1, evaluated_at=200.0)
    assert low.supported is False
    assert "support_lower_bound_not_met" in low.abstention_reason
    assert low.point == pytest.approx(0.1)


def test_sparse_stale_and_not_yet_valid_profiles_abstain_explicitly():
    fitted = profile()
    sparse = fitted.estimate(0.5, evaluated_at=200.0)
    assert sparse.supported is False
    assert "sparse_calibration_bin" in sparse.abstention_reason

    early = fitted.estimate(0.9, evaluated_at=99.0)
    assert early.supported is False
    assert "profile_not_yet_valid" in early.abstention_reason

    stale = fitted.estimate(0.9, evaluated_at=1_001.0)
    assert stale.supported is False
    assert "profile_expired" in stale.abstention_reason


def test_nondiscriminative_or_miscalibrated_profile_fails_admission():
    bad_rows = tuple(
        CalibrationObservation(
            observation_id=f"bad.{index:03d}",
            domain="general",
            predicted_probability=0.9,
            outcome=index % 2 == 0,
            prediction_receipt_sha256=text_sha256(f"bad-prediction:{index}"),
            outcome_receipt_sha256=text_sha256(f"bad-outcome:{index}"),
            outcome_verifier_id="heldout_exact_grader",
            outcome_verifier_version="v3",
            observed_at=float(index + 1),
        )
        for index in range(40)
    )
    fitted = profile(rows=bad_rows)
    assert fitted.passed is False
    assert set(fitted.failure_reasons) == {
        "brier_above_limit",
        "does_not_beat_constant_predictor",
        "ece_above_limit",
    }
    estimate = fitted.estimate(0.9, evaluated_at=200.0)
    assert estimate.supported is False
    assert "profile_not_admitted" in estimate.abstention_reason


def test_calibration_rejects_self_grading_duplicates_and_future_observations():
    rows = list(observations())
    rows[0] = CalibrationObservation(
        observation_id=rows[0].observation_id,
        domain=rows[0].domain,
        predicted_probability=rows[0].predicted_probability,
        outcome=rows[0].outcome,
        prediction_receipt_sha256=rows[0].prediction_receipt_sha256,
        outcome_receipt_sha256=rows[0].outcome_receipt_sha256,
        outcome_verifier_id="rlc_claim_head",
        outcome_verifier_version="v3",
        observed_at=rows[0].observed_at,
    )
    with pytest.raises(CalibrationError, match="verifier must differ"):
        profile(rows=tuple(rows))

    rows = list(observations())
    rows[1] = CalibrationObservation(
        observation_id=rows[1].observation_id,
        domain=rows[1].domain,
        predicted_probability=rows[1].predicted_probability,
        outcome=rows[1].outcome,
        prediction_receipt_sha256=rows[0].prediction_receipt_sha256,
        outcome_receipt_sha256=rows[1].outcome_receipt_sha256,
        outcome_verifier_id=rows[1].outcome_verifier_id,
        outcome_verifier_version=rows[1].outcome_verifier_version,
        observed_at=rows[1].observed_at,
    )
    with pytest.raises(CalibrationError, match="duplicate"):
        profile(rows=tuple(rows))

    rows = list(observations())
    rows[0] = CalibrationObservation(
        observation_id=rows[0].observation_id,
        domain=rows[0].domain,
        predicted_probability=rows[0].predicted_probability,
        outcome=rows[0].outcome,
        prediction_receipt_sha256=rows[0].prediction_receipt_sha256,
        outcome_receipt_sha256=rows[0].outcome_receipt_sha256,
        outcome_verifier_id=rows[0].outcome_verifier_id,
        outcome_verifier_version=rows[0].outcome_verifier_version,
        observed_at=101.0,
    )
    with pytest.raises(CalibrationError, match="after profile training"):
        profile(rows=tuple(rows))


def test_profile_rejects_tampered_metrics_unknown_fields_and_empty_training():
    fitted = profile()
    tampered = copy.deepcopy(fitted.to_dict())
    tampered["brier_score"] = 0.0
    with pytest.raises(CalibrationError, match="metrics do not match"):
        CalibrationProfile.from_dict(tampered)

    unknown = copy.deepcopy(fitted.to_dict())
    unknown["confidence"] = "trust me"
    with pytest.raises(CalibrationError, match="unknown"):
        CalibrationProfile.from_dict(unknown)

    with pytest.raises(CalibrationError, match="empty"):
        profile(rows=())


def test_observation_requires_boolean_ground_truth_and_valid_policy():
    with pytest.raises(CalibrationError, match="boolean ground truth"):
        CalibrationObservation(
            observation_id="obs.invalid",
            domain="general",
            predicted_probability=0.8,
            outcome=1,  # type: ignore[arg-type]
            prediction_receipt_sha256=text_sha256("prediction"),
            outcome_receipt_sha256=text_sha256("outcome"),
            outcome_verifier_id="grader",
            outcome_verifier_version="v1",
            observed_at=1.0,
        )
    with pytest.raises(CalibrationError, match="min_bin_samples"):
        CalibrationPolicy(min_samples=5, min_bin_samples=6)
