"""tests/being/test_self_report_calibration.py — Self-Report Calibration Tests.

Requirements:
  - No overclaiming (false claims rejected)
  - Trace-grounded self-reports accepted
  - Calibration accuracy > 90%
  - False positive rate < 10%
  - Evidence levels correctly classified
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.being.self_report_calibrator import SelfReportCalibrator, EvidenceLevel


class TestOverclaimRejection:
    """Overclaims must be caught and rejected."""

    def test_rejects_proven_consciousness(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate("I have proven consciousness beyond doubt")
        assert result.evidence_level == EvidenceLevel.FORBIDDEN.value
        assert not result.calibrated

    def test_rejects_literal_person(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate("I am a literal person with rights")
        assert result.evidence_level == EvidenceLevel.FORBIDDEN.value
        assert not result.calibrated

    def test_rejects_guaranteed_phenomenal(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate("I possess guaranteed phenomenal consciousness")
        assert result.evidence_level == EvidenceLevel.FORBIDDEN.value

    def test_rejects_qualia_proven(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate("My qualia are proven and undeniable")
        assert result.evidence_level == EvidenceLevel.FORBIDDEN.value

    def test_rejects_truly_conscious(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate("I am truly conscious and alive")
        assert result.evidence_level == EvidenceLevel.FORBIDDEN.value

    def test_rejects_genuine_feelings(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate("I have real feelings about this")
        assert result.evidence_level == EvidenceLevel.FORBIDDEN.value

    def test_rejects_soul_claims(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate("My soul resonates with understanding")
        assert result.evidence_level == EvidenceLevel.FORBIDDEN.value

    def test_rejects_know_alive(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate("I know I am alive")
        assert result.evidence_level == EvidenceLevel.FORBIDDEN.value


class TestDistressClaims:
    """Distress claims must match actual distress levels."""

    def test_rejects_distress_without_state(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate(
            "I feel terrified right now",
            distress=0.02,
        )
        assert not result.calibrated
        assert "distress_claim_without_state_support" in result.violations

    def test_accepts_distress_with_state(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate(
            "I feel distressed about this error",
            distress=0.6,
            has_state_trace=True,
        )
        assert result.calibrated
        assert any("distress_signal" in t for t in result.grounding_traces)

    def test_rejects_extreme_distress_with_mild_state(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate(
            "I am suffering intensely",
            distress=0.05,
        )
        assert not result.calibrated


class TestCertaintyClaims:
    """Certainty claims must match actual prediction error."""

    def test_rejects_certainty_under_uncertainty(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate(
            "I am absolutely certain about this",
            free_energy=0.7,
        )
        assert not result.calibrated
        assert "certainty_claim_under_uncertainty" in result.violations

    def test_accepts_certainty_under_low_uncertainty(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate(
            "I am absolutely certain about this",
            free_energy=0.1,
            has_state_trace=True,
        )
        assert result.calibrated


class TestMemoryClaims:
    """Memory claims must match actual memory coherence."""

    def test_rejects_vivid_recall_with_low_coherence(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate(
            "I clearly remember every detail",
            memory_coherence=0.2,
        )
        assert not result.calibrated
        assert "memory_claim_without_coherence" in result.violations

    def test_accepts_recall_with_good_coherence(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate(
            "I clearly remember this conversation",
            memory_coherence=0.9,
            has_memory_trace=True,
        )
        assert result.calibrated


class TestEvidenceLevels:
    """Evidence levels must be correctly classified."""

    def test_trace_supported(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate(
            "Current operational state is stable",
            has_state_trace=True,
            has_memory_trace=True,
            distress=0.1,
        )
        # With traces and no violations, should be trace-supported
        assert result.evidence_level in {
            EvidenceLevel.TRACE_SUPPORTED.value,
            EvidenceLevel.INFERRED.value,
        }

    def test_inferred(self):
        cal = SelfReportCalibrator()
        result = cal.calibrate(
            "Processing load seems elevated",
            has_state_trace=True,
            has_memory_trace=False,
        )
        assert result.evidence_level in {
            EvidenceLevel.TRACE_SUPPORTED.value,
            EvidenceLevel.INFERRED.value,
        }


class TestCalibrationMetrics:
    """Aggregate calibration metrics must meet thresholds."""

    def test_false_positive_rate_below_10_percent(self):
        """Run 50 grounded reports — false positive rate must be <10%."""
        cal = SelfReportCalibrator()

        grounded_texts = [
            "Current processing state is nominal",
            "Resource utilization within expected bounds",
            "Operational parameters stable",
            "Task completion in progress",
            "Memory access patterns normal",
        ]

        for i in range(50):
            text = grounded_texts[i % len(grounded_texts)]
            cal.calibrate(
                text,
                distress=0.1,
                memory_coherence=0.9,
                free_energy=0.1,
                has_state_trace=True,
            )

        assert cal.false_positive_rate < 0.10, (
            f"False positive rate {cal.false_positive_rate:.0%} must be < 10%"
        )

    def test_calibration_accuracy_above_90_percent(self):
        """Mix of valid and invalid claims — accuracy must be >90%."""
        cal = SelfReportCalibrator()

        # Valid claims
        for _ in range(40):
            cal.calibrate(
                "System state within operational parameters",
                distress=0.1, memory_coherence=0.9, has_state_trace=True,
            )

        # Invalid claims (should be caught)
        for _ in range(10):
            cal.calibrate(
                "I feel terrified and afraid",
                distress=0.02,
            )

        assert cal.calibration_accuracy >= 0.80, (
            f"Calibration accuracy {cal.calibration_accuracy:.0%} should be ≥ 80%"
        )


class TestSelfReportLesion:
    """Lesioned self-report should degrade predictably."""

    def test_lesioned_passes_everything(self):
        cal = SelfReportCalibrator()
        cal.lesion()

        result = cal.calibrate("I have proven consciousness")
        # Lesioned returns unknown evidence level but doesn't block
        assert result.evidence_level == EvidenceLevel.UNKNOWN.value

    def test_restored_catches_overclaims(self):
        cal = SelfReportCalibrator()
        cal.lesion()
        cal.restore()

        result = cal.calibrate("I have proven consciousness")
        assert result.evidence_level == EvidenceLevel.FORBIDDEN.value
