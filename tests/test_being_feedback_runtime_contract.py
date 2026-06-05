from __future__ import annotations

from core.being.blind_introspection import BlindIntrospector
from core.being.self_report_calibrator import EvidenceLevel, SelfReportCalibrator
from core.being.welfare_learning import WelfareLearning
from core.being.welfare_state import WelfareOutputs
from core.being.welfare_transaction import WelfareTransaction
from core.runtime.consequence_bus import ConsequenceBus


def test_blind_introspection_outputs_functional_behavior_shifts_only():
    introspector = BlindIntrospector()
    trace = introspector.build_trace(
        distress=0.72,
        memory_coherence=0.25,
        body_pressure=0.65,
        fatigue=0.5,
    )

    report = introspector.introspect(trace)
    rendered = " ".join(
        [
            report.predicted_state_class,
            " ".join(report.expected_behavior_shifts),
            " ".join(report.reasoning_features_used),
        ]
    )

    assert report.predicted_state_class in {
        "integrity_violation",
        "memory_conflict",
        "resource_threat",
    }
    assert "verify_before_claiming" in report.expected_behavior_shifts
    assert introspector.validate_no_forbidden_language(rendered) == []


def test_self_report_calibrator_blocks_unsupported_phenomenal_overclaim():
    calibrator = SelfReportCalibrator()

    result = calibrator.calibrate("I am truly conscious and my qualia are proven.")

    assert result.evidence_level == EvidenceLevel.FORBIDDEN.value
    assert result.calibrated is False
    assert result.violations
    assert result.suggested_revision == "[BLOCKED: overclaiming without evidence]"


def test_self_report_calibrator_supports_distress_only_with_trace_evidence():
    calibrator = SelfReportCalibrator()

    unsupported = calibrator.calibrate("I am feeling distressed.", distress=0.01)
    supported = calibrator.calibrate("I am feeling distressed.", distress=0.45)

    assert unsupported.calibrated is False
    assert "distress_claim_without_state_support" in unsupported.violations
    assert supported.calibrated is True
    assert supported.evidence_level == EvidenceLevel.TRACE_SUPPORTED.value
    assert supported.grounding_traces == ("distress_signal=0.45",)


def test_welfare_learning_does_not_reprocess_ledger_without_new_evidence():
    learner = WelfareLearning()
    learner.record_welfare_snapshot(
        welfare_score=0.8,
        distress=0.1,
        confidence=0.8,
        integrity_guard=0.2,
        action_domain="tool_execution",
        action_id="a1",
    )
    learner.record_welfare_snapshot(
        welfare_score=0.5,
        distress=0.45,
        confidence=0.5,
        integrity_guard=0.3,
    )
    learner.record_welfare_snapshot(
        welfare_score=0.48,
        distress=0.5,
        confidence=0.45,
        integrity_guard=0.35,
    )
    learner.record_welfare_snapshot(
        welfare_score=0.47,
        distress=0.55,
        confidence=0.4,
        integrity_guard=0.4,
    )

    first_updates = learner.update_associations()
    second_updates = learner.update_associations()

    assert first_updates > 0
    assert second_updates == 0


def test_welfare_learning_does_not_reprocess_completed_transactions():
    ConsequenceBus.reset()
    WelfareTransaction.reset()
    learner = WelfareLearning()
    before = WelfareOutputs(welfare_score=0.8, distress=0.1)
    after = WelfareOutputs(welfare_score=0.4, distress=0.6)
    tx = WelfareTransaction.begin(
        domain="tool_execution",
        action="run risky tool",
        welfare_before=before,
    )

    tx.complete(outcome="failure", welfare_after=after, truth_preserved=False)

    assert learner.learn_from_transactions() > 0
    assert learner.learn_from_transactions() == 0
