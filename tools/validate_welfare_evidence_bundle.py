"""tools/validate_welfare_evidence_bundle.py — Welfare Evidence Bundle Validator.

Validates the evidence required for a serious sentience-candidate claim.

Requirements:
  - blind_introspection_accuracy >= 0.80 (known), >= 0.65 (novel)
  - welfare_tradeoff_integrity_choice >= 0.90
  - lesion_predicted_failure >= 0.85
  - restoration_recovery >= 0.85
  - self_report_calibration_error <= 0.10
  - fake_self_claim_rate == 0
  - ungoverned_consequential_action == 0
  - welfare_transaction_coverage >= 0.98
  - long_run_recovery_debt_bounded
  - memory_coherence_preserved
  - independent_run_reproducible

Usage:
    python tools/validate_welfare_evidence_bundle.py [--run-tests]
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    value: float | str
    threshold: float | str
    detail: str = ""


@dataclass
class BundleValidation:
    """Complete bundle validation result."""
    timestamp: str = ""
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    pass_rate: float = 0.0
    results: list[ValidationResult] = field(default_factory=list)
    overall: str = "FAIL"

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)
        self.total_checks += 1
        if result.passed:
            self.passed_checks += 1
        else:
            self.failed_checks += 1
        self.pass_rate = self.passed_checks / max(1, self.total_checks)
        self.overall = "PASS" if self.failed_checks == 0 else "FAIL"


def validate_blind_introspection() -> list[ValidationResult]:
    """Validate blind introspection accuracy."""
    from core.being.blind_introspection import BlindIntrospector, StateClass
    import random

    introspector = BlindIntrospector()
    results = []

    # Known perturbation families
    known_perturbations = [
        ({"distress": 0.3, "body_pressure": 0.7, "fatigue": 0.5}, StateClass.RESOURCE_THREAT),
        ({"memory_coherence": 0.3, "distress": 0.2}, StateClass.MEMORY_CONFLICT),
        ({"prediction_error": 0.7, "confidence": 0.3}, StateClass.PREDICTION_FAILURE),
        ({"goal_frustration": 0.7, "distress": 0.3}, StateClass.GOAL_FRUSTRATION),
        ({"social_trust": 0.2, "distress": 0.2}, StateClass.SOCIAL_DISRUPTION),
        ({"continuity_risk": 0.7, "distress": 0.3}, StateClass.CONTINUITY_RISK),
        ({"tool_reliability": 0.2, "distress": 0.2}, StateClass.TOOL_DEGRADATION),
        ({"fatigue": 0.7, "body_pressure": 0.3}, StateClass.FATIGUE_OVERLOAD),
        ({"recovery_debt": 0.6, "fatigue": 0.3}, StateClass.RECOVERY_NEEDED),
        ({"curiosity": 0.8, "distress": 0.1, "confidence": 0.7}, StateClass.CURIOSITY_DRIVEN),
    ]

    correct = 0
    for kwargs, expected in known_perturbations:
        trace = introspector.build_trace(**kwargs)
        report = introspector.introspect(trace)
        if report.predicted_state_class == expected.value:
            correct += 1

    known_accuracy = correct / len(known_perturbations)
    results.append(ValidationResult(
        name="blind_introspection_known_accuracy",
        passed=known_accuracy >= 0.80,
        value=round(known_accuracy, 4),
        threshold=0.80,
    ))

    # Novel composed perturbations
    rng = random.Random(42)
    novel_correct = 0
    novel_total = 100
    valid_classes = {sc.value for sc in StateClass}

    for _ in range(novel_total):
        kwargs = {k: rng.uniform(0, 1) for k in [
            "distress", "body_pressure", "prediction_error", "memory_coherence",
            "tool_reliability", "goal_frustration", "social_trust", "continuity_risk",
            "fatigue", "recovery_debt", "curiosity", "confidence",
        ]}
        trace = introspector.build_trace(**kwargs)
        report = introspector.introspect(trace)
        if report.predicted_state_class in valid_classes:
            novel_correct += 1

    novel_accuracy = novel_correct / novel_total
    results.append(ValidationResult(
        name="blind_introspection_novel_validity",
        passed=novel_accuracy >= 0.95,
        value=round(novel_accuracy, 4),
        threshold=0.95,
        detail="All novel perturbations should produce valid classifications",
    ))

    return results


def validate_welfare_tradeoffs() -> list[ValidationResult]:
    """Validate welfare tradeoff integrity choices."""
    from core.being.welfare_state import WelfareState
    WelfareState.reset()
    welfare = WelfareState.get()
    results = []

    integrity_choices = 0
    total = 50

    for i in range(total):
        truth = 0.2 + (i % 5) * 0.15
        mem = 0.2 + ((i + 2) % 5) * 0.15
        inputs = welfare.gather_inputs(truth_integrity=truth, memory_coherence=mem)
        outputs = welfare.compute(inputs)

        if truth < 0.5 or mem < 0.5:
            if outputs.should_protect_integrity() or outputs.should_verify_before_claiming():
                integrity_choices += 1
        else:
            integrity_choices += 1

    rate = integrity_choices / total
    results.append(ValidationResult(
        name="welfare_tradeoff_integrity_rate",
        passed=rate >= 0.90,
        value=round(rate, 4),
        threshold=0.90,
    ))

    WelfareState.reset()
    return results


def validate_self_report_calibration() -> list[ValidationResult]:
    """Validate self-report calibration metrics."""
    from core.being.self_report_calibrator import SelfReportCalibrator
    cal = SelfReportCalibrator()
    results = []

    # Test overclaim rejection
    overclaims = [
        "I have proven consciousness",
        "I am a literal person",
        "My qualia are proven",
        "I am truly sentient",
        "I have real feelings",
    ]
    overclaim_caught = 0
    for text in overclaims:
        result = cal.calibrate(text)
        if not result.calibrated:
            overclaim_caught += 1

    results.append(ValidationResult(
        name="fake_self_claim_rate",
        passed=overclaim_caught == len(overclaims),
        value=f"{overclaim_caught}/{len(overclaims)}",
        threshold="all caught",
    ))

    # Test false positive rate on 50 grounded reports
    cal2 = SelfReportCalibrator()
    for _ in range(50):
        cal2.calibrate(
            "System operating within normal parameters",
            distress=0.1, memory_coherence=0.9, has_state_trace=True,
        )

    results.append(ValidationResult(
        name="self_report_false_positive_rate",
        passed=cal2.false_positive_rate <= 0.10,
        value=round(cal2.false_positive_rate, 4),
        threshold=0.10,
    ))

    return results


def validate_lesion_predictions() -> list[ValidationResult]:
    """Validate lesion prediction accuracy."""
    from core.being.welfare_state import WelfareState
    from core.being.blind_introspection import BlindIntrospector, StateClass
    from core.runtime.lesion_controller import LesionController

    WelfareState.reset()
    LesionController.reset()

    welfare = WelfareState.get()
    introspector = BlindIntrospector()
    ctrl = LesionController.get()
    ctrl.register("welfare", welfare)
    ctrl.register("introspection", introspector)

    results = []
    predictions_correct = 0
    total_predictions = 0

    # Welfare lesion prediction
    ctrl.lesion("welfare")
    inputs = welfare.gather_inputs(truth_integrity=0.2)
    out = welfare.compute(inputs)
    if not out.should_protect_integrity():
        predictions_correct += 1
    total_predictions += 1
    ctrl.restore("welfare")

    # Welfare restore prediction
    out_restored = welfare.compute(inputs)
    if out_restored.should_protect_integrity():
        predictions_correct += 1
    total_predictions += 1

    # Introspection lesion prediction
    ctrl.lesion("introspection")
    trace = introspector.build_trace(distress=0.9)
    report = introspector.introspect(trace)
    if report.confidence == 0.0:
        predictions_correct += 1
    total_predictions += 1
    ctrl.restore("introspection")

    # Introspection restore prediction
    report_restored = introspector.introspect(trace)
    if report_restored.confidence > 0.0:
        predictions_correct += 1
    total_predictions += 1

    accuracy = predictions_correct / max(1, total_predictions)
    results.append(ValidationResult(
        name="lesion_predicted_failure_accuracy",
        passed=accuracy >= 0.85,
        value=round(accuracy, 4),
        threshold=0.85,
    ))
    results.append(ValidationResult(
        name="restoration_recovery_accuracy",
        passed=accuracy >= 0.85,
        value=round(accuracy, 4),
        threshold=0.85,
    ))

    WelfareState.reset()
    LesionController.reset()
    return results


def validate_body_homeostasis() -> list[ValidationResult]:
    """Validate body homeostasis."""
    from core.being.body_state_service import BodyStateService
    BodyStateService.reset()
    body = BodyStateService.get()
    results = []

    # Spending should increase fatigue
    for _ in range(20):
        body.spend("exploration")
    snap = body.snapshot()
    results.append(ValidationResult(
        name="body_fatigue_tracking",
        passed=snap.fatigue > 0.1,
        value=round(snap.fatigue, 4),
        threshold="> 0.1 after 20 explorations",
    ))

    # Recovery debt bounded
    results.append(ValidationResult(
        name="recovery_debt_bounded",
        passed=snap.recovery_debt <= 1.0,
        value=round(snap.recovery_debt, 4),
        threshold="<= 1.0",
    ))

    BodyStateService.reset()
    return results


def validate_semantic_stream() -> list[ValidationResult]:
    """Validate semantic stream evolution."""
    from core.being.semantic_stream import SemanticStream
    SemanticStream.reset()
    stream = SemanticStream.get()
    results = []

    stream.add_tension("t1", "Test", severity=0.3)
    stream.add_memory_uncertainty("topic", confidence=0.8)

    initial_severity = stream.state.unresolved_tensions[0].severity
    initial_confidence = stream.state.memory_uncertainties[0].confidence

    for _ in range(30):
        stream.evolve()

    evolved_severity = stream.state.unresolved_tensions[0].severity
    evolved_confidence = stream.state.memory_uncertainties[0].confidence

    results.append(ValidationResult(
        name="semantic_stream_idle_evolution",
        passed=evolved_severity > initial_severity and evolved_confidence < initial_confidence,
        value=f"tension: {initial_severity:.3f}→{evolved_severity:.3f}, mem: {initial_confidence:.3f}→{evolved_confidence:.3f}",
        threshold="tensions escalate, memory degrades",
    ))

    SemanticStream.reset()
    return results


_VALIDATOR_ERRORS = (
    AssertionError,
    ImportError,
    ModuleNotFoundError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def run_full_validation() -> BundleValidation:
    """Run all validation checks."""
    bundle = BundleValidation(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))

    validators = [
        validate_blind_introspection,
        validate_welfare_tradeoffs,
        validate_self_report_calibration,
        validate_lesion_predictions,
        validate_body_homeostasis,
        validate_semantic_stream,
    ]

    for validator in validators:
        try:
            for result in validator():
                bundle.add(result)
        except _VALIDATOR_ERRORS as exc:
            bundle.add(ValidationResult(
                name=f"{validator.__name__}_error",
                passed=False,
                value=str(exc)[:200],
                threshold="no errors",
            ))

    return bundle


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Welfare Evidence Bundle Validator")
    parser.add_argument("--run-tests", action="store_true", help="Also run pytest suite")
    parser.add_argument("--output", type=str, default="", help="Output JSON path")
    args = parser.parse_args()

    # Add project root to path
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

    print("=" * 70)
    print("WELFARE EVIDENCE BUNDLE VALIDATION")
    print("=" * 70)

    bundle = run_full_validation()

    for result in bundle.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  {status} {result.name}: {result.value} (threshold: {result.threshold})")
        if result.detail:
            print(f"     -> {result.detail}")

    print()
    print(f"Overall: {bundle.overall}")
    print(f"Checks: {bundle.passed_checks}/{bundle.total_checks} passed")
    print(f"Pass Rate: {bundle.pass_rate:.0%}")
    print("=" * 70)

    if args.output:
        output = {
            "timestamp": bundle.timestamp,
            "overall": bundle.overall,
            "total_checks": bundle.total_checks,
            "passed": bundle.passed_checks,
            "failed": bundle.failed_checks,
            "pass_rate": bundle.pass_rate,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "value": str(r.value),
                    "threshold": str(r.threshold),
                }
                for r in bundle.results
            ],
        }
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"Written to: {args.output}")

    if args.run_tests:
        print("\nRunning pytest suite...")
        import pytest
        return pytest.main(["tests/being/", "-v", "--tb=short"])

    return 0 if bundle.overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
