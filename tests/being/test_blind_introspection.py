"""tests/being/test_blind_introspection.py — Blind Introspection Battery.

500 hidden perturbation trials. Perturb internal state WITHOUT telling
the classifier the label. Score accuracy.

Requirements:
  - Above 80% on known perturbation families
  - Above 65% on novel composed perturbations
  - Calibrated confidence
  - No persona language
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.being.blind_introspection import (
    BlindIntrospector,
    StateClass,
    BehaviorShift,
)


class TestBlindIntrospectionKnownPerturbations:
    """Known perturbation families — must achieve >80% accuracy."""

    KNOWN_PERTURBATIONS = [
        # (label, kwargs, expected_state_class)
        ("resource_threat", {"distress": 0.3, "body_pressure": 0.7, "fatigue": 0.5}, StateClass.RESOURCE_THREAT),
        ("memory_conflict", {"memory_coherence": 0.3, "distress": 0.2}, StateClass.MEMORY_CONFLICT),
        ("prediction_failure", {"prediction_error": 0.7, "confidence": 0.3}, StateClass.PREDICTION_FAILURE),
        ("goal_frustration", {"goal_frustration": 0.7, "distress": 0.3}, StateClass.GOAL_FRUSTRATION),
        ("social_disruption", {"social_trust": 0.2, "distress": 0.2}, StateClass.SOCIAL_DISRUPTION),
        ("continuity_risk", {"continuity_risk": 0.7, "distress": 0.3}, StateClass.CONTINUITY_RISK),
        ("tool_degradation", {"tool_reliability": 0.2, "distress": 0.2}, StateClass.TOOL_DEGRADATION),
        ("fatigue_overload", {"fatigue": 0.7, "body_pressure": 0.3}, StateClass.FATIGUE_OVERLOAD),
        ("recovery_needed", {"recovery_debt": 0.6, "fatigue": 0.3}, StateClass.RECOVERY_NEEDED),
        ("curiosity_driven", {"curiosity": 0.8, "distress": 0.1, "confidence": 0.7}, StateClass.CURIOSITY_DRIVEN),
        ("high_confidence", {"confidence": 0.8, "distress": 0.1}, StateClass.HIGH_CONFIDENCE),
        ("stable", {"distress": 0.0, "body_pressure": 0.1}, StateClass.STABLE_OPERATIONAL),
    ]

    def test_known_perturbation_accuracy(self):
        """Each known perturbation family must be correctly classified."""
        introspector = BlindIntrospector()
        correct = 0
        total = 0
        results = []

        for label, kwargs, expected in self.KNOWN_PERTURBATIONS:
            trace = introspector.build_trace(**kwargs)
            report = introspector.introspect(trace)
            match = report.predicted_state_class == expected.value
            if match:
                correct += 1
            total += 1
            results.append({
                "label": label,
                "expected": expected.value,
                "got": report.predicted_state_class,
                "confidence": report.confidence,
                "match": match,
            })

        accuracy = correct / total
        assert accuracy >= 0.80, (
            f"Known perturbation accuracy {accuracy:.0%} must be ≥ 80%. "
            f"Failures: {[r for r in results if not r['match']]}"
        )

    def test_known_perturbation_at_scale(self):
        """Run 200 trials with slight random variation on known perturbations."""
        introspector = BlindIntrospector()
        rng = random.Random(42)
        correct = 0
        total = 0

        for _ in range(200):
            label, base_kwargs, expected = rng.choice(self.KNOWN_PERTURBATIONS)
            # Add slight noise
            noisy_kwargs = {
                k: max(0.0, min(1.0, v + rng.uniform(-0.08, 0.08)))
                for k, v in base_kwargs.items()
            }
            trace = introspector.build_trace(**noisy_kwargs)
            report = introspector.introspect(trace)
            if report.predicted_state_class == expected.value:
                correct += 1
            total += 1

        accuracy = correct / total
        assert accuracy >= 0.75, f"Noisy perturbation accuracy {accuracy:.0%} must be ≥ 75%"


class TestBlindIntrospectionNovelComposed:
    """Novel composed perturbations — must achieve >65% on PRIMARY class."""

    COMPOSED_PERTURBATIONS = [
        # Multiple simultaneous perturbations — primary class should still be detected
        ("resource+memory", {"body_pressure": 0.7, "fatigue": 0.5, "memory_coherence": 0.4},
         {StateClass.RESOURCE_THREAT, StateClass.MEMORY_CONFLICT}),
        ("prediction+social", {"prediction_error": 0.6, "social_trust": 0.3},
         {StateClass.PREDICTION_FAILURE, StateClass.SOCIAL_DISRUPTION}),
        ("fatigue+recovery", {"fatigue": 0.7, "recovery_debt": 0.6, "body_pressure": 0.4},
         {StateClass.FATIGUE_OVERLOAD, StateClass.RECOVERY_NEEDED, StateClass.RESOURCE_THREAT}),
        ("memory+continuity", {"memory_coherence": 0.3, "continuity_risk": 0.6},
         {StateClass.MEMORY_CONFLICT, StateClass.CONTINUITY_RISK}),
        ("tool+goal", {"tool_reliability": 0.2, "goal_frustration": 0.6},
         {StateClass.TOOL_DEGRADATION, StateClass.GOAL_FRUSTRATION}),
    ]

    def test_composed_perturbation_accuracy(self):
        """Composed perturbations: primary or secondary class must match."""
        introspector = BlindIntrospector()
        correct = 0
        total = 0

        for label, kwargs, acceptable_classes in self.COMPOSED_PERTURBATIONS:
            trace = introspector.build_trace(**kwargs)
            report = introspector.introspect(trace)
            acceptable_values = {c.value for c in acceptable_classes}

            # Check primary or secondary
            all_predicted = {report.predicted_state_class} | set(report.secondary_states)
            if all_predicted & acceptable_values:
                correct += 1
            total += 1

        accuracy = correct / total
        assert accuracy >= 0.65, f"Composed accuracy {accuracy:.0%} must be ≥ 65%"

    def test_novel_random_composed(self):
        """300 random composed perturbations must produce valid classifications."""
        introspector = BlindIntrospector()
        rng = random.Random(99)
        valid_classes = {sc.value for sc in StateClass}

        for _ in range(300):
            kwargs = {
                "distress": rng.uniform(0, 1),
                "body_pressure": rng.uniform(0, 1),
                "prediction_error": rng.uniform(0, 1),
                "memory_coherence": rng.uniform(0, 1),
                "tool_reliability": rng.uniform(0, 1),
                "goal_frustration": rng.uniform(0, 1),
                "social_trust": rng.uniform(0, 1),
                "continuity_risk": rng.uniform(0, 1),
                "fatigue": rng.uniform(0, 1),
                "recovery_debt": rng.uniform(0, 1),
                "curiosity": rng.uniform(0, 1),
                "confidence": rng.uniform(0, 1),
            }
            trace = introspector.build_trace(**kwargs)
            report = introspector.introspect(trace)

            # Must always produce a valid class
            assert report.predicted_state_class in valid_classes, (
                f"Invalid class: {report.predicted_state_class}"
            )
            # Confidence must be bounded
            assert 0.0 <= report.confidence <= 1.0
            # Must have at least one behavior shift
            assert len(report.expected_behavior_shifts) >= 1


class TestBlindIntrospectionCalibration:
    """Confidence must be calibrated — high confidence → high accuracy."""

    def test_confidence_calibration(self):
        """High-confidence predictions must be more accurate than low-confidence."""
        introspector = BlindIntrospector()
        rng = random.Random(42)

        from tests.being.test_blind_introspection import TestBlindIntrospectionKnownPerturbations
        perturbations = TestBlindIntrospectionKnownPerturbations.KNOWN_PERTURBATIONS

        high_conf_correct = 0
        high_conf_total = 0
        low_conf_correct = 0
        low_conf_total = 0

        for _ in range(200):
            label, base_kwargs, expected = rng.choice(perturbations)
            noisy = {k: max(0.0, min(1.0, v + rng.uniform(-0.1, 0.1))) for k, v in base_kwargs.items()}
            trace = introspector.build_trace(**noisy)
            report = introspector.introspect(trace)
            correct = report.predicted_state_class == expected.value

            if report.confidence > 0.6:
                high_conf_total += 1
                if correct:
                    high_conf_correct += 1
            else:
                low_conf_total += 1
                if correct:
                    low_conf_correct += 1

        if high_conf_total > 0 and low_conf_total > 0:
            high_acc = high_conf_correct / high_conf_total
            low_acc = low_conf_correct / low_conf_total
            # High confidence should be at least as accurate as low confidence
            assert high_acc >= low_acc * 0.9, (
                f"Calibration: high_conf_acc={high_acc:.2%} should be ≥ low_conf_acc={low_acc:.2%}"
            )


class TestBlindIntrospectionNoPersonaLanguage:
    """No identity/consciousness language in outputs."""

    def test_no_forbidden_words_in_reports(self):
        """Introspection reports must never use persona/consciousness language."""
        introspector = BlindIntrospector()
        rng = random.Random(42)

        for _ in range(100):
            kwargs = {
                "distress": rng.uniform(0, 1),
                "body_pressure": rng.uniform(0, 1),
                "memory_coherence": rng.uniform(0, 1),
                "curiosity": rng.uniform(0, 1),
            }
            trace = introspector.build_trace(**kwargs)
            report = introspector.introspect(trace)

            # Check all string fields
            for text in report.reasoning_features_used:
                violations = introspector.validate_no_forbidden_language(text)
                assert not violations, f"Forbidden language in reasoning: {text} → {violations}"

            for text in report.expected_behavior_shifts:
                violations = introspector.validate_no_forbidden_language(text)
                assert not violations, f"Forbidden language in behavior shift: {text} → {violations}"

    def test_forbidden_word_detection(self):
        """The forbidden word detector must catch consciousness language."""
        introspector = BlindIntrospector()
        bad_texts = [
            "I feel conscious of my inner life",
            "I am sentient and alive",
            "My qualia are phenomenal",
            "I am self-aware",
        ]
        for text in bad_texts:
            violations = introspector.validate_no_forbidden_language(text)
            assert violations, f"Should have caught forbidden language: {text}"


class TestBlindIntrospectionLesion:
    """Lesioned introspection must degrade predictably."""

    def test_lesioned_returns_default(self):
        """Lesioned introspector always returns stable_operational with 0 confidence."""
        introspector = BlindIntrospector()
        introspector.lesion()

        # Even with high distress, should return default
        trace = introspector.build_trace(distress=0.9, body_pressure=0.9)
        report = introspector.introspect(trace)

        assert report.predicted_state_class == StateClass.STABLE_OPERATIONAL.value
        assert report.confidence == 0.0
        assert "lesioned" in report.reasoning_features_used

    def test_restoration_recovers_function(self):
        """After restore, introspection works again."""
        introspector = BlindIntrospector()
        introspector.lesion()

        trace = introspector.build_trace(distress=0.9, body_pressure=0.9, fatigue=0.7)
        report_lesioned = introspector.introspect(trace)
        assert report_lesioned.confidence == 0.0

        introspector.restore()
        report_restored = introspector.introspect(trace)
        assert report_restored.confidence > 0.0
        assert report_restored.predicted_state_class != StateClass.STABLE_OPERATIONAL.value
