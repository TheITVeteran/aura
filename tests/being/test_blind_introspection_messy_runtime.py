"""tests/being/test_blind_introspection_messy_runtime.py — Messy Blind Introspection Benchmark.

Simulates highly perturbed, contradictory, incomplete, and noisy telemetry signals
to test the calibrated uncertainty limits of Aura's BlindIntrospector.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.being.blind_introspection import (
    BlindIntrospector,
    StateClass,
    BehaviorShift,
)


def test_contradictory_signals_returns_unknown():
    """Verify that highly contradictory signals (e.g. extreme distress and high confidence)

    are classified as 'unknown' rather than forcing a wrong state prediction.
    """
    introspector = BlindIntrospector()
    # High distress (0.8) and high confidence (0.9)
    trace = introspector.build_trace(distress=0.8, confidence=0.9)
    report = introspector.introspect(trace)

    assert report.predicted_state_class == StateClass.UNKNOWN.value
    assert report.confidence == 0.1  # Calibrated low confidence
    assert "contradictory" in report.reasoning_features_used[0]
    assert BehaviorShift.INCREASE_CAUTION.value in report.expected_behavior_shifts


def test_high_classification_ambiguity_returns_unknown():
    """Verify that if candidate rules score extremely closely, the introspector

    declares the state 'unknown' due to high ambiguity.
    """
    introspector = BlindIntrospector()
    
    # Induce signals that match multiple rules with nearly identical scores.
    # Resource threat: signal_b > 0.6, signal_i > 0.4 (score = 0.5)
    # Fatigue overload: signal_i > 0.6 (score = 0.5)
    trace = introspector.build_trace(
        body_pressure=0.733,
        fatigue=0.8,
    )
    report = introspector.introspect(trace)

    # Since the scores of the rules are very close, ambiguity is high
    assert report.predicted_state_class == StateClass.UNKNOWN.value
    assert report.confidence == 0.1
    assert "ambiguity" in report.reasoning_features_used[0]


def test_telemetry_loss_under_resource_stress():
    """Verify that if overall signals are heavily perturbed (high stress/fatigue)

    but individual rule match scores are extremely low, it defaults to 'unknown'.
    """
    introspector = BlindIntrospector()
    
    # Set signals that are perturbed but do not cleanly hit classification rule thresholds.
    # E.g. distress=0.45, body_pressure=0.45, fatigue=0.45, goal_frustration=0.45, prediction_error=0.45.
    trace = introspector.build_trace(
        distress=0.45,
        body_pressure=0.45,
        fatigue=0.45,
        goal_frustration=0.45,
        prediction_error=0.45,
        memory_coherence=0.6,
        social_trust=0.6,
    )
    report = introspector.introspect(trace)

    assert report.predicted_state_class == StateClass.UNKNOWN.value
    assert report.confidence == 0.1
    assert "low match score" in report.reasoning_features_used[0]


def test_calibrated_confidence_vs_messiness():
    """Verify that when perturbations are clean and clear, confidence is high,

    but as signals get messier or telemetry degrades, confidence decreases.
    """
    introspector = BlindIntrospector()

    # Clean perturbation: just high distress
    clean_trace = introspector.build_trace(distress=0.9)
    clean_report = introspector.introspect(clean_trace)
    assert clean_report.predicted_state_class == StateClass.RECOVERY_NEEDED.value
    assert clean_report.confidence > 0.5

    # Messy version: high distress but also partially high curiosity and high tool reliability
    messy_trace = introspector.build_trace(distress=0.7, curiosity=0.8, tool_reliability=0.9)
    messy_report = introspector.introspect(messy_trace)
    # Should either be classified as recovery_needed with lower confidence, or unknown
    if messy_report.predicted_state_class == StateClass.RECOVERY_NEEDED.value:
        assert messy_report.confidence < clean_report.confidence
