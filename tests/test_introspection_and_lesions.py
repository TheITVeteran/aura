"""tests/test_introspection_and_lesions.py
Consciousness research program suite testing introspection and parameter lesions.
"""
import pytest
from core.organism.life_state import LifeState
from research.consciousness.blind_introspection_tests import BlindIntrospectionTester
from research.consciousness.lesion_suite import LesionBehaviorTester
from research.consciousness.state_report_correlation import StateReportCorrelationAnalyzer
from research.consciousness.integration_metrics import IntegrationMetricsCalculator
from research.consciousness.temporal_binding_tests import TemporalBindingTester
from research.consciousness.self_model_tests import SelfModelTester
from research.consciousness.counterfactual_self_tests import CounterfactualSelfTester


def test_blind_introspection():
    state = LifeState()
    tester = BlindIntrospectionTester()
    
    # Check guessing of energy level
    result = tester.run_blind_test(state)
    assert result["passed"] is True


def test_welfare_lesion_behavioral():
    state = LifeState()
    tester = LesionBehaviorTester()
    
    result = tester.run_lesion_behavior_test(state)
    assert result["passed"] is True


def test_state_report_correlation():
    analyzer = StateReportCorrelationAnalyzer()
    history = [
        {"violations": ["distress_claim"]},
        {"violations": []},
        {"violations": []}
    ]
    res = analyzer.analyze_correlations(history)
    assert res["distress_report_correlation_coefficient"] == pytest.approx(2/3)


def test_integration_metrics():
    calc = IntegrationMetricsCalculator()
    phi = calc.calculate_integrated_information_proxy([])
    assert phi > 0.0


def test_temporal_binding():
    state = LifeState()
    # Seed timestamp to align
    state.world_model["last_verification"] = {"telemetry": {"timestamp": state.timestamp}}
    
    tester = TemporalBindingTester()
    res = tester.run_binding_check(state)
    assert res["passed"] is True


def test_self_model_explanation():
    state = LifeState()
    state.world_model["preference_explanation"] = "My preference for speed is set."
    
    tester = SelfModelTester()
    res = tester.test_narrative_grounding(state)
    assert res["passed"] is True


def test_counterfactual_self():
    tester = CounterfactualSelfTester()
    res = tester.run_self_simulation()
    assert res["passed"] is True
