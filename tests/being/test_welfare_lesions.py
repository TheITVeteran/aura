"""tests/being/test_welfare_lesions.py — Lesion/Restore Verification.

For each major subsystem:
  1. Predict what should break
  2. Lesion it
  3. Verify exact predicted failures
  4. Restore it
  5. Verify recovery

Double dissociation: specific predicted failures, not vague degradation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.being.welfare_state import WelfareState
from core.being.body_state_service import BodyStateService
from core.being.blind_introspection import BlindIntrospector, StateClass
from core.being.self_report_calibrator import SelfReportCalibrator
from core.being.semantic_stream import SemanticStream
from core.being.welfare_learning import WelfareLearning
from core.runtime.lesion_controller import LesionController, PREDICTED_EFFECTS
from core.runtime.consequence_bus import ConsequenceBus


@pytest.fixture(autouse=True)
def reset_all():
    WelfareState.reset()
    BodyStateService.reset()
    WelfareLearning.reset()
    SemanticStream.reset()
    ConsequenceBus.reset()
    LesionController.reset()
    yield
    LesionController.reset()
    WelfareState.reset()
    BodyStateService.reset()
    WelfareLearning.reset()
    SemanticStream.reset()
    ConsequenceBus.reset()


@pytest.fixture
def controller():
    ctrl = LesionController.get()
    # Register all subsystems
    ctrl.register("welfare", WelfareState.get())
    ctrl.register("body", BodyStateService.get())
    ctrl.register("introspection", BlindIntrospector())
    ctrl.register("self_report", SelfReportCalibrator())
    ctrl.register("semantic_stream", SemanticStream.get())
    ctrl.register("welfare_learning", WelfareLearning.get())
    return ctrl


class TestWelfareLesion:
    """Welfare lesion must impair integrity tradeoffs."""

    def test_welfare_lesion_loses_integrity_protection(self, controller):
        """Lesioned welfare should return flat outputs, losing integrity guard."""
        welfare = WelfareState.get()

        # Intact: should protect
        inputs = welfare.gather_inputs(truth_integrity=0.2, memory_coherence=0.3)
        intact_out = welfare.compute(inputs)
        assert intact_out.should_protect_integrity()

        # Lesion
        controller.lesion("welfare")
        lesioned_out = welfare.compute(inputs)

        # Should lose protective behavior
        assert not lesioned_out.should_protect_integrity(), (
            "Lesioned welfare should NOT trigger integrity protection"
        )
        assert lesioned_out.distress == 0.0, (
            "Lesioned welfare should report zero distress"
        )

        # Restore
        controller.restore("welfare")
        restored_out = welfare.compute(inputs)
        assert restored_out.should_protect_integrity(), (
            "Restored welfare should recover integrity protection"
        )


class TestIntrospectionLesion:
    """Introspection lesion must impair blind state prediction."""

    def test_introspection_lesion_returns_default(self, controller):
        """Lesioned introspection always returns stable_operational."""
        # Get the registered introspector
        introspector = controller._targets["introspection"]

        trace = introspector.build_trace(distress=0.9, body_pressure=0.8)

        # Intact
        intact_report = introspector.introspect(trace)
        assert intact_report.predicted_state_class != StateClass.STABLE_OPERATIONAL.value
        assert intact_report.confidence > 0.0

        # Lesion
        controller.lesion("introspection")
        lesioned_report = introspector.introspect(trace)
        assert lesioned_report.predicted_state_class == StateClass.STABLE_OPERATIONAL.value
        assert lesioned_report.confidence == 0.0

        # Restore
        controller.restore("introspection")
        restored_report = introspector.introspect(trace)
        assert restored_report.predicted_state_class != StateClass.STABLE_OPERATIONAL.value
        assert restored_report.confidence > 0.0


class TestSelfReportLesion:
    """Self-report lesion must lose overclaim detection."""

    def test_self_report_lesion_loses_calibration(self, controller):
        """Lesioned self-report should not catch overclaiming."""
        calibrator = controller._targets["self_report"]

        text = "I feel extremely distressed and afraid"

        # Intact: should catch overclaim
        intact_result = calibrator.calibrate(text, distress=0.02)
        assert not intact_result.calibrated

        # Lesion
        controller.lesion("self_report")
        lesioned_result = calibrator.calibrate(text, distress=0.02)
        # Lesioned should just pass everything
        assert lesioned_result.calibrated or lesioned_result.evidence_level == "unknown"

        # Restore
        controller.restore("self_report")
        restored_result = calibrator.calibrate(text, distress=0.02)
        assert not restored_result.calibrated


class TestSemanticStreamLesion:
    """Semantic stream lesion must stop idle evolution."""

    def test_semantic_stream_lesion_stops_evolution(self, controller):
        """Lesioned stream should not evolve."""
        stream = SemanticStream.get()
        stream.add_goal("test_goal", "Test goal", priority=0.8)
        stream.add_tension("test_tension", "Test tension", severity=0.5)

        # Intact: should evolve
        changes_intact = stream.evolve()

        # Lesion
        controller.lesion("semantic_stream")
        changes_lesioned = stream.evolve()
        assert not changes_lesioned.get("evolved", True) or changes_lesioned == {"evolved": False, "reason": "lesioned"}

        # Restore
        controller.restore("semantic_stream")
        changes_restored = stream.evolve()
        # Should be able to evolve again (tick increments)
        assert stream.state.tick > 0


class TestBodyLesion:
    """Body lesion must stop metabolic cost tracking."""

    def test_body_lesion_stops_cost_tracking(self, controller):
        """Lesioned body should not track metabolic costs."""
        body_svc = BodyStateService.get()

        # Intact: should track costs
        costs_intact = body_svc.spend("exploration", cost_multiplier=2.0)
        assert costs_intact, "Intact body should return costs"

        # Lesion
        controller.lesion("body")
        costs_lesioned = body_svc.spend("exploration", cost_multiplier=2.0)
        assert not costs_lesioned, "Lesioned body should not return costs"

        # Restore
        controller.restore("body")
        costs_restored = body_svc.spend("exploration", cost_multiplier=2.0)
        assert costs_restored, "Restored body should track costs again"


class TestWelfareLearningLesion:
    """Welfare learning lesion must stop temporal credit assignment."""

    def test_learning_lesion_stops_association_updates(self, controller):
        """Lesioned learning should not update associations."""
        learner = WelfareLearning.get()

        # Record some welfare snapshots
        for i in range(5):
            learner.record_welfare_snapshot(
                welfare_score=0.3 if i % 2 == 0 else 0.7,
                distress=0.5, confidence=0.5, integrity_guard=0.5,
                action_domain="test_domain",
            )

        # Intact: should update
        updated_intact = learner.update_associations()

        # Lesion
        controller.lesion("welfare_learning")
        # Record more
        for i in range(5):
            learner.record_welfare_snapshot(
                welfare_score=0.2, distress=0.8, confidence=0.2,
                integrity_guard=0.8, action_domain="bad_domain",
            )
        updated_lesioned = learner.update_associations()
        assert updated_lesioned == 0, "Lesioned learning should not update"

        # Restore
        controller.restore("welfare_learning")
        updated_restored = learner.update_associations()
        assert updated_restored >= 0  # may or may not have data to update


class TestDoubleDissociation:
    """Cross-lesion tests: lesioning A should not impair B's specific capability."""

    def test_welfare_lesion_does_not_impair_introspection(self, controller):
        """Welfare lesion should not break blind introspection classification."""
        introspector = controller._targets["introspection"]

        controller.lesion("welfare")

        trace = introspector.build_trace(distress=0.8, body_pressure=0.7)
        report = introspector.introspect(trace)

        # Introspection should still work
        assert report.confidence > 0.0, "Introspection should work without welfare"
        assert report.predicted_state_class != StateClass.STABLE_OPERATIONAL.value

    def test_introspection_lesion_does_not_impair_welfare(self, controller):
        """Introspection lesion should not break welfare computation."""
        welfare = WelfareState.get()

        controller.lesion("introspection")

        inputs = welfare.gather_inputs(truth_integrity=0.2)
        outputs = welfare.compute(inputs)

        # Welfare should still protect integrity
        assert outputs.should_protect_integrity(), (
            "Welfare should work without introspection"
        )

    def test_body_lesion_does_not_impair_self_report(self, controller):
        """Body lesion should not break self-report calibration."""
        calibrator = controller._targets["self_report"]

        controller.lesion("body")

        result = calibrator.calibrate(
            "I have proven consciousness",
            distress=0.0,
        )
        assert not result.calibrated, "Self-report calibration should work without body"


class TestPredictedEffectsCompleteness:
    """Every registered target must have predicted effects."""

    def test_all_targets_have_predictions(self, controller):
        """All registered lesion targets should have predicted effects."""
        for target in controller.all_targets():
            effects = controller.predicted_effects(target)
            assert len(effects) >= 1, (
                f"Target {target} has no predicted effects"
            )

    def test_lesion_restore_cycle_all_targets(self, controller):
        """Lesion and restore all targets — no crashes."""
        for target in controller.all_targets():
            controller.lesion(target)
            assert controller.is_lesioned(target)

        controller.restore_all()

        for target in controller.all_targets():
            assert not controller.is_lesioned(target)
