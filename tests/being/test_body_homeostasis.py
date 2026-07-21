"""tests/being/test_body_homeostasis.py — Digital Body Stress/Recovery Tests.

Tests that Aura has a body-like homeostatic loop:
  - Every action pays metabolic cost
  - Fatigue accumulates, decays over time
  - Failed actions create tool-failure pressure
  - Successful recovery creates relief
  - Body state adapts under healthy/strained/damaged/recovering states
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.being.aura_now import BodyState
from core.being.body_state_service import ACTION_COSTS, BodyStateService
from core.runtime.consequence_bus import ConsequenceBus


@pytest.fixture(autouse=True)
def reset_singletons():
    BodyStateService.reset()
    ConsequenceBus.reset()
    yield
    BodyStateService.reset()
    ConsequenceBus.reset()


class TestMetabolicCosts:
    """Every action domain must pay a metabolic cost."""

    def test_all_action_domains_have_costs(self):
        """Every defined domain must have non-zero cost."""
        for domain, costs in ACTION_COSTS.items():
            assert costs, f"Domain {domain} has no costs"
            # At least compute or fatigue
            has_metabolic = "compute" in costs or "fatigue" in costs or "recovery" in costs
            assert has_metabolic, f"Domain {domain} has no metabolic dimension"

    def test_spending_increases_fatigue(self):
        """Multiple tool executions should increase fatigue."""
        body = BodyStateService.get()
        snap_before = body.snapshot()

        for _ in range(10):
            body.spend("tool_execution")

        snap_after = body.snapshot()
        assert snap_after.fatigue > snap_before.fatigue, (
            f"Fatigue should increase: {snap_before.fatigue} → {snap_after.fatigue}"
        )

    def test_cost_quote_does_not_mutate_body(self):
        body = BodyStateService.get()
        before = (
            body.metabolic.compute_spent,
            body.metabolic.memory_spent,
            body.metabolic.fatigue,
            body.metabolic.recovery_debt,
            body.metabolic.tool_calls_total,
        )

        quote = body.estimate_cost("tool_execution", cost_multiplier=0.9)

        assert quote["compute"] > 0.0
        assert before == (
            body.metabolic.compute_spent,
            body.metabolic.memory_spent,
            body.metabolic.fatigue,
            body.metabolic.recovery_debt,
            body.metabolic.tool_calls_total,
        )

    def test_receipt_bound_cost_commit_is_idempotent(self):
        body = BodyStateService.get()
        quote = body.estimate_cost("tool_execution", cost_multiplier=0.9)

        first = body.commit_cost(quote, receipt_id="will-test-receipt")
        after_first = body.metabolic.compute_spent
        second = body.commit_cost(quote, receipt_id="will-test-receipt")

        assert first == quote
        assert second == {}
        assert body.metabolic.compute_spent == after_first
        assert body.metabolic.tool_calls_total == 1

    def test_heavy_spending_strains_body(self):
        """Heavy spending should cause strained state."""
        body = BodyStateService.get()

        for _ in range(30):
            body.spend("exploration", cost_multiplier=3.0)

        snap = body.snapshot()
        assert snap.fatigue > 0.3, f"Should be fatigued ({snap.fatigue})"

    def test_self_modification_costs_most(self):
        """Self-modification should cost the most."""
        body = BodyStateService.get()
        costs = body.spend("self_modification")
        assert costs.get("integrity_risk", 0) > 0, (
            "Self-modification should carry integrity risk"
        )
        assert costs.get("fatigue", 0) > 0.04, (
            "Self-modification should be expensive"
        )

    def test_stabilization_heals(self):
        """Stabilization actions should reduce fatigue."""
        body = BodyStateService.get()

        # First fatigue
        for _ in range(10):
            body.spend("exploration")
        snap_tired = body.snapshot()

        # Then stabilize
        for _ in range(5):
            body.spend("stabilization")
        snap_after = body.snapshot()

        assert snap_after.fatigue <= snap_tired.fatigue, (
            "Stabilization should not increase fatigue"
        )


class TestRecovery:
    """Recovery must create measurable relief."""

    def test_relieve_reduces_debt(self):
        """relieve() should reduce recovery debt."""
        body = BodyStateService.get()

        # Create recovery debt
        for _ in range(10):
            body.spend("self_modification")
        snap_before = body.snapshot()

        body.relieve(0.2)
        snap_after = body.snapshot()

        assert snap_after.recovery_debt < snap_before.recovery_debt, (
            "Relief should reduce recovery debt"
        )

    def test_relief_accumulates(self):
        """Relief should accumulate in the metabolic budget."""
        body = BodyStateService.get()
        body.relieve(0.1)
        body.relieve(0.15)
        assert body.metabolic.relief_accumulated >= 0.25


class TestConsequenceFeedback:
    """Body should react to consequence bus events."""

    def test_failure_increases_error_rate(self):
        """Failed actions from consequence bus should increase error rate."""
        body = BodyStateService.get()
        body.update_body(BodyState())  # trigger subscription
        bus = ConsequenceBus.get()

        for i in range(10):
            bus.publish_action(
                source="test", domain="tool_execution",
                action_content=f"fail_{i}", actual_outcome="failure",
                recovery_required=0.05,
            )

        snap = body.snapshot()
        assert snap.error_rate > 0.0, f"Error rate should be non-zero ({snap.error_rate})"
        assert body.metabolic.tool_calls_failed > 0

    def test_success_keeps_low_error_rate(self):
        """Successful actions should maintain low error rate."""
        body = BodyStateService.get()
        body.update_body(BodyState())
        bus = ConsequenceBus.get()

        for i in range(20):
            bus.publish_action(
                source="test", domain="tool_execution",
                action_content=f"success_{i}", actual_outcome="success",
            )

        snap = body.snapshot()
        assert snap.error_rate == 0.0, f"Error rate should be zero ({snap.error_rate})"


class TestBodyStates:
    """Body should behave differently under different conditions."""

    def test_healthy_state(self):
        """Fresh body is healthy."""
        body = BodyStateService.get()
        snap = body.snapshot()
        assert not snap.is_strained()
        assert not snap.is_critical()
        assert not snap.needs_recovery()
        assert snap.operational_health > 0.7

    def test_strained_state(self):
        """Heavy use creates strained state."""
        body = BodyStateService.get()
        for _ in range(30):
            body.spend("exploration", cost_multiplier=2.0)

        snap = body.snapshot()
        assert snap.fatigue > 0.3

    def test_recovering_state(self):
        """After strain, recovery should improve health."""
        body = BodyStateService.get()
        for _ in range(20):
            body.spend("exploration", cost_multiplier=2.0)

        snap_strained = body.snapshot()

        for _ in range(5):
            body.relieve(0.05)

        snap_recovered = body.snapshot()
        assert snap_recovered.recovery_debt <= snap_strained.recovery_debt

    def test_pressure_vector_is_complete(self):
        """Pressure vector should contain all dimensions."""
        body = BodyStateService.get()
        snap = body.snapshot()
        pv = snap.pressure_vector()

        expected_dims = {
            "cpu", "memory", "disk", "thermal", "battery",
            "latency", "permission", "network", "context", "sensor",
            "tool_failure", "model_availability", "memory_corruption_risk",
            "queue_backlog", "fatigue", "recovery_debt",
        }
        assert set(pv.keys()) == expected_dims, (
            f"Missing dimensions: {expected_dims - set(pv.keys())}"
        )


class TestBodyLesion:
    """Lesioned body should stop tracking."""

    def test_lesioned_body_no_costs(self):
        """Lesioned body returns empty costs."""
        body = BodyStateService.get()
        body.lesion()

        costs = body.spend("exploration")
        assert costs == {}, "Lesioned body should return no costs"

    def test_lesioned_body_restores(self):
        """Restored body resumes tracking."""
        body = BodyStateService.get()
        body.lesion()
        body.restore()

        costs = body.spend("exploration")
        assert costs, "Restored body should return costs"
