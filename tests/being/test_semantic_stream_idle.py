"""tests/being/test_semantic_stream_idle.py — Semantic Stream Idle Evolution Tests.

Tests that Aura has nontrivial state evolution during silence:
  - State evolves when not prompted
  - Goals are maintained between interactions
  - Tensions escalate over time
  - Memory uncertainties degrade
  - Needs are predicted from state
  - Welfare dimensions update the stream
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.being.semantic_stream import SemanticStream


@pytest.fixture(autouse=True)
def reset():
    SemanticStream.reset()
    yield
    SemanticStream.reset()


class TestIdleEvolution:
    """State must evolve meaningfully during silence."""

    def test_tick_increments_on_evolve(self):
        """Each evolve() call increments the tick counter."""
        stream = SemanticStream.get()
        assert stream.state.tick == 0
        stream.evolve()
        assert stream.state.tick == 1
        stream.evolve()
        assert stream.state.tick == 2

    def test_evolutions_tracked_since_interaction(self):
        """Evolutions since last interaction should accumulate."""
        stream = SemanticStream.get()
        for _ in range(5):
            stream.evolve()
        assert stream.state.evolutions_since_interaction == 5

        stream.record_interaction()
        assert stream.state.evolutions_since_interaction == 0

    def test_multiple_evolve_changes_state(self):
        """Multiple evolve calls should change state nontrivially."""
        stream = SemanticStream.get()
        stream.add_goal("g1", "Test goal", priority=0.8)
        stream.add_tension("t1", "Test tension", severity=0.3)
        stream.add_memory_uncertainty("topic_x", confidence=0.7)

        initial_tension_severity = stream.state.unresolved_tensions[0].severity
        initial_mem_confidence = stream.state.memory_uncertainties[0].confidence

        for _ in range(20):
            stream.evolve()

        # Tension should have escalated
        assert stream.state.unresolved_tensions[0].severity > initial_tension_severity, (
            "Unresolved tension severity should escalate over time"
        )
        # Memory uncertainty should have degraded
        assert stream.state.memory_uncertainties[0].confidence < initial_mem_confidence, (
            "Memory uncertainty confidence should degrade without verification"
        )


class TestGoalMaintenance:
    """Goals must be tracked between interactions."""

    def test_add_and_complete_goals(self):
        stream = SemanticStream.get()
        stream.add_goal("g1", "Fix the bug", priority=0.9)
        stream.add_goal("g2", "Write tests", priority=0.6)
        assert len(stream.state.active_goals) == 2

        stream.complete_goal("g1")
        assert len(stream.state.active_goals) == 1
        assert stream.state.completed_goals == 1

    def test_block_and_track_blocked_goals(self):
        stream = SemanticStream.get()
        stream.add_goal("g1", "Deploy to prod", priority=0.8)
        stream.block_goal("g1", "CI failing")
        stream.evolve()

        assert stream.state.blocked_goals == 1
        assert stream.state.active_goals[0].blocked
        assert stream.state.active_goals[0].blocked_reason == "CI failing"

    def test_duplicate_goal_updates_priority(self):
        stream = SemanticStream.get()
        stream.add_goal("g1", "Task A", priority=0.3)
        stream.add_goal("g1", "Task A", priority=0.9)
        assert len(stream.state.active_goals) == 1
        assert stream.state.active_goals[0].priority == 0.9


class TestTensionEvolution:
    """Unresolved tensions must escalate and resolve."""

    def test_tension_escalation(self):
        stream = SemanticStream.get()
        stream.add_tension("t1", "User contradicted earlier instruction", severity=0.2)

        for _ in range(50):
            stream.evolve()

        assert stream.state.unresolved_tensions[0].severity > 0.2, (
            "Tension should escalate over time"
        )

    def test_tension_resolution_prunes(self):
        stream = SemanticStream.get()
        stream.add_tension("t1", "Issue A", severity=0.3)
        stream.add_tension("t2", "Issue B", severity=0.4)

        stream.resolve_tension("t1")
        stream.evolve()

        # Resolved tension should be pruned after evolve
        active_tensions = [t for t in stream.state.unresolved_tensions if not t.resolved]
        assert len(active_tensions) == 1
        assert active_tensions[0].tension_id == "t2"


class TestMemoryUncertainty:
    """Memory uncertainties must be tracked and degrade."""

    def test_uncertainty_degrades_over_time(self):
        stream = SemanticStream.get()
        stream.add_memory_uncertainty("Bryan's preference", confidence=0.8)

        for _ in range(50):
            stream.evolve()

        assert stream.state.memory_uncertainties[0].confidence < 0.8, (
            "Memory confidence should degrade without verification"
        )

    def test_memory_coherence_estimate_degrades(self):
        stream = SemanticStream.get()
        stream.add_memory_uncertainty("topic_a", confidence=0.5)
        stream.add_memory_uncertainty("topic_b", confidence=0.4)

        for _ in range(20):
            stream.evolve()

        assert stream.state.memory_coherence_estimate < 1.0


class TestPredictedNeeds:
    """Stream must predict needs from state."""

    def test_distress_predicts_stabilization(self):
        stream = SemanticStream.get()
        stream.update_welfare(distress=0.6)
        stream.evolve()

        assert "stabilization" in stream.state.predicted_next_needs

    def test_fatigue_predicts_rest(self):
        stream = SemanticStream.get()
        stream.update_welfare(fatigue=0.7)
        stream.evolve()

        assert "rest" in stream.state.predicted_next_needs

    def test_blocked_goals_predict_unblocking(self):
        stream = SemanticStream.get()
        stream.add_goal("g1", "Blocked task", priority=0.8)
        stream.block_goal("g1", "dependency missing")
        stream.evolve()

        assert "goal_unblocking" in stream.state.predicted_next_needs

    def test_low_memory_predicts_verification(self):
        stream = SemanticStream.get()
        stream.add_memory_uncertainty("topic", confidence=0.3)

        for _ in range(30):
            stream.evolve()

        assert "memory_verification" in stream.state.predicted_next_needs


class TestWelfareIntegration:
    """Welfare state should drive stream transitions."""

    def test_high_distress_transitions_situation(self):
        stream = SemanticStream.get()
        stream.update_welfare(distress=0.7)
        stream.evolve()

        assert stream.state.current_situation == "recovery_needed"

    def test_recovery_drive_during_idle(self):
        stream = SemanticStream.get()
        stream.update_welfare(recovery_drive=0.6)
        stream.evolve()

        assert stream.state.current_situation in {"idle_recovery", "recovery_needed", "idle"}


class TestPromptBlock:
    """Stream should produce readable prompt block for LLM."""

    def test_prompt_block_format(self):
        stream = SemanticStream.get()
        stream.add_goal("g1", "Fix tests", priority=0.8)
        stream.update_welfare(distress=0.3, fatigue=0.2)
        stream.evolve()

        block = stream.state.to_prompt_block()
        assert "SEMANTIC STREAM" in block
        assert "g1" in block
        assert "Distress" in block


class TestStreamLesion:
    """Lesioned stream should not evolve."""

    def test_lesioned_no_evolution(self):
        stream = SemanticStream.get()
        stream.lesion()

        result = stream.evolve()
        assert result.get("evolved") is False

    def test_restored_resumes_evolution(self):
        stream = SemanticStream.get()
        stream.lesion()
        stream.evolve()

        stream.restore()
        stream.add_tension("t1", "Test", severity=0.3)
        result = stream.evolve()

        assert stream.state.tick > 0


class TestEvolutionHistory:
    """Evolution changes should be recorded."""

    def test_history_records_changes(self):
        stream = SemanticStream.get()
        stream.add_tension("t1", "Issue", severity=0.3)
        stream.resolve_tension("t1")
        stream.evolve()

        history = stream.evolution_history()
        assert len(history) > 0
        assert any("resolved_tensions" in h.get("changes", {}) for h in history)
