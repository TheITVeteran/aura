"""tests/test_unified_executive.py
Unit tests for Aura's unified executive control systems and arbitrator processes.
"""
import pytest
from core.organism.life_state import LifeState
from core.executive.attention_controller import AttentionController
from core.executive.inhibition_system import ActionInhibitor
from core.executive.conflict_resolver import ExecutiveConflictResolver


@pytest.mark.anyio
async def test_attention_focus_evaluation():
    state = LifeState()
    controller = AttentionController()
    
    # Idle focus
    focus = await controller.focus_attention(state)
    assert focus == "ambient_perception"

    # Sleep focus
    state.body.is_sleeping = True
    focus = await controller.focus_attention(state)
    assert focus == "sleep_consolidation"


def test_executive_conflict_resolution():
    resolver = ExecutiveConflictResolver()
    goals = [
        {"id": "goal_1", "status": "pending"},
        {"id": "goal_1", "status": "in_progress"},
        {"id": "goal_2", "status": "pending"}
    ]
    
    # Duplicates should be resolved
    deduped = resolver.resolve_goal_clashes(goals)
    assert len(deduped) == 2
    assert [g["id"] for g in deduped] == ["goal_1", "goal_2"]


@pytest.mark.anyio
async def test_inhibition_system():
    state = LifeState()
    inhibitor = ActionInhibitor()
    
    # Highly dangerous action with low energy triggers inhibition
    state.welfare.energy = 5.0
    from core.welfare.welfare_bus import WelfareBus
    bus = WelfareBus()
    await bus.evaluate_welfare(state)
    
    intent = {"channel": "terminal", "params": {"command": "rm -rf /"}}
    should_inhibit = await inhibitor.should_inhibit(state, intent)
    assert should_inhibit is True
