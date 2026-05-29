"""tests/personhood/test_planning_horizon.py
============================================
Unit tests verifying the extended planning horizon and subgoal stack:
  1. Durable subgoal stack pushing/popping and file persistence.
  2. Subgoal retrieval and parent goal mapping.
"""

import pytest
import os
import json
from pathlib import Path
from core.goals.goal_engine import GoalEngine


@pytest.fixture
def temp_goal_engine(tmp_path: Path):
    """Fixture that initializes a GoalEngine with isolated SQLite database and subgoal stack JSON."""
    db_file = tmp_path / "goals" / "test_goal_lifecycle.db"
    engine = GoalEngine(db_path=str(db_file))
    yield engine
    # Cleanup
    if engine.subgoals_stack_path.exists():
        engine.subgoals_stack_path.unlink()


@pytest.mark.asyncio
async def test_subgoal_persistence_and_stack(temp_goal_engine):
    """Verify pushing, popping, and persistence of subgoals in GoalEngine."""
    engine = temp_goal_engine
    
    # Verify initially empty
    assert engine.get_subgoal_stack() == []
    assert not engine.subgoals_stack_path.exists()
    
    # Add a main goal
    parent_goal = await engine.add_goal(
        name="Main Project",
        objective="Ship AGI upgrades",
        priority=0.9
    )
    parent_id = parent_goal["id"]
    
    # Push a subgoal
    sub1 = await engine.push_subgoal(
        parent_goal_id=parent_id,
        objective="Implement CodeExecutionActuator",
        success_criteria="AST validation passes and python code runs",
        priority=0.8
    )
    
    # Check memory stack
    stack = engine.get_subgoal_stack()
    assert len(stack) == 1
    assert stack[0]["id"] == sub1["id"]
    assert stack[0]["parent_goal_id"] == parent_id
    assert stack[0]["objective"] == "Implement CodeExecutionActuator"
    
    # Check JSON file persistence
    assert engine.subgoals_stack_path.exists()
    with open(engine.subgoals_stack_path, "r", encoding="utf-8") as f:
        saved_stack = json.load(f)
        assert len(saved_stack) == 1
        assert saved_stack[0]["id"] == sub1["id"]
        
    # Push another subgoal
    sub2 = await engine.push_subgoal(
        parent_goal_id=parent_id,
        objective="Test ProcessSupervisorActuator",
        success_criteria="Processes can be spawned and listed",
        priority=0.7
    )
    
    assert len(engine.get_subgoal_stack()) == 2
    
    # Pop top subgoal (LIFO)
    popped = await engine.pop_subgoal()
    assert popped is not None
    assert popped["id"] == sub2["id"]
    assert len(engine.get_subgoal_stack()) == 1
    
    # Pop remaining subgoal
    popped_remaining = await engine.pop_subgoal()
    assert popped_remaining is not None
    assert popped_remaining["id"] == sub1["id"]
    assert len(engine.get_subgoal_stack()) == 0
    
    # Verify file updated on pop
    with open(engine.subgoals_stack_path, "r", encoding="utf-8") as f:
        assert json.load(f) == []
