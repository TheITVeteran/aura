"""Contracts for evaluator-only recurrent machine-state targets."""

from __future__ import annotations

import pytest

from core.learning.recurrence_curriculum import TASK_GENERATORS
from core.learning.recurrent_state_schema import (
    STATE_CARDINALITY,
    STATE_SLOT_NAMES,
    state_targets_from_trace,
)


def test_state_targets_share_one_schema_across_distinct_families() -> None:
    khop = TASK_GENERATORS["khop"](3, 17).transition_trace
    registers = TASK_GENERATORS["register_trace"](3, 18).transition_trace
    assert khop is not None and registers is not None

    khop_targets = state_targets_from_trace(khop, 5)
    register_targets = state_targets_from_trace(registers, 5)
    assert len(khop_targets.values) == len(register_targets.values) == 5
    assert len(khop_targets.initial_values) == len(STATE_SLOT_NAMES)
    assert khop_targets.initial_values[0] == 0
    assert khop_targets.initial_values[-1] == 0
    assert all(len(row) == len(STATE_SLOT_NAMES) for row in khop_targets.values)
    assert khop_targets.masks[0] == (True, True, False, False, True)
    assert register_targets.masks[0] == (True, True, True, True, True)
    assert all(
        0 <= value < STATE_CARDINALITY
        for row in khop_targets.values + register_targets.values
        for value in row
    )


def test_state_targets_repeat_terminal_state_after_program_completion() -> None:
    trace = TASK_GENERATORS["khop"](2, 19).transition_trace
    assert trace is not None
    targets = state_targets_from_trace(trace, 5)
    assert targets.values[1] == targets.values[2] == targets.values[3] == targets.values[4]
    assert targets.values[-1][-1] == 1
    commitment = targets.commitment()
    assert commitment["private_values_exposed"] is False
    assert "values" not in commitment
    assert "initial_values" not in commitment
    assert commitment["trace_sha256"] == trace.trace_sha256


def test_state_target_contract_rejects_invalid_depth_or_value() -> None:
    trace = TASK_GENERATORS["register_trace"](2, 20).transition_trace
    assert trace is not None
    with pytest.raises(ValueError, match="positive"):
        state_targets_from_trace(trace, 0)
