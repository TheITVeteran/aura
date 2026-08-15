from __future__ import annotations

import mlx.core as mx
import pytest

from core.learning.frontier_process_supervision import frontier_process_task_battery
from core.learning.public_frontier_action_compiler import compile_public_frontier_actions
from core.learning.recurrent_state_schema import state_targets_from_trace
from core.learning.semantic_neural_machine import SemanticNeuralMachine


def _fresh_tasks(*, seed: int, per_cell: int = 1):
    return frontier_process_task_battery(
        ("coding", "calibration", "misleading_premise"),
        (1, 2, 3),
        per_cell,
        seed=seed,
    )


def _execute(machine: SemanticNeuralMachine, task):
    actions = compile_public_frontier_actions(task.prompt, task.family).values
    targets = state_targets_from_trace(task.transition_trace, task.depth, state_slots=11)
    state = targets.initial_values
    results = []
    for action in actions:
        result = machine.transition(state, action)
        results.append(result)
        state = result.next_state
    return state, targets.values[-1], tuple(results)


def test_semantic_neural_machine_executes_fresh_closed_loop_programs_exactly():
    machine = SemanticNeuralMachine()
    learned_operations = 0
    for task in _fresh_tasks(seed=1547, per_cell=2):
        observed, expected, results = _execute(machine, task)
        assert observed == expected
        learned_operations += sum(row.learned_operation_count for row in results)
        assert all(row.receipt()["teacher_available"] is False for row in results)
        assert all(row.receipt()["private_trace_available"] is False for row in results)
    assert learned_operations > 1_000


def test_semantic_neural_machine_tissue_lesion_removes_exact_execution():
    machine = SemanticNeuralMachine()
    original_identity = machine.tissue_sha256
    lesion_tissue = SemanticNeuralMachine().tissue
    lesion_tissue.raw_coefficients = lesion_tissue.raw_coefficients.at[1, 2].add(
        -lesion_tissue.raw_coefficients[1, 2]
    )
    machine = SemanticNeuralMachine(lesion_tissue)
    failures = 0
    for task in _fresh_tasks(seed=2547, per_cell=1):
        try:
            observed, expected, _results = _execute(machine, task)
        except (RuntimeError, ValueError):
            failures += 1
        else:
            failures += int(observed != expected)
    assert failures >= 6
    assert original_identity != machine.tissue_sha256


def test_semantic_neural_machine_rejects_nonquantizable_tissue():
    tissue = SemanticNeuralMachine().tissue
    tissue.raw_coefficients = tissue.raw_coefficients.at[0, 0].add(mx.array(0.02))
    with pytest.raises(RuntimeError, match="not quantizable"):
        SemanticNeuralMachine(tissue)
