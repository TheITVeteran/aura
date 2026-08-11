"""Contracts for typed recurrent actions and their causal state write."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from core.learning.recurrence_curriculum import TASK_GENERATORS  # noqa: E402
from core.learning.recurrent_action_schema import (  # noqa: E402
    ACTION_NULL,
    ACTION_SLOT_NAMES,
    OP_ADD_MOD,
    OP_BOOL_AND,
    OP_BOOL_NOT,
    OP_BOOL_OR,
    OP_BOOL_XOR,
    OP_COPY_VALUE,
    OP_MUL_MOD,
    OP_REGISTER_AFFINE,
    OP_SUB_MOD,
    action_targets_from_program,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)


def test_actions_share_one_schema_and_null_post_completion() -> None:
    khop = TASK_GENERATORS["khop"](1, 41).transition_program
    registers = TASK_GENERATORS["register_trace"](2, 42).transition_program
    assert khop is not None and registers is not None
    khop_targets = action_targets_from_program(khop, 3)
    register_targets = action_targets_from_program(registers, 3)
    assert len(khop_targets.values) == len(register_targets.values) == 3
    assert len(khop_targets.values[0]) == len(ACTION_SLOT_NAMES)
    assert khop_targets.masks[0] == (
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
    )
    assert khop_targets.values[0][0] == OP_COPY_VALUE
    assert khop_targets.values[0][2:-1] == (ACTION_NULL,) * 5
    assert khop_targets.values[1] == (ACTION_NULL,) * len(ACTION_SLOT_NAMES)
    assert not any(khop_targets.masks[1])
    assert register_targets.masks[0] == (True,) * len(ACTION_SLOT_NAMES)
    assert register_targets.values[0][0] == OP_REGISTER_AFFINE
    assert register_targets.commitment()["private_values_exposed"] is False


def test_typed_action_is_a_causal_input_to_state_transition() -> None:
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(hidden_size=32, correction_rank=4)
    )
    problem = mx.random.normal((1, 7, 32))
    hidden = mx.random.normal((1, 10, 32))
    left = controller.teacher_action_state((1, 2, 3, 4, 5, 6, 7, 8))
    right = controller.teacher_action_state((8, 7, 6, 5, 4, 3, 2, 1))
    left_logits = controller.state_transition_logits(
        problem, hidden, state_slot_start=3, step=1, action_state=left
    )
    right_logits = controller.state_transition_logits(
        problem, hidden, state_slot_start=3, step=1, action_state=right
    )
    mx.eval(left_logits, right_logits)
    assert not bool(mx.array_equal(left_logits, right_logits))


def test_canonical_microcode_executes_all_structured_curriculum_families() -> None:
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(hidden_size=32, correction_rank=4)
    )
    families = ("khop", "boolean", "modular", "register_trace")
    for offset, family in enumerate(families):
        task = TASK_GENERATORS[family](1, 700 + offset)
        assert task.transition_trace is not None
        assert task.transition_program is not None
        action_targets = action_targets_from_program(task.transition_program, 1)
        initial = task.transition_trace.states[0]
        expected = task.transition_trace.states[1]
        interior = initial[1:-1]
        expected_interior = expected[1:-1]
        initial_values = (
            initial[0],
            *interior,
            *([0] * (3 - len(interior))),
            initial[-1],
        )
        expected_values = (
            expected[0],
            *expected_interior,
            *([0] * (3 - len(expected_interior))),
            expected[-1],
        )
        state_probabilities = controller.exact_probabilities(
            initial_values,
            slots=controller.config.state_slots,
            cardinality=controller.config.state_cardinality,
        )
        action_probabilities = controller.exact_probabilities(
            action_targets.values[0],
            slots=controller.config.action_slots,
            cardinality=controller.config.action_cardinality,
        )
        logits, recognized = controller.microcode_transition_logits(
            state_probabilities,
            action_probabilities,
        )
        mx.eval(logits, recognized)
        assert bool(recognized.item())
        produced = tuple(int(value) for value in mx.argmax(logits[0], axis=-1).tolist())
        assert produced == expected_values


def test_completed_state_stutters_when_no_instruction_remains() -> None:
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(hidden_size=32, correction_rank=4)
    )
    hidden = mx.random.normal((1, 10, 32))
    problem = mx.random.normal((1, 7, 32))
    state = controller.exact_probabilities(
        (1, 7, 8, 9, 1),
        slots=controller.config.state_slots,
        cardinality=controller.config.state_cardinality,
    )
    action = controller.exact_probabilities(
        (OP_ADD_MOD, 3, 11, ACTION_NULL, ACTION_NULL, ACTION_NULL, ACTION_NULL, 0),
        slots=controller.config.action_slots,
        cardinality=controller.config.action_cardinality,
    )
    logits = controller.state_transition_logits(
        problem,
        hidden,
        state_slot_start=3,
        step=2,
        state_probabilities=state,
        action_probabilities=action,
    )
    mx.eval(logits)
    assert tuple(int(value) for value in mx.argmax(logits[0], axis=-1).tolist()) == (
        1,
        7,
        8,
        9,
        1,
    )


def test_tokenizer_literal_observation_causally_changes_problem_evidence() -> None:
    digit_ids = tuple(range(100, 110))
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=32,
            correction_rank=4,
            literal_digit_token_ids=digit_ids,
        )
    )
    problem = mx.zeros((1, 4, 32))
    grounded_12 = controller.ground_literal_evidence(
        problem, mx.array([[50, 101, 102, 51]])
    )
    grounded_13 = controller.ground_literal_evidence(
        problem, mx.array([[50, 101, 103, 51]])
    )
    mx.eval(grounded_12, grounded_13)
    assert bool(mx.all(grounded_12[:, :2, :] == 0.0))
    assert bool(mx.all(grounded_12[:, 3:, :] == 0.0))
    assert not bool(mx.array_equal(grounded_12[:, 2, :], grounded_13[:, 2, :]))
    initial_12 = controller.initial_state_logits(
        grounded_12, mx.array([[50, 101, 102, 51]])
    )
    initial_13 = controller.initial_state_logits(
        grounded_13, mx.array([[50, 101, 103, 51]])
    )
    mx.eval(initial_12, initial_13)
    assert not bool(mx.array_equal(initial_12, initial_13))


def test_tokenizer_opcode_observation_causally_selects_the_operation() -> None:
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=32,
            correction_rank=4,
            opcode_token_patterns=(
                (OP_COPY_VALUE, (70,)),
                (OP_ADD_MOD, (71,)),
                (OP_MUL_MOD, (72,)),
                (OP_SUB_MOD, (73,)),
                (OP_BOOL_NOT, (74,)),
                (OP_BOOL_AND, (75,)),
                (OP_BOOL_OR, (76,)),
                (OP_BOOL_XOR, (77,)),
            ),
            opcode_context_patterns=(
                ("graph", (40,)),
                ("graph_edges_start", (41,)),
                ("graph_edges_end", (42,)),
                ("modular_start", (43,)),
                ("modular_end", (44,)),
                ("boolean_start", (45,)),
                ("boolean_end", (46,)),
                ("register", (47,)),
                ("register_ops_start", (48,)),
                ("register_ops_end", (49,)),
            ),
        )
    )
    problem = mx.zeros((1, 3, 32), dtype=mx.float32)
    hidden = mx.zeros((1, 8, 32), dtype=mx.float32)
    copy_logits = controller.action_logits(
        problem,
        hidden,
        state_slot_start=3,
        step=1,
        token_ids=mx.array([[40, 70, 11]]),
    )
    multiply_logits = controller.action_logits(
        problem,
        hidden,
        state_slot_start=3,
        step=1,
        token_ids=mx.array([[43, 72, 44]]),
    )
    mx.eval(copy_logits, multiply_logits)
    assert int(mx.argmax(copy_logits[0, 0]).item()) == OP_COPY_VALUE
    assert int(mx.argmax(multiply_logits[0, 0]).item()) == OP_MUL_MOD
    assert not bool(mx.array_equal(copy_logits[:, 0], multiply_logits[:, 0]))
