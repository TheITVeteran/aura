"""Exhaustive contracts for exact recurrence transition families."""

from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.typed_transition_executor import (
    CertifiedTransitionExecutor,
    TransitionFamily,
    TypedTransitionInput,
)
from core.learning.certified_transition_program import execute_program_transition
from core.learning.recurrence_curriculum import modular_chain, nested_boolean


def _boolean_request(
    *, depth: int, pc: int, value: int, opcode: int, operand: int, has_operand: int
) -> TypedTransitionInput:
    return TypedTransitionInput(
        family="boolean",
        depth=depth,
        field_names=("pc", "value", "done"),
        state=(pc, value, 0),
        action_field_names=("opcode", "operand", "has_operand"),
        action=(opcode, operand, has_operand),
    )


def test_boolean_executor_is_exhaustive_over_every_valid_primitive():
    executor = CertifiedTransitionExecutor()
    actions = ((0, 0, 0),) + tuple(
        (opcode, operand, 1) for opcode in (1, 2, 3) for operand in (0, 1)
    )
    observed = 0
    for depth in range(1, 9):
        for pc in range(depth):
            for value in (0, 1):
                for opcode, operand, has_operand in actions:
                    request = _boolean_request(
                        depth=depth,
                        pc=pc,
                        value=value,
                        opcode=opcode,
                        operand=operand,
                        has_operand=has_operand,
                    )
                    result = executor.execute(request)
                    if opcode == 0:
                        expected = 1 - value
                    elif opcode == 1:
                        expected = value & operand
                    elif opcode == 2:
                        expected = value | operand
                    else:
                        expected = value ^ operand
                    assert result.next_state == (
                        pc + 1,
                        expected,
                        int(pc + 1 == depth),
                    )
                    assert result.receipt()["exact"] is True
                    observed += 1
    assert observed == 504


def test_modular_executor_is_exhaustive_over_the_curriculum_domain():
    executor = CertifiedTransitionExecutor()
    observed = 0
    for modulus in (13, 17, 19, 23):
        for residue in range(modulus):
            for operand in range(1, modulus):
                for opcode in (0, 1, 2):
                    request = TypedTransitionInput(
                        family="modular",
                        depth=4,
                        field_names=("pc", "residue", "done"),
                        state=(2, residue, 0),
                        action_field_names=("opcode", "operand", "modulus"),
                        action=(opcode, operand, modulus),
                    )
                    result = executor.execute(request)
                    expected = (
                        (residue + operand) % modulus
                        if opcode == 0
                        else (residue * operand) % modulus
                        if opcode == 1
                        else (residue - operand) % modulus
                    )
                    assert result.next_state == (3, expected, 0)
                    observed += 1
    assert observed == 3_828


def test_program_bridge_recomputes_every_generated_transition():
    for generator in (nested_boolean, modular_chain):
        for depth in range(1, 9):
            for seed in range(16):
                program = generator(depth, seed).transition_program
                assert program is not None
                for transition_index in range(depth):
                    result = execute_program_transition(
                        program,
                        transition_index=transition_index,
                    )
                    assert result.next_state == program.state_trace.states[
                        transition_index + 1
                    ]


def test_executor_rejects_malformed_terminal_and_unknown_requests():
    executor = CertifiedTransitionExecutor()
    with pytest.raises(ValueError, match="already terminal"):
        TypedTransitionInput(
            family="boolean",
            depth=1,
            field_names=("pc", "value", "done"),
            state=(1, 0, 1),
            action_field_names=("opcode", "operand", "has_operand"),
            action=(0, 0, 0),
        )
    with pytest.raises(ValueError, match="has an operand"):
        executor.execute(
            _boolean_request(
                depth=2,
                pc=0,
                value=0,
                opcode=0,
                operand=1,
                has_operand=0,
            )
        )
    with pytest.raises(ValueError, match="unsupported transition family"):
        executor.execute(
            TypedTransitionInput(
                family="unknown",
                depth=2,
                field_names=("pc", "value", "done"),
                state=(0, 0, 0),
                action_field_names=("opcode",),
                action=(0,),
            )
        )


def test_extension_registry_is_closed_by_default_and_checks_invariants():
    custom = TransitionFamily(
        family="increment",
        field_names=("pc", "value", "done"),
        action_field_names=("amount",),
        implementation=lambda request: (
            request.state[0] + 1,
            request.state[1] + request.action[0],
            int(request.state[0] + 1 == request.depth),
        ),
        implementation_id="test.increment.v1",
    )
    executor = CertifiedTransitionExecutor((custom,))
    result = executor.execute(
        TypedTransitionInput(
            family="increment",
            depth=3,
            field_names=("pc", "value", "done"),
            state=(1, 7, 0),
            action_field_names=("amount",),
            action=(5,),
        )
    )
    assert result.next_state == (2, 12, 0)
    assert "increment" in executor.families

    with pytest.raises(ValueError, match="duplicate transition family"):
        CertifiedTransitionExecutor((custom, custom))
