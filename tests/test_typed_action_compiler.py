"""Contracts for public-evidence compilation into typed recurrent actions."""

from __future__ import annotations

import inspect
import json

import pytest

from core.brain.llm.latent_cortex.objective_program_verifier import (
    verify_objective_program,
)
from core.brain.llm.latent_cortex.neural_objective_producer import (  # noqa: E402
    solve_objective_program_neural as solve_objective_program,
)
from core.brain.llm.latent_cortex.typed_action_compiler import (
    compile_boolean_expression,
    compile_modular_operations,
    compile_public_transition_program,
)
from core.learning.certified_transition_program import execute_compiled_action_program
from core.learning.recurrence_curriculum import modular_chain, nested_boolean


def test_public_compiler_signature_has_no_answer_or_private_trace_channel():
    parameters = inspect.signature(compile_public_transition_program).parameters
    assert tuple(parameters) == ("prompt",)
    assert "answer" not in parameters
    assert "trace" not in parameters


def test_compiler_recovers_fresh_programs_from_public_prompts_only():
    observed = 0
    for generator in (nested_boolean, modular_chain):
        for depth in (1, 2, 4, 8, 16, 32):
            for seed in range(24):
                task = generator(depth, 50_000 + seed)
                compiled = compile_public_transition_program(task.prompt)
                execution = execute_compiled_action_program(compiled)
                trace = task.transition_trace
                assert trace is not None
                assert execution.states == trace.states
                assert compiled.public_source_sha256
                observed += depth
    assert observed == 3_024


def test_compiler_receipt_hides_initial_state_and_actions():
    task = modular_chain(4, 71)
    compiled = compile_public_transition_program(task.prompt)
    receipt = compiled.public_receipt()
    encoded = json.dumps(receipt, sort_keys=True)

    assert "initial_state" not in receipt
    assert "actions" not in receipt
    assert json.dumps(list(compiled.initial_state)) not in encoded
    assert json.dumps([list(action) for action in compiled.actions]) not in encoded


def test_compiler_is_deterministic_and_semantic_mutation_changes_execution():
    source = "0" * 64
    first = compile_modular_operations(
        initial=1,
        modulus=13,
        operations=("+1", "*2", "-3"),
        public_source_sha256=source,
    )
    sham = compile_modular_operations(
        initial=1,
        modulus=13,
        operations=("+1", "*2", "-3"),
        public_source_sha256=source,
    )
    lesion = compile_modular_operations(
        initial=1,
        modulus=13,
        operations=("+1", "*3", "-3"),
        public_source_sha256=source,
    )

    assert first == sham
    assert first.program_sha256 == sham.program_sha256
    assert first.program_sha256 != lesion.program_sha256
    assert (
        execute_compiled_action_program(first).terminal_state
        != execute_compiled_action_program(lesion).terminal_state
    )


def test_boolean_recursive_descent_rejects_non_linear_or_trailing_syntax():
    source = "1" * 64
    valid = compile_boolean_expression(
        "((not 1) xor 0)",
        public_source_sha256=source,
    )
    assert valid.depth == 2
    with pytest.raises(ValueError, match="right operand"):
        compile_boolean_expression(
            "(1 and (not 0))",
            public_source_sha256=source,
        )
    with pytest.raises(ValueError, match="trailing"):
        compile_boolean_expression(
            "(1 and 0) 1",
            public_source_sha256=source,
        )


def test_public_adapter_refuses_unknown_tampered_and_depth_drift():
    with pytest.raises(ValueError, match="unsupported or ambiguous"):
        compile_public_transition_program("Please calculate one plus one.")

    task = nested_boolean(3, 81)
    tampered = task.prompt.replace("3-operation", "4-operation", 1)
    with pytest.raises(ValueError, match="declared Boolean depth"):
        compile_public_transition_program(tampered)

    modular = modular_chain(2, 82)
    tampered = modular.prompt.replace("Operations: ", "Operations: +999, ", 1)
    with pytest.raises(ValueError, match="outside its modulus"):
        compile_public_transition_program(tampered)


@pytest.mark.parametrize("generator", (nested_boolean, modular_chain))
def test_complete_engine_uses_certified_recurrence_for_declared_prompt(generator):
    task = generator(8, 90_001)
    solved = solve_objective_program(task.prompt)
    assert solved is not None
    candidate, receipt = solved
    execution = receipt["execution"]

    expected_engine = (
        "neural_transition_tissue.v1"
        if generator is nested_boolean
        else "systematic_neural_alu.v1"
    )
    assert execution["engine"] == expected_engine
    assert execution["teacher_available"] is False
    assert execution["student_rollin"]["student_rollin"] is True
    assert execution["student_rollin"]["transition_count"] == 8
    assert execution["independent_crosscheck_match"] is True
    assert task.answer in candidate
    verdict = verify_objective_program(candidate, objective=task.prompt)
    assert verdict is not None
    assert verdict["outcome"] == "verified"
    wire = json.dumps(receipt, sort_keys=True)
    assert '"actions"' not in wire
    assert '"initial_state"' not in wire
