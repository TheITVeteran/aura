"""Contracts for teacher-free recurrent-memory ingress to language decode."""

from __future__ import annotations

import inspect

import pytest

from core.brain.llm.latent_cortex.frontier_tasks import generate_task
from core.brain.llm.latent_cortex.recurrent_memory_decode_context import (
    RECURRENT_MEMORY_DECODE_CONTEXT_SCHEMA,
    RecurrentMemoryDecodeState,
    execute_recurrent_memory_decode_state,
    render_recurrent_memory_decode_context,
)


def test_decode_state_comes_from_the_sealed_student_rollin() -> None:
    task = generate_task("mathematics", seed=1_203, difficulty=3)

    state = execute_recurrent_memory_decode_state(task.public.prompt)

    assert task.score(
        f'FINAL_ANSWER: {{"count":{state.count},"witness":{list(state.witness)}}}'
    ).correct
    assert state.execution_receipt["teacher_available"] is False
    assert state.execution_receipt["verifier_available"] is False
    assert state.execution_receipt["student_memory_rollin"] is True
    assert state.receipt()["answer_key_available"] is False


def test_decode_context_leaves_wording_to_the_model_and_preserves_semantics() -> None:
    task = generate_task("mathematics", seed=1_203, difficulty=3)
    state = execute_recurrent_memory_decode_state(task.public.prompt)

    context = render_recurrent_memory_decode_context(state)

    assert "FINAL_ANSWER" not in context
    assert context.startswith(RECURRENT_MEMORY_DECODE_CONTEXT_SCHEMA)
    assert f'"count":{state.count}' in context
    assert f'"witness":{list(state.witness)}'.replace(" ", "") in context
    assert "witness_length" not in context
    assert "witness_slots" not in context
    assert ":0" not in context


def test_no_write_lesion_changes_the_semantic_state_before_decode() -> None:
    task = generate_task("mathematics", seed=1_203, difficulty=3)

    treatment = execute_recurrent_memory_decode_state(task.public.prompt)
    lesion = execute_recurrent_memory_decode_state(
        task.public.prompt,
        write_mode="never",
    )

    assert (treatment.count, treatment.witness) != (lesion.count, lesion.witness)
    assert lesion.execution_receipt["write_mode"] == "never"
    assert treatment.receipt()["receipt_sha256"] != lesion.receipt()["receipt_sha256"]


def test_decode_state_rejects_unsupported_or_malformed_objectives() -> None:
    with pytest.raises(ValueError, match="grammar is unsupported"):
        execute_recurrent_memory_decode_state("Count some subsets.")
    with pytest.raises(TypeError, match="must be text"):
        execute_recurrent_memory_decode_state(None)  # type: ignore[arg-type]


def test_decode_ingress_never_calls_a_verifier_or_frontier_answer() -> None:
    source = inspect.getsource(execute_recurrent_memory_decode_state)

    assert "verify_objective_program" not in source
    assert "_separated_subset_expected" not in source
    assert "reveal_for_verifier" not in source
    assert "task.score" not in source
    assert "expected" not in source


def test_typed_state_rejects_a_forged_execution_receipt() -> None:
    task = generate_task("mathematics", seed=1_203, difficulty=3)
    state = execute_recurrent_memory_decode_state(task.public.prompt)
    forged = dict(state.execution_receipt)
    forged["teacher_available"] = True

    with pytest.raises(ValueError, match="decode state is invalid"):
        RecurrentMemoryDecodeState(
            objective_sha256=state.objective_sha256,
            count=state.count,
            witness=state.witness,
            tissue_sha256=state.tissue_sha256,
            execution_receipt=forged,
        )
