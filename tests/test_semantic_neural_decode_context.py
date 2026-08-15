from __future__ import annotations

import json

from core.brain.llm.latent_cortex.semantic_neural_decode_context import (
    execute_semantic_neural_decode_state,
    normalize_semantic_neural_response,
    render_semantic_neural_answer,
    render_semantic_neural_decode_context,
    render_semantic_neural_decode_correction,
    semantic_result_matches_response,
)
from core.learning.frontier_process_supervision import frontier_process_task_battery


def test_semantic_neural_decode_state_matches_fresh_answers_without_teacher():
    tasks = frontier_process_task_battery(
        ("coding", "calibration", "misleading_premise", "scientific_inference"),
        (1, 2, 3),
        2,
        seed=1548,
    )
    for task in tasks:
        state = execute_semantic_neural_decode_state(task.prompt, task.family)
        expected = json.loads(task.answer.removeprefix("FINAL_ANSWER: "))
        assert state.semantic_result == expected
        receipt = state.receipt()
        assert receipt["teacher_available"] is False
        assert receipt["private_trace_available"] is False
        assert receipt["verifier_available"] is False
        assert receipt["answer_key_available"] is False
        context = render_semantic_neural_decode_context(state)
        assert json.dumps(expected, sort_keys=True, separators=(",", ":")) in context


def test_semantic_neural_decode_state_is_replayable():
    task = frontier_process_task_battery(("coding",), (3,), 1, seed=2548)[0]
    first = execute_semantic_neural_decode_state(task.prompt, task.family)
    second = execute_semantic_neural_decode_state(task.prompt, task.family)
    assert first == second
    assert first.receipt() == second.receipt()


def test_semantic_neural_decode_serialization_check_uses_state_not_task_oracle():
    task = frontier_process_task_battery(("calibration",), (1,), 1, seed=915)[0]
    state = execute_semantic_neural_decode_state(task.prompt, task.family)
    encoded = json.dumps(state.semantic_result, sort_keys=True, separators=(",", ":"))
    assert semantic_result_matches_response(state, f"FINAL_ANSWER: {encoded}") is True
    assert semantic_result_matches_response(state, "FINAL_ANSWER: {}") is False
    correction = render_semantic_neural_decode_correction(state)
    assert encoded in correction
    assert "preceding serialization" in correction
    assert render_semantic_neural_answer(state) == f"FINAL_ANSWER: {encoded}"


def test_semantic_neural_wire_normalization_only_trims_exact_object_overrun():
    task = frontier_process_task_battery(("coding",), (1,), 1, seed=1553)[0]
    state = execute_semantic_neural_decode_state(task.prompt, task.family)
    canonical = render_semantic_neural_answer(state)
    normalized, changed = normalize_semantic_neural_response(
        state,
        canonical + '},{"token_boundary_overrun":',
    )
    assert changed is True
    assert normalized == canonical
    wrong, changed = normalize_semantic_neural_response(
        state,
        'FINAL_ANSWER: {}},{"token_boundary_overrun":',
    )
    assert changed is False
    assert wrong == 'FINAL_ANSWER: {}},{"token_boundary_overrun":'
