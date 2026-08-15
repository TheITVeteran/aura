from __future__ import annotations

import json

from core.brain.llm.latent_cortex.semantic_neural_decode_context import (
    execute_semantic_neural_decode_state,
    render_semantic_neural_decode_context,
)
from core.learning.frontier_process_supervision import frontier_process_task_battery


def test_semantic_neural_decode_state_matches_fresh_answers_without_teacher():
    tasks = frontier_process_task_battery(
        ("coding", "calibration", "misleading_premise"),
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
