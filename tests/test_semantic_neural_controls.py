from __future__ import annotations

from core.brain.llm.latent_cortex.semantic_neural_decode_context import (
    execute_semantic_neural_decode_state,
)
from core.learning.frontier_process_supervision import frontier_process_task_battery
from core.learning.semantic_neural_controls import (
    SEMANTIC_FAMILY_LESIONS,
    classify_public_semantic_objective,
    semantic_neural_family_lesion_machine,
)


def _tasks():
    return frontier_process_task_battery(
        ("coding", "calibration", "misleading_premise", "scientific_inference"),
        (1, 2, 3),
        1,
        seed=2026081558,
    )


def test_semantic_objective_classifier_requires_canonical_complete_prompt():
    for task in _tasks():
        assert classify_public_semantic_objective(task.prompt) == task.family
        assert classify_public_semantic_objective("prefix " + task.prompt) is None
        assert classify_public_semantic_objective(task.prompt + " suffix") is None
    assert classify_public_semantic_objective("Calculate 2 + 2.") is None


def test_family_targeted_lesions_remove_each_supported_execution_path():
    disrupted = {family: 0 for family in SEMANTIC_FAMILY_LESIONS}
    for task in _tasks():
        expected = execute_semantic_neural_decode_state(task.prompt, task.family)
        machine = semantic_neural_family_lesion_machine(task.family)
        try:
            observed = execute_semantic_neural_decode_state(
                task.prompt,
                task.family,
                machine=machine,
            )
        except (RuntimeError, ValueError):
            disrupted[task.family] += 1
        else:
            disrupted[task.family] += int(observed.semantic_result != expected.semantic_result)
    assert disrupted == {family: 3 for family in SEMANTIC_FAMILY_LESIONS}
