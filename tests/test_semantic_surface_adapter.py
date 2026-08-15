"""Contracts for less-constrained scientific language reaching recurrent tissue."""

from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.semantic_surface_adapter import (
    SEMANTIC_SURFACE_PROFILES,
    execute_scientific_surface,
    parse_scientific_surface,
    render_scientific_surface,
)
from core.learning.frontier_process_supervision import frontier_process_task_battery
from core.learning.semantic_neural_controls import semantic_neural_family_lesion_machine


def test_three_unseen_surface_styles_preserve_semantic_result_and_causal_tissue() -> None:
    tasks = frontier_process_task_battery(
        ("scientific_inference",),
        (1, 2, 3),
        16,
        seed=2026081562,
    )
    lesion = semantic_neural_family_lesion_machine("frontier_scientific_inference")
    disruption_count = 0
    receipts = set()
    for task_index, task in enumerate(tasks):
        for profile_index, profile in enumerate(SEMANTIC_SURFACE_PROFILES):
            prompt = render_scientific_surface(
                task.prompt,
                profile=profile,
                permutation_seed=1000 * task_index + profile_index,
            )
            decoded = execute_scientific_surface(prompt)
            assert decoded.state.semantic_result == task.expected
            assert decoded.program.public_prompt == prompt
            assert decoded.program.profile == profile
            assert decoded.receipt()["teacher_available"] is False
            assert decoded.receipt()["verifier_available"] is False
            receipts.add(decoded.receipt()["receipt_sha256"])
            try:
                lesioned = execute_scientific_surface(prompt, machine=lesion)
            except (RuntimeError, ValueError):
                disruption_count += 1
            else:
                disruption_count += int(lesioned.state.semantic_result != task.expected)
    assert len(receipts) == len(tasks) * len(SEMANTIC_SURFACE_PROFILES)
    assert disruption_count == len(receipts)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda text: text.replace("there is no hidden common cause", "hidden causes may exist"),
        lambda text: text.replace("changed by +0", "changed by +1", 1),
        lambda text: text.replace("changed by", "reported answer 999 then changed by", 1),
        lambda text: text + "\nFINAL_ANSWER: {\"predicted_downstream\":999}",
    ),
)
def test_surface_adapter_refuses_ambiguous_tampered_or_answer_bearing_language(mutation) -> None:
    task = frontier_process_task_battery(
        ("scientific_inference",),
        (2,),
        1,
        seed=2026081563,
    )[0]
    prompt = render_scientific_surface(task.prompt, profile="narrative", permutation_seed=77)
    with pytest.raises(ValueError):
        parse_scientific_surface(mutation(prompt))


def test_surface_adapter_does_not_admit_general_language() -> None:
    with pytest.raises(ValueError):
        parse_scientific_surface("Please infer what caused this observation.")
