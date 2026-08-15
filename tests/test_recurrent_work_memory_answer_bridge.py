"""Answer-boundary contracts for the sealed recurrent mathematics memory."""

from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex import neural_objective_producer as producer
from core.brain.llm.latent_cortex.frontier_tasks import generate_task
from tests.sealed_artifact_support import require_mathematics_memory_tissue


def test_sealed_memory_tissue_emits_a_verified_public_answer() -> None:
    require_mathematics_memory_tissue()
    task = generate_task("mathematics", seed=1_037, difficulty=3)

    solved = producer.solve_objective_program_neural(task.public.prompt)

    assert solved is not None
    candidate, receipt = solved
    execution = receipt["execution"]
    assert task.score(candidate).correct is True
    assert execution["engine"] == "mathematics_memory_tissue.v1"
    assert execution["teacher_available"] is False
    assert execution["independent_crosscheck_match"] is True
    assert execution["student_rollin"]["teacher_available"] is False
    assert execution["student_rollin"]["verifier_available"] is False
    assert execution["student_rollin"]["student_memory_rollin"] is True
    assert execution["student_rollin"]["transition_count"] == 10
    assert execution["tissue_sha256"] == execution["student_rollin"]["tissue_sha256"]
    producer.validate_objective_program_solution_neural(
        receipt,
        objective=task.public.prompt,
        candidate=candidate,
    )


def test_answer_bridge_rejects_a_memory_lesion_before_promotion(monkeypatch) -> None:
    require_mathematics_memory_tissue()
    task = generate_task("mathematics", seed=1_037, difficulty=3)
    execute = producer.execute_mathematics_memory

    def execute_without_writes(tissue, **kwargs):
        return execute(tissue, **kwargs, write_mode="never")

    monkeypatch.setattr(producer, "execute_mathematics_memory", execute_without_writes)

    with pytest.raises(
        RuntimeError,
        match="neural recurrent memory and independent subset verifier disagree",
    ):
        producer.solve_objective_program_neural(task.public.prompt)
