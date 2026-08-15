"""Causal contracts for teacher-removed mathematics memory tissue."""

from __future__ import annotations

import shutil
from pathlib import Path

import mlx.core as mx
import pytest

from core.learning.recurrent_work_memory_tissue import (
    DEFAULT_MATHEMATICS_MEMORY_ARTIFACT,
    MATHEMATICS_MEMORY_EXECUTION_SCHEMA,
    MathematicsMemoryTissue,
    execute_mathematics_memory,
    load_mathematics_memory_tissue,
)
from core.learning.recurrent_work_memory_training import (
    autonomous_execution_metrics,
    build_mathematics_memory_registry,
    train_mathematics_memory_tissue,
)


@pytest.fixture(scope="module")
def trained_memory_tissue() -> tuple:
    training, training_tasks = build_mathematics_memory_registry(
        seeds=range(12),
        difficulties=(1, 2, 3),
    )
    _heldout, heldout_tasks = build_mathematics_memory_registry(
        seeds=range(100, 112),
        difficulties=(1, 2, 3),
    )
    tissue, receipt = train_mathematics_memory_tissue(
        training,
        steps=400,
        learning_rate=0.01,
    )
    return tissue, receipt, training_tasks, heldout_tasks


def test_predicate_training_reaches_exact_teacher_removed_execution(
    trained_memory_tissue: tuple,
) -> None:
    tissue, receipt, training_tasks, heldout_tasks = trained_memory_tissue

    assert receipt["initial_metrics"]["write_accuracy"] < 0.5
    assert receipt["final_metrics"]["write_accuracy"] == 1.0
    assert receipt["final_metrics"]["read_accuracy"] > 0.999
    assert receipt["teacher_removed_before_evaluation"] is True
    assert autonomous_execution_metrics(tissue, training_tasks)[
        "exact_accuracy"
    ] == 1.0
    heldout = autonomous_execution_metrics(tissue, heldout_tasks)
    assert heldout["exact_accuracy"] == 1.0
    assert heldout["teacher_removed"] is True
    assert heldout["student_memory_rollin"] is True


def test_matched_control_and_memory_lesions_remove_the_gain(
    trained_memory_tissue: tuple,
) -> None:
    tissue, _receipt, _training_tasks, heldout_tasks = trained_memory_tissue
    matched_control = MathematicsMemoryTissue(
        hidden_size=tissue.hidden_size,
        seed=tissue.seed,
    )

    assert autonomous_execution_metrics(matched_control, heldout_tasks)[
        "exact_accuracy"
    ] == 0.0
    for lesion in (
        {"write_mode": "never"},
        {"read_mode": "never"},
        {"write_mode": "always"},
        {"routing_mode": "rotated"},
        {"memory_mode": "reset_each_step"},
    ):
        assert autonomous_execution_metrics(tissue, heldout_tasks, **lesion)[
            "exact_accuracy"
        ] < 1.0


def test_runtime_receipt_has_no_teacher_or_verifier(
    trained_memory_tissue: tuple,
) -> None:
    tissue, _receipt, _training_tasks, _heldout_tasks = trained_memory_tissue
    result = execute_mathematics_memory(
        tissue,
        choose=3,
        gap=2,
        low=15,
        high=30,
        values=(3, 8, 11, 15),
    )
    receipt = result.receipt()

    assert receipt["schema"] == MATHEMATICS_MEMORY_EXECUTION_SCHEMA
    assert receipt["teacher_available"] is False
    assert receipt["verifier_available"] is False
    assert receipt["student_memory_rollin"] is True
    assert receipt["generic_address_bus"] is True
    assert receipt["transition_count"] == 4


def test_wrong_write_lesion_remains_observable_instead_of_being_guarded() -> None:
    tissue = MathematicsMemoryTissue()

    result = execute_mathematics_memory(
        tissue,
        choose=3,
        gap=20,
        low=0,
        high=60,
        values=(3, 8, 11, 15),
        write_mode="always",
        read_mode="always",
    )

    assert result.count > 0


def test_sealed_tissue_reloads_and_replays_the_fresh_registry() -> None:
    tissue = load_mathematics_memory_tissue()
    _private_labels, heldout_tasks = build_mathematics_memory_registry(
        seeds=range(1_000, 1_100),
        difficulties=(1, 2, 3),
    )

    metrics = autonomous_execution_metrics(tissue, heldout_tasks)

    assert metrics["exact"] == 300
    assert metrics["exact_accuracy"] == 1.0


def test_sealed_tissue_rejects_weight_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    shutil.copytree(DEFAULT_MATHEMATICS_MEMORY_ARTIFACT, artifact)
    weights = artifact / "weights.safetensors"
    tensors = mx.load(str(weights))
    tensors["read_bias"] = tensors["read_bias"] + 0.001
    mx.save_safetensors(str(weights), tensors)

    with pytest.raises(RuntimeError, match="weights commitment differs"):
        load_mathematics_memory_tissue(artifact)
