from __future__ import annotations

import hashlib
from pathlib import Path

import mlx.core as mx
import pytest

from core.brain.llm.latent_cortex.systematic_neural_alu import (
    SystematicNeuralALU,
    execute_systematic_neural_program,
    load_systematic_neural_alu,
)
from core.brain.llm.latent_cortex.typed_action_compiler import (
    compile_modular_operations,
)
from core.learning.systematic_neural_alu_training import (
    DEVELOPMENT_MODULI,
    TRAIN_MODULI,
    exact_accuracy,
    train_and_write_systematic_neural_alu_artifact,
    train_systematic_neural_alu,
)


@pytest.fixture(scope="module")
def trained_tissue():
    return train_systematic_neural_alu()


def test_training_admits_only_exact_seen_and_development_moduli(trained_tissue) -> None:
    tissue, receipt = trained_tissue
    assert receipt["train_exact_accuracy"] == 1.0
    assert receipt["development_exact_accuracy"] == 1.0
    assert exact_accuracy(tissue, TRAIN_MODULI) == 1.0
    assert exact_accuracy(tissue, DEVELOPMENT_MODULI) == 1.0


def test_systematic_tissue_composes_with_student_rollin(trained_tissue) -> None:
    tissue, _receipt = trained_tissue
    program = compile_modular_operations(
        initial=18,
        modulus=19,
        operations=("+0", "*7", "-4", "*0", "+18", "-0"),
        public_source_sha256=hashlib.sha256(b"student-rollin").hexdigest(),
    )
    result = execute_systematic_neural_program(program, tissue)
    assert result.terminal_state == (6, 18, 1)
    assert len(result.transition_receipts) == 6
    assert all(row["teacher_available"] is False for row in result.transition_receipts)
    assert all(
        row["exact_operator_available"] is False
        for row in result.transition_receipts
    )


def test_systematic_tissue_refuses_requests_outside_declared_support() -> None:
    tissue = SystematicNeuralALU()
    with pytest.raises(ValueError, match="outside support"):
        tissue.transition(depth=1, state=(0, 0, 0), action=(0, 0, 64))
    with pytest.raises(ValueError, match="outside support"):
        tissue.transition(depth=1, state=(0, 1, 0), action=(0, 0, 1))


def test_sealed_artifact_is_source_and_weight_bound(tmp_path: Path) -> None:
    manifest = train_and_write_systematic_neural_alu_artifact(tmp_path)
    tissue = load_systematic_neural_alu(tmp_path)
    assert tissue.tissue_sha256 == manifest["weights_sha256"]
    weights = tmp_path / "weights.safetensors"
    tensors = mx.load(str(weights))
    tensors["raw_coefficients"] = tensors["raw_coefficients"] + 0.001
    mx.save_safetensors(str(weights), tensors)
    with pytest.raises(RuntimeError, match="weights commitment differs"):
        load_systematic_neural_alu(tmp_path)
