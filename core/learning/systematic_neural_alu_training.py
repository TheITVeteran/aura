"""Train the systematic neural ALU on one-step trajectories and unseen moduli."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Final

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from core.brain.llm.latent_cortex.systematic_neural_alu import (
    SYSTEMATIC_NEURAL_ALU_SOURCE_FILES,
    SystematicNeuralALU,
    write_systematic_neural_alu_manifest,
)

SYSTEMATIC_NEURAL_ALU_TRAINING_SCHEMA: Final = "aura.systematic_neural_alu_training.v1"
TRAIN_MODULI: Final = (5, 7, 11, 13, 17)
DEVELOPMENT_MODULI: Final = (19, 23)
FROZEN_TEST_MODULI: Final = (29, 31, 37, 41, 43)
REPO_ROOT: Final = Path(__file__).resolve().parents[2]


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _rows(moduli: tuple[int, ...]) -> tuple[tuple[int, int, int, int, int, int], ...]:
    rows = []
    for modulus in moduli:
        for residue in range(modulus):
            for operand in range(modulus):
                for opcode in (0, 1, 2):
                    raw = (
                        residue + operand
                        if opcode == 0
                        else residue * operand
                        if opcode == 1
                        else residue - operand
                    )
                    rows.append((opcode, residue, operand, modulus, raw % modulus, raw))
    return tuple(rows)


def _arrays(rows: tuple[tuple[int, int, int, int, int, int], ...]) -> tuple[Any, ...]:
    return (
        mx.array([row[0] for row in rows], dtype=mx.int32),
        mx.array([row[1] for row in rows], dtype=mx.float32),
        mx.array([row[2] for row in rows], dtype=mx.float32),
        mx.array([row[3] for row in rows], dtype=mx.float32),
        mx.array([row[4] for row in rows], dtype=mx.int32),
        mx.array([row[5] for row in rows], dtype=mx.float32),
    )


def exact_accuracy(tissue: SystematicNeuralALU, moduli: tuple[int, ...]) -> float:
    data = _arrays(_rows(moduli))
    predictions = mx.argmax(tissue.logits_batch(*data[:4]), axis=-1)
    accuracy = mx.mean(predictions == data[4])
    mx.eval(accuracy)
    return float(accuracy.item())


def train_systematic_neural_alu(
    *,
    raw_steps: int = 6_000,
    decoder_steps: int = 800,
) -> tuple[SystematicNeuralALU, dict[str, Any]]:
    if (
        type(raw_steps) is not int
        or not 100 <= raw_steps <= 100_000
        or type(decoder_steps) is not int
        or not 100 <= decoder_steps <= 100_000
    ):
        raise ValueError("systematic neural ALU training schedule is invalid")
    train_rows = _rows(TRAIN_MODULI)
    train = _arrays(train_rows)
    tissue = SystematicNeuralALU()
    raw_optimizer = optim.Adam(learning_rate=0.01)

    def raw_objective(candidate: SystematicNeuralALU) -> Any:
        opcodes, residues, operands, _moduli, _targets, raw_targets = train
        difference = candidate.raw_batch(opcodes, residues, operands) - raw_targets
        scales = mx.array((25.0, 225.0, 25.0), dtype=mx.float32)[opcodes]
        return mx.mean(mx.square(difference) / scales)

    raw_loss_and_grad = nn.value_and_grad(tissue, raw_objective)
    initial_raw_loss = float(raw_objective(tissue).item())
    for _step in range(raw_steps):
        loss, gradients = raw_loss_and_grad(tissue)
        raw_optimizer.update(tissue, gradients)
        mx.eval(tissue.parameters(), raw_optimizer.state, loss)
        if not math.isfinite(float(loss.item())):
            raise FloatingPointError("systematic neural ALU raw loss is non-finite")
    final_raw_loss = float(raw_objective(tissue).item())

    tissue.freeze(keys=["raw_coefficients"])
    decoder_optimizer = optim.Adam(learning_rate=0.03)

    def decoder_objective(candidate: SystematicNeuralALU) -> Any:
        return nn.losses.cross_entropy(
            candidate.logits_batch(*train[:4]),
            train[4],
            reduction="mean",
        )

    decoder_loss_and_grad = nn.value_and_grad(tissue, decoder_objective)
    for _step in range(decoder_steps):
        loss, gradients = decoder_loss_and_grad(tissue)
        decoder_optimizer.update(tissue, gradients)
        mx.eval(tissue.parameters(), decoder_optimizer.state, loss)
        if not math.isfinite(float(loss.item())):
            raise FloatingPointError("systematic neural ALU decoder loss is non-finite")
    tissue.unfreeze()
    train_accuracy = exact_accuracy(tissue, TRAIN_MODULI)
    development_accuracy = exact_accuracy(tissue, DEVELOPMENT_MODULI)
    if train_accuracy != 1.0 or development_accuracy != 1.0:
        raise RuntimeError("systematic neural ALU failed its pre-registered admission")
    source_sha256s = {
        relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        for relative in SYSTEMATIC_NEURAL_ALU_SOURCE_FILES
    }
    body = {
        "schema": SYSTEMATIC_NEURAL_ALU_TRAINING_SCHEMA,
        "train_moduli": list(TRAIN_MODULI),
        "development_moduli": list(DEVELOPMENT_MODULI),
        "frozen_test_moduli": list(FROZEN_TEST_MODULI),
        "train_example_count": len(train_rows),
        "raw_steps": raw_steps,
        "decoder_steps": decoder_steps,
        "initial_raw_loss": initial_raw_loss,
        "final_raw_loss": final_raw_loss,
        "train_exact_accuracy": train_accuracy,
        "development_exact_accuracy": development_accuracy,
        "source_sha256s": source_sha256s,
        "teacher_removed_before_evaluation": True,
        "claim_boundary": (
            "operation and periodic decoding weights learned from one-step traces; "
            "frozen moduli and program depths remain unopened"
        ),
    }
    return tissue, {**body, "receipt_sha256": _canonical_sha256(body)}


def train_and_write_systematic_neural_alu_artifact(
    out_dir: Path,
) -> dict[str, Any]:
    target = out_dir.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    tissue, receipt = train_systematic_neural_alu()
    scratch = target / f".weights.{os.getpid()}.safetensors"
    mx.save_safetensors(
        str(scratch),
        {
            "raw_coefficients": tissue.raw_coefficients,
            "harmonic_weights": tissue.harmonic_weights,
        },
    )
    os.replace(scratch, target / "weights.safetensors")
    return write_systematic_neural_alu_manifest(target, training_receipt=receipt)


__all__ = [
    "DEVELOPMENT_MODULI",
    "FROZEN_TEST_MODULI",
    "SYSTEMATIC_NEURAL_ALU_TRAINING_SCHEMA",
    "TRAIN_MODULI",
    "exact_accuracy",
    "train_and_write_systematic_neural_alu_artifact",
    "train_systematic_neural_alu",
]
