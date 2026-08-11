"""Train bounded recurrent neural tissue from independently certified transitions."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from core.brain.llm.latent_cortex.neural_transition_tissue import (
    NEURAL_TRANSITION_SOURCE_FILES,
    SUPPORTED_MODULI,
    NeuralTransitionTissue,
    write_neural_transition_manifest,
)
from core.brain.llm.latent_cortex.typed_transition_executor import (
    CertifiedTransitionExecutor,
    TypedTransitionInput,
)

NEURAL_TRANSITION_TRAINING_SCHEMA: Final = "aura.neural_transition_training.v1"
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


def _source_sha256s() -> dict[str, str]:
    return {
        relative: hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        for relative in NEURAL_TRANSITION_SOURCE_FILES
    }


@dataclass(frozen=True, slots=True)
class TransitionTrainingBatch:
    boolean_keys: tuple[int, ...]
    boolean_targets: tuple[int, ...]
    modular_keys: tuple[int, ...]
    modular_targets: tuple[int, ...]
    teacher_receipt_sha256: str

    @property
    def example_count(self) -> int:
        return len(self.boolean_keys) + len(self.modular_keys)


def build_certified_transition_batch() -> TransitionTrainingBatch:
    teacher = CertifiedTransitionExecutor()
    boolean_keys: list[int] = []
    boolean_targets: list[int] = []
    boolean_actions = ((0, 0, 0),) + tuple(
        (opcode, operand, 1) for opcode in (1, 2, 3) for operand in (0, 1)
    )
    receipt_hashes: list[str] = []
    for value in (0, 1):
        for opcode, operand, has_operand in boolean_actions:
            result = teacher.execute(
                TypedTransitionInput(
                    family="boolean",
                    depth=2,
                    field_names=("pc", "value", "done"),
                    state=(0, value, 0),
                    action_field_names=("opcode", "operand", "has_operand"),
                    action=(opcode, operand, has_operand),
                )
            )
            boolean_keys.append(
                NeuralTransitionTissue.boolean_key(
                    opcode=opcode,
                    value=value,
                    operand=operand,
                    has_operand=has_operand,
                )
            )
            boolean_targets.append(result.next_state[1])
            receipt_hashes.append(result.receipt()["receipt_sha256"])

    modular_keys: list[int] = []
    modular_targets: list[int] = []
    for modulus_index, modulus in enumerate(SUPPORTED_MODULI):
        for residue in range(modulus):
            for operand in range(1, modulus):
                for opcode in (0, 1, 2):
                    result = teacher.execute(
                        TypedTransitionInput(
                            family="modular",
                            depth=2,
                            field_names=("pc", "residue", "done"),
                            state=(0, residue, 0),
                            action_field_names=("opcode", "operand", "modulus"),
                            action=(opcode, operand, modulus),
                        )
                    )
                    modular_keys.append(
                        NeuralTransitionTissue.modular_key(
                            modulus_index=modulus_index,
                            opcode=opcode,
                            residue=residue,
                            operand=operand,
                        )
                    )
                    modular_targets.append(result.next_state[1])
                    receipt_hashes.append(result.receipt()["receipt_sha256"])
    body = {
        "schema": NEURAL_TRANSITION_TRAINING_SCHEMA,
        "boolean_examples": len(boolean_keys),
        "modular_examples": len(modular_keys),
        "ordered_teacher_receipt_sha256s": receipt_hashes,
    }
    return TransitionTrainingBatch(
        boolean_keys=tuple(boolean_keys),
        boolean_targets=tuple(boolean_targets),
        modular_keys=tuple(modular_keys),
        modular_targets=tuple(modular_targets),
        teacher_receipt_sha256=_canonical_sha256(body),
    )


def _batch_tensors(batch: TransitionTrainingBatch) -> tuple[Any, Any, Any, Any]:
    return (
        mx.array(batch.boolean_keys, dtype=mx.int32),
        mx.array(batch.boolean_targets, dtype=mx.int32),
        mx.array(batch.modular_keys, dtype=mx.int32),
        mx.array(batch.modular_targets, dtype=mx.int32),
    )


def transition_training_metrics(
    tissue: NeuralTransitionTissue,
    batch: TransitionTrainingBatch,
) -> dict[str, float]:
    boolean_keys, boolean_targets, modular_keys, modular_targets = _batch_tensors(batch)
    boolean_logits = tissue.boolean_batch(boolean_keys)
    modular_logits = tissue.modular_batch(modular_keys)
    boolean_loss = nn.losses.cross_entropy(
        boolean_logits,
        boolean_targets,
        reduction="mean",
    )
    modular_loss = nn.losses.cross_entropy(
        modular_logits,
        modular_targets,
        reduction="mean",
    )
    boolean_accuracy = mx.mean(mx.argmax(boolean_logits, axis=-1) == boolean_targets)
    modular_accuracy = mx.mean(mx.argmax(modular_logits, axis=-1) == modular_targets)
    mx.eval(boolean_loss, modular_loss, boolean_accuracy, modular_accuracy)
    return {
        "loss": float((boolean_loss + modular_loss).item()),
        "boolean_accuracy": float(boolean_accuracy.item()),
        "modular_accuracy": float(modular_accuracy.item()),
        "exact_accuracy": float(
            (
                mx.sum(mx.argmax(boolean_logits, axis=-1) == boolean_targets)
                + mx.sum(mx.argmax(modular_logits, axis=-1) == modular_targets)
            ).item()
            / batch.example_count
        ),
    }


def train_neural_transition_tissue(
    *,
    steps: int = 32,
    learning_rate: float = 0.2,
) -> tuple[NeuralTransitionTissue, dict[str, Any]]:
    if (
        type(steps) is not int
        or not 1 <= steps <= 1_000
        or isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or not 1e-5 <= float(learning_rate) <= 1.0
    ):
        raise ValueError("neural transition training configuration is invalid")
    batch = build_certified_transition_batch()
    boolean_keys, boolean_targets, modular_keys, modular_targets = _batch_tensors(batch)
    tissue = NeuralTransitionTissue()
    optimizer = optim.Adam(learning_rate=float(learning_rate))

    def objective(candidate: NeuralTransitionTissue) -> Any:
        boolean_loss = nn.losses.cross_entropy(
            candidate.boolean_batch(boolean_keys),
            boolean_targets,
            reduction="mean",
        )
        modular_loss = nn.losses.cross_entropy(
            candidate.modular_batch(modular_keys),
            modular_targets,
            reduction="mean",
        )
        return boolean_loss + modular_loss

    initial = transition_training_metrics(tissue, batch)
    loss_and_grad = nn.value_and_grad(tissue, objective)
    loss_history: list[float] = []
    for _step in range(steps):
        loss, gradients = loss_and_grad(tissue)
        optimizer.update(tissue, gradients)
        mx.eval(tissue.parameters(), optimizer.state, loss)
        value = float(loss.item())
        if not math.isfinite(value):
            raise FloatingPointError("neural transition training loss is non-finite")
        loss_history.append(value)
    final = transition_training_metrics(tissue, batch)
    if final["exact_accuracy"] != 1.0:
        raise RuntimeError("neural transition tissue did not learn every primitive")
    receipt_body = {
        "schema": NEURAL_TRANSITION_TRAINING_SCHEMA,
        "steps": steps,
        "learning_rate": float(learning_rate),
        "example_count": batch.example_count,
        "boolean_example_count": len(batch.boolean_keys),
        "modular_example_count": len(batch.modular_keys),
        "teacher_receipt_sha256": batch.teacher_receipt_sha256,
        "source_sha256s": _source_sha256s(),
        "initial_metrics": initial,
        "final_metrics": final,
        "loss_history_sha256": _canonical_sha256(loss_history),
        "teacher_removed_before_evaluation": True,
    }
    return tissue, {
        **receipt_body,
        "receipt_sha256": _canonical_sha256(receipt_body),
    }


def train_and_write_neural_transition_artifact(
    out_dir: Path,
    *,
    steps: int = 32,
    learning_rate: float = 0.2,
) -> dict[str, Any]:
    target = out_dir.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    tissue, receipt = train_neural_transition_tissue(
        steps=steps,
        learning_rate=learning_rate,
    )
    scratch = target / f".weights.{os.getpid()}.safetensors"
    mx.save_safetensors(
        str(scratch),
        {
            "boolean_logits": tissue.boolean_logits,
            "modular_logits": tissue.modular_logits,
        },
    )
    os.replace(scratch, target / "weights.safetensors")
    return write_neural_transition_manifest(target, training_receipt=receipt)


__all__ = [
    "NEURAL_TRANSITION_TRAINING_SCHEMA",
    "TransitionTrainingBatch",
    "build_certified_transition_batch",
    "train_and_write_neural_transition_artifact",
    "train_neural_transition_tissue",
    "transition_training_metrics",
]
