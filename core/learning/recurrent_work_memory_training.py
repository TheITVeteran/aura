"""Train and evaluate bounded mathematics memory predicates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from safetensors.numpy import save as save_safetensors_bytes

from core.brain.llm.latent_cortex.frontier_tasks import generate_task
from core.brain.llm.latent_cortex.persistence import LatentCortexPersistence
from core.learning.frontier_process_supervision import (
    compile_frontier_process_supervision,
)
from core.learning.recurrent_work_memory import MathematicsWorkMemoryTrace
from core.learning.recurrent_work_memory_tissue import (
    MathematicsMemoryTissue,
    build_mathematics_memory_manifest,
    execute_mathematics_memory,
)

MATHEMATICS_MEMORY_TRAINING_SCHEMA: Final = (
    "aura.mathematics_memory_training.v1"
)


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


@dataclass(frozen=True, slots=True)
class MathematicsMemoryPredicateBatch:
    write_features: tuple[tuple[float, ...], ...]
    write_labels: tuple[int, ...]
    read_features: tuple[tuple[float, ...], ...]
    read_labels: tuple[int, ...]
    task_ids: tuple[str, ...]
    supervision_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.write_features
            or len(self.write_features) != len(self.write_labels)
            or not self.read_features
            or len(self.read_features) != len(self.read_labels)
            or any(label not in (0, 1) for label in self.write_labels)
            or any(label not in (0, 1) for label in self.read_labels)
            or not self.task_ids
            or len(set(self.task_ids)) != len(self.task_ids)
            or len(self.supervision_sha256) != 64
        ):
            raise ValueError("mathematics memory predicate batch is invalid")


@dataclass(frozen=True, slots=True)
class MathematicsMemoryEvaluationTask:
    task_id: str
    choose: int
    gap: int
    low: int
    high: int
    values: tuple[int, ...]
    expected_count: int
    expected_witness: tuple[int, ...]

    @property
    def public_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "choose": self.choose,
            "gap": self.gap,
            "low": self.low,
            "high": self.high,
            "values": list(self.values),
        }


def _trace_objective(trace: MathematicsWorkMemoryTrace) -> dict[str, Any]:
    final = trace.states[-1]
    return {
        "choose": final.choose,
        "gap": final.gap,
        "low": final.low,
        "high": final.high,
        "values": final.processed_values,
    }


def build_mathematics_memory_registry(
    *,
    seeds: range,
    difficulties: tuple[int, ...] = (1, 2, 3),
) -> tuple[MathematicsMemoryPredicateBatch, tuple[MathematicsMemoryEvaluationTask, ...]]:
    """Compile private predicate labels and a separately scored task registry."""

    if (
        not isinstance(seeds, range)
        or not seeds
        or not difficulties
        or any(difficulty not in (1, 2, 3) for difficulty in difficulties)
    ):
        raise ValueError("mathematics memory registry selection is invalid")
    write_features: list[tuple[float, ...]] = []
    write_labels: list[int] = []
    read_features: list[tuple[float, ...]] = []
    read_labels: list[int] = []
    task_ids: list[str] = []
    evaluation: list[MathematicsMemoryEvaluationTask] = []
    trace_hashes: list[str] = []
    for difficulty in difficulties:
        for seed in seeds:
            source = generate_task("mathematics", seed=seed, difficulty=difficulty)
            compiled = compile_frontier_process_supervision(source)
            trace = compiled.work_memory_trace
            if trace is None:
                raise RuntimeError("mathematics work-memory teacher disappeared")
            objective = _trace_objective(trace)
            values = objective["values"]
            for index, value in enumerate(values):
                state = trace.states[index + trace.configuration_steps]
                for cell in state.cells:
                    write_features.append(
                        MathematicsMemoryTissue.write_features(
                            cell,
                            value=value,
                            choose=state.choose,
                            gap=state.gap,
                            processed_count=len(state.processed_values),
                        )
                    )
                    selected = cell.address.selected_count
                    write_labels.append(
                        int(
                            selected < state.choose
                            and (
                                selected == 0
                                or value - cell.address.last_value >= state.gap
                            )
                        )
                    )
            final = trace.states[-1]
            for cell in final.cells:
                read_features.append(
                    MathematicsMemoryTissue.read_features(
                        cell,
                        choose=final.choose,
                        low=final.low,
                        high=final.high,
                    )
                )
                read_labels.append(
                    int(
                        cell.address.selected_count == final.choose
                        and final.low <= cell.address.total_sum <= final.high
                    )
                )
            expected = source.reveal_for_verifier()["expected"]
            task_ids.append(source.task_id)
            trace_hashes.append(trace.trace_sha256)
            evaluation.append(
                MathematicsMemoryEvaluationTask(
                    task_id=source.task_id,
                    choose=final.choose,
                    gap=final.gap,
                    low=final.low,
                    high=final.high,
                    values=final.processed_values,
                    expected_count=expected["count"],
                    expected_witness=tuple(expected["witness"]),
                )
            )
    supervision_body = {
        "schema": MATHEMATICS_MEMORY_TRAINING_SCHEMA,
        "task_ids": task_ids,
        "trace_sha256s": trace_hashes,
        "write_examples": len(write_labels),
        "read_examples": len(read_labels),
    }
    return (
        MathematicsMemoryPredicateBatch(
            write_features=tuple(write_features),
            write_labels=tuple(write_labels),
            read_features=tuple(read_features),
            read_labels=tuple(read_labels),
            task_ids=tuple(task_ids),
            supervision_sha256=_canonical_sha256(supervision_body),
        ),
        tuple(evaluation),
    )


def _balanced_binary_loss(logits: Any, labels: Any) -> Any:
    positives = mx.maximum(mx.sum(labels), 1.0)
    negatives = mx.maximum(mx.sum(1.0 - labels), 1.0)
    weights = mx.where(labels > 0.5, negatives / positives, mx.ones_like(labels))
    return nn.losses.binary_cross_entropy(
        logits,
        labels,
        weights=weights,
        with_logits=True,
        reduction="mean",
    )


def predicate_metrics(
    tissue: MathematicsMemoryTissue,
    batch: MathematicsMemoryPredicateBatch,
) -> dict[str, float]:
    write_features = mx.array(batch.write_features, dtype=mx.float32)
    write_labels = mx.array(batch.write_labels, dtype=mx.float32)
    read_features = mx.array(batch.read_features, dtype=mx.float32)
    read_labels = mx.array(batch.read_labels, dtype=mx.float32)
    write_logits = tissue.write_logits(write_features)
    read_logits = tissue.read_logits(read_features)
    write_loss = _balanced_binary_loss(write_logits, write_labels)
    read_loss = _balanced_binary_loss(read_logits, read_labels)
    write_accuracy = mx.mean((write_logits > 0.0) == (write_labels > 0.5))
    read_accuracy = mx.mean((read_logits > 0.0) == (read_labels > 0.5))
    mx.eval(write_loss, read_loss, write_accuracy, read_accuracy)
    return {
        "loss": float((write_loss + read_loss).item()),
        "write_accuracy": float(write_accuracy.item()),
        "read_accuracy": float(read_accuracy.item()),
    }


def autonomous_execution_metrics(
    tissue: MathematicsMemoryTissue,
    tasks: tuple[MathematicsMemoryEvaluationTask, ...],
    *,
    write_mode: str = "learned",
    read_mode: str = "learned",
    routing_mode: str = "identity",
    memory_mode: str = "active",
) -> dict[str, Any]:
    if not tasks:
        raise ValueError("mathematics memory evaluation task set is empty")
    correct: list[str] = []
    incorrect: list[str] = []
    for task in tasks:
        result = execute_mathematics_memory(
            tissue,
            choose=task.choose,
            gap=task.gap,
            low=task.low,
            high=task.high,
            values=task.values,
            write_mode=write_mode,
            read_mode=read_mode,
            routing_mode=routing_mode,
            memory_mode=memory_mode,
        )
        target = (
            task.expected_count,
            task.expected_witness,
        )
        observed = (result.count, result.witness)
        (correct if observed == target else incorrect).append(task.task_id)
    body = {
        "schema": MATHEMATICS_MEMORY_TRAINING_SCHEMA,
        "examples": len(tasks),
        "exact": len(correct),
        "exact_accuracy": len(correct) / len(tasks),
        "correct_task_ids": correct,
        "incorrect_task_ids": incorrect,
        "write_mode": write_mode,
        "read_mode": read_mode,
        "routing_mode": routing_mode,
        "memory_mode": memory_mode,
        "teacher_removed": True,
        "student_memory_rollin": True,
    }
    return {**body, "receipt_sha256": _canonical_sha256(body)}


def train_mathematics_memory_tissue(
    batch: MathematicsMemoryPredicateBatch,
    *,
    steps: int = 400,
    learning_rate: float = 0.01,
    hidden_size: int = 32,
    seed: int = 2026081507,
) -> tuple[MathematicsMemoryTissue, dict[str, Any]]:
    if (
        type(steps) is not int
        or not 1 <= steps <= 10_000
        or isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or not 1e-6 <= float(learning_rate) <= 1.0
    ):
        raise ValueError("mathematics memory training configuration is invalid")
    tissue = MathematicsMemoryTissue(hidden_size=hidden_size, seed=seed)
    optimizer = optim.Adam(learning_rate=float(learning_rate))
    write_features = mx.array(batch.write_features, dtype=mx.float32)
    write_labels = mx.array(batch.write_labels, dtype=mx.float32)
    read_features = mx.array(batch.read_features, dtype=mx.float32)
    read_labels = mx.array(batch.read_labels, dtype=mx.float32)

    def objective(candidate: MathematicsMemoryTissue) -> Any:
        return _balanced_binary_loss(
            candidate.write_logits(write_features),
            write_labels,
        ) + _balanced_binary_loss(
            candidate.read_logits(read_features),
            read_labels,
        )

    initial = predicate_metrics(tissue, batch)
    loss_and_grad = nn.value_and_grad(tissue, objective)
    history: list[float] = []
    for _step in range(steps):
        loss, gradients = loss_and_grad(tissue)
        optimizer.update(tissue, gradients)
        mx.eval(tissue.parameters(), optimizer.state, loss)
        observed = float(loss.item())
        if not math.isfinite(observed):
            raise FloatingPointError("mathematics memory training became non-finite")
        history.append(observed)
    final = predicate_metrics(tissue, batch)
    body = {
        "schema": MATHEMATICS_MEMORY_TRAINING_SCHEMA,
        "steps": steps,
        "learning_rate": float(learning_rate),
        "hidden_size": hidden_size,
        "seed": seed,
        "supervision_sha256": batch.supervision_sha256,
        "task_count": len(batch.task_ids),
        "write_example_count": len(batch.write_labels),
        "read_example_count": len(batch.read_labels),
        "initial_metrics": initial,
        "final_metrics": final,
        "loss_history_sha256": _canonical_sha256(history),
        "teacher_removed_before_evaluation": True,
    }
    return tissue, {**body, "receipt_sha256": _canonical_sha256(body)}


def train_and_write_mathematics_memory_artifact(
    out_dir: Path,
    *,
    canary_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Reproduce the admitted tissue and publish one atomic runtime artifact."""

    training, _tasks = build_mathematics_memory_registry(
        seeds=range(0, 40),
        difficulties=(1, 2, 3),
    )
    tissue, training_receipt = train_mathematics_memory_tissue(
        training,
        steps=400,
        learning_rate=0.01,
        hidden_size=32,
        seed=2026081507,
    )
    if training_receipt != canary_receipt.get("training"):
        raise RuntimeError("mathematics memory tissue is not a canary reproduction")
    tensors = {
        name: np.asarray(value)
        for name, value in tree_flatten(tissue.parameters())
    }
    weights_payload = save_safetensors_bytes(tensors)
    manifest = build_mathematics_memory_manifest(
        weights_sha256=hashlib.sha256(weights_payload).hexdigest(),
        training_receipt=training_receipt,
        canary_receipt=canary_receipt,
    )
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True).encode("ascii") + b"\n"
    )
    LatentCortexPersistence().publish_neural_tissue_artifact(
        out_dir.expanduser().resolve(),
        weights_payload=weights_payload,
        manifest_payload=manifest_payload,
    )
    return manifest


__all__ = [
    "MATHEMATICS_MEMORY_TRAINING_SCHEMA",
    "MathematicsMemoryEvaluationTask",
    "MathematicsMemoryPredicateBatch",
    "autonomous_execution_metrics",
    "build_mathematics_memory_registry",
    "predicate_metrics",
    "train_mathematics_memory_tissue",
    "train_and_write_mathematics_memory_artifact",
]
