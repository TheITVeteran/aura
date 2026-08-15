#!/usr/bin/env python3
"""Bounded model-free acquisition gate for semantic recurrent microcode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from core.learning.recurrent_action_schema import SEMANTIC_MICRO_OPCODES  # noqa: E402
from core.learning.semantic_micro_curriculum import (  # noqa: E402
    semantic_micro_batch,
    semantic_micro_batch_receipt,
)
from core.learning.unified_intrinsic_objective import (  # noqa: E402
    unified_semantic_micro_primitive_loss,
)
from core.learning.unified_intrinsic_recurrence import (  # noqa: E402
    TRANSITION_OPCODE_EXPERT_PARAMETER_NAMES,
    TRANSITION_PROCESSOR_PARAMETER_NAMES,
    UnifiedRecurrenceConfig,
    UnifiedRecurrentController,
)
from core.runtime.atomic_writer import (  # noqa: E402
    atomic_write_text,
    ensure_private_directory,
)

SEMANTIC_MICRO_PROCESSOR_TRAINING_SCHEMA = (
    "aura.semantic_micro_processor_training.v1"
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


def _processor_tensors(controller: UnifiedRecurrentController) -> dict[str, Any]:
    allowed = set(TRANSITION_PROCESSOR_PARAMETER_NAMES) | set(
        TRANSITION_OPCODE_EXPERT_PARAMETER_NAMES
    )
    selected = {
        name: value
        for name, value in tree_flatten(controller.parameters())
        if name in allowed
    }
    if set(selected) != allowed:
        raise RuntimeError("semantic processor tensor inventory differs")
    return selected


def _evaluate(
    controller: UnifiedRecurrentController,
    *,
    seed: int,
    batch_size: int,
    batch_index: int,
) -> dict[str, Any]:
    examples = semantic_micro_batch(
        seed=seed,
        batch_size=batch_size,
        batch_index=batch_index,
    )
    loss, metrics = unified_semantic_micro_primitive_loss(controller, examples)
    mx.eval(loss)
    return {
        **metrics,
        "batch": semantic_micro_batch_receipt(examples),
    }


def train_semantic_micro_processor(
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    train_seed: int,
    heldout_seed: int,
    evaluation_interval: int,
    exact_patience: int,
    initialization_seed: int,
) -> tuple[UnifiedRecurrentController, dict[str, Any]]:
    opcode_count = len(SEMANTIC_MICRO_OPCODES)
    if (
        type(steps) is not int
        or not 1 <= steps <= 20_000
        or type(batch_size) is not int
        or batch_size < opcode_count
        or isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or not 1e-6 <= float(learning_rate) <= 0.1
        or type(train_seed) is not int
        or train_seed < 0
        or type(heldout_seed) is not int
        or heldout_seed < 0
        or train_seed == heldout_seed
        or type(evaluation_interval) is not int
        or not 1 <= evaluation_interval <= steps
        or type(exact_patience) is not int
        or exact_patience < 1
        or type(initialization_seed) is not int
        or initialization_seed < 0
    ):
        raise ValueError("semantic processor training configuration is invalid")
    controller = UnifiedRecurrentController(
        UnifiedRecurrenceConfig(
            hidden_size=64,
            correction_rank=33,
            state_slots=11,
            initialization_seed=initialization_seed,
        )
    )
    optimizer = optim.Adam(learning_rate=float(learning_rate))

    def objective(candidate: UnifiedRecurrentController, update_index: int) -> Any:
        examples = semantic_micro_batch(
            seed=train_seed,
            batch_size=batch_size,
            batch_index=update_index,
        )
        return unified_semantic_micro_primitive_loss(candidate, examples)[0]

    loss_and_grad = nn.value_and_grad(controller, objective)
    initial_parameter_sha256 = _canonical_sha256(
        {
            name: hashlib.sha256(bytes(memoryview(value.astype(mx.float32)))).hexdigest()
            for name, value in _processor_tensors(controller).items()
        }
    )
    history: list[dict[str, Any]] = []
    exact_streak = 0
    started = time.monotonic()
    completed_steps = 0
    for step in range(steps):
        loss, gradients = loss_and_grad(controller, step)
        optimizer.update(controller, gradients)
        mx.eval(controller.parameters(), optimizer.state, loss)
        loss_value = float(loss.item())
        if not math.isfinite(loss_value):
            raise FloatingPointError("semantic processor loss is non-finite")
        completed_steps = step + 1
        if completed_steps % evaluation_interval and completed_steps != steps:
            continue
        evaluation_batch = opcode_count * 16
        train_eval = _evaluate(
            controller,
            seed=train_seed,
            batch_size=evaluation_batch,
            batch_index=10_000 + completed_steps,
        )
        heldout_eval = _evaluate(
            controller,
            seed=heldout_seed,
            batch_size=evaluation_batch,
            batch_index=20_000 + completed_steps,
        )
        row = {
            "step": completed_steps,
            "optimization_loss": loss_value,
            "train": train_eval,
            "heldout": heldout_eval,
        }
        history.append(row)
        print(
            "[semantic-micro] "
            f"step={completed_steps} loss={loss_value:.6f} "
            f"train={train_eval['exact_example_accuracy']:.4f} "
            f"heldout={heldout_eval['exact_example_accuracy']:.4f}",
            flush=True,
        )
        exact = (
            train_eval["exact_example_accuracy"] == 1.0
            and heldout_eval["exact_example_accuracy"] == 1.0
        )
        exact_streak = exact_streak + 1 if exact else 0
        if exact_streak >= exact_patience:
            break
    final = history[-1] if history else {
        "step": completed_steps,
        "train": _evaluate(
            controller,
            seed=train_seed,
            batch_size=opcode_count * 16,
            batch_index=10_000 + completed_steps,
        ),
        "heldout": _evaluate(
            controller,
            seed=heldout_seed,
            batch_size=opcode_count * 16,
            batch_index=20_000 + completed_steps,
        ),
    }
    tensors = _processor_tensors(controller)
    tensor_sha256s = {
        name: hashlib.sha256(bytes(memoryview(value.astype(mx.float32)))).hexdigest()
        for name, value in tensors.items()
    }
    body = {
        "schema": SEMANTIC_MICRO_PROCESSOR_TRAINING_SCHEMA,
        "steps_requested": steps,
        "steps_completed": completed_steps,
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "train_seed": train_seed,
        "heldout_seed": heldout_seed,
        "initialization_seed": initialization_seed,
        "evaluation_interval": evaluation_interval,
        "exact_patience": exact_patience,
        "elapsed_seconds": time.monotonic() - started,
        "initial_parameter_sha256": initial_parameter_sha256,
        "tensor_sha256s": tensor_sha256s,
        "history": history,
        "final": final,
        "admission": {
            "train_exact": final["train"]["exact_example_accuracy"] == 1.0,
            "heldout_exact": final["heldout"]["exact_example_accuracy"] == 1.0,
            "admitted": exact_streak >= exact_patience,
        },
        "teacher_removed_before_evaluation": True,
        "microcode_available_to_treatment": False,
        "model_loaded": False,
        "claim_boundary": (
            "local semantic micro-operation acquisition on fresh operand "
            "combinations; no multi-step or decoded reasoning claim"
        ),
    }
    return controller, {**body, "receipt_sha256": _canonical_sha256(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--train-seed", type=int, default=202_608_155_470)
    parser.add_argument("--heldout-seed", type=int, default=202_608_155_471)
    parser.add_argument("--evaluation-interval", type=int, default=25)
    parser.add_argument("--exact-patience", type=int, default=3)
    parser.add_argument("--initialization-seed", type=int, default=547)
    args = parser.parse_args()
    target = args.out.expanduser().resolve()
    ensure_private_directory(target)
    controller, receipt = train_semantic_micro_processor(
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_seed=args.train_seed,
        heldout_seed=args.heldout_seed,
        evaluation_interval=args.evaluation_interval,
        exact_patience=args.exact_patience,
        initialization_seed=args.initialization_seed,
    )
    tensors = _processor_tensors(controller)
    scratch = target / f"processor.{os.getpid()}.tmp.safetensors"
    mx.save_safetensors(str(scratch), tensors)
    os.replace(scratch, target / "processor.safetensors")
    atomic_write_text(
        target / "receipt.json",
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        mode=0o600,
    )
    print(json.dumps(receipt["admission"], sort_keys=True), flush=True)
    return 0 if receipt["admission"]["admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
