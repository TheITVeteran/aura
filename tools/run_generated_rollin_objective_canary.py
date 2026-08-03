#!/usr/bin/env python3
"""Run a bounded real-checkpoint canary for the generated-prefix RLC objective.

This is an engineering discriminator, not a reasoning-gain experiment. It
loads a small MLX checkpoint, attaches the production recurrent adapter
topology, measures a held-out row, performs a bounded number of updates, and
remeasures the same row. The durable receipt proves that adapter tensors moved,
the base checkpoint files did not, generated-prefix evidence validates, and all
losses remained finite. It never promotes or fuses an adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.execution_spec import RLCExecutionSpec  # noqa: E402
from core.brain.llm.latent_cortex.recurrence_adapter_identity_v2 import (  # noqa: E402
    full_weight_checkpoint_identity,
)
from core.learning.recurrence_native_objective_v5 import (  # noqa: E402
    RECURRENCE_NATIVE_OBJECTIVE_V5_SCHEMA,
    GeneratedRollinSelectionConfig,
    derive_rollin_seed,
    generated_rollin_live_path_loss,
    generated_rollin_live_path_value_and_grad,
    validate_generated_rollin_receipt,
)
from core.learning.recurrent_grpo import (  # noqa: E402
    attach_recurrent_policy_adapters,
)
from core.learning.recurrent_sft_execution import (  # noqa: E402
    adapter_tensor_dict,
    adapter_tensor_fingerprint,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

CANARY_SCHEMA: Final = "aura.generated_rollin_objective_canary.v1"
SOURCE_PATHS: Final = (
    "core/learning/recurrence_native_objective_v2.py",
    "core/learning/recurrence_native_objective_v5.py",
    "core/learning/recurrent_grpo.py",
    "core/brain/llm/latent_cortex/recurrence_adapter.py",
    "core/learning/depth_conditioned_lora.py",
    "tools/run_generated_rollin_objective_canary.py",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_state() -> tuple[str, dict[str, dict[str, Any]]]:
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError("canary requires a clean source worktree")
    head = _git("rev-parse", "HEAD")
    origin_main = _git("rev-parse", "origin/main")
    if head != origin_main:
        raise RuntimeError("canary source commit is not published on origin/main")
    bindings: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        payload = path.read_bytes()
        committed = subprocess.run(
            ["git", "show", f"{head}:{relative}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if payload != committed:
            raise RuntimeError(f"canary source differs from commit: {relative}")
        bindings[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    return head, bindings


def _tokenize(tokenizer: Any, prompt: str, answer: str) -> tuple[list[int], list[int]]:
    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    try:
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    except TypeError:
        answer_tokens = tokenizer.encode(answer)
    eos = getattr(tokenizer, "eos_token_id", None)
    normalized_answer = [int(token) for token in answer_tokens]
    if eos is not None and (not normalized_answer or normalized_answer[-1] != int(eos)):
        normalized_answer.append(int(eos))
    return [int(token) for token in prompt_tokens], normalized_answer


def _evaluate(
    model: Any,
    row: dict[str, Any],
    *,
    spec: RLCExecutionSpec,
    config: GeneratedRollinSelectionConfig,
    campaign_seed: int,
) -> dict[str, Any]:
    seed = derive_rollin_seed(
        campaign_seed=campaign_seed,
        phase="validation",
        example_id=row["task_id"],
        sample_ordinal=0,
        execution_spec_sha256=spec.sha256,
    )
    evaluation = generated_rollin_live_path_loss(
        model,
        row["prompt_tokens"],
        row["answer_tokens"],
        spec=spec,
        base_seed=seed,
        config=config,
    )
    receipt = validate_generated_rollin_receipt(evaluation.receipt())
    return {
        "task_id": row["task_id"],
        "loss": evaluation.value,
        "branch_values": list(evaluation.branch_values),
        "branch_weights": list(evaluation.branch_weights),
        "rollin_base_seed": seed,
        "objective_receipt": receipt,
    }


def run_canary(
    *,
    model_path: Path,
    out_dir: Path,
    steps: int,
    seed: int,
    memory_fraction: float,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm import load

    from core.learning.recurrence_curriculum import task_battery

    if type(steps) is not int or not 1 <= steps <= 8:
        raise ValueError("steps must be inside [1, 8]")
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be inside [0, 2^63-1]")
    started = time.time()
    source_commit, source_bindings = _source_state()
    base_before = full_weight_checkpoint_identity(model_path)
    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution", "critical_audit"),
        recurrent_steps=2,
        exchange_interval=1,
    )
    objective_config = GeneratedRollinSelectionConfig(
        student_forcing_probability=1.0,
        sampling_temperature=0.0,
        branch_softmin_temperature=0.5,
    )
    with (
        standalone_model_lane(
            owner_id=f"generated-rollin-canary:{out_dir.name}",
            model_path=str(model_path),
            purpose="training",
            preemptible=False,
            metadata={
                "tool": "run_generated_rollin_objective_canary",
                "source_commit": source_commit,
            },
        ),
        mlx_memory_envelope(
            fraction=memory_fraction,
            restore_limits_on_exit=True,
        ),
    ):
        model, tokenizer = load(str(model_path))
        attach_recurrent_policy_adapters(
            model,
            spec,
            lora_rank=2,
            lora_layers=2,
            lora_targets=("o_proj",),
            initialization_seed=(seed ^ 0x51F7A11) & 0xFFFFFFFF,
            lora_scale=1.0,
            depth_conditioned_steps=spec.recurrent_steps,
        )
        adapter_before = adapter_tensor_fingerprint(adapter_tensor_dict(model))
        tasks = task_battery(["boolean"], [2], 2, seed=seed)
        if len(tasks) != 2:
            raise RuntimeError("canary task battery did not produce two tasks")
        rows = []
        for task in tasks:
            prompt_tokens, answer_tokens = _tokenize(
                tokenizer,
                task.prompt,
                str(task.answer),
            )
            rows.append(
                {
                    "task_id": task.task_id,
                    "prompt_tokens": prompt_tokens,
                    "answer_tokens": answer_tokens,
                }
            )
        training_row, validation_row = rows
        before = _evaluate(
            model,
            validation_row,
            spec=spec,
            config=objective_config,
            campaign_seed=seed,
        )
        optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.0)
        optimizer.init(model.trainable_parameters())
        loss_trail: list[dict[str, Any]] = []
        for step in range(1, steps + 1):
            rollin_seed = derive_rollin_seed(
                campaign_seed=seed,
                phase="train",
                example_id=training_row["task_id"],
                sample_ordinal=step,
                execution_spec_sha256=spec.sha256,
            )
            result = generated_rollin_live_path_value_and_grad(
                model,
                training_row["prompt_tokens"],
                training_row["answer_tokens"],
                spec=spec,
                base_seed=rollin_seed,
                config=objective_config,
            )
            receipt = validate_generated_rollin_receipt(
                result.evaluation.receipt()
            )
            optimizer.update(model, result.gradients)
            mx.eval(model.trainable_parameters(), optimizer.state)
            loss_trail.append(
                {
                    "step": step,
                    "loss": result.value,
                    "branch_values": list(result.branch_values),
                    "branch_weights": list(result.branch_weights),
                    "rollin_base_seed": rollin_seed,
                    "objective_receipt": receipt,
                    "adapter_sha256": adapter_tensor_fingerprint(
                        adapter_tensor_dict(model)
                    ),
                }
            )
        after = _evaluate(
            model,
            validation_row,
            spec=spec,
            config=objective_config,
            campaign_seed=seed,
        )
        adapter = adapter_tensor_dict(model)
        adapter_after = adapter_tensor_fingerprint(adapter)
        out_dir.mkdir(parents=True, exist_ok=False)
        mx.save_safetensors(str(out_dir / "adapter.safetensors"), adapter)

    base_after = full_weight_checkpoint_identity(model_path)
    finite_losses = [
        before["loss"],
        after["loss"],
        *(entry["loss"] for entry in loss_trail),
    ]
    gates = {
        "base_checkpoint_immutable": base_before == base_after,
        "adapter_mutated": adapter_before != adapter_after,
        "losses_finite": all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in finite_losses
        ),
        "generated_prefix_exercised": all(
            any(
                branch["student_forced_positions"]
                for branch in entry["objective_receipt"]["branches"]
            )
            for entry in loss_trail
        ),
        "branch_credit_normalized": all(
            math.isclose(
                sum(entry["branch_weights"]),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for entry in loss_trail
        ),
    }
    body = {
        "schema": CANARY_SCHEMA,
        "objective_schema": RECURRENCE_NATIVE_OBJECTIVE_V5_SCHEMA,
        "source_commit": source_commit,
        "source_bindings": source_bindings,
        "model_path": str(model_path),
        "base_before": base_before,
        "base_after": base_after,
        "execution_spec": spec.to_dict(),
        "execution_spec_sha256": spec.sha256,
        "objective_config": objective_config.to_dict(),
        "seed": seed,
        "steps": steps,
        "adapter_before_sha256": adapter_before,
        "adapter_after_sha256": adapter_after,
        "training_task_id": training_row["task_id"],
        "validation_task_id": validation_row["task_id"],
        "validation_before": before,
        "loss_trail": loss_trail,
        "validation_after": after,
        "validation_loss_delta": after["loss"] - before["loss"],
        "gates": gates,
        "passed": all(gates.values()),
        "claim_state": {
            "reasoning_gain_proven": False,
            "frontier_level_proven": False,
            "promotion_allowed": False,
            "fusion_allowed": False,
        },
        "elapsed_s": time.time() - started,
    }
    receipt = {
        **body,
        "receipt_sha256": hashlib.sha256(_canonical_json_bytes(body)).hexdigest(),
    }
    atomic_write_bytes(
        out_dir / "receipt.json",
        _canonical_json_bytes(receipt),
        mode=0o600,
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026080207)
    parser.add_argument("--memory-fraction", type=float, default=0.35)
    args = parser.parse_args()
    receipt = run_canary(
        model_path=args.model.expanduser().resolve(strict=True),
        out_dir=args.out_dir.expanduser().resolve(strict=False),
        steps=args.steps,
        seed=args.seed,
        memory_fraction=args.memory_fraction,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
