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
from core.learning.recurrence_native_objective_v2 import (  # noqa: E402
    ExactAdjointTrajectoryConfig,
)
from core.learning.recurrence_native_objective_v5 import (  # noqa: E402
    GeneratedRollinSelectionConfig,
    derive_rollin_seed,
)
from core.learning.recurrence_native_objective_v6 import (  # noqa: E402
    RECURRENCE_NATIVE_OBJECTIVE_V6_SCHEMA,
    BranchSpecializationConfig,
    branch_specialization_live_path_loss,
    branch_specialization_live_path_value_and_grad,
    generated_rollin_specialization_loss,
    generated_rollin_specialization_value_and_grad,
    validate_branch_specialization_receipt,
    validate_generated_rollin_specialization_receipt,
)
from core.learning.recurrent_checkpoint_admission import (  # noqa: E402
    build_checkpoint_behavioral_admission,
    build_free_generation_report,
    build_recurrence_task_manifest,
    validate_checkpoint_behavioral_admission,
)
from core.learning.recurrent_grpo import (  # noqa: E402
    RecurrentSamplingConfig,
    attach_recurrent_policy_adapters,
    cortex_config_from_execution_spec,
    recurrent_policy_sha256,
)
from core.learning.recurrent_sft_execution import (  # noqa: E402
    adapter_tensor_dict,
    adapter_tensor_fingerprint,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

CANARY_SCHEMA: Final = "aura.generated_rollin_objective_canary.v2"
SOURCE_PATHS: Final = (
    "core/learning/recurrence_native_objective_v2.py",
    "core/learning/recurrence_native_objective_v5.py",
    "core/learning/recurrence_native_objective_v4.py",
    "core/learning/recurrence_native_objective_v6.py",
    "core/learning/role_conditioned_lora.py",
    "core/learning/recurrent_grpo.py",
    "core/learning/recurrent_checkpoint_admission.py",
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
    generated_config: GeneratedRollinSelectionConfig,
    specialization_config: BranchSpecializationConfig,
    trajectory_config: ExactAdjointTrajectoryConfig,
    campaign_seed: int,
) -> dict[str, Any]:
    seed = derive_rollin_seed(
        campaign_seed=campaign_seed,
        phase="validation",
        example_id=row["task_id"],
        sample_ordinal=0,
        execution_spec_sha256=spec.sha256,
    )
    evaluation = generated_rollin_specialization_loss(
        model,
        row["prompt_tokens"],
        row["answer_tokens"],
        spec=spec,
        base_seed=seed,
        generated_config=generated_config,
        specialization_config=specialization_config,
        trajectory_config=trajectory_config,
        trajectory_policy_sha256=recurrent_policy_sha256(model, spec),
    )
    receipt = validate_generated_rollin_specialization_receipt(
        evaluation.receipt()
    )
    return {
        "task_id": row["task_id"],
        "loss": evaluation.value,
        "lexical_loss": evaluation.generated.value,
        "specialization_loss": evaluation.specialization.value,
        "trajectory_loss": evaluation.trajectory.value,
        "branch_separations": list(evaluation.specialization.separations),
        "branch_values": list(evaluation.branch_values),
        "branch_weights": list(evaluation.branch_weights),
        "rollin_base_seed": seed,
        "objective_receipt": receipt,
    }


def _branch_separations(
    model: Any,
    row: dict[str, Any],
    *,
    spec: RLCExecutionSpec,
) -> list[float]:
    import mlx.core as mx

    from core.learning.recurrence_native_objective_v2 import live_path_forward
    from core.learning.recurrence_native_objective_v4 import pairwise_separations

    forward = live_path_forward(
        model,
        row["prompt_tokens"],
        row["answer_tokens"],
        spec=spec,
    )
    values = pairwise_separations(forward, comm_slot=spec.comm_slot)
    mx.eval(values)
    result = [float(value) for value in values]
    del forward, values
    mx.clear_cache()
    return result


def _branch_specialization_gates(
    loss_trail: list[dict[str, Any]],
    separation_after: list[float],
) -> dict[str, bool]:
    return {
        "branch_generated_prefix_distinct": bool(
            loss_trail
            and all(
                len(
                    {
                        branch["generated_tokens_sha256"]
                        for branch in entry["objective_receipt"][
                            "generated_receipt"
                        ]["branches"]
                    }
                )
                == len(
                    entry["objective_receipt"]["generated_receipt"]["branches"]
                )
                for entry in loss_trail
            )
        ),
        "branch_state_specialized": bool(
            separation_after and min(separation_after) >= 0.30
        ),
    }


def _paired_generation_seed(
    campaign_seed: int,
    task_ordinal: int,
    task_id: str,
    depth: int,
) -> int:
    """Use one random stream per task/depth coordinate across both arms."""

    material = f"{campaign_seed}:{task_ordinal}:{task_id}:{depth}"
    return int.from_bytes(hashlib.sha256(material.encode("ascii")).digest()[:4], "big")


def _free_generation_sampling_config() -> RecurrentSamplingConfig:
    """Use the exact categorical policy required by recurrent proof runs."""

    return RecurrentSamplingConfig(
        max_tokens=96,
        temperature=1.0,
        top_p=1.0,
    )


def _free_generation_report(
    model: Any,
    tokenizer: Any,
    tasks: list[Any],
    *,
    spec: RLCExecutionSpec,
    arm: str,
    adapter_sha256: str,
    task_manifest_sha256: str,
    seed: int,
) -> dict[str, Any]:
    """Run exact held-out generations at shallow and full recurrent depth."""

    import mlx.core as mx

    from core.brain.llm.latent_cortex.engine import LatentCortexEngine

    depths = tuple(sorted({1, spec.recurrent_steps}))
    records: list[dict[str, Any]] = []
    for task_ordinal, task in enumerate(tasks):
        prompt_tokens, _answer_tokens = _tokenize(
            tokenizer,
            task.prompt,
            task.answer,
        )
        for depth in depths:
            depth_spec = spec.with_depth(depth)
            config = cortex_config_from_execution_spec(
                depth_spec,
                sampling=_free_generation_sampling_config(),
            )
            config.decode_contract = "final_answer_v1"
            config.decode_contract_grace_tokens = 0
            config.decode_incumbent_policy = "latent"
            engine = LatentCortexEngine(
                model,
                tokenizer=tokenizer,
                config=config,
                schedule_library=None,
            )
            mx.random.seed(
                _paired_generation_seed(seed, task_ordinal, task.task_id, depth)
            )
            result = engine.reason(
                token_ids=prompt_tokens,
                decode_max_tokens=96,
                decode_sentence_grace_tokens=0,
            )
            grade = dict(task.grade(result.text if result.ok else ""))
            grade["correct"] = bool(grade.get("correct"))
            receipt_payload = result.receipt.to_dict()
            records.append(
                {
                    "task_id": task.task_id,
                    "depth": depth,
                    "response_sha256": hashlib.sha256(
                        result.text.encode("utf-8")
                    ).hexdigest(),
                    "response_text": result.text,
                    "tokens_sha256": hashlib.sha256(
                        _canonical_json_bytes(result.tokens)
                    ).hexdigest(),
                    "tokens": list(result.tokens),
                    "token_count": len(result.tokens),
                    "correct": bool(result.ok and grade["correct"]),
                    "grade_receipt": {
                        **grade,
                        "correct": bool(result.ok and grade["correct"]),
                    },
                    "episode_ok": bool(result.ok),
                    "episode_reason": str(result.reason or ""),
                    "decode_termination": str(
                        result.receipt.decode_termination or "not_reached"
                    ),
                    "branch_selection_admitted": bool(
                        result.receipt.branch_selection_admitted
                    ),
                    "decode_incumbent_policy": (
                        result.receipt.decode_incumbent_policy
                    ),
                    "episode_receipt_sha256": hashlib.sha256(
                        _canonical_json_bytes(receipt_payload)
                    ).hexdigest(),
                }
            )
            del engine, result
            mx.synchronize()
            mx.clear_cache()
    return build_free_generation_report(
        arm=arm,
        adapter_sha256=adapter_sha256,
        execution_spec_sha256=spec.sha256,
        task_manifest_sha256=task_manifest_sha256,
        task_ids=[task.task_id for task in tasks],
        depths=depths,
        records=records,
    )


def run_canary(
    *,
    model_path: Path,
    out_dir: Path,
    steps: int,
    seed: int,
    memory_fraction: float,
    student_forcing_probability: float,
    sampling_temperature: float,
    specialization_weight: float,
    warmup_steps: int,
    warmup_learning_rate: float,
    joint_learning_rate: float,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm import load

    from core.learning.recurrence_curriculum import task_battery

    if type(steps) is not int or not 1 <= steps <= 8:
        raise ValueError("steps must be inside [1, 8]")
    if type(warmup_steps) is not int or not 1 <= warmup_steps <= 8:
        raise ValueError("warmup_steps must be inside [1, 8]")
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be inside [0, 2^63-1]")
    for name, value in (
        ("warmup_learning_rate", warmup_learning_rate),
        ("joint_learning_rate", joint_learning_rate),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 1e-6 <= float(value) <= 1e-2
        ):
            raise ValueError(f"{name} must be inside [1e-6, 1e-2]")
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
        student_forcing_probability=student_forcing_probability,
        sampling_temperature=sampling_temperature,
        branch_softmin_temperature=0.5,
    )
    specialization_config = BranchSpecializationConfig(
        weight=specialization_weight,
        target_separation=0.30,
    )
    trajectory_config = ExactAdjointTrajectoryConfig(
        probe_steps=(1, 2),
        improvement_weight=1.0,
        improvement_margin=0.05,
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
            role_conditioned_branches=len(spec.branch_roles),
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
        proxy_tasks = task_battery(
            ["boolean", "modular"],
            [2],
            2,
            seed=seed + 7_919,
            excluded_prompts=tuple(task.prompt for task in tasks),
            excluded_task_ids=tuple(task.task_id for task in tasks),
        )
        proxy_manifest, proxy_manifest_sha256 = build_recurrence_task_manifest(
            proxy_tasks
        )
        free_generation_before = _free_generation_report(
            model,
            tokenizer,
            proxy_tasks,
            spec=spec,
            arm="initial_adapter",
            adapter_sha256=adapter_before,
            task_manifest_sha256=proxy_manifest_sha256,
            seed=seed,
        )
        before = _evaluate(
            model,
            validation_row,
            spec=spec,
            generated_config=objective_config,
            specialization_config=specialization_config,
            trajectory_config=trajectory_config,
            campaign_seed=seed,
        )
        separation_before = _branch_separations(
            model,
            validation_row,
            spec=spec,
        )
        warmup_optimizer = optim.AdamW(
            learning_rate=warmup_learning_rate,
            weight_decay=0.0,
        )
        warmup_optimizer.init(model.trainable_parameters())
        warmup_trail: list[dict[str, Any]] = []
        for warmup_step in range(1, warmup_steps + 1):
            result = branch_specialization_live_path_value_and_grad(
                model,
                training_row["prompt_tokens"],
                spec=spec,
                config=specialization_config,
            )
            structural_receipt = validate_branch_specialization_receipt(
                result.evaluation.receipt()
            )
            optimizer_before = adapter_tensor_fingerprint(
                adapter_tensor_dict(model)
            )
            warmup_optimizer.update(model, result.gradients)
            mx.eval(model.trainable_parameters(), warmup_optimizer.state)
            post_update = branch_specialization_live_path_loss(
                model,
                training_row["prompt_tokens"],
                spec=spec,
                config=specialization_config,
            )
            warmup_trail.append(
                {
                    "step": warmup_step,
                    "loss_before": result.value,
                    "separations_before": list(
                        result.evaluation.separations
                    ),
                    "separations_after": list(post_update.separations),
                    "objective_receipt": structural_receipt,
                    "adapter_before_sha256": optimizer_before,
                    "adapter_after_sha256": adapter_tensor_fingerprint(
                        adapter_tensor_dict(model)
                    ),
                }
            )
            if min(post_update.separations) >= float(
                specialization_config.target_separation
            ):
                break
        warmup_validation = _evaluate(
            model,
            validation_row,
            spec=spec,
            generated_config=objective_config,
            specialization_config=specialization_config,
            trajectory_config=trajectory_config,
            campaign_seed=seed,
        )
        # Reset momentum when the structural constraint is met. Continuing an
        # Adam trajectory after the hinge reaches zero overshoots the target.
        optimizer = optim.AdamW(
            learning_rate=joint_learning_rate,
            weight_decay=0.0,
        )
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
            result = generated_rollin_specialization_value_and_grad(
                model,
                training_row["prompt_tokens"],
                training_row["answer_tokens"],
                spec=spec,
                base_seed=rollin_seed,
                generated_config=objective_config,
                specialization_config=specialization_config,
                trajectory_config=trajectory_config,
                trajectory_policy_sha256=recurrent_policy_sha256(model, spec),
            )
            receipt = validate_generated_rollin_specialization_receipt(
                result.evaluation.receipt()
            )
            optimizer.update(model, result.gradients)
            mx.eval(model.trainable_parameters(), optimizer.state)
            loss_trail.append(
                {
                    "step": step,
                    "loss": result.value,
                    "lexical_loss": result.evaluation.generated.value,
                    "specialization_loss": result.evaluation.specialization.value,
                    "trajectory_loss": result.evaluation.trajectory.value,
                    "branch_separations": list(
                        result.evaluation.specialization.separations
                    ),
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
            generated_config=objective_config,
            specialization_config=specialization_config,
            trajectory_config=trajectory_config,
            campaign_seed=seed,
        )
        separation_after = _branch_separations(
            model,
            validation_row,
            spec=spec,
        )
        adapter = adapter_tensor_dict(model)
        adapter_after = adapter_tensor_fingerprint(adapter)
        free_generation_after = _free_generation_report(
            model,
            tokenizer,
            proxy_tasks,
            spec=spec,
            arm="trained_adapter",
            adapter_sha256=adapter_after,
            task_manifest_sha256=proxy_manifest_sha256,
            seed=seed,
        )
        behavioral_admission = build_checkpoint_behavioral_admission(
            initial_report=free_generation_before,
            trained_report=free_generation_after,
            task_manifest=proxy_manifest,
        )
        validate_checkpoint_behavioral_admission(
            behavioral_admission,
            initial_report=free_generation_before,
            trained_report=free_generation_after,
            task_manifest=proxy_manifest,
        )
        out_dir.mkdir(parents=True, exist_ok=False)
        mx.save_safetensors(str(out_dir / "adapter.safetensors"), adapter)

    base_after = full_weight_checkpoint_identity(model_path)
    finite_losses = [
        before["loss"],
        warmup_validation["loss"],
        after["loss"],
        *(entry["loss_before"] for entry in warmup_trail),
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
                for branch in entry["objective_receipt"]["generated_receipt"][
                    "branches"
                ]
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
        "warmup_target_reached": bool(
            warmup_trail
            and min(warmup_trail[-1]["separations_after"])
            >= float(specialization_config.target_separation)
        ),
        **_branch_specialization_gates(loss_trail, separation_after),
        "heldout_lexical_non_regression": after["lexical_loss"]
        <= before["lexical_loss"] + 1e-6,
        "heldout_depth_improvement_non_regression": after["trajectory_loss"]
        <= before["trajectory_loss"] + 1e-6,
        "heldout_free_generation_strict_gain": behavioral_admission["admitted"],
    }
    body = {
        "schema": CANARY_SCHEMA,
        "objective_schema": RECURRENCE_NATIVE_OBJECTIVE_V6_SCHEMA,
        "source_commit": source_commit,
        "source_bindings": source_bindings,
        "model_path": str(model_path),
        "base_before": base_before,
        "base_after": base_after,
        "execution_spec": spec.to_dict(),
        "execution_spec_sha256": spec.sha256,
        "objective_config": {
            "generated": objective_config.to_dict(),
            "specialization": specialization_config.to_dict(),
            "trajectory": trajectory_config.to_dict(),
            "warmup_steps": warmup_steps,
            "warmup_learning_rate": float(warmup_learning_rate),
            "joint_learning_rate": float(joint_learning_rate),
        },
        "seed": seed,
        "steps": steps,
        "adapter_before_sha256": adapter_before,
        "adapter_after_sha256": adapter_after,
        "training_task_id": training_row["task_id"],
        "validation_task_id": validation_row["task_id"],
        "proxy_task_manifest": proxy_manifest,
        "proxy_task_manifest_sha256": proxy_manifest_sha256,
        "free_generation_before": free_generation_before,
        "free_generation_after": free_generation_after,
        "checkpoint_behavioral_admission": behavioral_admission,
        "validation_before": before,
        "branch_separation_before": separation_before,
        "warmup_trail": warmup_trail,
        "validation_after_warmup": warmup_validation,
        "loss_trail": loss_trail,
        "validation_after": after,
        "branch_separation_after": separation_after,
        "validation_loss_delta": after["loss"] - before["loss"],
        "validation_lexical_loss_delta": (
            after["lexical_loss"] - before["lexical_loss"]
        ),
        "gates": gates,
        "passed": all(gates.values()),
        "claim_state": {
            "reasoning_gain_proven": False,
            "frontier_level_proven": False,
            "promotion_allowed": False,
            "fusion_allowed": False,
            "resident_campaign_admitted": False,
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
    parser.add_argument("--student-forcing-probability", type=float, default=0.5)
    parser.add_argument("--sampling-temperature", type=float, default=0.8)
    parser.add_argument("--specialization-weight", type=float, default=8.0)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--warmup-learning-rate", type=float, default=1e-3)
    parser.add_argument("--joint-learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    receipt = run_canary(
        model_path=args.model.expanduser().resolve(strict=True),
        out_dir=args.out_dir.expanduser().resolve(strict=False),
        steps=args.steps,
        seed=args.seed,
        memory_fraction=args.memory_fraction,
        student_forcing_probability=args.student_forcing_probability,
        sampling_temperature=args.sampling_temperature,
        specialization_weight=args.specialization_weight,
        warmup_steps=args.warmup_steps,
        warmup_learning_rate=args.warmup_learning_rate,
        joint_learning_rate=args.joint_learning_rate,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
