#!/usr/bin/env python3
"""Run a bounded on-policy recurrent-GRPO behavioral canary.

The generated-rollin canary proved that lower teacher-forced loss can coexist
with worse free generation. This discriminator instead samples complete answers
from the recurrent policy, grades those exact answers with deterministic task
programs, and differentiates their admitted token traces. A separate disjoint
probe still decides whether the resulting checkpoint improved. It never fuses,
promotes, or activates an adapter.
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
from core.learning.grpo import group_advantages, reward_from_verdict  # noqa: E402
from core.learning.recurrent_behavioral_probe import (  # noqa: E402
    build_behavioral_probe_report,
    canonical_json_bytes,
    free_generation_sampling_config,
    tokenize_task,
)
from core.learning.recurrent_checkpoint_admission import (  # noqa: E402
    RecurrentCheckpointAdmissionError,
    build_checkpoint_behavioral_admission,
    build_recurrence_task_manifest,
    validate_checkpoint_behavioral_admission,
)
from core.learning.recurrent_grpo import (  # noqa: E402
    RecurrentGRPOConfig,
    RecurrentSamplingAdmissionError,
    attach_recurrent_policy_adapters,
    build_recurrent_policy_optimizer,
    exact_adjoint_sampled_group_value_and_grad,
    recurrent_policy_sha256,
    sample_final_recurrent_transition_completion,
)
from core.learning.recurrent_sft_execution import (  # noqa: E402
    adapter_tensor_dict,
    adapter_tensor_fingerprint,
)
from core.runtime.atomic_writer import atomic_write_bytes  # noqa: E402
from core.runtime.mlx_memory_guard import mlx_memory_envelope  # noqa: E402
from core.runtime.model_lane_control import standalone_model_lane  # noqa: E402

CANARY_SCHEMA: Final = "aura.recurrent_grpo_behavioral_canary.v1"
SOURCE_PATHS: Final = (
    "core/learning/grpo.py",
    "core/learning/recurrence_curriculum.py",
    "core/learning/recurrent_behavioral_probe.py",
    "core/learning/recurrent_checkpoint_admission.py",
    "core/learning/recurrent_grpo.py",
    "core/learning/recurrence_native_objective_v2.py",
    "core/learning/recurrent_sft_execution.py",
    "core/brain/llm/latent_cortex/execution_spec.py",
    "core/brain/llm/latent_cortex/engine.py",
    "core/brain/llm/latent_cortex/recurrence_adapter.py",
    "core/learning/depth_conditioned_lora.py",
    "core/learning/role_conditioned_lora.py",
    "tools/run_recurrent_grpo_behavioral_canary.py",
)


class RecurrentCanarySamplingError(RuntimeError):
    """A bounded group could not produce enough proof-admissible samples."""

    def __init__(
        self,
        *,
        admitted: int,
        requested: int,
        rejected: list[dict[str, Any]],
    ) -> None:
        self.admitted = admitted
        self.requested = requested
        self.rejected = rejected
        self.diagnostics = [
            _sampling_rejection_diagnostics(receipt) for receipt in rejected
        ]
        reasons = sorted(
            {
                reason
                for diagnostic in self.diagnostics
                for reason in diagnostic["failed_gates"]
            }
        )
        super().__init__(
            "recurrent GRPO canary exhausted admissible samples: "
            f"admitted={admitted} requested={requested} "
            f"rejected={len(rejected)} failed_gates={','.join(reasons) or 'unknown'}"
        )


def _sampling_rejection_diagnostics(receipt: dict[str, Any]) -> dict[str, Any]:
    """Reduce a large sample receipt to the exact local admission predicates."""

    sampling = dict(receipt.get("sampling_config") or {})
    activation = dict(receipt.get("cached_recurrence_adapter") or {})
    episode = dict(receipt.get("episode_receipt") or {})
    honest_flags = list(episode.get("honest_flags") or [])
    checks = {
        "max_abs_logprob_drift": float(receipt.get("max_abs_logprob_drift", math.inf))
        <= float(sampling.get("max_abs_logprob_drift", -math.inf)),
        "mean_abs_logprob_drift": float(receipt.get("mean_abs_logprob_drift", math.inf))
        <= float(sampling.get("max_mean_abs_logprob_drift", -math.inf)),
        "clipped_token_fraction": float(receipt.get("clipped_token_fraction", math.inf))
        <= float(sampling.get("max_clipped_token_fraction", -math.inf)),
        "old_policy_approx_kl": float(receipt.get("old_policy_approx_kl", math.inf))
        <= float(sampling.get("max_old_policy_approx_kl", -math.inf)),
        "params_unchanged": receipt.get("cached_params_unchanged") is True,
        "recurrence_adapter_active": bool(
            activation.get("active") is True
            and int(activation.get("calls", 0) or 0) > 0
            and int(activation.get("adapted_positions", 0) or 0) > 0
        ),
        "nonparametric_memory_disabled": (
            receipt.get("cached_nonparametric_memory_status") == "disabled_by_policy"
        ),
        "no_fallback": not any(str(flag).startswith("fallback_") for flag in honest_flags),
        "behavior_admitted": receipt.get("behavior_admitted") is True,
    }
    return {
        "episode_id": receipt.get("episode_id"),
        "seed": receipt.get("seed"),
        "token_count": receipt.get("token_count"),
        "max_abs_logprob_drift": receipt.get("max_abs_logprob_drift"),
        "mean_abs_logprob_drift": receipt.get("mean_abs_logprob_drift"),
        "clipped_token_fraction": receipt.get("clipped_token_fraction"),
        "old_policy_approx_kl": receipt.get("old_policy_approx_kl"),
        "checks": checks,
        "failed_gates": [name for name, passed in checks.items() if not passed],
        "runtime_integrity": receipt.get("cached_runtime_integrity"),
        "recurrence_adapter": activation,
        "nonparametric_memory_status": receipt.get(
            "cached_nonparametric_memory_status"
        ),
        "honest_flags": honest_flags,
    }


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
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("canary requires a clean source worktree")
    head = _git("rev-parse", "HEAD")
    if head != _git("rev-parse", "origin/main"):
        raise RuntimeError("canary source commit is not published on origin/main")
    bindings: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_PATHS:
        payload = (REPO_ROOT / relative).read_bytes()
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


def _stable_seed(*parts: Any) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _cyclic_task(tasks: list[Any], *, one_based_step: int) -> Any:
    if not tasks or type(one_based_step) is not int or one_based_step < 1:
        raise ValueError("training task cycle coordinates are invalid")
    return tasks[(one_based_step - 1) % len(tasks)]


def _grade_reward(
    task: Any,
    response_text: str,
    *,
    format_credit: float,
) -> tuple[dict[str, Any], float]:
    verdict = dict(task.grade(response_text))
    verdict["correct"] = bool(verdict.get("correct"))
    reward = reward_from_verdict(verdict, format_credit=format_credit)
    return verdict, float(reward)


def _status(out_dir: Path, phase: str, **detail: Any) -> None:
    body = {
        "schema": "aura.recurrent_grpo_behavioral_canary.status.v1",
        "phase": phase,
        "updated_at_unix_ns": time.time_ns(),
        **detail,
    }
    payload = {**body, "status_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    atomic_write_bytes(out_dir / "status.json", canonical_json_bytes(payload), mode=0o600)


def _sample_group(
    model: Any,
    tokenizer: Any,
    task: Any,
    *,
    spec: RLCExecutionSpec,
    group_size: int,
    campaign_seed: int,
    step: int,
    model_path: str,
) -> tuple[list[int], list[Any], list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_tokens, _answer_tokens = tokenize_task(tokenizer, task.prompt, task.answer)
    sampling = free_generation_sampling_config()
    samples: list[Any] = []
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    max_attempts = group_size * 8
    for attempt in range(max_attempts):
        if len(samples) >= group_size:
            break
        sample_seed = _stable_seed(campaign_seed, "grpo", step, task.task_id, attempt)
        try:
            branch_index = attempt % len(spec.branch_roles)
            sample = sample_final_recurrent_transition_completion(
                model,
                prompt_tokens,
                spec=spec,
                branch_index=branch_index,
                seed=sample_seed,
                sampling=sampling,
                tokenizer=tokenizer,
                model_path=model_path,
                episode_id=(f"grpo-canary-{campaign_seed}-{step}-{attempt}"),
            )
        except RecurrentSamplingAdmissionError as exc:
            rejected.append(exc.sample.receipt())
            continue
        response_text = tokenizer.decode(list(sample.tokens))
        verdict, reward = _grade_reward(
            task,
            response_text,
            format_credit=0.1,
        )
        samples.append(sample)
        rows.append(
            {
                "sample_seed": sample_seed,
                "branch_index": sample.branch_index,
                "response_text": response_text,
                "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
                "verdict": verdict,
                "reward": reward,
                "sample_receipt": sample.receipt(),
            }
        )
    if len(samples) != group_size:
        raise RecurrentCanarySamplingError(
            admitted=len(samples),
            requested=group_size,
            rejected=rejected,
        )
    return prompt_tokens, samples, rows, rejected


def run_canary(
    *,
    model_path: Path,
    out_dir: Path,
    steps: int,
    group_size: int,
    seed: int,
    learning_rate: float,
    memory_fraction: float,
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load

    from core.learning.recurrence_curriculum import task_battery

    if type(steps) is not int or not 1 <= steps <= 16:
        raise ValueError("steps must be inside [1, 16]")
    if type(group_size) is not int or not 2 <= group_size <= 8:
        raise ValueError("group_size must be inside [2, 8]")
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be inside [0, 2^63-1]")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or not 1e-7 <= float(learning_rate) <= 1e-3
    ):
        raise ValueError("learning_rate must be inside [1e-7, 1e-3]")

    started = time.time()
    source_commit, source_bindings = _source_state()
    out_dir.mkdir(parents=True, exist_ok=False)
    _status(out_dir, "source_bound", source_commit=source_commit)
    base_before = full_weight_checkpoint_identity(model_path)
    spec = RLCExecutionSpec(
        n_slots=4,
        branch_roles=("constructive_solution", "critical_audit"),
        recurrent_steps=2,
        exchange_interval=1,
    )
    grpo_config = RecurrentGRPOConfig(
        kl_coefficient=0.04,
        advantage_clip=4.0,
    )
    with (
        standalone_model_lane(
            owner_id=f"recurrent-grpo-canary:{out_dir.name}",
            model_path=str(model_path),
            purpose="training",
            preemptible=False,
            metadata={
                "tool": "run_recurrent_grpo_behavioral_canary",
                "source_commit": source_commit,
            },
        ),
        mlx_memory_envelope(
            fraction=memory_fraction,
            restore_limits_on_exit=True,
        ),
    ):
        _status(out_dir, "loading_model")
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
        training_tasks = task_battery(
            ["boolean", "modular"],
            [2],
            4,
            seed=seed,
        )
        proxy_tasks = task_battery(
            ["boolean", "modular"],
            [2],
            2,
            seed=seed + 7_919,
            excluded_prompts=tuple(task.prompt for task in training_tasks),
            excluded_task_ids=tuple(task.task_id for task in training_tasks),
        )
        proxy_manifest, proxy_manifest_sha256 = build_recurrence_task_manifest(proxy_tasks)
        _status(out_dir, "baseline_probe")
        before = build_behavioral_probe_report(
            model,
            tokenizer,
            proxy_tasks,
            spec=spec,
            arm="initial_adapter",
            adapter_sha256=adapter_before,
            task_manifest_sha256=proxy_manifest_sha256,
            seed=seed,
        )
        optimizer = build_recurrent_policy_optimizer(float(learning_rate))
        optimizer.init(model.trainable_parameters())
        step_receipts: list[dict[str, Any]] = []
        optimizer_updates = 0
        all_samples_admitted = True
        for step in range(1, steps + 1):
            task = _cyclic_task(training_tasks, one_based_step=step)
            _status(
                out_dir,
                "sampling",
                step=step,
                steps=steps,
                task_id=task.task_id,
                optimizer_updates=optimizer_updates,
            )
            policy_before = recurrent_policy_sha256(model, spec)
            prompt_tokens, samples, rows, rejected = _sample_group(
                model,
                tokenizer,
                task,
                spec=spec,
                group_size=group_size,
                campaign_seed=seed,
                step=step,
                model_path=str(model_path),
            )
            rewards = [float(row["reward"]) for row in rows]
            advantage = group_advantages(
                rewards,
                clip=grpo_config.advantage_clip,
            )
            adapter_step_before = adapter_tensor_fingerprint(adapter_tensor_dict(model))
            objective: dict[str, Any] | None = None
            if not advantage["degenerate"]:
                _status(
                    out_dir,
                    "optimizing",
                    step=step,
                    steps=steps,
                    task_id=task.task_id,
                    optimizer_updates=optimizer_updates,
                )
                result = exact_adjoint_sampled_group_value_and_grad(
                    model,
                    prompt_tokens,
                    samples,
                    rewards,
                    spec=spec,
                    config=grpo_config,
                )
                optimizer.update(model, result.gradients)
                mx.eval(model.trainable_parameters(), optimizer.state)
                optimizer_updates += 1
                objective = {
                    "advantage_report": result.advantage_report,
                    "reference_kl": result.reference_kl,
                    "old_policy_approx_kl": result.old_policy_approx_kl,
                    "clip_fraction": result.clip_fraction,
                    "policy_loss": result.policy_loss,
                    "objective_at_sampling": result.objective_at_sampling,
                    "gradient_surrogate_value": result.gradient_surrogate_value,
                    "completion_count": result.completion_count,
                    "token_count": result.token_count,
                    "branch_indices": list(result.branch_indices),
                }
                del result
            policy_after = recurrent_policy_sha256(model, spec)
            adapter_step_after = adapter_tensor_fingerprint(adapter_tensor_dict(model))
            if advantage["degenerate"]:
                if policy_after != policy_before or adapter_step_after != adapter_step_before:
                    raise RuntimeError("degenerate recurrent group mutated the policy")
            elif policy_after == policy_before or adapter_step_after == adapter_step_before:
                raise RuntimeError("recurrent optimizer update did not mutate the policy")
            all_samples_admitted = all_samples_admitted and all(
                bool(sample.behavior_admitted) for sample in samples
            )
            step_receipts.append(
                {
                    "step": step,
                    "task_id": task.task_id,
                    "policy_before_sha256": policy_before,
                    "policy_after_sha256": policy_after,
                    "adapter_before_sha256": adapter_step_before,
                    "adapter_after_sha256": adapter_step_after,
                    "advantage_report": advantage,
                    "optimizer_updated": not advantage["degenerate"],
                    "objective": objective,
                    "samples": rows,
                    "rejected_sample_receipts": rejected,
                }
            )
            journal_body = {
                "schema": "aura.recurrent_grpo_behavioral_canary.journal.v1",
                "source_commit": source_commit,
                "execution_spec_sha256": spec.sha256,
                "seed": seed,
                "configured_steps": steps,
                "completed_steps": step,
                "optimizer_updates": optimizer_updates,
                "step_receipts": step_receipts,
            }
            journal = {
                **journal_body,
                "journal_sha256": hashlib.sha256(canonical_json_bytes(journal_body)).hexdigest(),
            }
            atomic_write_bytes(
                out_dir / "step_journal.json",
                canonical_json_bytes(journal),
                mode=0o600,
            )
            print(
                "[recurrent-grpo-canary] "
                f"step={step}/{steps} task={task.task_id} "
                f"mean_reward={advantage['mean_reward']:.3f} "
                f"reward_std={advantage['reward_std']:.3f} "
                f"updates={optimizer_updates}",
                flush=True,
            )
            del samples, rows
            mx.synchronize()
            mx.clear_cache()
        adapter = adapter_tensor_dict(model)
        adapter_after = adapter_tensor_fingerprint(adapter)
        _status(
            out_dir,
            "trained_probe",
            optimizer_updates=optimizer_updates,
        )
        after = build_behavioral_probe_report(
            model,
            tokenizer,
            proxy_tasks,
            spec=spec,
            arm="trained_adapter",
            adapter_sha256=adapter_after,
            task_manifest_sha256=proxy_manifest_sha256,
            seed=seed,
        )
        admission: dict[str, Any] | None = None
        admission_error = ""
        try:
            admission = build_checkpoint_behavioral_admission(
                initial_report=before,
                trained_report=after,
                task_manifest=proxy_manifest,
            )
            validate_checkpoint_behavioral_admission(
                admission,
                initial_report=before,
                trained_report=after,
                task_manifest=proxy_manifest,
            )
        except RecurrentCheckpointAdmissionError as exc:
            admission_error = str(exc)
        mx.save_safetensors(str(out_dir / "adapter.safetensors"), adapter)

    base_after = full_weight_checkpoint_identity(model_path)
    gates = {
        "base_checkpoint_immutable": base_before == base_after,
        "all_samples_admitted": all_samples_admitted,
        "optimizer_signal_observed": optimizer_updates > 0,
        "adapter_mutated": adapter_before != adapter_after,
        "heldout_free_generation_strict_gain": bool(
            admission is not None and admission["admitted"]
        ),
    }
    body = {
        "schema": CANARY_SCHEMA,
        "source_commit": source_commit,
        "source_bindings": source_bindings,
        "model_path": str(model_path),
        "base_before": base_before,
        "base_after": base_after,
        "execution_spec": spec.to_dict(),
        "execution_spec_sha256": spec.sha256,
        "grpo_config": {
            "clip_epsilon": grpo_config.clip_epsilon,
            "kl_coefficient": grpo_config.kl_coefficient,
            "advantage_clip": grpo_config.advantage_clip,
            "group_size": group_size,
            "learning_rate": float(learning_rate),
            "sampling": free_generation_sampling_config().to_dict(),
            "reward": "exact_correctness_plus_at_most_0.1_parseability_credit",
        },
        "seed": seed,
        "steps": steps,
        "optimizer_updates": optimizer_updates,
        "adapter_before_sha256": adapter_before,
        "adapter_after_sha256": adapter_after,
        "training_task_ids": [task.task_id for task in training_tasks],
        "proxy_task_manifest": proxy_manifest,
        "proxy_task_manifest_sha256": proxy_manifest_sha256,
        "free_generation_before": before,
        "step_receipts": step_receipts,
        "free_generation_after": after,
        "checkpoint_behavioral_admission": admission,
        "checkpoint_behavioral_admission_error": admission_error,
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
        "receipt_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }
    atomic_write_bytes(
        out_dir / "receipt.json",
        canonical_json_bytes(receipt),
        mode=0o600,
    )
    _status(
        out_dir,
        "completed",
        passed=receipt["passed"],
        optimizer_updates=optimizer_updates,
        receipt_sha256=receipt["receipt_sha256"],
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026080701)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--memory-fraction", type=float, default=0.35)
    args = parser.parse_args()
    out_dir = args.out_dir.expanduser().resolve(strict=False)
    try:
        receipt = run_canary(
            model_path=args.model.expanduser().resolve(strict=True),
            out_dir=out_dir,
            steps=args.steps,
            group_size=args.group_size,
            seed=args.seed,
            learning_rate=args.learning_rate,
            memory_fraction=args.memory_fraction,
        )
    except Exception as exc:
        if out_dir.is_dir():
            failure = {
                "schema": "aura.recurrent_grpo_behavioral_canary.failure.v1",
                "error_type": type(exc).__name__,
                "error": str(exc)[:4_000],
                "recorded_at_unix_ns": time.time_ns(),
            }
            if isinstance(exc, RecurrentCanarySamplingError):
                failure["sampling_failure"] = {
                    "admitted": exc.admitted,
                    "requested": exc.requested,
                    "rejected": len(exc.rejected),
                    "diagnostics": exc.diagnostics,
                    "rejected_sample_receipts": exc.rejected,
                }
            atomic_write_bytes(
                out_dir / "failure.json",
                canonical_json_bytes(failure),
                mode=0o600,
            )
        raise
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "passed": receipt["passed"],
                "optimizer_updates": receipt["optimizer_updates"],
                "gates": receipt["gates"],
                "elapsed_s": receipt["elapsed_s"],
                "receipt_sha256": receipt["receipt_sha256"],
                "receipt_path": str(out_dir / "receipt.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
