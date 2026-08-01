"""Shared producer/verifier contract for recurrent-GRPO artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, Never

from core.learning.grpo import GRPO_SCHEMA, group_advantages

PROTOCOL_SCHEMA_V5: Final = "aura.grpo_protocol.v5"
PROTOCOL_SCHEMA_V6: Final = "aura.grpo_protocol.v6"
PROTOCOL_SCHEMA_V7: Final = "aura.grpo_protocol.v7"
PROTOCOL_SCHEMA_V8: Final = "aura.grpo_protocol.v8"
PROTOCOL_SCHEMA: Final = "aura.grpo_protocol.v9"
TRAINING_RECEIPT_SCHEMA: Final = "aura.grpo_training.v5"
STEP_RECEIPT_SCHEMA: Final = "aura.recurrent_grpo_step.v1"
TRAINING_ADEQUACY_SCHEMA: Final = "aura.recurrent_grpo.training_adequacy.v1"
TRAINING_ADEQUACY_MIN_UPDATE_FRACTION: Final = 0.25
TRAINING_ADEQUACY_MIN_UPDATES_PER_WINDOW: Final = 1

PROTOCOL_TRAINING_KEYS_V5: Final = frozenset(
    {
        "execution_mode",
        "execution_spec",
        "execution_spec_sha256",
        "verified_transition_provider_contract_sha256",
        "domains",
        "depths",
        "train_per_cell",
        "holdout_per_cell",
        "group_size",
        "temperature",
        "max_tokens",
        "kl_coefficient",
        "format_credit",
        "trajectory_credit",
        "trajectory_shaping_weight",
        "lora_rank",
        "lora_targets",
        "lora_layers",
        "lora_initialization_seed",
        "learning_rate",
        "max_steps",
        "eval_every",
        "checkpoint_every",
        "min_signal_groups",
        "calibrate",
        "calibrate_samples",
        "calibrate_group",
        "calibrate_tokens",
        "calibrate_minutes",
        "cot",
        "seed",
        "memory_fraction",
        "rng_strategy",
    }
)
PROTOCOL_TRAINING_KEYS_V8: Final = PROTOCOL_TRAINING_KEYS_V5 | {
    "advantage_clip",
    "verified_trajectory_config",
    "verified_trajectory_config_sha256",
    "max_invocation_steps",
}
PROTOCOL_TRAINING_KEYS: Final = PROTOCOL_TRAINING_KEYS_V8 | {
    "warm_start_contract_sha256",
}

STEP_RECEIPT_KEYS: Final = frozenset(
    {
        "schema",
        "step",
        "task_id",
        "sample_seed",
        "execution_spec_sha256",
        "samples",
        "rewards",
        "verifier_rewards",
        "answer_channel",
        "verifier_advantage_report",
        "trajectory_credit",
        "advantage_report",
        "step_kind",
        "update",
        "policy_after_sha256",
    }
)

_ANSWER_CHANNEL_KEYS: Final = frozenset(
    {
        "completions",
        "parseable",
        "unparseable",
        "correct",
        "parseable_fraction",
        "correct_fraction",
        "grade_reasons",
    }
)
_ADVANTAGE_KEYS: Final = frozenset(
    {
        "schema",
        "advantages",
        "mean_reward",
        "reward_std",
        "degenerate",
        "all_correct",
        "all_wrong",
        "uniform_partial",
    }
)
_SHAPED_ADVANTAGE_KEYS: Final = _ADVANTAGE_KEYS | {
    "shaped_mean_reward",
    "shaped_reward_std",
    "shaped_degenerate",
    "shaped_all_correct",
    "shaped_all_wrong",
    "shaped_uniform_partial",
    "verifier_reward_std",
    "verifier_degenerate",
    "trajectory_shaped",
}
_TRAJECTORY_KEYS: Final = frozenset(
    {
        "schema",
        "shaped_rewards",
        "rows",
        "shaping_weight",
        "shaping_reordered",
        "ce_trails",
        "score_trails",
    }
)
_TRAJECTORY_ROW_KEYS: Final = frozenset({"final_reward", "shaping", "shaped_reward", "steps"})
_FLOAT_TOLERANCE: Final = 1e-9
_ROUNDED_TOLERANCE: Final = 1.1e-6


class RecurrentGRPOArtifactSchemaError(ValueError):
    """Stable artifact-contract failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise RecurrentGRPOArtifactSchemaError(code)


def recurrent_training_adequacy_policy() -> dict[str, Any]:
    """Return the source-bound minimum dose required for adapter publication."""

    return {
        "schema": TRAINING_ADEQUACY_SCHEMA,
        "schedule": "one_complete_unique_task_pass",
        "minimum_optimizer_update_fraction": TRAINING_ADEQUACY_MIN_UPDATE_FRACTION,
        "minimum_optimizer_updates_per_evaluation_window": (
            TRAINING_ADEQUACY_MIN_UPDATES_PER_WINDOW
        ),
        "evaluation_schedule": "every_eval_interval_plus_terminal_step",
        "policy_transition_requirement": "distinct_digest_per_optimizer_update",
        "learning_signal_required": True,
    }


def recurrent_training_adequacy_report(
    *,
    step_receipts: Sequence[Mapping[str, Any]],
    scheduled_task_ids: Sequence[str],
    max_steps: int,
    eval_every: int,
    evaluation_steps: Sequence[int],
    learning_signal: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute whether one proof-grade recurrent training dose was adequate."""

    if (
        type(max_steps) is not int
        or max_steps <= 0
        or type(eval_every) is not int
        or eval_every <= 0
        or any(not isinstance(task_id, str) or not task_id for task_id in scheduled_task_ids)
        or any(type(step) is not int or step <= 0 for step in evaluation_steps)
    ):
        _fail("training_adequacy_input_invalid")
    expected_tasks = list(scheduled_task_ids)
    observed_tasks = [receipt.get("task_id") for receipt in step_receipts]
    observed_steps = [receipt.get("step") for receipt in step_receipts]
    update_receipts = [
        receipt
        for receipt in step_receipts
        if receipt.get("step_kind") in {"optimizer_update", "verified_optimizer_update"}
    ]
    update_steps = [receipt.get("step") for receipt in update_receipts]
    update_policies = [receipt.get("policy_after_sha256") for receipt in update_receipts]
    expected_evaluations = list(range(eval_every, max_steps + 1, eval_every))
    if not expected_evaluations or expected_evaluations[-1] != max_steps:
        expected_evaluations.append(max_steps)
    observed_evaluations = sorted(set(evaluation_steps))
    minimum_updates = max(
        1,
        math.ceil(max_steps * TRAINING_ADEQUACY_MIN_UPDATE_FRACTION),
    )
    window_counts: list[dict[str, int]] = []
    window_start = 1
    for window_end in expected_evaluations:
        window_counts.append(
            {
                "start_step": window_start,
                "end_step": window_end,
                "optimizer_updates": sum(
                    type(step) is int and window_start <= step <= window_end
                    for step in update_steps
                ),
            }
        )
        window_start = window_end + 1

    checks = {
        "one_complete_pass": (
            len(step_receipts) == max_steps
            and len(expected_tasks) == max_steps
            and observed_steps == list(range(1, max_steps + 1))
        ),
        "exact_task_schedule": observed_tasks == expected_tasks,
        "unique_task_coverage": (
            len(expected_tasks) == len(set(expected_tasks))
            and len(observed_tasks) == len(set(observed_tasks))
        ),
        "minimum_optimizer_updates": len(update_receipts) >= minimum_updates,
        "distributed_update_activity": all(
            window["optimizer_updates"]
            >= TRAINING_ADEQUACY_MIN_UPDATES_PER_WINDOW
            for window in window_counts
        ),
        "distinct_policy_transitions": (
            len(update_policies) == len(set(update_policies))
            and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in update_policies)
        ),
        "evaluation_schedule_complete": observed_evaluations == expected_evaluations,
        "learning_signal": learning_signal.get("learning_signal") is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "schema": TRAINING_ADEQUACY_SCHEMA,
        "policy": recurrent_training_adequacy_policy(),
        "max_steps": max_steps,
        "scheduled_tasks": len(expected_tasks),
        "optimizer_updates": len(update_receipts),
        "minimum_optimizer_updates": minimum_updates,
        "optimizer_update_fraction": round(len(update_receipts) / max_steps, 6),
        "evaluation_steps": observed_evaluations,
        "expected_evaluation_steps": expected_evaluations,
        "update_windows": window_counts,
        "checks": checks,
        "failed_checks": failed,
        "admitted": not failed,
    }


def protocol_semantic_sha256(value: Any) -> str:
    """Hash the exact canonical JSON bytes persisted by the trainer."""

    try:
        payload = (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
        raise RecurrentGRPOArtifactSchemaError("protocol_semantic_json_invalid") from exc
    return hashlib.sha256(payload).hexdigest()


def _finite(value: Any, *, role: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(f"{role}_invalid")
    return float(value)


def _finite_vector(
    value: Any,
    *,
    role: str,
    length: int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        _fail(f"{role}_invalid")
    result = [_finite(item, role=role) for item in value]
    if minimum is not None and any(item < minimum for item in result):
        _fail(f"{role}_invalid")
    if maximum is not None and any(item > maximum for item in result):
        _fail(f"{role}_invalid")
    return result


def _close(left: float, right: float, *, tolerance: float = _FLOAT_TOLERANCE) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _expected_shaped_advantage(
    effective_rewards: Sequence[float],
    verifier_report: Mapping[str, Any],
    *,
    clip: float,
) -> dict[str, Any]:
    report = group_advantages(effective_rewards, clip=clip)
    expected = dict(report)
    expected["shaped_mean_reward"] = report["mean_reward"]
    expected["shaped_reward_std"] = report["reward_std"]
    expected["shaped_degenerate"] = report["degenerate"]
    expected["shaped_all_correct"] = report["all_correct"]
    expected["shaped_all_wrong"] = report["all_wrong"]
    expected["shaped_uniform_partial"] = report["uniform_partial"]
    expected["mean_reward"] = verifier_report["mean_reward"]
    expected["verifier_reward_std"] = verifier_report["reward_std"]
    expected["verifier_degenerate"] = verifier_report["degenerate"]
    if expected["degenerate"]:
        expected["all_correct"] = verifier_report["all_correct"]
        expected["all_wrong"] = verifier_report["all_wrong"]
        expected["uniform_partial"] = verifier_report["uniform_partial"]
    expected["trajectory_shaped"] = True
    return expected


def _validate_answer_channel(value: Any, *, group_size: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ANSWER_CHANNEL_KEYS:
        _fail("step_answer_channel_schema_invalid")
    counts: dict[str, int] = {}
    for key in ("completions", "parseable", "unparseable", "correct"):
        item = value.get(key)
        if type(item) is not int or item < 0:
            _fail("step_answer_channel_count_invalid")
        counts[key] = item
    if (
        counts["completions"] != group_size
        or counts["parseable"] + counts["unparseable"] != group_size
        or counts["correct"] > counts["parseable"]
    ):
        _fail("step_answer_channel_count_mismatch")
    expected_parseable = round(counts["parseable"] / group_size, 4)
    expected_correct = round(counts["correct"] / group_size, 4)
    if not _close(
        _finite(value.get("parseable_fraction"), role="step_parseable_fraction"),
        expected_parseable,
    ) or not _close(
        _finite(value.get("correct_fraction"), role="step_correct_fraction"),
        expected_correct,
    ):
        _fail("step_answer_channel_fraction_mismatch")
    reasons = value.get("grade_reasons")
    if (
        not isinstance(reasons, Mapping)
        or any(
            not isinstance(reason, str) or not reason or type(count) is not int or count <= 0
            for reason, count in reasons.items()
        )
        or sum(reasons.values()) != group_size
    ):
        _fail("step_answer_channel_reasons_invalid")
    return dict(value)


def _validate_trajectory_credit(
    value: Any,
    *,
    verifier_rewards: Sequence[float],
    effective_rewards: Sequence[float],
    shaping_weight: float,
) -> dict[str, Any]:
    group_size = len(verifier_rewards)
    if not isinstance(value, Mapping) or set(value) != _TRAJECTORY_KEYS:
        _fail("step_trajectory_credit_schema_invalid")
    if (
        value.get("schema") != GRPO_SCHEMA
        or not _close(
            _finite(value.get("shaping_weight"), role="step_trajectory_weight"),
            shaping_weight,
        )
        or type(value.get("shaping_reordered")) is not bool
    ):
        _fail("step_trajectory_credit_identity_invalid")
    shaped = _finite_vector(
        value.get("shaped_rewards"),
        role="step_trajectory_shaped_rewards",
        length=group_size,
        minimum=-shaping_weight,
        maximum=1.0 + shaping_weight,
    )
    if any(
        not _close(observed, expected)
        for observed, expected in zip(shaped, effective_rewards, strict=True)
    ):
        _fail("step_trajectory_effective_reward_mismatch")
    rows = value.get("rows")
    ce_trails = value.get("ce_trails")
    score_trails = value.get("score_trails")
    if (
        not isinstance(rows, list)
        or len(rows) != group_size
        or not isinstance(ce_trails, list)
        or len(ce_trails) != group_size
        or not isinstance(score_trails, list)
        or len(score_trails) != group_size
    ):
        _fail("step_trajectory_group_invalid")
    for index, (row, ce_raw, score_raw) in enumerate(
        zip(rows, ce_trails, score_trails, strict=True)
    ):
        if not isinstance(row, Mapping) or set(row) != _TRAJECTORY_ROW_KEYS:
            _fail("step_trajectory_row_schema_invalid")
        if (
            not isinstance(ce_raw, list)
            or not ce_raw
            or not isinstance(score_raw, list)
            or len(score_raw) != len(ce_raw)
        ):
            _fail("step_trajectory_trail_invalid")
        ce = [_finite(item, role="step_trajectory_ce") for item in ce_raw]
        scores = [_finite(item, role="step_trajectory_score") for item in score_raw]
        if any(item < 0.0 for item in ce) or any(not 0.0 <= item <= 1.0 for item in scores):
            _fail("step_trajectory_trail_invalid")
        if any(
            not _close(
                score,
                round(math.exp(-loss), 6),
                tolerance=_ROUNDED_TOLERANCE,
            )
            for loss, score in zip(ce, scores, strict=True)
        ):
            _fail("step_trajectory_score_replay_mismatch")
        verifier = verifier_rewards[index]
        effective = effective_rewards[index]
        if (
            row.get("steps") != len(scores)
            or not _close(
                _finite(row.get("final_reward"), role="step_trajectory_final"),
                round(verifier, 6),
                tolerance=_ROUNDED_TOLERANCE,
            )
            or not _close(
                _finite(row.get("shaping"), role="step_trajectory_shaping"),
                round(effective - verifier, 6),
                tolerance=_ROUNDED_TOLERANCE,
            )
            or not _close(
                _finite(row.get("shaped_reward"), role="step_trajectory_shaped"),
                round(effective, 6),
                tolerance=_ROUNDED_TOLERANCE,
            )
            or abs(effective - verifier) > shaping_weight + _ROUNDED_TOLERANCE
        ):
            _fail("step_trajectory_row_mismatch")
    order_by_final = sorted(range(group_size), key=lambda index: (verifier_rewards[index], index))
    order_by_shaped = sorted(range(group_size), key=lambda index: (effective_rewards[index], index))
    if value["shaping_reordered"] is not (order_by_final != order_by_shaped):
        _fail("step_trajectory_order_mismatch")
    return dict(value)


def validate_step_reward_channels(
    step: Mapping[str, Any],
    *,
    group_size: int,
    trajectory_credit_enabled: bool,
    shaping_weight: float,
    advantage_clip: float,
) -> dict[str, Any]:
    """Replay all reward channels and reject producer/verifier drift."""

    if not isinstance(step, Mapping) or set(step) != STEP_RECEIPT_KEYS:
        _fail("step_receipt_schema_invalid")
    if step.get("schema") != STEP_RECEIPT_SCHEMA:
        _fail("step_receipt_schema_unsupported")
    if type(trajectory_credit_enabled) is not bool:
        _fail("step_trajectory_policy_invalid")
    shaping_weight = _finite(shaping_weight, role="step_trajectory_weight")
    advantage_clip = _finite(advantage_clip, role="step_advantage_clip")
    if not 0.0 <= shaping_weight <= 0.49 or advantage_clip <= 0.0:
        _fail("step_reward_policy_invalid")

    verifier = _finite_vector(
        step.get("verifier_rewards"),
        role="step_verifier_rewards",
        length=group_size,
        minimum=0.0,
        maximum=1.0,
    )
    effective = _finite_vector(
        step.get("rewards"),
        role="step_effective_rewards",
        length=group_size,
        minimum=-shaping_weight,
        maximum=1.0 + shaping_weight,
    )
    _validate_answer_channel(step.get("answer_channel"), group_size=group_size)

    expected_verifier_report = group_advantages(verifier, clip=advantage_clip)
    verifier_report = step.get("verifier_advantage_report")
    if (
        not isinstance(verifier_report, Mapping)
        or set(verifier_report) != _ADVANTAGE_KEYS
        or dict(verifier_report) != expected_verifier_report
    ):
        _fail("step_verifier_advantage_replay_mismatch")

    trajectory = step.get("trajectory_credit")
    if trajectory is None:
        if any(
            not _close(observed, expected)
            for observed, expected in zip(effective, verifier, strict=True)
        ):
            _fail("step_unreceipted_reward_shaping")
        expected_advantage = expected_verifier_report
        expected_advantage_keys = _ADVANTAGE_KEYS
    else:
        if not trajectory_credit_enabled or expected_verifier_report["degenerate"] is not True:
            _fail("step_trajectory_credit_not_authorized")
        _validate_trajectory_credit(
            trajectory,
            verifier_rewards=verifier,
            effective_rewards=effective,
            shaping_weight=shaping_weight,
        )
        expected_advantage = _expected_shaped_advantage(
            effective,
            expected_verifier_report,
            clip=advantage_clip,
        )
        expected_advantage_keys = _SHAPED_ADVANTAGE_KEYS
    advantage = step.get("advantage_report")
    if (
        not isinstance(advantage, Mapping)
        or set(advantage) != expected_advantage_keys
        or dict(advantage) != expected_advantage
    ):
        _fail("step_effective_advantage_replay_mismatch")
    expected_kind = "degenerate_group" if expected_advantage["degenerate"] else "optimizer_update"
    if step.get("step_kind") != expected_kind:
        _fail("step_kind_reward_mismatch")
    return {
        "verifier_rewards": verifier,
        "effective_rewards": effective,
        "trajectory_credit_applied": trajectory is not None,
        "verifier_advantage_report": expected_verifier_report,
        "advantage_report": expected_advantage,
        "step_kind": expected_kind,
    }


__all__ = [
    "PROTOCOL_SCHEMA",
    "PROTOCOL_SCHEMA_V5",
    "PROTOCOL_SCHEMA_V6",
    "PROTOCOL_SCHEMA_V7",
    "PROTOCOL_SCHEMA_V8",
    "PROTOCOL_TRAINING_KEYS",
    "PROTOCOL_TRAINING_KEYS_V5",
    "PROTOCOL_TRAINING_KEYS_V8",
    "STEP_RECEIPT_KEYS",
    "STEP_RECEIPT_SCHEMA",
    "TRAINING_RECEIPT_SCHEMA",
    "RecurrentGRPOArtifactSchemaError",
    "validate_step_reward_channels",
]
