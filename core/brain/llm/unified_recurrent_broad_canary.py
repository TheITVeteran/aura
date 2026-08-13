"""Model-free contracts for broad unified-recurrence transfer canaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from core.brain.llm.latent_cortex.frontier_tasks import FRONTIER_DOMAINS

PLAN_SCHEMA: Final = "aura.unified_intrinsic.broad_canary_plan.v1"
RESULT_SCHEMA: Final = "aura.unified_intrinsic.broad_canary_result.v1"
VERDICT_SCHEMA: Final = "aura.unified_intrinsic.broad_canary_verdict.v1"
ARMS: Final = (
    "base_greedy",
    "initial_t4",
    "trained_t1",
    "trained_t4",
)
SUPPORTED: Final = "broad_general_channel_canary_supported"
REFUTED: Final = "broad_general_channel_canary_refuted"
CLAIM_BOUNDARY: Final = (
    "A supported canary is exploratory evidence that the frozen trained recurrent "
    "controller improves fresh broad-task exact correctness over initialization-"
    "matched recurrence and base greedy decoding without observed regression. It "
    "is not powered replication, ordinary-chat authority, frontier performance, "
    "static fusion, or a WOW Signal."
)


class UnifiedRecurrentBroadCanaryError(ValueError):
    """A broad canary plan or result violates its frozen contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def seal_broad_canary_plan(
    *,
    package_id: str,
    manifest_sha256: str,
    controller_sha256: str,
    model_manifest_sha256: str,
    recurrence_depth: int,
    max_tokens: int,
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze one answer-blind broad canary before model execution."""

    rows = [dict(task) for task in tasks]
    body = {
        "schema": PLAN_SCHEMA,
        "package_id": package_id,
        "manifest_sha256": manifest_sha256,
        "controller_sha256": controller_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "recurrence_depth": recurrence_depth,
        "max_tokens": max_tokens,
        "arms": list(ARMS),
        "domains": list(FRONTIER_DOMAINS),
        "tasks": rows,
        "typed_state_slots_enabled": False,
        "terminal_grammar_enabled": False,
        "answer_digit_pointer_enabled": False,
        "serving_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    result = {**body, "plan_sha256": _sha(body)}
    errors = broad_canary_plan_errors(result)
    if errors:
        raise UnifiedRecurrentBroadCanaryError(",".join(errors))
    return result


def broad_canary_plan_errors(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["broad_canary_plan_not_mapping"]
    expected = {
        "schema",
        "package_id",
        "manifest_sha256",
        "controller_sha256",
        "model_manifest_sha256",
        "recurrence_depth",
        "max_tokens",
        "arms",
        "domains",
        "tasks",
        "typed_state_slots_enabled",
        "terminal_grammar_enabled",
        "answer_digit_pointer_enabled",
        "serving_authority",
        "claim_boundary",
        "plan_sha256",
    }
    if set(value) != expected:
        return ["broad_canary_plan_fields_differ"]
    body = {key: item for key, item in value.items() if key != "plan_sha256"}
    errors: list[str] = []
    if value.get("schema") != PLAN_SCHEMA or value.get("plan_sha256") != _sha(body):
        errors.append("broad_canary_plan_identity_differs")
    if (
        not isinstance(value.get("package_id"), str)
        or not value["package_id"]
        or not all(
            _is_sha(value.get(name))
            for name in (
                "manifest_sha256",
                "controller_sha256",
                "model_manifest_sha256",
            )
        )
    ):
        errors.append("broad_canary_package_identity_invalid")
    if (
        type(value.get("recurrence_depth")) is not int
        or value["recurrence_depth"] < 2
        or type(value.get("max_tokens")) is not int
        or not 16 <= value["max_tokens"] <= 2048
    ):
        errors.append("broad_canary_compute_contract_invalid")
    if value.get("arms") != list(ARMS) or value.get("domains") != list(FRONTIER_DOMAINS):
        errors.append("broad_canary_arm_or_domain_contract_differs")
    tasks = value.get("tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) < len(FRONTIER_DOMAINS)
        or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "task_id",
                "domain",
                "task_payload_sha256",
                "answer_commitment_sha256",
                "prompt_sha256",
            }
            or not isinstance(row.get("task_id"), str)
            or row.get("domain") not in FRONTIER_DOMAINS
            or not all(
                _is_sha(row.get(name))
                for name in (
                    "task_payload_sha256",
                    "answer_commitment_sha256",
                    "prompt_sha256",
                )
            )
            for row in tasks
        )
        or len({row["task_id"] for row in tasks if isinstance(row, Mapping)})
        != len(tasks)
        or {row["domain"] for row in tasks if isinstance(row, Mapping)}
        != set(FRONTIER_DOMAINS)
    ):
        errors.append("broad_canary_task_contract_invalid")
    if any(
        value.get(name) is not False
        for name in (
            "typed_state_slots_enabled",
            "terminal_grammar_enabled",
            "answer_digit_pointer_enabled",
            "serving_authority",
        )
    ) or value.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("broad_canary_authority_or_claim_differs")
    return errors


def seal_broad_canary_result(
    plan: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Adjudicate complete answer-hidden candidate rows against a frozen plan."""

    errors = broad_canary_plan_errors(plan)
    if errors:
        raise UnifiedRecurrentBroadCanaryError(",".join(errors))
    task_rows = {str(row["task_id"]): row for row in plan["tasks"]}
    observed: dict[tuple[str, str], Mapping[str, Any]] = {}
    candidate_errors: list[str] = []
    for row in candidates:
        if not isinstance(row, Mapping) or set(row) != {
            "task_id",
            "domain",
            "arm",
            "correct",
            "parsed",
            "response_sha256",
            "generated_tokens",
            "stopped",
            "latency_ms",
        }:
            candidate_errors.append("broad_canary_candidate_fields_differ")
            continue
        task_id = row.get("task_id")
        arm = row.get("arm")
        key = (str(task_id), str(arm))
        if (
            task_id not in task_rows
            or arm not in ARMS
            or row.get("domain") != task_rows.get(task_id, {}).get("domain")
            or type(row.get("correct")) is not bool
            or type(row.get("parsed")) is not bool
            or not _is_sha(row.get("response_sha256"))
            or type(row.get("generated_tokens")) is not int
            or not 0 <= row["generated_tokens"] <= plan["max_tokens"]
            or type(row.get("stopped")) is not bool
            or type(row.get("latency_ms")) is not int
            or row["latency_ms"] < 0
            or key in observed
        ):
            candidate_errors.append("broad_canary_candidate_invalid")
            continue
        observed[key] = row
    expected = {(task_id, arm) for task_id in task_rows for arm in ARMS}
    complete = not candidate_errors and set(observed) == expected

    counts = {
        arm: sum(
            observed.get((task_id, arm), {}).get("correct") is True
            for task_id in task_rows
        )
        for arm in ARMS
    }

    def transitions(left: str, right: str) -> tuple[int, int]:
        gains = regressions = 0
        for task_id in task_rows:
            before = observed.get((task_id, left), {}).get("correct") is True
            after = observed.get((task_id, right), {}).get("correct") is True
            gains += not before and after
            regressions += before and not after
        return gains, regressions

    initial_gain, initial_regression = transitions("initial_t4", "trained_t4")
    base_gain, base_regression = transitions("base_greedy", "trained_t4")
    depth_gain, depth_regression = transitions("trained_t1", "trained_t4")
    task_count = len(task_rows)
    checks = {
        "complete_candidate_matrix": complete,
        "base_is_not_floor_or_ceiling": 0 < counts["base_greedy"] < task_count,
        "trained_beats_initial_matched_control": (
            counts["trained_t4"] > counts["initial_t4"] and initial_gain > 0
        ),
        "trained_has_zero_initial_control_regressions": initial_regression == 0,
        "trained_preserves_base_successes": base_regression == 0,
        "trained_depth_four_beats_depth_one": (
            counts["trained_t4"] > counts["trained_t1"]
            and depth_gain > 0
            and depth_regression == 0
        ),
    }
    supported = all(checks.values())
    body = {
        "schema": RESULT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "package_id": plan["package_id"],
        "controller_sha256": plan["controller_sha256"],
        "candidate_count": len(observed),
        "task_count": task_count,
        "counts": counts,
        "transitions": {
            "initial_to_trained": {
                "wrong_to_right": initial_gain,
                "right_to_wrong": initial_regression,
            },
            "base_to_trained": {
                "wrong_to_right": base_gain,
                "right_to_wrong": base_regression,
            },
            "trained_t1_to_t4": {
                "wrong_to_right": depth_gain,
                "right_to_wrong": depth_regression,
            },
        },
        "checks": checks,
        "candidate_errors": sorted(candidate_errors),
        "supported": supported,
        "verdict": SUPPORTED if supported else REFUTED,
        "serving_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    verdict = {**body, "result_sha256": _sha(body)}
    if broad_canary_result_errors(verdict, plan=plan):
        raise UnifiedRecurrentBroadCanaryError(
            ",".join(broad_canary_result_errors(verdict, plan=plan))
        )
    return verdict


def broad_canary_result_errors(
    value: Any,
    *,
    plan: Mapping[str, Any],
) -> list[str]:
    if not isinstance(value, Mapping):
        return ["broad_canary_result_not_mapping"]
    body = {key: item for key, item in value.items() if key != "result_sha256"}
    if (
        value.get("schema") != RESULT_SCHEMA
        or value.get("plan_sha256") != plan.get("plan_sha256")
        or value.get("package_id") != plan.get("package_id")
        or value.get("controller_sha256") != plan.get("controller_sha256")
        or value.get("result_sha256") != _sha(body)
        or value.get("supported") is not all(
            item is True for item in value.get("checks", {}).values()
        )
        or value.get("verdict")
        != (SUPPORTED if value.get("supported") is True else REFUTED)
        or value.get("serving_authority") is not False
        or value.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        return ["broad_canary_result_identity_or_verdict_differs"]
    return []


__all__ = [
    "ARMS",
    "CLAIM_BOUNDARY",
    "PLAN_SCHEMA",
    "REFUTED",
    "RESULT_SCHEMA",
    "SUPPORTED",
    "UnifiedRecurrentBroadCanaryError",
    "broad_canary_plan_errors",
    "broad_canary_result_errors",
    "seal_broad_canary_plan",
    "seal_broad_canary_result",
]
