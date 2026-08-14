"""Model-free contract for parent-to-treatment recurrent transfer canaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from core.brain.llm.latent_cortex.frontier_tasks import FRONTIER_DOMAINS

PLAN_SCHEMA: Final = "aura.unified_intrinsic.transfer_canary_plan.v1"
RESULT_SCHEMA: Final = "aura.unified_intrinsic.transfer_canary_result.v1"
ARMS: Final = (
    "base_greedy",
    "parent_typed",
    "treatment_typed",
    "action_lesion",
)
SUPPORTED: Final = "broad_process_transfer_canary_supported"
REFUTED: Final = "broad_process_transfer_canary_refuted"
INCONCLUSIVE: Final = "broad_process_transfer_canary_inconclusive"
CLAIM_BOUNDARY: Final = (
    "A supported result is exploratory task-disjoint evidence that broad-process "
    "adaptation improves exact correctness over the frozen imported parent and "
    "base model, and that removing the typed action channel removes at least one "
    "gain. It is not powered replication, resident-32B evidence, serving "
    "authority, static fusion, frontier performance, or a WOW Signal."
)


class UnifiedRecurrentTransferCanaryError(ValueError):
    """A transfer canary violates its frozen evidence contract."""


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


def _is_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def seal_transfer_canary_plan(
    *,
    campaign_identity_sha256: str,
    parent_checkpoint_sha256: str,
    parent_controller_sha256: str,
    treatment_checkpoint_sha256: str,
    treatment_controller_sha256: str,
    recurrence_depth: int,
    max_tokens: int,
    tasks: Sequence[Mapping[str, Any]],
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": PLAN_SCHEMA,
        "campaign_identity_sha256": campaign_identity_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_controller_sha256": parent_controller_sha256,
        "treatment_checkpoint_sha256": treatment_checkpoint_sha256,
        "treatment_controller_sha256": treatment_controller_sha256,
        "recurrence_depth": recurrence_depth,
        "max_tokens": max_tokens,
        "arms": list(ARMS),
        "domains": list(FRONTIER_DOMAINS),
        "tasks": [dict(task) for task in tasks],
        "source_binding": dict(source_binding),
        "typed_state_slots_enabled": True,
        "terminal_grammar_enabled": False,
        "answer_digit_pointer_enabled": False,
        "runtime_teacher_available": False,
        "serving_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    plan = {**body, "plan_sha256": _sha(body)}
    errors = transfer_canary_plan_errors(plan)
    if errors:
        raise UnifiedRecurrentTransferCanaryError(",".join(errors))
    return plan


def transfer_canary_plan_errors(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["transfer_canary_plan_not_mapping"]
    expected = {
        "schema",
        "campaign_identity_sha256",
        "parent_checkpoint_sha256",
        "parent_controller_sha256",
        "treatment_checkpoint_sha256",
        "treatment_controller_sha256",
        "recurrence_depth",
        "max_tokens",
        "arms",
        "domains",
        "tasks",
        "source_binding",
        "typed_state_slots_enabled",
        "terminal_grammar_enabled",
        "answer_digit_pointer_enabled",
        "runtime_teacher_available",
        "serving_authority",
        "claim_boundary",
        "plan_sha256",
    }
    if set(value) != expected:
        return ["transfer_canary_plan_fields_differ"]
    body = {key: item for key, item in value.items() if key != "plan_sha256"}
    errors: list[str] = []
    identity_fields = (
        "campaign_identity_sha256",
        "parent_checkpoint_sha256",
        "parent_controller_sha256",
        "treatment_checkpoint_sha256",
        "treatment_controller_sha256",
    )
    if (
        value.get("schema") != PLAN_SCHEMA
        or value.get("plan_sha256") != _sha(body)
        or any(not _is_sha(value.get(name)) for name in identity_fields)
        or value.get("parent_controller_sha256")
        == value.get("treatment_controller_sha256")
    ):
        errors.append("transfer_canary_identity_invalid")
    if (
        type(value.get("recurrence_depth")) is not int
        or value["recurrence_depth"] < 2
        or type(value.get("max_tokens")) is not int
        or not 16 <= value["max_tokens"] <= 2048
        or value.get("arms") != list(ARMS)
        or value.get("domains") != list(FRONTIER_DOMAINS)
    ):
        errors.append("transfer_canary_compute_or_arm_contract_invalid")
    tasks = value.get("tasks")
    task_fields = {
        "task_id",
        "domain",
        "task_payload_sha256",
        "answer_commitment_sha256",
        "prompt_sha256",
    }
    if (
        not isinstance(tasks, list)
        or len(tasks) < len(FRONTIER_DOMAINS)
        or any(
            not isinstance(row, Mapping)
            or set(row) != task_fields
            or not isinstance(row.get("task_id"), str)
            or row.get("domain") not in FRONTIER_DOMAINS
            or any(
                not _is_sha(row.get(name))
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
        errors.append("transfer_canary_task_contract_invalid")
    source = value.get("source_binding")
    if (
        not isinstance(source, Mapping)
        or set(source) != {"git_commit", "implementation_sha256s"}
        or not _is_git_commit(source.get("git_commit"))
        or not isinstance(source.get("implementation_sha256s"), Mapping)
        or not source["implementation_sha256s"]
        or any(
            not isinstance(path, str) or not path or not _is_sha(digest)
            for path, digest in source["implementation_sha256s"].items()
        )
    ):
        errors.append("transfer_canary_source_binding_invalid")
    if (
        value.get("typed_state_slots_enabled") is not True
        or any(
            value.get(name) is not False
            for name in (
                "terminal_grammar_enabled",
                "answer_digit_pointer_enabled",
                "runtime_teacher_available",
                "serving_authority",
            )
        )
        or value.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        errors.append("transfer_canary_mechanism_or_claim_differs")
    return errors


def seal_transfer_canary_result(
    plan: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    errors = transfer_canary_plan_errors(plan)
    if errors:
        raise UnifiedRecurrentTransferCanaryError(",".join(errors))
    task_rows = {str(row["task_id"]): row for row in plan["tasks"]}
    candidate_fields = {
        "task_id",
        "domain",
        "arm",
        "correct",
        "parsed",
        "response_sha256",
        "generated_tokens",
        "stopped",
        "latency_ms",
    }
    observed: dict[tuple[str, str], Mapping[str, Any]] = {}
    candidate_errors: list[str] = []
    for row in candidates:
        if not isinstance(row, Mapping) or set(row) != candidate_fields:
            candidate_errors.append("transfer_canary_candidate_fields_differ")
            continue
        key = (str(row.get("task_id")), str(row.get("arm")))
        task = task_rows.get(key[0])
        if (
            task is None
            or key[1] not in ARMS
            or row.get("domain") != task.get("domain")
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
            candidate_errors.append("transfer_canary_candidate_invalid")
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

    def transitions(left: str, right: str) -> dict[str, int]:
        wrong_to_right = right_to_wrong = 0
        for task_id in task_rows:
            before = observed.get((task_id, left), {}).get("correct") is True
            after = observed.get((task_id, right), {}).get("correct") is True
            wrong_to_right += not before and after
            right_to_wrong += before and not after
        return {"wrong_to_right": wrong_to_right, "right_to_wrong": right_to_wrong}

    parent_effect = transitions("parent_typed", "treatment_typed")
    base_effect = transitions("base_greedy", "treatment_typed")
    lesion_effect = transitions("action_lesion", "treatment_typed")
    task_count = len(task_rows)
    terminal_complete = complete and all(
        observed[key]["stopped"] is True for key in expected
    )
    checks = {
        "complete_candidate_matrix": complete,
        "all_candidates_reached_terminal_contract": terminal_complete,
        "base_is_not_floor_or_ceiling": 0 < counts["base_greedy"] < task_count,
        "treatment_beats_frozen_parent": (
            counts["treatment_typed"] > counts["parent_typed"]
            and parent_effect["wrong_to_right"] > 0
        ),
        "treatment_has_zero_parent_regressions": (
            parent_effect["right_to_wrong"] == 0
        ),
        "treatment_preserves_base_successes": base_effect["right_to_wrong"] == 0,
        "action_lesion_removes_treatment_gain": (
            counts["treatment_typed"] > counts["action_lesion"]
            and lesion_effect["wrong_to_right"] > 0
            and lesion_effect["right_to_wrong"] == 0
        ),
    }
    conclusive = (
        checks["complete_candidate_matrix"]
        and checks["all_candidates_reached_terminal_contract"]
        and checks["base_is_not_floor_or_ceiling"]
    )
    supported = all(checks.values())
    verdict = SUPPORTED if supported else REFUTED if conclusive else INCONCLUSIVE
    body = {
        "schema": RESULT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "candidate_count": len(observed),
        "task_count": task_count,
        "counts": counts,
        "transitions": {
            "parent_to_treatment": parent_effect,
            "base_to_treatment": base_effect,
            "lesion_to_treatment": lesion_effect,
        },
        "checks": checks,
        "candidate_errors": sorted(candidate_errors),
        "conclusive": conclusive,
        "supported": supported,
        "verdict": verdict,
        "serving_authority": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    result = {**body, "result_sha256": _sha(body)}
    if transfer_canary_result_errors(result, plan=plan):
        raise UnifiedRecurrentTransferCanaryError(
            ",".join(transfer_canary_result_errors(result, plan=plan))
        )
    return result


def transfer_canary_result_errors(
    value: Any,
    *,
    plan: Mapping[str, Any],
) -> list[str]:
    if not isinstance(value, Mapping):
        return ["transfer_canary_result_not_mapping"]
    body = {key: item for key, item in value.items() if key != "result_sha256"}
    checks = value.get("checks", {})
    expected_conclusive = (
        checks.get("complete_candidate_matrix") is True
        and checks.get("all_candidates_reached_terminal_contract") is True
        and checks.get("base_is_not_floor_or_ceiling") is True
    )
    expected_supported = bool(checks) and all(item is True for item in checks.values())
    expected_verdict = (
        SUPPORTED
        if expected_supported
        else REFUTED
        if expected_conclusive
        else INCONCLUSIVE
    )
    if (
        value.get("schema") != RESULT_SCHEMA
        or value.get("plan_sha256") != plan.get("plan_sha256")
        or value.get("result_sha256") != _sha(body)
        or value.get("conclusive") is not expected_conclusive
        or value.get("supported") is not expected_supported
        or value.get("verdict") != expected_verdict
        or value.get("serving_authority") is not False
        or value.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        return ["transfer_canary_result_identity_or_verdict_differs"]
    return []


__all__ = [
    "ARMS",
    "CLAIM_BOUNDARY",
    "INCONCLUSIVE",
    "REFUTED",
    "SUPPORTED",
    "UnifiedRecurrentTransferCanaryError",
    "seal_transfer_canary_plan",
    "seal_transfer_canary_result",
    "transfer_canary_plan_errors",
    "transfer_canary_result_errors",
]
