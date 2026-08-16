"""Pure-data authority contract for domain-qualified recurrent serving."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, Final

ACTIVATION_SCHEMA: Final = "aura.unified_intrinsic.qualified_activation.v2"
LOAD_SCHEMA: Final = "aura.unified_intrinsic.qualified_activation_load.v1"
LIFECYCLE_SCHEMA: Final = "aura.unified_intrinsic.shadow_lifecycle_run.v1"
QUALIFIED_CANARY_SCHEMA: Final = "aura.unified_intrinsic.qualified_serving_canary.v3"
_HEX = frozenset("0123456789abcdef")
_QUALIFIED_FAMILIES: Final = frozenset({"khop", "modular", "register_trace"})
_FIELDS: Final = {
    "schema",
    "package_id",
    "manifest_sha256",
    "checkpoint_sha256",
    "controller_sha256",
    "pointer_sha256",
    "lifecycle_result_sha256",
    "canary_plan_sha256",
    "candidate_canary_sha256",
    "qualified_canary_sha256",
    "families",
    "task_depths",
    "recurrence_depth",
    "mode",
    "ordinary_chat_authorized",
    "arbitrary_reasoning_authorized",
    "serving_authority",
    "activation_sha256",
}
_LIFECYCLE_CHECKS: Final = {
    "durable_pointer_reopened",
    "first_cold_load_supported",
    "restart_cold_load_supported",
    "restart_identity_stable",
    "pointer_rollback_completed",
    "post_rollback_worker_inactive",
}
_LOAD_FIELDS: Final = {
    "schema",
    "configured",
    "loaded",
    "reason",
    "activation",
    "serving_authority",
    "receipt_sha256",
}
_CANARY_FIELDS: Final = {
    "schema",
    "package_id",
    "manifest_sha256",
    "checkpoint_sha256",
    "controller_sha256",
    "activation_sha256",
    "battery_sha256",
    "started_at_unix",
    "completed_at_unix",
    "case_count",
    "exact_count",
    "total_latency_ms",
    "maximum_latency_ms",
    "evidence",
    "supported",
    "serving_authority",
    "authority_remains_active",
    "canary_authority_was_request_scoped",
    "output_exposed",
    "result_sha256",
}
_CANARY_EVIDENCE_FIELDS: Final = {
    "index",
    "task_id",
    "family",
    "task_depth",
    "request_sha256",
    "expected_token_ids_sha256",
    "generated_token_ids_sha256",
    "qualified_result_sha256",
    "latency_ms",
    "exact",
}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX for c in value)


def _unique_sorted_strings(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item for item in value)
        and len(value) == len(set(value))
        and value == sorted(value)
    )


def _unique_sorted_depths(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(type(item) is int and item >= 1 for item in value)
        and len(value) == len(set(value))
        and value == sorted(value)
    )


def qualified_activation_errors(value: Any) -> list[str]:
    if not isinstance(value, Mapping) or set(value) != _FIELDS:
        return ["qualified_activation_fields_differ"]
    body = {key: item for key, item in value.items() if key != "activation_sha256"}
    errors: list[str] = []
    if value.get("schema") != ACTIVATION_SCHEMA or value.get("activation_sha256") != _sha(body):
        errors.append("qualified_activation_identity_differs")
    if (
        not isinstance(value.get("package_id"), str)
        or not value["package_id"]
        or any(
            not _is_sha(value.get(key))
            for key in (
                "manifest_sha256",
                "checkpoint_sha256",
                "controller_sha256",
                "pointer_sha256",
                "lifecycle_result_sha256",
                "canary_plan_sha256",
            )
        )
    ):
        errors.append("qualified_activation_evidence_identity_invalid")
    families = value.get("families")
    depths = value.get("task_depths")
    if (
        not _unique_sorted_strings(families)
        or any(family not in _QUALIFIED_FAMILIES for family in families)
        or not _unique_sorted_depths(depths)
        or type(value.get("recurrence_depth")) is not int
        or value["recurrence_depth"] < 2
        or (depths and max(depths) > value["recurrence_depth"])
    ):
        errors.append("qualified_activation_domain_invalid")
    candidate = (
        value.get("mode") == "qualified_canary_only"
        and value.get("serving_authority") is False
        and value.get("candidate_canary_sha256") == ""
        and value.get("qualified_canary_sha256") == ""
    )
    pending = (
        value.get("mode") == "qualified_typed_pending"
        and value.get("serving_authority") is False
        and _is_sha(value.get("candidate_canary_sha256"))
        and value.get("qualified_canary_sha256") == ""
    )
    durable = (
        value.get("mode") == "qualified_typed_only"
        and value.get("serving_authority") is True
        and _is_sha(value.get("candidate_canary_sha256"))
        and _is_sha(value.get("qualified_canary_sha256"))
    )
    if (
        not (candidate or pending or durable)
        or value.get("ordinary_chat_authorized") is not False
        or value.get("arbitrary_reasoning_authorized") is not False
    ):
        errors.append("qualified_activation_authority_invalid")
    return errors


def qualified_serving_canary_errors(
    value: Any,
    *,
    expected_activation: Mapping[str, Any] | None = None,
    expected_battery: Mapping[str, Any] | None = None,
) -> list[str]:
    """Validate the complete candidate or durable cold-load canary artifact."""

    if not isinstance(value, Mapping) or set(value) != _CANARY_FIELDS:
        return ["qualified_serving_canary_fields_differ"]
    body = {key: item for key, item in value.items() if key != "result_sha256"}
    evidence = value.get("evidence")
    started = value.get("started_at_unix")
    completed = value.get("completed_at_unix")
    counts = (value.get("case_count"), value.get("exact_count"))
    latencies = (value.get("total_latency_ms"), value.get("maximum_latency_ms"))
    errors: list[str] = []
    if (
        value.get("schema") != QUALIFIED_CANARY_SCHEMA
        or value.get("result_sha256") != _sha(body)
        or not isinstance(value.get("package_id"), str)
        or not value["package_id"]
        or any(
            not _is_sha(value.get(key))
            for key in (
                "manifest_sha256",
                "checkpoint_sha256",
                "controller_sha256",
                "activation_sha256",
                "battery_sha256",
            )
        )
        or any(type(item) is not int or item < 0 for item in (*counts, *latencies))
        or not isinstance(started, (int, float))
        or isinstance(started, bool)
        or not isinstance(completed, (int, float))
        or isinstance(completed, bool)
        or not math.isfinite(float(started))
        or not math.isfinite(float(completed))
        or float(completed) < float(started)
        or value.get("supported") is not True
        or value.get("output_exposed") is not False
    ):
        errors.append("qualified_serving_canary_identity_invalid")
    request_scoped = value.get("canary_authority_was_request_scoped")
    candidate = (
        value.get("serving_authority") is False
        and value.get("authority_remains_active") is False
        and request_scoped is True
    )
    durable = (
        value.get("serving_authority") is True
        and value.get("authority_remains_active") is True
        and request_scoped is False
    )
    if not (candidate or durable):
        errors.append("qualified_serving_canary_authority_invalid")
    if not isinstance(evidence, list) or not evidence:
        errors.append("qualified_serving_canary_evidence_invalid")
    else:
        task_ids: set[str] = set()
        observed_latencies: list[int] = []
        for index, row in enumerate(evidence):
            if (
                not isinstance(row, Mapping)
                or set(row) != _CANARY_EVIDENCE_FIELDS
                or row.get("index") != index
                or not isinstance(row.get("task_id"), str)
                or not row["task_id"]
                or row["task_id"] in task_ids
                or row.get("family") not in _QUALIFIED_FAMILIES
                or type(row.get("task_depth")) is not int
                or row["task_depth"] < 1
                or not _is_sha(row.get("request_sha256"))
                or not _is_sha(row.get("expected_token_ids_sha256"))
                or not _is_sha(row.get("generated_token_ids_sha256"))
                or row.get("generated_token_ids_sha256")
                != row.get("expected_token_ids_sha256")
                or not _is_sha(row.get("qualified_result_sha256"))
                or type(row.get("latency_ms")) is not int
                or row["latency_ms"] < 0
                or row.get("exact") is not True
            ):
                errors.append("qualified_serving_canary_evidence_invalid")
                break
            task_ids.add(row["task_id"])
            observed_latencies.append(row["latency_ms"])
        if observed_latencies and (
            counts[0] != len(evidence)
            or counts[1] != len(evidence)
            or latencies[0] != sum(observed_latencies)
            or latencies[1] != max(observed_latencies)
        ):
            errors.append("qualified_serving_canary_aggregate_invalid")
    if isinstance(expected_activation, Mapping) and any(
        value.get(key) != expected_activation.get(key)
        for key in (
            "package_id",
            "manifest_sha256",
            "checkpoint_sha256",
            "controller_sha256",
            "activation_sha256",
        )
    ):
        errors.append("qualified_serving_canary_activation_differs")
    if isinstance(expected_activation, Mapping) and isinstance(evidence, list) and any(
        not isinstance(row, Mapping)
        or row.get("family") not in set(expected_activation.get("families") or ())
        or row.get("task_depth") not in set(expected_activation.get("task_depths") or ())
        for row in evidence
    ):
        errors.append("qualified_serving_canary_domain_differs")
    if isinstance(expected_battery, Mapping):
        try:
            from core.brain.llm.unified_recurrent_shadow_battery import (
                validate_shadow_canary_battery,
            )

            battery = validate_shadow_canary_battery(expected_battery)
        except (ImportError, TypeError, ValueError):
            errors.append("qualified_serving_canary_battery_invalid")
        else:
            cases = battery["cases"]
            if (
                value.get("battery_sha256") != battery.get("battery_sha256")
                or not isinstance(evidence, list)
                or len(evidence) != len(cases)
            ):
                errors.append("qualified_serving_canary_battery_differs")
            elif any(
                row.get("index") != index
                or row.get("task_id") != case.get("task_id")
                or row.get("family") != case.get("family")
                or row.get("task_depth") != case.get("task_depth")
                or row.get("request_sha256") != case.get("request_sha256")
                or row.get("expected_token_ids_sha256")
                != _sha(case.get("expected_token_ids"))
                or row.get("generated_token_ids_sha256")
                != row.get("expected_token_ids_sha256")
                or row.get("exact") is not True
                for index, (row, case) in enumerate(zip(evidence, cases, strict=True))
            ):
                errors.append("qualified_serving_canary_case_inventory_differs")
    return list(dict.fromkeys(errors))


def seal_qualified_activation_load_receipt(
    *,
    configured: bool,
    loaded: bool,
    reason: str,
    activation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Seal explicit active or inactive worker state without inherited authority."""

    normalized = dict(activation) if isinstance(activation, Mapping) else None
    body = {
        "schema": LOAD_SCHEMA,
        "configured": configured,
        "loaded": loaded,
        "reason": reason,
        "activation": normalized,
        "serving_authority": bool(
            loaded and normalized and normalized.get("serving_authority") is True
        ),
    }
    receipt = {**body, "receipt_sha256": _sha(body)}
    errors = qualified_activation_load_receipt_errors(receipt)
    if errors:
        raise ValueError(",".join(errors))
    return receipt


def qualified_activation_load_receipt_errors(value: Any) -> list[str]:
    """Return every reason a worker activation receipt cannot be trusted."""

    if not isinstance(value, Mapping) or set(value) != _LOAD_FIELDS:
        return ["qualified_activation_load_fields_differ"]
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    errors: list[str] = []
    configured = value.get("configured")
    loaded = value.get("loaded")
    activation = value.get("activation")
    if (
        value.get("schema") != LOAD_SCHEMA
        or type(configured) is not bool
        or type(loaded) is not bool
        or not isinstance(value.get("reason"), str)
        or not value["reason"]
        or value.get("receipt_sha256") != _sha(body)
    ):
        errors.append("qualified_activation_load_identity_invalid")
        return errors
    if loaded:
        if not configured:
            errors.append("loaded_qualified_activation_not_configured")
        activation_errors = qualified_activation_errors(activation)
        if activation_errors:
            errors.extend(activation_errors)
        if value.get("serving_authority") is not True:
            errors.append("loaded_qualified_activation_lacks_authority")
    elif configured:
        activation_errors = qualified_activation_errors(activation)
        pending = (
            isinstance(activation, Mapping)
            and activation.get("mode") == "qualified_typed_pending"
            and activation.get("serving_authority") is False
            and value.get("reason") == "qualified_activation_pending_canary"
        )
        unavailable = (
            isinstance(activation, Mapping)
            and activation.get("mode") in {
                "qualified_typed_pending",
                "qualified_typed_only",
            }
            and value.get("reason") == "qualified_activation_shadow_unavailable"
        )
        if activation_errors or not (pending or unavailable) or value.get(
            "serving_authority"
        ) is not False:
            errors.append("pending_qualified_activation_state_invalid")
    elif (
        activation is not None
        or value.get("serving_authority") is not False
        or value.get("reason") != "not_configured"
    ):
        errors.append("inactive_qualified_activation_claims_state")
    return errors


def seal_qualified_activation(
    manifest: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    pointer: Mapping[str, Any],
) -> dict[str, Any]:
    """Issue typed authority only from matching, supported lifecycle evidence."""

    domain = manifest.get("domain_contract")
    checks = lifecycle.get("checks")
    retired = lifecycle.get("activation_pointer")
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    lifecycle_body = {
        key: value for key, value in lifecycle.items() if key != "result_sha256"
    }
    pointer_body = {
        key: value for key, value in pointer.items() if key != "pointer_sha256"
    }
    if (
        not isinstance(domain, Mapping)
        or domain.get("qualification") != "generator_and_grammar_bound"
        or not isinstance(domain.get("families"), list)
        or not domain["families"]
        or not isinstance(domain.get("task_depths"), list)
        or not domain["task_depths"]
        or type(domain.get("recurrence_depth")) is not int
        or lifecycle.get("schema") != LIFECYCLE_SCHEMA
        or lifecycle.get("supported") is not True
        or lifecycle.get("serving_authority") is not False
        or lifecycle.get("output_exposed") is not False
        or not isinstance(checks, Mapping)
        or set(checks) != _LIFECYCLE_CHECKS
        or any(value is not True for value in checks.values())
        or lifecycle.get("package_id") != manifest.get("package_id")
        or lifecycle.get("manifest_sha256") != manifest.get("manifest_sha256")
        or not isinstance(retired, Mapping)
        or dict(retired) != dict(pointer)
        or pointer.get("package_id") != manifest.get("package_id")
        or pointer.get("manifest_sha256") != manifest.get("manifest_sha256")
        or not _is_sha(lifecycle.get("result_sha256"))
        or not _is_sha(lifecycle.get("canary_plan_sha256"))
        or not _is_sha(lifecycle.get("controller_sha256"))
        or not _is_sha(pointer.get("pointer_sha256"))
        or not _is_sha(manifest.get("checkpoint_sha256"))
        or manifest.get("manifest_sha256") != _sha(manifest_body)
        or lifecycle.get("result_sha256") != _sha(lifecycle_body)
        or pointer.get("pointer_sha256") != _sha(pointer_body)
    ):
        raise ValueError("qualified_activation_evidence_is_not_admissible")
    body = {
        "schema": ACTIVATION_SCHEMA,
        "package_id": manifest["package_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "controller_sha256": lifecycle["controller_sha256"],
        "pointer_sha256": pointer["pointer_sha256"],
        "lifecycle_result_sha256": lifecycle["result_sha256"],
        "canary_plan_sha256": lifecycle["canary_plan_sha256"],
        "families": list(domain["families"]),
        "task_depths": list(domain["task_depths"]),
        "recurrence_depth": int(domain["recurrence_depth"]),
        "candidate_canary_sha256": "",
        "qualified_canary_sha256": "",
        "mode": "qualified_canary_only",
        "ordinary_chat_authorized": False,
        "arbitrary_reasoning_authorized": False,
        "serving_authority": False,
    }
    activation = {**body, "activation_sha256": _sha(body)}
    errors = qualified_activation_errors(activation)
    if errors:
        raise ValueError(",".join(errors))
    return activation


def seal_verified_qualified_activation(
    candidate: Mapping[str, Any],
    canary: Mapping[str, Any],
    *,
    expected_battery: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal inert persisted authority from an exact in-memory canary."""

    if not isinstance(expected_battery, Mapping):
        raise ValueError("qualified_canary_battery_is_not_admissible")
    errors = qualified_activation_errors(candidate)
    if (
        errors
        or candidate.get("mode") != "qualified_canary_only"
        or candidate.get("serving_authority") is not False
        or qualified_serving_canary_errors(
            canary,
            expected_activation=candidate,
            expected_battery=expected_battery,
        )
        or canary.get("canary_authority_was_request_scoped") is not True
    ):
        raise ValueError("qualified_canary_evidence_is_not_admissible")
    body = {
        key: value
        for key, value in candidate.items()
        if key != "activation_sha256"
    }
    body.update(
        {
            "candidate_canary_sha256": canary["result_sha256"],
            "mode": "qualified_typed_pending",
        }
    )
    activation = {**body, "activation_sha256": _sha(body)}
    errors = qualified_activation_errors(activation)
    if errors:
        raise ValueError(",".join(errors))
    return activation


def seal_serving_qualified_activation(
    pending: Mapping[str, Any],
    canary: Mapping[str, Any],
    *,
    expected_battery: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote a cold-loaded pending document only after its exact canary."""

    if not isinstance(expected_battery, Mapping):
        raise ValueError("qualified_canary_battery_is_not_admissible")
    if (
        qualified_activation_errors(pending)
        or pending.get("mode") != "qualified_typed_pending"
        or pending.get("serving_authority") is not False
        or qualified_serving_canary_errors(
            canary,
            expected_activation=pending,
            expected_battery=expected_battery,
        )
        or canary.get("canary_authority_was_request_scoped") is not True
    ):
        raise ValueError("qualified_pending_canary_evidence_is_not_admissible")
    body = {
        key: value
        for key, value in pending.items()
        if key != "activation_sha256"
    }
    body.update(
        {
            "qualified_canary_sha256": canary["result_sha256"],
            "mode": "qualified_typed_only",
            "serving_authority": True,
        }
    )
    activation = {**body, "activation_sha256": _sha(body)}
    errors = qualified_activation_errors(activation)
    if errors:
        raise ValueError(",".join(errors))
    return activation


def pending_activation_from_serving(
    serving: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the exact pending identity committed by serving authority."""

    if (
        qualified_activation_errors(serving)
        or serving.get("mode") != "qualified_typed_only"
        or serving.get("serving_authority") is not True
    ):
        raise ValueError("serving_qualified_activation_is_not_admissible")
    body = {
        key: value
        for key, value in serving.items()
        if key != "activation_sha256"
    }
    body.update(
        {
            "qualified_canary_sha256": "",
            "mode": "qualified_typed_pending",
            "serving_authority": False,
        }
    )
    pending = {**body, "activation_sha256": _sha(body)}
    errors = qualified_activation_errors(pending)
    if errors:
        raise ValueError(",".join(errors))
    return pending


def candidate_activation_from_pending(
    pending: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the exact ephemeral candidate committed by pending state."""

    if (
        qualified_activation_errors(pending)
        or pending.get("mode") != "qualified_typed_pending"
        or pending.get("serving_authority") is not False
    ):
        raise ValueError("pending_qualified_activation_is_not_admissible")
    body = {
        key: value
        for key, value in pending.items()
        if key != "activation_sha256"
    }
    body.update(
        {
            "candidate_canary_sha256": "",
            "mode": "qualified_canary_only",
        }
    )
    candidate = {**body, "activation_sha256": _sha(body)}
    errors = qualified_activation_errors(candidate)
    if errors:
        raise ValueError(",".join(errors))
    return candidate


def activation_matches_shadow_receipt(
    activation: Mapping[str, Any],
    shadow_receipt: Mapping[str, Any],
) -> bool:
    """Match runtime package/controller/domain identity before any output."""

    return not qualified_activation_errors(activation) and all(
        (
            activation.get("package_id") == shadow_receipt.get("package_id"),
            activation.get("manifest_sha256") == shadow_receipt.get("manifest_sha256"),
            activation.get("checkpoint_sha256") == shadow_receipt.get("checkpoint_sha256"),
            activation.get("controller_sha256") == shadow_receipt.get("controller_sha256"),
            activation.get("families") == shadow_receipt.get("families"),
            activation.get("task_depths") == shadow_receipt.get("task_depths"),
            activation.get("recurrence_depth") == shadow_receipt.get("recurrence_depth"),
        )
    )


__all__ = [
    "ACTIVATION_SCHEMA",
    "LOAD_SCHEMA",
    "LIFECYCLE_SCHEMA",
    "activation_matches_shadow_receipt",
    "candidate_activation_from_pending",
    "qualified_activation_errors",
    "qualified_activation_load_receipt_errors",
    "qualified_serving_canary_errors",
    "pending_activation_from_serving",
    "seal_qualified_activation",
    "seal_qualified_activation_load_receipt",
    "seal_serving_qualified_activation",
    "seal_verified_qualified_activation",
]
