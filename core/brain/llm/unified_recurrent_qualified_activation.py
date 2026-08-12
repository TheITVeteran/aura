"""Pure-data authority contract for domain-qualified recurrent serving."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Final

ACTIVATION_SCHEMA: Final = "aura.unified_intrinsic.qualified_activation.v1"
LOAD_SCHEMA: Final = "aura.unified_intrinsic.qualified_activation_load.v1"
LIFECYCLE_SCHEMA: Final = "aura.unified_intrinsic.shadow_lifecycle_run.v1"
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
    if (
        value.get("mode") != "qualified_typed_only"
        or value.get("ordinary_chat_authorized") is not False
        or value.get("arbitrary_reasoning_authorized") is not False
        or value.get("serving_authority") is not True
    ):
        errors.append("qualified_activation_authority_invalid")
    return errors


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
    elif (
        activation is not None
        or value.get("serving_authority") is not False
        or (configured and value.get("reason") == "not_configured")
        or (not configured and value.get("reason") != "not_configured")
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
        "mode": "qualified_typed_only",
        "ordinary_chat_authorized": False,
        "arbitrary_reasoning_authorized": False,
        "serving_authority": True,
    }
    activation = {**body, "activation_sha256": _sha(body)}
    errors = qualified_activation_errors(activation)
    if errors:
        raise ValueError(",".join(errors))
    return activation


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
    "qualified_activation_errors",
    "qualified_activation_load_receipt_errors",
    "seal_qualified_activation",
    "seal_qualified_activation_load_receipt",
]
