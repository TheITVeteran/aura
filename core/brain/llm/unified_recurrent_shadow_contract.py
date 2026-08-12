"""Pure-data contract for worker-side unified-recurrence shadow loading."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Final

LOAD_SCHEMA: Final = "aura.unified_intrinsic.shadow_load.v1"
_RECEIPT_FIELDS: Final = {
    "schema",
    "configured",
    "loaded",
    "reason",
    "package_id",
    "manifest_sha256",
    "checkpoint_sha256",
    "controller_sha256",
    "families",
    "task_depths",
    "recurrence_depth",
    "model_identity_strength",
    "mode",
    "serving_authority",
    "receipt_sha256",
}


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


def seal_shadow_load_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    """Seal one exact receipt body; callers cannot omit or add authority fields."""

    expected_body = _RECEIPT_FIELDS - {"receipt_sha256"}
    if set(body) != expected_body:
        raise ValueError("unified_recurrent_shadow_receipt_body_fields_differ")
    sealed = {**dict(body), "receipt_sha256": _canonical_sha256(dict(body))}
    errors = shadow_load_receipt_errors(sealed)
    if errors:
        raise ValueError(",".join(errors))
    return sealed


def shadow_load_receipt_errors(value: Any) -> list[str]:
    """Return every reason a worker shadow-load receipt is not trustworthy."""

    if not isinstance(value, dict):
        return ["unified_recurrent_shadow_not_mapping"]
    if set(value) != _RECEIPT_FIELDS:
        return ["unified_recurrent_shadow_fields_differ"]
    errors: list[str] = []
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    configured = value.get("configured")
    loaded = value.get("loaded")
    if value.get("schema") != LOAD_SCHEMA:
        errors.append("unified_recurrent_shadow_schema_differs")
    if type(configured) is not bool or type(loaded) is not bool:
        errors.append("unified_recurrent_shadow_state_invalid")
        return errors
    if (
        not isinstance(value.get("reason"), str)
        or not value["reason"]
        or value.get("mode") != "shadow_only"
        or value.get("serving_authority") is not False
        or value.get("receipt_sha256") != _canonical_sha256(body)
    ):
        errors.append("unified_recurrent_shadow_receipt_invalid")
    if loaded:
        if not configured:
            errors.append("loaded_unified_recurrent_shadow_not_configured")
        for key in (
            "manifest_sha256",
            "checkpoint_sha256",
            "controller_sha256",
        ):
            field = value.get(key)
            if (
                not isinstance(field, str)
                or len(field) != 64
                or any(character not in "0123456789abcdef" for character in field)
            ):
                errors.append(f"unified_recurrent_shadow_{key}_invalid")
        families = value.get("families")
        task_depths = value.get("task_depths")
        if (
            not isinstance(value.get("package_id"), str)
            or not value["package_id"]
            or not isinstance(families, list)
            or not families
            or any(not isinstance(family, str) or not family for family in families)
            or not isinstance(task_depths, list)
            or not task_depths
            or any(type(depth) is not int or depth < 1 for depth in task_depths)
            or type(value.get("recurrence_depth")) is not int
            or value["recurrence_depth"] < 2
            or value.get("model_identity_strength") != "config_behavior_hash_and_weight_extent"
        ):
            errors.append("unified_recurrent_shadow_domain_invalid")
    elif configured:
        errors.append("configured_unified_recurrent_shadow_not_loaded")
    elif (
        any(
            value.get(key)
            for key in (
                "package_id",
                "manifest_sha256",
                "checkpoint_sha256",
                "controller_sha256",
                "families",
                "task_depths",
                "recurrence_depth",
            )
        )
        or value.get("model_identity_strength") != "none"
    ):
        errors.append("inactive_unified_recurrent_shadow_claims_state")
    return errors


__all__ = [
    "LOAD_SCHEMA",
    "seal_shadow_load_receipt",
    "shadow_load_receipt_errors",
]
