"""Pure-data contracts for non-serving unified-recurrence shadow probes."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

REQUEST_SCHEMA: Final = "aura.unified_intrinsic.shadow_probe_request.v1"
RECEIPT_SCHEMA: Final = "aura.unified_intrinsic.shadow_probe_receipt.v1"
MAX_PUBLIC_TOKENS: Final = 4096
MAX_OUTPUT_TOKENS: Final = 32
_HEX = frozenset("0123456789abcdef")
_STATUSES = frozenset({"completed", "abstained", "unavailable", "cancelled"})
_REQUEST_FIELDS = {
    "schema",
    "public_token_ids",
    "expected_token_ids",
    "max_tokens",
    "request_sha256",
}
_RECEIPT_FIELDS = {
    "schema",
    "request_sha256",
    "status",
    "reason",
    "package_id",
    "controller_sha256",
    "family",
    "recurrence_depth",
    "input_token_count",
    "expected_token_count",
    "max_tokens",
    "base_token_count",
    "base_output_sha256",
    "base_exact_match",
    "base_stopped_on_eos",
    "base_latency_ms",
    "shadow_token_count",
    "shadow_output_sha256",
    "shadow_exact_match",
    "shadow_stopped_on_eos",
    "shadow_latency_ms",
    "outputs_equal",
    "output_exposed",
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


def _token_ids(value: Any, *, maximum: int, allow_empty: bool = False) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or (not value and not allow_empty):
        return None
    if len(value) > maximum or any(
        type(token_id) is not int or not 0 <= token_id < 2**31 for token_id in value
    ):
        return None
    return [int(token_id) for token_id in value]


def seal_shadow_probe_request(
    public_token_ids: Sequence[int],
    expected_token_ids: Sequence[int],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    """Normalize and commit one bounded, answer-bearing probe request."""

    public = _token_ids(public_token_ids, maximum=MAX_PUBLIC_TOKENS)
    expected = _token_ids(expected_token_ids, maximum=MAX_OUTPUT_TOKENS)
    if public is None:
        raise ValueError("unified_recurrent_shadow_probe_public_tokens_invalid")
    if expected is None:
        raise ValueError("unified_recurrent_shadow_probe_expected_tokens_invalid")
    if type(max_tokens) is not int or not len(expected) <= max_tokens <= MAX_OUTPUT_TOKENS:
        raise ValueError("unified_recurrent_shadow_probe_max_tokens_invalid")
    body = {
        "schema": REQUEST_SCHEMA,
        "public_token_ids": public,
        "expected_token_ids": expected,
        "max_tokens": max_tokens,
    }
    return {**body, "request_sha256": _canonical_sha256(body)}


def shadow_probe_request_errors(value: Any) -> list[str]:
    if not isinstance(value, dict) or set(value) != _REQUEST_FIELDS:
        return ["unified_recurrent_shadow_probe_request_fields_differ"]
    public = _token_ids(value.get("public_token_ids"), maximum=MAX_PUBLIC_TOKENS)
    expected = _token_ids(value.get("expected_token_ids"), maximum=MAX_OUTPUT_TOKENS)
    max_tokens = value.get("max_tokens")
    body = {key: item for key, item in value.items() if key != "request_sha256"}
    errors: list[str] = []
    if value.get("schema") != REQUEST_SCHEMA:
        errors.append("unified_recurrent_shadow_probe_request_schema_differs")
    if public is None or expected is None:
        errors.append("unified_recurrent_shadow_probe_tokens_invalid")
    if (
        type(max_tokens) is not int
        or expected is None
        or not len(expected) <= max_tokens <= MAX_OUTPUT_TOKENS
    ):
        errors.append("unified_recurrent_shadow_probe_budget_invalid")
    if value.get("request_sha256") != _canonical_sha256(body):
        errors.append("unified_recurrent_shadow_probe_request_commitment_differs")
    return errors


def seal_shadow_probe_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    if set(body) != _RECEIPT_FIELDS - {"receipt_sha256"}:
        raise ValueError("unified_recurrent_shadow_probe_receipt_body_fields_differ")
    sealed = {**dict(body), "receipt_sha256": _canonical_sha256(dict(body))}
    errors = shadow_probe_receipt_errors(sealed)
    if errors:
        raise ValueError(",".join(errors))
    return sealed


def shadow_probe_receipt_errors(
    value: Any,
    *,
    expected_request_sha256: str = "",
    expected_package_id: str = "",
    expected_controller_sha256: str = "",
) -> list[str]:
    """Validate a no-output receipt and optionally bind it to parent state."""

    if not isinstance(value, dict) or set(value) != _RECEIPT_FIELDS:
        return ["unified_recurrent_shadow_probe_receipt_fields_differ"]
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    errors: list[str] = []
    status = value.get("status")
    if value.get("schema") != RECEIPT_SCHEMA or status not in _STATUSES:
        errors.append("unified_recurrent_shadow_probe_receipt_schema_differs")
    for key in ("request_sha256", "controller_sha256"):
        item = value.get(key)
        if not isinstance(item, str) or len(item) != 64 or any(ch not in _HEX for ch in item):
            errors.append(f"unified_recurrent_shadow_probe_{key}_invalid")
    if (
        not isinstance(value.get("reason"), str)
        or not value["reason"]
        or not isinstance(value.get("package_id"), str)
        or not isinstance(value.get("family"), str)
        or type(value.get("recurrence_depth")) is not int
        or value["recurrence_depth"] < 0
        or value.get("output_exposed") is not False
        or value.get("serving_authority") is not False
        or value.get("receipt_sha256") != _canonical_sha256(body)
    ):
        errors.append("unified_recurrent_shadow_probe_receipt_invalid")
    integer_fields = (
        "input_token_count",
        "expected_token_count",
        "max_tokens",
        "base_token_count",
        "base_latency_ms",
        "shadow_token_count",
        "shadow_latency_ms",
    )
    if any(type(value.get(key)) is not int or value[key] < 0 for key in integer_fields):
        errors.append("unified_recurrent_shadow_probe_measurement_invalid")
    for key in ("base_exact_match", "base_stopped_on_eos", "shadow_exact_match", "shadow_stopped_on_eos", "outputs_equal"):
        if type(value.get(key)) is not bool:
            errors.append("unified_recurrent_shadow_probe_boolean_invalid")
            break
    for key in ("base_output_sha256", "shadow_output_sha256"):
        digest = value.get(key)
        if not isinstance(digest, str) or (digest and (len(digest) != 64 or any(ch not in _HEX for ch in digest))):
            errors.append(f"unified_recurrent_shadow_probe_{key}_invalid")
    completed = status == "completed"
    if completed:
        if (
            not value["package_id"]
            or not value["family"]
            or value["recurrence_depth"] < 2
            or not value["base_output_sha256"]
            or not value["shadow_output_sha256"]
            or value["input_token_count"] < 1
            or value["expected_token_count"] < 1
            or not value["expected_token_count"] <= value["max_tokens"] <= MAX_OUTPUT_TOKENS
            or value["base_token_count"] > value["max_tokens"]
            or value["shadow_token_count"] > value["max_tokens"]
        ):
            errors.append("unified_recurrent_shadow_probe_completed_state_invalid")
        digests_equal = value["base_output_sha256"] == value["shadow_output_sha256"]
        if value["outputs_equal"] is not digests_equal:
            errors.append("unified_recurrent_shadow_probe_output_equality_differs")
        if value["base_exact_match"] and (
            value["base_token_count"] != value["expected_token_count"]
            or value["base_stopped_on_eos"] is not True
        ):
            errors.append("unified_recurrent_shadow_probe_base_exactness_invalid")
        if value["shadow_exact_match"] and (
            value["shadow_token_count"] != value["expected_token_count"]
            or value["shadow_stopped_on_eos"] is not True
        ):
            errors.append("unified_recurrent_shadow_probe_shadow_exactness_invalid")
        if value["outputs_equal"] and (
            value["base_token_count"] != value["shadow_token_count"]
            or value["base_exact_match"] is not value["shadow_exact_match"]
            or value["base_stopped_on_eos"] is not value["shadow_stopped_on_eos"]
        ):
            errors.append("unified_recurrent_shadow_probe_equal_output_state_differs")
    elif any(
        value.get(key)
        for key in (
            "family",
            "base_token_count",
            "base_output_sha256",
            "base_exact_match",
            "base_stopped_on_eos",
            "base_latency_ms",
            "shadow_token_count",
            "shadow_output_sha256",
            "shadow_exact_match",
            "shadow_stopped_on_eos",
            "shadow_latency_ms",
            "outputs_equal",
        )
    ):
        errors.append("unified_recurrent_shadow_probe_noncompleted_claims_measurement")
    if expected_request_sha256 and value.get("request_sha256") != expected_request_sha256:
        errors.append("unified_recurrent_shadow_probe_request_binding_differs")
    if expected_package_id and value.get("package_id") != expected_package_id:
        errors.append("unified_recurrent_shadow_probe_package_binding_differs")
    if expected_controller_sha256 and value.get("controller_sha256") != expected_controller_sha256:
        errors.append("unified_recurrent_shadow_probe_controller_binding_differs")
    return errors


def token_sequence_sha256(token_ids: Sequence[int], *, key: bytes) -> str:
    """Commit token identity with an undisclosed per-probe HMAC key."""

    normalized = _token_ids(token_ids, maximum=MAX_OUTPUT_TOKENS, allow_empty=True)
    if normalized is None or not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("unified_recurrent_shadow_probe_output_tokens_invalid")
    payload = json.dumps(normalized, separators=(",", ":")).encode("ascii")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


__all__ = [
    "MAX_OUTPUT_TOKENS",
    "MAX_PUBLIC_TOKENS",
    "RECEIPT_SCHEMA",
    "REQUEST_SCHEMA",
    "seal_shadow_probe_receipt",
    "seal_shadow_probe_request",
    "shadow_probe_receipt_errors",
    "shadow_probe_request_errors",
    "token_sequence_sha256",
]
