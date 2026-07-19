"""Pre-authorized ephemeral worker identity and signed result chains."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping
from typing import Any, Never

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    VerifiedCampaignTrustPolicy,
    verify_role_attestation,
)

WORKER_AUTHORIZATION_PAYLOAD_SCHEMA = (
    "aura.latent_cortex.worker_authorization_payload.v1"
)
WORKER_RESULT_ORIGIN_SCHEMA = "aura.latent_cortex.worker_result_origin.v1"
WORKER_RESULT_SIGNED_PAYLOAD_SCHEMA = (
    "aura.latent_cortex.worker_result_signed_payload.v1"
)
ZERO_SHA256 = "0" * 64

_AUTHORIZATION_KEYS = {
    "schema",
    "campaign_name",
    "policy_sha256",
    "protocol_sha256",
    "plan_sha256",
    "arm",
    "worker_source_sha256",
    "worker_command_sha256",
    "model_identity_sha256",
    "adapter_identity_sha256",
    "worker_public_key_b64",
    "worker_key_id",
}
_ORIGIN_KEYS = {
    "schema",
    "signed_payload",
    "signed_payload_sha256",
    "signature_b64",
    "origin_sha256",
}
_RESULT_PAYLOAD_KEYS = {
    "schema",
    "authorization_attestation_sha256",
    "plan_sha256",
    "arm",
    "cell_id",
    "attempt_id",
    "worker_boot_id",
    "worker_key_id",
    "sequence",
    "previous_origin_sha256",
    "result_body_sha256",
}


class WorkerOriginError(ValueError):
    """Stable fail-closed worker-origin validation error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise WorkerOriginError(code)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identifier(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 512
    ):
        _fail(f"{role}_invalid")
    return value


def _worker_boot_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail("worker_boot_id_invalid")
    return value


def _sequence(value: Any, *, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{role}_invalid")
    return value


def _normalize(value: Any, *, role: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail(f"{role}_invalid")


def _decode_public_key(value: Any) -> tuple[bytes, str]:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail("worker_public_key_invalid")
    try:
        raw = base64.b64decode(value, validate=True)
    except (TypeError, ValueError, binascii.Error):
        _fail("worker_public_key_invalid")
    if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != value:
        _fail("worker_public_key_invalid")
    return raw, _sha256_bytes(raw)


def _validated_authorization_payload(value: Any) -> dict[str, Any]:
    authorization = _normalize(value, role="worker_authorization_payload")
    if (
        not isinstance(authorization, dict)
        or set(authorization) != _AUTHORIZATION_KEYS
        or authorization.get("schema") != WORKER_AUTHORIZATION_PAYLOAD_SCHEMA
    ):
        _fail("worker_authorization_payload_invalid")
    for role in ("campaign_name", "arm"):
        _identifier(authorization.get(role), role=role)
    for role in (
        "policy_sha256",
        "protocol_sha256",
        "plan_sha256",
        "worker_source_sha256",
        "worker_command_sha256",
        "model_identity_sha256",
        "adapter_identity_sha256",
    ):
        if not _is_sha256(authorization.get(role)):
            _fail(f"{role}_invalid")
    _public_raw, key_id = _decode_public_key(
        authorization.get("worker_public_key_b64")
    )
    if authorization.get("worker_key_id") != key_id:
        _fail("worker_authorization_key_mismatch")
    return authorization


def build_worker_authorization_payload(
    *,
    campaign_name: str,
    policy_sha256: str,
    protocol_sha256: str,
    plan_sha256: str,
    arm: str,
    worker_source_sha256: str,
    worker_command_sha256: str,
    model_identity_sha256: str,
    adapter_identity_sha256: str,
    worker_public_key_raw: bytes,
) -> dict[str, Any]:
    """Build the exact runner-authorized worker identity contract."""

    if not isinstance(worker_public_key_raw, bytes) or len(worker_public_key_raw) != 32:
        _fail("worker_public_key_invalid")
    for role, value in (
        ("campaign_name", campaign_name),
        ("arm", arm),
    ):
        _identifier(value, role=role)
    for role, value in (
        ("policy_sha256", policy_sha256),
        ("protocol_sha256", protocol_sha256),
        ("plan_sha256", plan_sha256),
        ("worker_source_sha256", worker_source_sha256),
        ("worker_command_sha256", worker_command_sha256),
        ("model_identity_sha256", model_identity_sha256),
        ("adapter_identity_sha256", adapter_identity_sha256),
    ):
        if not _is_sha256(value):
            _fail(f"{role}_invalid")
    return {
        "schema": WORKER_AUTHORIZATION_PAYLOAD_SCHEMA,
        "campaign_name": campaign_name,
        "policy_sha256": policy_sha256,
        "protocol_sha256": protocol_sha256,
        "plan_sha256": plan_sha256,
        "arm": arm,
        "worker_source_sha256": worker_source_sha256,
        "worker_command_sha256": worker_command_sha256,
        "model_identity_sha256": model_identity_sha256,
        "adapter_identity_sha256": adapter_identity_sha256,
        "worker_public_key_b64": base64.b64encode(worker_public_key_raw).decode(
            "ascii"
        ),
        "worker_key_id": _sha256_bytes(worker_public_key_raw),
    }


def verify_worker_authorization(
    policy: VerifiedCampaignTrustPolicy,
    attestation: Mapping[str, Any],
    *,
    expected_payload: Mapping[str, Any],
    not_after_unix: int | None = None,
) -> dict[str, Any]:
    """Verify one pre-inference runner authorization for an ephemeral key."""

    expected = _validated_authorization_payload(expected_payload)
    try:
        return verify_role_attestation(
            policy,
            attestation,
            role=CAMPAIGN_RUNNER,
            expected_payload=expected,
            not_after_unix=not_after_unix,
        )
    except ValueError as exc:
        raise WorkerOriginError("worker_authorization_attestation_invalid") from exc


def build_worker_result_origin(
    *,
    authorization_attestation: Mapping[str, Any],
    authorization_payload: Mapping[str, Any],
    private_key: Any,
    result_body: Mapping[str, Any],
    cell_id: str,
    attempt_id: str,
    worker_boot_id: str,
    sequence: int,
    previous_origin_sha256: str = ZERO_SHA256,
) -> dict[str, Any]:
    """Sign one result body and bind it into the worker's ordered chain."""

    authorization = _validated_authorization_payload(authorization_payload)
    _sequence(sequence, role="worker_result_sequence")
    if not _is_sha256(previous_origin_sha256):
        _fail("worker_result_previous_origin_invalid")
    for role, value in (("cell_id", cell_id), ("attempt_id", attempt_id)):
        _identifier(value, role=role)
    _worker_boot_id(worker_boot_id)
    result = _normalize(result_body, role="worker_result_body")
    if not isinstance(result, dict) or "worker_origin" in result:
        _fail("worker_result_body_invalid")
    from cryptography.hazmat.primitives import serialization

    public_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if _sha256_bytes(public_raw) != authorization.get("worker_key_id"):
        _fail("worker_private_key_mismatch")
    signed_payload = {
        "schema": WORKER_RESULT_SIGNED_PAYLOAD_SCHEMA,
        "authorization_attestation_sha256": _sha256_bytes(
            canonical_json_bytes(authorization_attestation)
        ),
        "plan_sha256": authorization["plan_sha256"],
        "arm": authorization["arm"],
        "cell_id": cell_id,
        "attempt_id": attempt_id,
        "worker_boot_id": worker_boot_id,
        "worker_key_id": authorization["worker_key_id"],
        "sequence": sequence,
        "previous_origin_sha256": previous_origin_sha256,
        "result_body_sha256": _sha256_bytes(canonical_json_bytes(result)),
    }
    signed_bytes = canonical_json_bytes(signed_payload)
    material = {
        "schema": WORKER_RESULT_ORIGIN_SCHEMA,
        "signed_payload": signed_payload,
        "signed_payload_sha256": _sha256_bytes(signed_bytes),
        "signature_b64": base64.b64encode(private_key.sign(signed_bytes)).decode(
            "ascii"
        ),
    }
    return {
        **material,
        "origin_sha256": _sha256_bytes(canonical_json_bytes(material)),
    }


def verify_worker_result_origin(
    policy: VerifiedCampaignTrustPolicy,
    *,
    authorization_attestation: Mapping[str, Any],
    expected_authorization_payload: Mapping[str, Any],
    result: Mapping[str, Any],
    expected_cell_id: str,
    expected_attempt_id: str,
    expected_sequence: int,
    expected_previous_origin_sha256: str,
    authorization_not_after_unix: int | None = None,
) -> dict[str, Any]:
    """Verify authorization, result-body signature, and chain position."""

    _identifier(expected_cell_id, role="expected_cell_id")
    _identifier(expected_attempt_id, role="expected_attempt_id")
    _sequence(expected_sequence, role="worker_result_expected_sequence")
    if not _is_sha256(expected_previous_origin_sha256):
        _fail("worker_result_expected_previous_origin_invalid")
    authorization = _validated_authorization_payload(
        expected_authorization_payload
    )
    verify_worker_authorization(
        policy,
        authorization_attestation,
        expected_payload=authorization,
        not_after_unix=authorization_not_after_unix,
    )
    normalized_result = _normalize(result, role="worker_result")
    if not isinstance(normalized_result, dict):
        _fail("worker_result_invalid")
    origin = normalized_result.get("worker_origin")
    if not isinstance(origin, dict) or set(origin) != _ORIGIN_KEYS:
        _fail("worker_result_origin_invalid")
    if origin.get("schema") != WORKER_RESULT_ORIGIN_SCHEMA:
        _fail("worker_result_origin_invalid")
    material = dict(origin)
    origin_sha256 = material.pop("origin_sha256", None)
    if origin_sha256 != _sha256_bytes(canonical_json_bytes(material)):
        _fail("worker_result_origin_digest_invalid")
    signed_payload = origin.get("signed_payload")
    if not isinstance(signed_payload, dict) or set(signed_payload) != _RESULT_PAYLOAD_KEYS:
        _fail("worker_result_signed_payload_invalid")
    result_body = dict(normalized_result)
    result_body.pop("worker_origin", None)
    expected = {
        "schema": WORKER_RESULT_SIGNED_PAYLOAD_SCHEMA,
        "authorization_attestation_sha256": _sha256_bytes(
            canonical_json_bytes(authorization_attestation)
        ),
        "plan_sha256": authorization["plan_sha256"],
        "arm": authorization["arm"],
        "cell_id": expected_cell_id,
        "attempt_id": expected_attempt_id,
        "worker_boot_id": signed_payload.get("worker_boot_id"),
        "worker_key_id": authorization["worker_key_id"],
        "sequence": expected_sequence,
        "previous_origin_sha256": expected_previous_origin_sha256,
        "result_body_sha256": _sha256_bytes(canonical_json_bytes(result_body)),
    }
    if signed_payload != expected:
        _fail("worker_result_binding_invalid")
    _identifier(signed_payload.get("cell_id"), role="cell_id")
    _identifier(signed_payload.get("attempt_id"), role="attempt_id")
    _worker_boot_id(signed_payload.get("worker_boot_id"))
    _sequence(signed_payload.get("sequence"), role="worker_result_sequence")
    for role in (
        "authorization_attestation_sha256",
        "plan_sha256",
        "worker_key_id",
        "previous_origin_sha256",
        "result_body_sha256",
    ):
        if not _is_sha256(signed_payload.get(role)):
            _fail(f"worker_result_{role}_invalid")
    runtime_identity = result_body.get("runtime_model_identity")
    if (
        not isinstance(runtime_identity, dict)
        or runtime_identity.get("worker_boot_id")
        != signed_payload.get("worker_boot_id")
    ):
        _fail("worker_result_boot_identity_mismatch")
    signed_bytes = canonical_json_bytes(signed_payload)
    if origin.get("signed_payload_sha256") != _sha256_bytes(signed_bytes):
        _fail("worker_result_signed_payload_digest_invalid")
    try:
        signature = base64.b64decode(origin.get("signature_b64"), validate=True)
    except (TypeError, ValueError, binascii.Error):
        _fail("worker_result_signature_invalid")
    if (
        len(signature) != 64
        or base64.b64encode(signature).decode("ascii")
        != origin.get("signature_b64")
    ):
        _fail("worker_result_signature_invalid")
    public_raw, _key_id = _decode_public_key(
        authorization.get("worker_public_key_b64")
    )
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(public_raw).verify(signature, signed_bytes)
    except (InvalidSignature, ValueError):
        _fail("worker_result_signature_invalid")
    return dict(signed_payload)


__all__ = [
    "WORKER_AUTHORIZATION_PAYLOAD_SCHEMA",
    "WORKER_RESULT_ORIGIN_SCHEMA",
    "WORKER_RESULT_SIGNED_PAYLOAD_SCHEMA",
    "ZERO_SHA256",
    "WorkerOriginError",
    "build_worker_authorization_payload",
    "build_worker_result_origin",
    "verify_worker_authorization",
    "verify_worker_result_origin",
]
