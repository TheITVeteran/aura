"""Pure protocol assembly and validation for detached worker origins."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Never

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    VerifiedCampaignTrustPolicy,
    verify_role_attestation,
)

WORKER_AUTHORIZATION_PAYLOAD_SCHEMA = (
    "aura.latent_cortex.worker_authorization_payload.v4"
)
WORKER_RESULT_SIGNED_PAYLOAD_SCHEMA = (
    "aura.latent_cortex.worker_result_signed_payload.v3"
)
WORKER_RESULT_ORIGIN_SCHEMA = "aura.latent_cortex.worker_result_origin.v2"
WORKER_LIFECYCLE_EVENT_PAYLOAD_SCHEMA = (
    "aura.latent_cortex.worker_lifecycle_event_payload.v1"
)
WORKER_LIFECYCLE_EVENT_ORIGIN_SCHEMA = (
    "aura.latent_cortex.worker_lifecycle_event_origin.v1"
)

WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR = "detached_supervisor_memory_only"
WORKER_KEY_CUSTODY_PRODUCER_SOFTWARE = "producer_process_exportable"
ZERO_SHA256 = "0" * 64
MAX_WORKER_PROTOCOL_VALUE_BYTES = 1_048_576
MAX_WORKER_ALLOWED_CELLS = 16_384

_SUPPORTED_CUSTODY_CLASSES = {
    WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
    WORKER_KEY_CUSTODY_PRODUCER_SOFTWARE,
}
_AUTHORIZATION_KEYS = {
    "schema",
    "campaign_name",
    "policy_sha256",
    "protocol_sha256",
    "detached_plan_sha256",
    "broker_policy_sha256",
    "executable_binding_sha256",
    "environment_sha256",
    "sandbox_sha256",
    "source_manifest_sha256",
    "session_id",
    "supervisor_attempt",
    "arm",
    "worker_attempt_slot",
    "allowed_cell_digest",
    "model_identity_sha256",
    "adapter_identity_sha256",
    "worker_key_custody",
    "worker_public_key_b64",
    "worker_key_id",
}
_RESULT_PAYLOAD_KEYS = {
    "schema",
    "authorization_attestation_sha256",
    "authorization_payload_sha256",
    "detached_plan_sha256",
    "session_id",
    "supervisor_attempt",
    "arm",
    "worker_attempt_slot",
    "cell_id",
    "cell_type",
    "attempt_id",
    "worker_key_id",
    "sequence",
    "previous_origin_sha256",
    "result_body_sha256",
}
_RESULT_ORIGIN_KEYS = {
    "schema",
    "signed_payload",
    "signed_payload_sha256",
    "signature_b64",
    "origin_sha256",
}
_LIFECYCLE_PAYLOAD_KEYS = {
    "schema",
    "authorization_attestation_sha256",
    "authorization_payload_sha256",
    "detached_plan_sha256",
    "session_id",
    "supervisor_attempt",
    "arm",
    "worker_attempt_slot",
    "worker_key_id",
    "event_type",
    "prior_state",
    "result_count",
    "previous_origin_sha256",
    "completed_cell_digest",
    "occurred_at_unix",
    "return_code",
    "reason",
}
_LIFECYCLE_ORIGIN_KEYS = {
    "schema",
    "signed_payload",
    "signed_payload_sha256",
    "signature_b64",
    "event_sha256",
}
_RESULT_BODY_BINDING_KEYS = {
    "cell_id",
    "cell_type",
    "attempt_id",
    "origin_session_id",
}
_LIFECYCLE_EVENT_TYPES = {"terminal", "abandoned"}
_LIFECYCLE_PRIOR_STATES = {
    "prepared",
    "awaiting_external_signature",
    "authorized",
    "running",
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


def _sha256(value: Any, *, role: str) -> str:
    if not _is_sha256(value):
        _fail(f"{role}_invalid")
    return value


def _identifier(value: Any, *, role: str, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        _fail(f"{role}_invalid")
    return value


def _hex_identifier(value: Any, *, role: str, length: int = 32) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{role}_invalid")
    return value


def _positive_int(value: Any, *, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{role}_invalid")
    return value


def _nonnegative_int(value: Any, *, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{role}_invalid")
    return value


def _normalize(value: Any, *, role: str) -> Any:
    try:
        payload = canonical_json_bytes(value)
        if len(payload) > MAX_WORKER_PROTOCOL_VALUE_BYTES:
            _fail(f"{role}_too_large")
        return json.loads(payload)
    except WorkerOriginError:
        raise
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


def _signature_bytes(value: Any, *, role: str) -> bytes:
    if isinstance(value, bytes):
        signature = value
    elif isinstance(value, str) and value == value.strip():
        try:
            signature = base64.b64decode(value, validate=True)
        except (TypeError, ValueError, binascii.Error):
            _fail(f"{role}_invalid")
    else:
        _fail(f"{role}_invalid")
    if len(signature) != 64:
        _fail(f"{role}_invalid")
    return signature


def _verify_signature(
    *,
    public_key_b64: str,
    signature_b64: Any,
    signed_bytes: bytes,
    role: str,
) -> None:
    signature = _signature_bytes(signature_b64, role=role)
    public_raw, _key_id = _decode_public_key(public_key_b64)
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(public_raw).verify(
            signature, signed_bytes
        )
    except (InvalidSignature, ValueError):
        _fail(f"{role}_invalid")


def compute_allowed_cell_digest(
    allowed_cells: Sequence[Mapping[str, str]],
) -> str:
    """Hash one ordered, unique set of typed cells."""

    if isinstance(allowed_cells, (str, bytes)) or not isinstance(
        allowed_cells, Sequence
    ):
        _fail("worker_allowed_cells_invalid")
    if len(allowed_cells) > MAX_WORKER_ALLOWED_CELLS:
        _fail("worker_allowed_cells_too_large")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in allowed_cells:
        if not isinstance(raw, Mapping) or set(raw) != {"cell_id", "cell_type"}:
            _fail("worker_allowed_cells_invalid")
        cell_id = _identifier(raw.get("cell_id"), role="worker_cell_id")
        cell_type = _identifier(raw.get("cell_type"), role="worker_cell_type")
        if cell_id in seen:
            _fail("worker_allowed_cell_duplicate")
        seen.add(cell_id)
        normalized.append({"cell_id": cell_id, "cell_type": cell_type})
    if not normalized:
        _fail("worker_allowed_cells_invalid")
    return _sha256_bytes(canonical_json_bytes(normalized))


def compute_completed_cell_digest(completed_cell_ids: Sequence[str]) -> str:
    """Hash an ordered completed-cell prefix."""

    if isinstance(completed_cell_ids, (str, bytes)) or not isinstance(
        completed_cell_ids, Sequence
    ):
        _fail("worker_completed_cells_invalid")
    normalized: list[str] = []
    seen: set[str] = set()
    for cell_id in completed_cell_ids:
        parsed = _identifier(cell_id, role="worker_completed_cell_id")
        if parsed in seen:
            _fail("worker_completed_cell_duplicate")
        seen.add(parsed)
        normalized.append(parsed)
    return _sha256_bytes(canonical_json_bytes(normalized))


def _validated_authorization_payload(value: Any) -> dict[str, Any]:
    authorization = _normalize(value, role="worker_authorization_payload")
    if not isinstance(authorization, dict):
        _fail("worker_authorization_payload_invalid")
    schema = authorization.get("schema")
    if schema != WORKER_AUTHORIZATION_PAYLOAD_SCHEMA:
        if isinstance(schema, str) and schema.startswith(
            "aura.latent_cortex.worker_authorization_payload.v"
        ):
            _fail("worker_authorization_payload_version_incompatible")
        _fail("worker_authorization_payload_invalid")
    if set(authorization) != _AUTHORIZATION_KEYS:
        _fail("worker_authorization_payload_invalid")
    for role in ("campaign_name", "arm"):
        _identifier(authorization.get(role), role=role)
    _hex_identifier(authorization.get("session_id"), role="worker_session_id")
    _positive_int(
        authorization.get("supervisor_attempt"), role="worker_supervisor_attempt"
    )
    _positive_int(
        authorization.get("worker_attempt_slot"), role="worker_attempt_slot"
    )
    for role in (
        "policy_sha256",
        "protocol_sha256",
        "detached_plan_sha256",
        "broker_policy_sha256",
        "executable_binding_sha256",
        "environment_sha256",
        "sandbox_sha256",
        "source_manifest_sha256",
        "allowed_cell_digest",
        "model_identity_sha256",
        "adapter_identity_sha256",
    ):
        _sha256(authorization.get(role), role=role)
    custody = authorization.get("worker_key_custody")
    if custody not in _SUPPORTED_CUSTODY_CLASSES:
        _fail("worker_key_custody_invalid")
    _public_raw, key_id = _decode_public_key(
        authorization.get("worker_public_key_b64")
    )
    if authorization.get("worker_key_id") != key_id:
        _fail("worker_authorization_key_mismatch")
    return authorization


def validate_worker_authorization_payload(
    value: Mapping[str, Any],
    *,
    require_claim_proof_custody: bool = True,
) -> dict[str, Any]:
    """Validate a v4 payload and enforce claim-proof key custody by default."""

    if not isinstance(require_claim_proof_custody, bool):
        _fail("worker_claim_proof_custody_flag_invalid")
    authorization = _validated_authorization_payload(value)
    if (
        require_claim_proof_custody
        and authorization["worker_key_custody"]
        == WORKER_KEY_CUSTODY_PRODUCER_SOFTWARE
    ):
        _fail("worker_key_custody_claim_incompatible")
    return authorization


def build_worker_authorization_payload(
    *,
    campaign_name: str,
    policy_sha256: str,
    protocol_sha256: str,
    detached_plan_sha256: str,
    broker_policy_sha256: str,
    executable_binding_sha256: str,
    environment_sha256: str,
    sandbox_sha256: str,
    source_manifest_sha256: str,
    session_id: str,
    supervisor_attempt: int,
    arm: str,
    worker_attempt_slot: int,
    allowed_cell_digest: str,
    model_identity_sha256: str,
    adapter_identity_sha256: str,
    worker_key_custody: str,
    worker_public_key_raw: bytes,
) -> dict[str, Any]:
    """Build the exact externally authorized supervisor execution contract."""

    if not isinstance(worker_public_key_raw, bytes) or len(worker_public_key_raw) != 32:
        _fail("worker_public_key_invalid")
    payload = {
        "schema": WORKER_AUTHORIZATION_PAYLOAD_SCHEMA,
        "campaign_name": campaign_name,
        "policy_sha256": policy_sha256,
        "protocol_sha256": protocol_sha256,
        "detached_plan_sha256": detached_plan_sha256,
        "broker_policy_sha256": broker_policy_sha256,
        "executable_binding_sha256": executable_binding_sha256,
        "environment_sha256": environment_sha256,
        "sandbox_sha256": sandbox_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "session_id": session_id,
        "supervisor_attempt": supervisor_attempt,
        "arm": arm,
        "worker_attempt_slot": worker_attempt_slot,
        "allowed_cell_digest": allowed_cell_digest,
        "model_identity_sha256": model_identity_sha256,
        "adapter_identity_sha256": adapter_identity_sha256,
        "worker_key_custody": worker_key_custody,
        "worker_public_key_b64": base64.b64encode(worker_public_key_raw).decode(
            "ascii"
        ),
        "worker_key_id": _sha256_bytes(worker_public_key_raw),
    }
    return validate_worker_authorization_payload(
        payload,
        require_claim_proof_custody=False,
    )


def verify_worker_authorization(
    policy: VerifiedCampaignTrustPolicy,
    attestation: Mapping[str, Any],
    *,
    expected_payload: Mapping[str, Any],
    not_before_unix: int | None = None,
    not_after_unix: int | None = None,
    require_claim_proof_custody: bool = True,
) -> dict[str, Any]:
    """Verify exact CAMPAIGN_RUNNER authorization for a v4 supervisor key."""

    expected = validate_worker_authorization_payload(
        expected_payload,
        require_claim_proof_custody=require_claim_proof_custody,
    )
    if expected["policy_sha256"] != policy.policy_sha256:
        _fail("worker_authorization_policy_mismatch")
    try:
        return verify_role_attestation(
            policy,
            attestation,
            role=CAMPAIGN_RUNNER,
            expected_payload=expected,
            not_before_unix=not_before_unix,
            not_after_unix=not_after_unix,
        )
    except ValueError as exc:
        raise WorkerOriginError("worker_authorization_attestation_invalid") from exc


def validate_worker_result_body(
    result_body: Mapping[str, Any],
    *,
    expected_cell_id: str,
    expected_cell_type: str,
    expected_attempt_id: str,
    expected_session_id: str,
) -> dict[str, Any]:
    """Validate one typed result body's mandatory origin bindings."""

    cell_id = _identifier(expected_cell_id, role="expected_cell_id")
    cell_type = _identifier(expected_cell_type, role="expected_cell_type")
    attempt_id = _identifier(expected_attempt_id, role="expected_attempt_id")
    session_id = _hex_identifier(
        expected_session_id, role="expected_worker_session_id"
    )
    result = _normalize(result_body, role="worker_result_body")
    if (
        not isinstance(result, dict)
        or "worker_origin" in result
        or not _RESULT_BODY_BINDING_KEYS.issubset(result)
        or result.get("cell_id") != cell_id
        or result.get("cell_type") != cell_type
        or result.get("attempt_id") != attempt_id
        or result.get("origin_session_id") != session_id
    ):
        _fail("worker_result_body_binding_invalid")
    return result


def build_worker_result_signed_payload(
    *,
    authorization_attestation: Mapping[str, Any],
    authorization_payload: Mapping[str, Any],
    result_body: Mapping[str, Any],
    cell_id: str,
    cell_type: str,
    attempt_id: str,
    sequence: int,
    previous_origin_sha256: str = ZERO_SHA256,
) -> dict[str, Any]:
    """Build canonical bytes for one typed, ordered worker result."""

    authorization = validate_worker_authorization_payload(authorization_payload)
    sequence_value = _positive_int(sequence, role="worker_result_sequence")
    previous = _sha256(
        previous_origin_sha256, role="worker_result_previous_origin"
    )
    result = validate_worker_result_body(
        result_body,
        expected_cell_id=cell_id,
        expected_cell_type=cell_type,
        expected_attempt_id=attempt_id,
        expected_session_id=authorization["session_id"],
    )
    return {
        "schema": WORKER_RESULT_SIGNED_PAYLOAD_SCHEMA,
        "authorization_attestation_sha256": _sha256_bytes(
            canonical_json_bytes(authorization_attestation)
        ),
        "authorization_payload_sha256": _sha256_bytes(
            canonical_json_bytes(authorization)
        ),
        "detached_plan_sha256": authorization["detached_plan_sha256"],
        "session_id": authorization["session_id"],
        "supervisor_attempt": authorization["supervisor_attempt"],
        "arm": authorization["arm"],
        "worker_attempt_slot": authorization["worker_attempt_slot"],
        "cell_id": cell_id,
        "cell_type": cell_type,
        "attempt_id": attempt_id,
        "worker_key_id": authorization["worker_key_id"],
        "sequence": sequence_value,
        "previous_origin_sha256": previous,
        "result_body_sha256": _sha256_bytes(canonical_json_bytes(result)),
    }


def assemble_worker_result_origin(
    signed_payload: Mapping[str, Any],
    *,
    signature: bytes | str,
) -> dict[str, Any]:
    """Assemble an origin envelope from an externally produced signature."""

    payload = _normalize(signed_payload, role="worker_result_signed_payload")
    if (
        not isinstance(payload, dict)
        or set(payload) != _RESULT_PAYLOAD_KEYS
        or payload.get("schema") != WORKER_RESULT_SIGNED_PAYLOAD_SCHEMA
    ):
        _fail("worker_result_signed_payload_invalid")
    signature_raw = _signature_bytes(signature, role="worker_result_signature")
    signed_bytes = canonical_json_bytes(payload)
    material = {
        "schema": WORKER_RESULT_ORIGIN_SCHEMA,
        "signed_payload": payload,
        "signed_payload_sha256": _sha256_bytes(signed_bytes),
        "signature_b64": base64.b64encode(signature_raw).decode("ascii"),
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
    expected_cell_type: str,
    expected_attempt_id: str,
    expected_sequence: int,
    expected_previous_origin_sha256: str,
    authorization_not_before_unix: int | None = None,
    authorization_not_after_unix: int | None = None,
) -> dict[str, Any]:
    """Verify authorization, typed result signature, and chain position."""

    authorization = validate_worker_authorization_payload(
        expected_authorization_payload
    )
    verify_worker_authorization(
        policy,
        authorization_attestation,
        expected_payload=authorization,
        not_before_unix=authorization_not_before_unix,
        not_after_unix=authorization_not_after_unix,
    )
    normalized_result = _normalize(result, role="worker_result")
    if not isinstance(normalized_result, dict):
        _fail("worker_result_invalid")
    origin = normalized_result.get("worker_origin")
    if not isinstance(origin, dict) or set(origin) != _RESULT_ORIGIN_KEYS:
        _fail("worker_result_origin_invalid")
    if origin.get("schema") != WORKER_RESULT_ORIGIN_SCHEMA:
        _fail("worker_result_origin_version_incompatible")
    material = dict(origin)
    origin_sha256 = material.pop("origin_sha256", None)
    if origin_sha256 != _sha256_bytes(canonical_json_bytes(material)):
        _fail("worker_result_origin_digest_invalid")
    result_body = dict(normalized_result)
    result_body.pop("worker_origin", None)
    expected_payload = build_worker_result_signed_payload(
        authorization_attestation=authorization_attestation,
        authorization_payload=authorization,
        result_body=result_body,
        cell_id=expected_cell_id,
        cell_type=expected_cell_type,
        attempt_id=expected_attempt_id,
        sequence=expected_sequence,
        previous_origin_sha256=expected_previous_origin_sha256,
    )
    signed_payload = origin.get("signed_payload")
    if signed_payload != expected_payload:
        _fail("worker_result_binding_invalid")
    signed_bytes = canonical_json_bytes(signed_payload)
    if origin.get("signed_payload_sha256") != _sha256_bytes(signed_bytes):
        _fail("worker_result_signed_payload_digest_invalid")
    _verify_signature(
        public_key_b64=authorization["worker_public_key_b64"],
        signature_b64=origin.get("signature_b64"),
        signed_bytes=signed_bytes,
        role="worker_result_signature",
    )
    return dict(signed_payload)


def build_worker_lifecycle_event_payload(
    *,
    authorization_attestation: Mapping[str, Any] | None,
    authorization_payload: Mapping[str, Any],
    event_type: str,
    prior_state: str,
    result_count: int,
    previous_origin_sha256: str,
    completed_cell_ids: Sequence[str],
    occurred_at_unix: int,
    return_code: int | None,
    reason: str | None,
) -> dict[str, Any]:
    """Build a terminal or abandoned supervisor event payload."""

    authorization = validate_worker_authorization_payload(authorization_payload)
    event = _identifier(event_type, role="worker_lifecycle_event_type")
    state = _identifier(prior_state, role="worker_lifecycle_prior_state")
    if event not in _LIFECYCLE_EVENT_TYPES:
        _fail("worker_lifecycle_event_type_invalid")
    if state not in _LIFECYCLE_PRIOR_STATES:
        _fail("worker_lifecycle_prior_state_invalid")
    count = _nonnegative_int(result_count, role="worker_lifecycle_result_count")
    previous = _sha256(
        previous_origin_sha256, role="worker_lifecycle_previous_origin"
    )
    occurred = _positive_int(
        occurred_at_unix, role="worker_lifecycle_occurred_at_unix"
    )
    completed_digest = compute_completed_cell_digest(completed_cell_ids)
    if len(completed_cell_ids) != count:
        _fail("worker_lifecycle_result_count_mismatch")
    if event == "terminal":
        if state != "running":
            _fail("worker_lifecycle_terminal_state_invalid")
        if (
            isinstance(return_code, bool)
            or not isinstance(return_code, int)
            or return_code != 0
            or reason is not None
        ):
            _fail("worker_lifecycle_terminal_fields_invalid")
        if authorization_attestation is None:
            _fail("worker_lifecycle_authorization_missing")
    else:
        if return_code is not None:
            _fail("worker_lifecycle_abandoned_fields_invalid")
        _identifier(reason, role="worker_lifecycle_reason", maximum=2048)
    attestation_sha256 = (
        ZERO_SHA256
        if authorization_attestation is None
        else _sha256_bytes(canonical_json_bytes(authorization_attestation))
    )
    return {
        "schema": WORKER_LIFECYCLE_EVENT_PAYLOAD_SCHEMA,
        "authorization_attestation_sha256": attestation_sha256,
        "authorization_payload_sha256": _sha256_bytes(
            canonical_json_bytes(authorization)
        ),
        "detached_plan_sha256": authorization["detached_plan_sha256"],
        "session_id": authorization["session_id"],
        "supervisor_attempt": authorization["supervisor_attempt"],
        "arm": authorization["arm"],
        "worker_attempt_slot": authorization["worker_attempt_slot"],
        "worker_key_id": authorization["worker_key_id"],
        "event_type": event,
        "prior_state": state,
        "result_count": count,
        "previous_origin_sha256": previous,
        "completed_cell_digest": completed_digest,
        "occurred_at_unix": occurred,
        "return_code": return_code,
        "reason": reason,
    }


def assemble_worker_lifecycle_event_origin(
    signed_payload: Mapping[str, Any],
    *,
    signature: bytes | str,
) -> dict[str, Any]:
    """Assemble a signed terminal/abandoned lifecycle envelope."""

    payload = _normalize(signed_payload, role="worker_lifecycle_signed_payload")
    if (
        not isinstance(payload, dict)
        or set(payload) != _LIFECYCLE_PAYLOAD_KEYS
        or payload.get("schema") != WORKER_LIFECYCLE_EVENT_PAYLOAD_SCHEMA
    ):
        _fail("worker_lifecycle_signed_payload_invalid")
    signature_raw = _signature_bytes(signature, role="worker_lifecycle_signature")
    signed_bytes = canonical_json_bytes(payload)
    material = {
        "schema": WORKER_LIFECYCLE_EVENT_ORIGIN_SCHEMA,
        "signed_payload": payload,
        "signed_payload_sha256": _sha256_bytes(signed_bytes),
        "signature_b64": base64.b64encode(signature_raw).decode("ascii"),
    }
    return {
        **material,
        "event_sha256": _sha256_bytes(canonical_json_bytes(material)),
    }


def verify_worker_lifecycle_event_origin(
    *,
    policy: VerifiedCampaignTrustPolicy | None = None,
    authorization_payload: Mapping[str, Any],
    authorization_attestation: Mapping[str, Any] | None,
    event_origin: Mapping[str, Any],
    expected_event_type: str,
    expected_prior_state: str,
    expected_result_count: int,
    expected_previous_origin_sha256: str,
    expected_completed_cell_ids: Sequence[str],
    expected_occurred_at_unix: int,
    expected_return_code: int | None,
    expected_reason: str | None,
) -> dict[str, Any]:
    """Verify a supervisor terminal/abandoned event and exact bindings."""

    authorization = validate_worker_authorization_payload(authorization_payload)
    if authorization_attestation is not None:
        if policy is None:
            _fail("worker_lifecycle_authorization_policy_missing")
        verify_worker_authorization(
            policy,
            authorization_attestation,
            expected_payload=authorization,
        )
    expected = build_worker_lifecycle_event_payload(
        authorization_attestation=authorization_attestation,
        authorization_payload=authorization,
        event_type=expected_event_type,
        prior_state=expected_prior_state,
        result_count=expected_result_count,
        previous_origin_sha256=expected_previous_origin_sha256,
        completed_cell_ids=expected_completed_cell_ids,
        occurred_at_unix=expected_occurred_at_unix,
        return_code=expected_return_code,
        reason=expected_reason,
    )
    origin = _normalize(event_origin, role="worker_lifecycle_origin")
    if (
        not isinstance(origin, dict)
        or set(origin) != _LIFECYCLE_ORIGIN_KEYS
        or origin.get("schema") != WORKER_LIFECYCLE_EVENT_ORIGIN_SCHEMA
    ):
        _fail("worker_lifecycle_origin_invalid")
    material = dict(origin)
    event_sha256 = material.pop("event_sha256", None)
    if event_sha256 != _sha256_bytes(canonical_json_bytes(material)):
        _fail("worker_lifecycle_origin_digest_invalid")
    signed_payload = origin.get("signed_payload")
    if signed_payload != expected:
        _fail("worker_lifecycle_binding_invalid")
    signed_bytes = canonical_json_bytes(signed_payload)
    if origin.get("signed_payload_sha256") != _sha256_bytes(signed_bytes):
        _fail("worker_lifecycle_signed_payload_digest_invalid")
    _verify_signature(
        public_key_b64=authorization["worker_public_key_b64"],
        signature_b64=origin.get("signature_b64"),
        signed_bytes=signed_bytes,
        role="worker_lifecycle_signature",
    )
    return dict(signed_payload)


__all__ = [
    "MAX_WORKER_ALLOWED_CELLS",
    "MAX_WORKER_PROTOCOL_VALUE_BYTES",
    "WORKER_AUTHORIZATION_PAYLOAD_SCHEMA",
    "WORKER_RESULT_ORIGIN_SCHEMA",
    "WORKER_RESULT_SIGNED_PAYLOAD_SCHEMA",
    "WORKER_LIFECYCLE_EVENT_PAYLOAD_SCHEMA",
    "WORKER_LIFECYCLE_EVENT_ORIGIN_SCHEMA",
    "WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR",
    "WORKER_KEY_CUSTODY_PRODUCER_SOFTWARE",
    "ZERO_SHA256",
    "WorkerOriginError",
    "assemble_worker_lifecycle_event_origin",
    "assemble_worker_result_origin",
    "build_worker_authorization_payload",
    "build_worker_lifecycle_event_payload",
    "build_worker_result_signed_payload",
    "compute_allowed_cell_digest",
    "compute_completed_cell_digest",
    "validate_worker_authorization_payload",
    "validate_worker_result_body",
    "verify_worker_authorization",
    "verify_worker_lifecycle_event_origin",
    "verify_worker_result_origin",
]
