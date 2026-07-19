"""Externally rooted, role-separated trust for RLC evidence campaigns.

The evidence bundle is not allowed to choose its own trust anchors.  A caller
must supply the root public key independently; that root authenticates a
time-bounded policy which pins every campaign role before inference begins.
Role attestations then bind canonical payloads to those pre-authorized keys.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Never

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes

CAMPAIGN_TRUST_POLICY_SCHEMA = "aura.latent_cortex.campaign_trust_policy.v1"
CAMPAIGN_ROLE_ATTESTATION_SCHEMA = (
    "aura.latent_cortex.campaign_role_attestation.v1"
)
CAMPAIGN_ROLE_PAYLOAD_SCHEMA = "aura.latent_cortex.campaign_role_payload.v1"

TASK_ISSUER = "task_issuer"
CAMPAIGN_RUNNER = "campaign_runner"
CONTAMINATION_AUDITOR = "contamination_auditor"
EVIDENCE_VERIFIER = "evidence_verifier"
CAMPAIGN_TRUST_ROLES = (
    TASK_ISSUER,
    CAMPAIGN_RUNNER,
    CONTAMINATION_AUDITOR,
    EVIDENCE_VERIFIER,
)

_POLICY_KEYS = {
    "schema",
    "policy_id",
    "campaign_name",
    "issued_at_unix",
    "not_before_unix",
    "expires_at_unix",
    "roles",
    "root_signature",
}
_ROLE_PIN_KEYS = {
    "signer_id",
    "organization_id",
    "public_key_b64",
    "key_id",
    "implementation_sha256",
    "release_sha256",
}
_ROOT_SIGNATURE_KEYS = {
    "algorithm",
    "key_id",
    "signature_b64",
    "signed_payload_sha256",
}
_ATTESTATION_KEYS = {
    "schema",
    "signed_payload",
    "signed_payload_sha256",
    "signature_b64",
}
_SIGNED_PAYLOAD_KEYS = {
    "schema",
    "policy_sha256",
    "role",
    "signer_id",
    "signed_at_unix",
    "payload",
}


class CampaignTrustError(ValueError):
    """Stable fail-closed trust-policy or attestation error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise CampaignTrustError(code)


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
        or len(value) > 200
    ):
        _fail(f"{role}_invalid")
    return value


def _unix_time(value: Any, *, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{role}_invalid")
    return value


def _normalized_json(value: Any, *, role: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail(f"{role}_invalid")


def _decode_public_key(value: Any, *, role: str) -> tuple[bytes, str]:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(f"{role}_public_key_invalid")
    try:
        raw = base64.b64decode(value, validate=True)
    except (TypeError, ValueError, binascii.Error):
        _fail(f"{role}_public_key_invalid")
    if len(raw) != 32 or base64.b64encode(raw).decode("ascii") != value:
        _fail(f"{role}_public_key_invalid")
    return raw, hashlib.sha256(raw).hexdigest()


def load_ed25519_public_key(public_key_pem: bytes, *, role: str) -> tuple[Any, bytes, str]:
    """Load a PEM Ed25519 key and return the object, raw key, and key id."""

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        public_key = serialization.load_pem_public_key(public_key_pem)
    except (TypeError, ValueError):
        _fail(f"{role}_trust_root_invalid")
    if not isinstance(public_key, Ed25519PublicKey):
        _fail(f"{role}_trust_root_not_ed25519")
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return public_key, raw, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedCampaignTrustPolicy:
    """Normalized policy plus identities derived from verified bytes."""

    document: dict[str, Any]
    policy_sha256: str
    root_key_id: str

    def role_pin(self, role: str) -> dict[str, str]:
        if role not in CAMPAIGN_TRUST_ROLES:
            _fail("campaign_role_invalid")
        return dict(self.document["roles"][role])


def policy_signed_payload(policy_document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact policy material authenticated by the external root."""

    if not isinstance(policy_document, Mapping):
        _fail("campaign_trust_policy_invalid")
    return {
        key: policy_document[key]
        for key in (
            "schema",
            "policy_id",
            "campaign_name",
            "issued_at_unix",
            "not_before_unix",
            "expires_at_unix",
            "roles",
        )
        if key in policy_document
    }


def validate_campaign_trust_policy(
    raw: Any,
    *,
    trusted_root_public_key_pem: bytes,
    expected_campaign_name: str | None = None,
    now_unix: int | None = None,
) -> VerifiedCampaignTrustPolicy:
    """Authenticate one strict policy against a separately supplied root key."""

    if not isinstance(raw, Mapping) or set(raw) != _POLICY_KEYS:
        _fail("campaign_trust_policy_schema_invalid")
    document = _normalized_json(raw, role="campaign_trust_policy")
    if not isinstance(document, dict) or document.get("schema") != CAMPAIGN_TRUST_POLICY_SCHEMA:
        _fail("campaign_trust_policy_schema_invalid")
    _identifier(document.get("policy_id"), role="campaign_trust_policy_id")
    campaign_name = _identifier(
        document.get("campaign_name"), role="campaign_trust_campaign_name"
    )
    if expected_campaign_name is not None and campaign_name != expected_campaign_name:
        _fail("campaign_trust_campaign_name_mismatch")

    issued_at = _unix_time(
        document.get("issued_at_unix"), role="campaign_trust_issued_at"
    )
    not_before = _unix_time(
        document.get("not_before_unix"), role="campaign_trust_not_before"
    )
    expires_at = _unix_time(
        document.get("expires_at_unix"), role="campaign_trust_expires_at"
    )
    if issued_at > not_before or not_before >= expires_at:
        _fail("campaign_trust_validity_window_invalid")
    if now_unix is not None:
        observed = _unix_time(now_unix, role="campaign_trust_observed_at")
        if observed < not_before:
            _fail("campaign_trust_policy_not_yet_valid")
        if observed >= expires_at:
            _fail("campaign_trust_policy_expired")

    roles = document.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(CAMPAIGN_TRUST_ROLES):
        _fail("campaign_trust_roles_invalid")
    signer_ids: set[str] = set()
    organization_ids: set[str] = set()
    role_keys: set[bytes] = set()
    for role in CAMPAIGN_TRUST_ROLES:
        pin = roles.get(role)
        if not isinstance(pin, dict) or set(pin) != _ROLE_PIN_KEYS:
            _fail(f"campaign_trust_{role}_pin_invalid")
        signer_id = _identifier(pin.get("signer_id"), role=f"{role}_signer_id")
        organization_id = _identifier(
            pin.get("organization_id"), role=f"{role}_organization_id"
        )
        public_raw, key_id = _decode_public_key(pin.get("public_key_b64"), role=role)
        if pin.get("key_id") != key_id:
            _fail(f"campaign_trust_{role}_key_id_mismatch")
        if not _is_sha256(pin.get("implementation_sha256")):
            _fail(f"campaign_trust_{role}_implementation_invalid")
        if not _is_sha256(pin.get("release_sha256")):
            _fail(f"campaign_trust_{role}_release_invalid")
        if signer_id in signer_ids:
            _fail("campaign_trust_signer_identity_reused")
        if organization_id in organization_ids:
            _fail("campaign_trust_organization_reused")
        if public_raw in role_keys:
            _fail("campaign_trust_role_key_reused")
        signer_ids.add(signer_id)
        organization_ids.add(organization_id)
        role_keys.add(public_raw)

    root_signature = document.get("root_signature")
    if (
        not isinstance(root_signature, dict)
        or set(root_signature) != _ROOT_SIGNATURE_KEYS
        or root_signature.get("algorithm") != "Ed25519"
        or not isinstance(root_signature.get("signature_b64"), str)
    ):
        _fail("campaign_trust_root_signature_invalid")
    root_key, root_raw, root_key_id = load_ed25519_public_key(
        trusted_root_public_key_pem,
        role="campaign",
    )
    if root_raw in role_keys:
        _fail("campaign_trust_root_role_key_reused")
    if root_signature.get("key_id") != root_key_id:
        _fail("campaign_trust_root_key_mismatch")
    signed_payload = canonical_json_bytes(policy_signed_payload(document))
    signed_payload_sha256 = hashlib.sha256(signed_payload).hexdigest()
    if root_signature.get("signed_payload_sha256") != signed_payload_sha256:
        _fail("campaign_trust_root_payload_mismatch")
    try:
        signature = base64.b64decode(
            root_signature["signature_b64"], validate=True
        )
    except (TypeError, ValueError, binascii.Error):
        _fail("campaign_trust_root_signature_invalid")
    from cryptography.exceptions import InvalidSignature

    try:
        root_key.verify(signature, signed_payload)
    except InvalidSignature:
        _fail("campaign_trust_root_signature_invalid")

    return VerifiedCampaignTrustPolicy(
        document=document,
        policy_sha256=hashlib.sha256(canonical_json_bytes(document)).hexdigest(),
        root_key_id=root_key_id,
    )


def build_role_attestation(
    policy: VerifiedCampaignTrustPolicy,
    *,
    role: str,
    payload: Mapping[str, Any],
    signed_at_unix: int,
    private_key: Any,
) -> dict[str, Any]:
    """Sign one canonical role payload with the policy-pinned private key."""

    pin = policy.role_pin(role)
    signed_at = _unix_time(signed_at_unix, role="campaign_attestation_signed_at")
    normalized_payload = _normalized_json(payload, role="campaign_attestation_payload")
    if not isinstance(normalized_payload, dict):
        _fail("campaign_attestation_payload_invalid")
    public_key = private_key.public_key()
    from cryptography.hazmat.primitives import serialization

    public_raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if hashlib.sha256(public_raw).hexdigest() != pin["key_id"]:
        _fail("campaign_attestation_private_key_mismatch")
    signed_payload = {
        "schema": CAMPAIGN_ROLE_PAYLOAD_SCHEMA,
        "policy_sha256": policy.policy_sha256,
        "role": role,
        "signer_id": pin["signer_id"],
        "signed_at_unix": signed_at,
        "payload": normalized_payload,
    }
    signed_bytes = canonical_json_bytes(signed_payload)
    return {
        "schema": CAMPAIGN_ROLE_ATTESTATION_SCHEMA,
        "signed_payload": signed_payload,
        "signed_payload_sha256": hashlib.sha256(signed_bytes).hexdigest(),
        "signature_b64": base64.b64encode(private_key.sign(signed_bytes)).decode(
            "ascii"
        ),
    }


def verify_role_attestation(
    policy: VerifiedCampaignTrustPolicy,
    raw: Any,
    *,
    role: str,
    expected_payload: Mapping[str, Any],
    not_before_unix: int | None = None,
    not_after_unix: int | None = None,
) -> dict[str, Any]:
    """Verify a role envelope and return its normalized signed payload."""

    if role not in CAMPAIGN_TRUST_ROLES:
        _fail("campaign_role_invalid")
    if not isinstance(raw, Mapping) or set(raw) != _ATTESTATION_KEYS:
        _fail("campaign_attestation_schema_invalid")
    attestation = _normalized_json(raw, role="campaign_attestation")
    if (
        not isinstance(attestation, dict)
        or attestation.get("schema") != CAMPAIGN_ROLE_ATTESTATION_SCHEMA
    ):
        _fail("campaign_attestation_schema_invalid")
    signed_payload = attestation.get("signed_payload")
    if not isinstance(signed_payload, dict) or set(signed_payload) != _SIGNED_PAYLOAD_KEYS:
        _fail("campaign_attestation_payload_schema_invalid")
    pin = policy.role_pin(role)
    if (
        signed_payload.get("schema") != CAMPAIGN_ROLE_PAYLOAD_SCHEMA
        or signed_payload.get("policy_sha256") != policy.policy_sha256
        or signed_payload.get("role") != role
        or signed_payload.get("signer_id") != pin["signer_id"]
    ):
        _fail("campaign_attestation_identity_mismatch")
    expected = _normalized_json(expected_payload, role="campaign_expected_payload")
    if signed_payload.get("payload") != expected:
        _fail("campaign_attestation_payload_mismatch")
    signed_at = _unix_time(
        signed_payload.get("signed_at_unix"), role="campaign_attestation_signed_at"
    )
    if not policy.document["not_before_unix"] <= signed_at < policy.document["expires_at_unix"]:
        _fail("campaign_attestation_outside_policy_window")
    if not_before_unix is not None and signed_at < _unix_time(
        not_before_unix, role="campaign_attestation_minimum_time"
    ):
        _fail("campaign_attestation_too_early")
    if not_after_unix is not None and signed_at > _unix_time(
        not_after_unix, role="campaign_attestation_maximum_time"
    ):
        _fail("campaign_attestation_too_late")

    signed_bytes = canonical_json_bytes(signed_payload)
    if attestation.get("signed_payload_sha256") != hashlib.sha256(
        signed_bytes
    ).hexdigest():
        _fail("campaign_attestation_digest_mismatch")
    try:
        signature = base64.b64decode(
            str(attestation.get("signature_b64")), validate=True
        )
    except (TypeError, ValueError, binascii.Error):
        _fail("campaign_attestation_signature_invalid")
    public_raw, _key_id = _decode_public_key(pin["public_key_b64"], role=role)
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(public_raw).verify(signature, signed_bytes)
    except InvalidSignature:
        _fail("campaign_attestation_signature_invalid")
    return dict(signed_payload)


__all__ = [
    "CAMPAIGN_ROLE_ATTESTATION_SCHEMA",
    "CAMPAIGN_ROLE_PAYLOAD_SCHEMA",
    "CAMPAIGN_RUNNER",
    "CAMPAIGN_TRUST_POLICY_SCHEMA",
    "CAMPAIGN_TRUST_ROLES",
    "CONTAMINATION_AUDITOR",
    "CampaignTrustError",
    "EVIDENCE_VERIFIER",
    "TASK_ISSUER",
    "VerifiedCampaignTrustPolicy",
    "build_role_attestation",
    "load_ed25519_public_key",
    "policy_signed_payload",
    "validate_campaign_trust_policy",
    "verify_role_attestation",
]
