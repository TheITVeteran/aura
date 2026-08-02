"""Cryptographic authority binding for persistent physical attachments."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Never, Protocol, runtime_checkable

from core.governance.capability_chain import (
    CapabilityVerifier,
    SignedCapability,
    capability_chain_status,
    compute_action_digest,
    get_capability_verifier,
)

ATTACHMENT_AUTHORITY_ACTION = "reality_attachment.authorize"
ATTACHMENT_AUTHORITY_SCHEMA = "aura.reality-attachment-authority.intent.v1"
ATTACHMENT_AUTHORITY_EVIDENCE_SCHEMA = "aura.reality-attachment-authority.evidence.v1"
ATTACHMENT_AUTHORITY_DOMAIN = "environment_action"


class AttachmentAuthorityError(PermissionError):
    """Stable fail-closed error for attachment authority verification."""


def _fail(code: str) -> Never:
    raise AttachmentAuthorityError(code)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AttachmentAuthorityError("attachment_authority_json_invalid") from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _scope_for_access(access: tuple[str, ...]) -> str:
    return (
        "reality_attachment.control"
        if "control" in access
        else "reality_attachment.observe"
    )


def build_attachment_authority_intent(
    *,
    request_id: str,
    candidate_sha256: str,
    identity_fingerprint: str,
    connector_id: str,
    manifest_sha256: str,
    requested_access: tuple[str, ...],
    persistent: bool,
    grant_ttl_s: int,
) -> dict[str, Any]:
    """Build the exact payload a Will capability must authorize."""

    access = tuple(sorted(dict.fromkeys(str(item) for item in requested_access)))
    if not access or any(item not in {"observe", "control"} for item in access):
        raise ValueError("attachment authority access is invalid")
    if isinstance(grant_ttl_s, bool) or not isinstance(grant_ttl_s, int) or grant_ttl_s <= 0:
        raise ValueError("attachment authority grant lifetime is invalid")
    return {
        "schema": ATTACHMENT_AUTHORITY_SCHEMA,
        "action": ATTACHMENT_AUTHORITY_ACTION,
        "request_id": str(request_id),
        "candidate_sha256": str(candidate_sha256),
        "identity_fingerprint": str(identity_fingerprint),
        "connector_id": str(connector_id),
        "manifest_sha256": str(manifest_sha256),
        "requested_access": list(access),
        "persistent": bool(persistent),
        "grant_ttl_s": grant_ttl_s,
        "scope": _scope_for_access(access),
    }


@runtime_checkable
class WillReceiptSource(Protocol):
    def verify_receipt_signature(self, receipt_id: str) -> bool: ...

    def get_receipt_verification_material(self, receipt_id: str) -> dict[str, Any]: ...


@runtime_checkable
class PhysicalAuthorityVerifier(Protocol):
    def verify(
        self,
        capability: Mapping[str, Any] | SignedCapability,
        *,
        intent: Mapping[str, Any],
        persistent: bool,
    ) -> Mapping[str, Any]: ...

    def validate_persisted(
        self,
        evidence: Mapping[str, Any],
        *,
        intent: Mapping[str, Any],
        persistent: bool,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class AttachmentAuthorityPolicy:
    require_asymmetric_persistent_authority: bool = True
    accepted_will_signature_schemes: tuple[str, ...] = ("ed25519", "hmac-sha256")
    maximum_clock_skew_s: float = 5.0


class AttachmentCapabilityAuthorityVerifier:
    """Verify one-use Will capabilities at the attachment sink.

    The capability verifier establishes signature, expiry, exact action digest,
    domain, and durable nonce consumption.  This verifier additionally binds
    physical scope and independently checks the originating Will receipt before
    returning evidence suitable for encrypted persistence.
    """

    def __init__(
        self,
        *,
        capability_verifier: CapabilityVerifier | None = None,
        will_receipts: WillReceiptSource | None = None,
        policy: AttachmentAuthorityPolicy | None = None,
    ) -> None:
        self._capability_verifier = capability_verifier or get_capability_verifier()
        if will_receipts is None:
            from core.governance.will import get_will

            will_receipts = get_will()
        if not isinstance(will_receipts, WillReceiptSource):
            raise TypeError("will_receipts must satisfy WillReceiptSource")
        self._will_receipts = will_receipts
        self._policy = policy or AttachmentAuthorityPolicy()

    @staticmethod
    def _expected(intent: Mapping[str, Any]) -> tuple[str, str]:
        payload = dict(intent)
        if payload.get("schema") != ATTACHMENT_AUTHORITY_SCHEMA:
            _fail("attachment_authority_intent_schema_invalid")
        scope = str(payload.get("scope") or "")
        access = payload.get("requested_access")
        if not isinstance(access, list) or scope != _scope_for_access(tuple(str(x) for x in access)):
            _fail("attachment_authority_scope_invalid")
        return compute_action_digest(ATTACHMENT_AUTHORITY_ACTION, payload), scope

    def _verify_will_material(
        self,
        receipt_id: str,
        *,
        capability: SignedCapability,
    ) -> dict[str, Any]:
        if not receipt_id or not self._will_receipts.verify_receipt_signature(receipt_id):
            _fail("attachment_authority_will_receipt_unverified")
        material = self._will_receipts.get_receipt_verification_material(receipt_id)
        if not isinstance(material, dict):
            _fail("attachment_authority_will_material_missing")
        try:
            payload = json.loads(str(material.get("payload") or ""))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise AttachmentAuthorityError("attachment_authority_will_material_invalid") from exc
        scheme = str(material.get("signature_scheme") or "")
        if (
            not isinstance(payload, dict)
            or payload.get("receipt_id") != receipt_id
            or str(payload.get("outcome") or "").lower() != capability.outcome
            or str(payload.get("domain") or "").lower() != capability.domain
            or scheme not in self._policy.accepted_will_signature_schemes
            or not str(material.get("signature") or "")
        ):
            _fail("attachment_authority_will_material_invalid")
        return {
            "receipt_id": receipt_id,
            "payload": str(material["payload"]),
            "signature": str(material["signature"]),
            "signature_scheme": scheme,
            "material_sha256": _digest(
                {
                    "receipt_id": receipt_id,
                    "payload": str(material["payload"]),
                    "signature": str(material["signature"]),
                    "signature_scheme": scheme,
                }
            ),
        }

    def verify(
        self,
        capability: Mapping[str, Any] | SignedCapability,
        *,
        intent: Mapping[str, Any],
        persistent: bool,
    ) -> Mapping[str, Any]:
        action_digest, scope = self._expected(intent)
        preflight = self._capability_verifier.verify(
            capability,
            expected_domain=ATTACHMENT_AUTHORITY_DOMAIN,
            expected_action_digest=action_digest,
            consume=False,
        )
        if not preflight.ok or preflight.capability is None:
            denial = preflight.denial.value if preflight.denial is not None else "invalid"
            _fail(f"attachment_authority_capability_{denial}")
        verified = preflight.capability
        if verified.scope != scope:
            _fail("attachment_authority_capability_scope_mismatch")
        authority_status = capability_chain_status()
        if persistent and self._policy.require_asymmetric_persistent_authority:
            if not bool(authority_status.get("asymmetric")) or not bool(
                authority_status.get("authority_durable")
            ):
                _fail("attachment_authority_persistent_root_not_durable")
        will_material = self._verify_will_material(
            verified.receipt_id,
            capability=verified,
        )
        consumed = self._capability_verifier.verify(
            capability,
            expected_domain=ATTACHMENT_AUTHORITY_DOMAIN,
            expected_action_digest=action_digest,
            consume=True,
        )
        if not consumed.ok or consumed.capability is None:
            denial = consumed.denial.value if consumed.denial is not None else "invalid"
            _fail(f"attachment_authority_capability_{denial}")
        if consumed.capability.capability_id != verified.capability_id:
            _fail("attachment_authority_capability_identity_changed")
        verified = consumed.capability
        verified_at_ns = max(1, time.time_ns())
        body = {
            "schema": ATTACHMENT_AUTHORITY_EVIDENCE_SCHEMA,
            "intent_sha256": _digest(dict(intent)),
            "action_digest": action_digest,
            "scope": scope,
            "persistent": bool(persistent),
            "verified_at_ns": verified_at_ns,
            "authority_durable": bool(authority_status.get("authority_durable")),
            "capability": verified.to_dict(),
            "will_receipt": will_material,
        }
        return {**body, "evidence_sha256": _digest(body)}

    def validate_persisted(
        self,
        evidence: Mapping[str, Any],
        *,
        intent: Mapping[str, Any],
        persistent: bool,
    ) -> Mapping[str, Any]:
        value = dict(evidence)
        expected_fields = {
            "schema",
            "intent_sha256",
            "action_digest",
            "scope",
            "persistent",
            "verified_at_ns",
            "authority_durable",
            "capability",
            "will_receipt",
            "evidence_sha256",
        }
        body = {key: item for key, item in value.items() if key != "evidence_sha256"}
        action_digest, scope = self._expected(intent)
        verified_at_ns = value.get("verified_at_ns")
        if (
            set(value) != expected_fields
            or value.get("schema") != ATTACHMENT_AUTHORITY_EVIDENCE_SCHEMA
            or value.get("intent_sha256") != _digest(dict(intent))
            or value.get("action_digest") != action_digest
            or value.get("scope") != scope
            or value.get("persistent") is not bool(persistent)
            or not isinstance(verified_at_ns, int)
            or isinstance(verified_at_ns, bool)
            or verified_at_ns <= 0
            or verified_at_ns > time.time_ns() + int(self._policy.maximum_clock_skew_s * 1e9)
            or not isinstance(value.get("evidence_sha256"), str)
            or not hmac.compare_digest(str(value["evidence_sha256"]), _digest(body))
        ):
            _fail("attachment_authority_evidence_invalid")
        capability = value.get("capability")
        result = self._capability_verifier.verify(
            capability,
            expected_domain=ATTACHMENT_AUTHORITY_DOMAIN,
            expected_action_digest=action_digest,
            consume=False,
            now=float(verified_at_ns) / 1e9,
        )
        if not result.ok or result.capability is None:
            _fail("attachment_authority_persisted_capability_invalid")
        verified = result.capability
        if verified.scope != scope:
            _fail("attachment_authority_persisted_scope_mismatch")
        will_material = value.get("will_receipt")
        if not isinstance(will_material, dict):
            _fail("attachment_authority_persisted_will_material_invalid")
        material_body = {
            "receipt_id": will_material.get("receipt_id"),
            "payload": will_material.get("payload"),
            "signature": will_material.get("signature"),
            "signature_scheme": will_material.get("signature_scheme"),
        }
        try:
            will_payload = json.loads(str(will_material.get("payload") or ""))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise AttachmentAuthorityError(
                "attachment_authority_persisted_will_material_invalid"
            ) from exc
        if (
            will_material.get("receipt_id") != verified.receipt_id
            or not isinstance(will_payload, dict)
            or will_payload.get("receipt_id") != verified.receipt_id
            or str(will_payload.get("outcome") or "").lower() != verified.outcome
            or str(will_payload.get("domain") or "").lower() != verified.domain
            or will_material.get("signature_scheme")
            not in self._policy.accepted_will_signature_schemes
            or not str(will_material.get("signature") or "")
            or will_material.get("material_sha256") != _digest(material_body)
        ):
            _fail("attachment_authority_persisted_will_material_invalid")
        if persistent and self._policy.require_asymmetric_persistent_authority:
            if not bool(value.get("authority_durable")) or not verified.key_id.startswith(
                "ed25519-"
            ):
                _fail("attachment_authority_persisted_root_not_durable")
        return value


__all__ = [
    "ATTACHMENT_AUTHORITY_ACTION",
    "ATTACHMENT_AUTHORITY_DOMAIN",
    "ATTACHMENT_AUTHORITY_EVIDENCE_SCHEMA",
    "ATTACHMENT_AUTHORITY_SCHEMA",
    "AttachmentAuthorityError",
    "AttachmentAuthorityPolicy",
    "AttachmentCapabilityAuthorityVerifier",
    "PhysicalAuthorityVerifier",
    "WillReceiptSource",
    "build_attachment_authority_intent",
]
