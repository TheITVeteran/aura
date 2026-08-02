"""Cryptographic authority binding for persistent physical attachments."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
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
    get_nonce_ledger,
)

ATTACHMENT_AUTHORITY_ACTION = "reality_attachment.authorize"
ATTACHMENT_AUTHORITY_SCHEMA = "aura.reality-attachment-authority.intent.v1"
ATTACHMENT_AUTHORITY_EVIDENCE_SCHEMA = "aura.reality-attachment-authority.evidence.v1"
ATTACHMENT_AUTHORITY_DOMAIN = "environment_action"
MANIFEST_MIGRATION_AUTHORITY_ACTION = "reality_attachment.migrate_manifest"
MANIFEST_MIGRATION_AUTHORITY_SCHEMA = (
    "aura.reality-manifest-migration-authority.intent.v1"
)
MANIFEST_MIGRATION_AUTHORITY_EVIDENCE_SCHEMA = (
    "aura.reality-manifest-migration-authority.evidence.v1"
)
MANIFEST_MIGRATION_AUTHORITY_SCOPE = "reality_attachment.manifest_migration"

_CANONICAL_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MANIFEST_MIGRATION_INTENT_FIELDS = frozenset(
    {
        "schema",
        "action",
        "request_id",
        "identity_fingerprint",
        "connector_id",
        "expected_manifest_sha256",
        "new_manifest_sha256",
        "persistent",
        "scope",
    }
)


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


def _canonical_identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _CANONICAL_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a canonical identifier of at most 128 characters")
    return value


def _sha256_digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_DIGEST.fullmatch(value):
        raise ValueError(f"{name} must be a canonical sha256 digest")
    return value


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


def build_manifest_migration_authority_intent(
    *,
    request_id: str,
    identity_fingerprint: str,
    connector_id: str,
    expected_manifest_sha256: str,
    new_manifest_sha256: str,
    persistent: bool,
) -> dict[str, Any]:
    """Build the exact compare-and-swap transition a capability authorizes."""

    request = _canonical_identifier(request_id, name="request_id")
    identity = _sha256_digest(identity_fingerprint, name="identity_fingerprint")
    connector = _canonical_identifier(connector_id, name="connector_id")
    expected_manifest = _sha256_digest(
        expected_manifest_sha256,
        name="expected_manifest_sha256",
    )
    new_manifest = _sha256_digest(new_manifest_sha256, name="new_manifest_sha256")
    if expected_manifest == new_manifest:
        raise ValueError("manifest migration must change the manifest")
    if not isinstance(persistent, bool):
        raise TypeError("manifest migration persistence must be boolean")
    return {
        "schema": MANIFEST_MIGRATION_AUTHORITY_SCHEMA,
        "action": MANIFEST_MIGRATION_AUTHORITY_ACTION,
        "request_id": request,
        "identity_fingerprint": identity,
        "connector_id": connector,
        "expected_manifest_sha256": expected_manifest,
        "new_manifest_sha256": new_manifest,
        "persistent": persistent,
        "scope": MANIFEST_MIGRATION_AUTHORITY_SCOPE,
    }


def _manifest_migration_binding(intent: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(intent)
    if set(payload) != _MANIFEST_MIGRATION_INTENT_FIELDS:
        _fail("manifest_migration_authority_intent_shape_invalid")
    if (
        payload.get("schema") != MANIFEST_MIGRATION_AUTHORITY_SCHEMA
        or payload.get("action") != MANIFEST_MIGRATION_AUTHORITY_ACTION
        or payload.get("scope") != MANIFEST_MIGRATION_AUTHORITY_SCOPE
    ):
        _fail("manifest_migration_authority_intent_contract_invalid")
    try:
        request_id = _canonical_identifier(payload.get("request_id"), name="request_id")
        identity = _sha256_digest(
            payload.get("identity_fingerprint"),
            name="identity_fingerprint",
        )
        connector_id = _canonical_identifier(
            payload.get("connector_id"),
            name="connector_id",
        )
        expected_manifest = _sha256_digest(
            payload.get("expected_manifest_sha256"),
            name="expected_manifest_sha256",
        )
        new_manifest = _sha256_digest(
            payload.get("new_manifest_sha256"),
            name="new_manifest_sha256",
        )
    except (TypeError, ValueError) as exc:
        raise AttachmentAuthorityError(
            "manifest_migration_authority_intent_binding_invalid"
        ) from exc
    persistent = payload.get("persistent")
    if not isinstance(persistent, bool):
        _fail("manifest_migration_authority_persistence_invalid")
    if expected_manifest == new_manifest:
        _fail("manifest_migration_authority_transition_is_noop")
    return {
        "request_id": request_id,
        "identity_fingerprint": identity,
        "connector_id": connector_id,
        "expected_manifest_sha256": expected_manifest,
        "new_manifest_sha256": new_manifest,
        "persistent": persistent,
    }


@runtime_checkable
class WillReceiptSource(Protocol):
    def verify_receipt_signature(self, receipt_id: str) -> bool: ...

    def get_receipt_verification_material(self, receipt_id: str) -> dict[str, Any]: ...

    def verify_persisted_receipt_material(self, material: Mapping[str, Any]) -> bool: ...


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


@runtime_checkable
class ManifestMigrationAuthorityVerifier(Protocol):
    def verify_manifest_migration(
        self,
        capability: Mapping[str, Any] | SignedCapability,
        *,
        intent: Mapping[str, Any],
        persistent: bool,
    ) -> Mapping[str, Any]: ...

    def validate_persisted_manifest_migration(
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

    @staticmethod
    def _expected_manifest_migration(
        intent: Mapping[str, Any],
    ) -> tuple[str, str, dict[str, Any]]:
        payload = dict(intent)
        binding = _manifest_migration_binding(payload)
        return (
            compute_action_digest(MANIFEST_MIGRATION_AUTHORITY_ACTION, payload),
            MANIFEST_MIGRATION_AUTHORITY_SCOPE,
            binding,
        )

    def _verify_will_material(
        self,
        receipt_id: str,
        *,
        capability: SignedCapability,
        persistent: bool,
    ) -> dict[str, Any]:
        if not receipt_id or not self._will_receipts.verify_receipt_signature(receipt_id):
            _fail("attachment_authority_will_receipt_unverified")
        material = self._will_receipts.get_receipt_verification_material(receipt_id)
        expected_fields = {
            "receipt_id",
            "payload",
            "signature",
            "signature_scheme",
            "signature_key_id",
            "trust_root_durable",
        }
        if not isinstance(material, dict) or set(material) != expected_fields:
            _fail("attachment_authority_will_material_missing")
        return self._validate_will_material(
            material,
            capability=capability,
            persistent=persistent,
            stored=False,
            failure_code="attachment_authority_will_material_invalid",
        )

    def _validate_will_material(
        self,
        material: Mapping[str, Any],
        *,
        capability: SignedCapability,
        persistent: bool,
        stored: bool,
        failure_code: str,
    ) -> dict[str, Any]:
        value = dict(material)
        body_fields = {
            "receipt_id",
            "payload",
            "signature",
            "signature_scheme",
            "signature_key_id",
            "trust_root_durable",
        }
        expected_fields = body_fields | ({"material_sha256"} if stored else set())
        if set(value) != expected_fields:
            _fail(failure_code)
        body = {key: value[key] for key in body_fields}
        try:
            payload = json.loads(str(value.get("payload") or ""))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise AttachmentAuthorityError(failure_code) from exc
        scheme = str(value.get("signature_scheme") or "")
        key_id = str(value.get("signature_key_id") or "")
        root_durable = value.get("trust_root_durable")
        if (
            not isinstance(payload, dict)
            or value.get("receipt_id") != capability.receipt_id
            or payload.get("receipt_id") != capability.receipt_id
            or str(payload.get("outcome") or "").lower() != capability.outcome
            or str(payload.get("domain") or "").lower() != capability.domain
            or scheme not in self._policy.accepted_will_signature_schemes
            or not str(value.get("signature") or "")
            or not key_id
            or not isinstance(root_durable, bool)
            or (stored and value.get("material_sha256") != _digest(body))
            or not self._will_receipts.verify_persisted_receipt_material(value)
        ):
            _fail(failure_code)
        if persistent and self._policy.require_asymmetric_persistent_authority:
            if scheme != "ed25519" or not key_id.startswith("ed25519-") or root_durable is not True:
                _fail(failure_code)
        return {**body, "material_sha256": _digest(body)}

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
            persistent=persistent,
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
        if not isinstance(will_material, Mapping):
            _fail("attachment_authority_persisted_will_material_invalid")
        self._validate_will_material(
            will_material,
            capability=verified,
            persistent=persistent,
            stored=True,
            failure_code="attachment_authority_persisted_will_material_invalid",
        )
        if persistent and self._policy.require_asymmetric_persistent_authority:
            if not bool(value.get("authority_durable")) or not verified.key_id.startswith(
                "ed25519-"
            ):
                _fail("attachment_authority_persisted_root_not_durable")
        return value

    def verify_manifest_migration(
        self,
        capability: Mapping[str, Any] | SignedCapability,
        *,
        intent: Mapping[str, Any],
        persistent: bool,
    ) -> Mapping[str, Any]:
        """Consume authority for one exact manifest compare-and-swap transition."""

        action_digest, scope, binding = self._expected_manifest_migration(intent)
        if not isinstance(persistent, bool) or binding["persistent"] is not persistent:
            _fail("manifest_migration_authority_persistence_mismatch")
        preflight = self._capability_verifier.verify(
            capability,
            expected_domain=ATTACHMENT_AUTHORITY_DOMAIN,
            expected_action_digest=action_digest,
            consume=False,
        )
        if not preflight.ok or preflight.capability is None:
            denial = preflight.denial.value if preflight.denial is not None else "invalid"
            _fail(f"manifest_migration_authority_capability_{denial}")
        verified = preflight.capability
        if verified.scope != scope:
            _fail("manifest_migration_authority_capability_scope_mismatch")

        authority_status = capability_chain_status()
        if persistent and self._policy.require_asymmetric_persistent_authority:
            if not bool(authority_status.get("asymmetric")) or not bool(
                authority_status.get("authority_durable")
            ):
                _fail("manifest_migration_authority_persistent_root_not_durable")
        will_material = self._verify_will_material(
            verified.receipt_id,
            capability=verified,
            persistent=persistent,
        )

        consumed = self._capability_verifier.verify(
            capability,
            expected_domain=ATTACHMENT_AUTHORITY_DOMAIN,
            expected_action_digest=action_digest,
            consume=True,
        )
        if not consumed.ok or consumed.capability is None:
            denial = consumed.denial.value if consumed.denial is not None else "invalid"
            _fail(f"manifest_migration_authority_capability_{denial}")
        if consumed.capability.capability_id != verified.capability_id:
            _fail("manifest_migration_authority_capability_identity_changed")
        verified = consumed.capability

        verified_at_ns = max(1, time.time_ns())
        expires_at_ns = int(verified.expires_at * 1_000_000_000)
        if verified_at_ns >= expires_at_ns:
            _fail("manifest_migration_authority_capability_expired")
        nonce_ledger = get_nonce_ledger()
        nonce_seen = nonce_ledger.seen(verified.nonce)
        nonce_status = nonce_ledger.status()
        if not nonce_seen or not bool(nonce_status.get("healthy")):
            _fail("manifest_migration_authority_nonce_not_durable")

        nonce_consumption = {
            "capability_id": verified.capability_id,
            "nonce_sha256": _digest({"nonce": verified.nonce}),
            "consumed": True,
            "ledger_durable": True,
            "checked_at_ns": verified_at_ns,
            "expires_at_ns": expires_at_ns,
        }
        body = {
            "schema": MANIFEST_MIGRATION_AUTHORITY_EVIDENCE_SCHEMA,
            "intent_sha256": _digest(dict(intent)),
            "migration": binding,
            "action_digest": action_digest,
            "scope": scope,
            "persistent": persistent,
            "verified_at_ns": verified_at_ns,
            "capability_expires_at_ns": expires_at_ns,
            "authority_durable": bool(authority_status.get("authority_durable")),
            "nonce_consumption": nonce_consumption,
            "capability": verified.to_dict(),
            "will_receipt": will_material,
        }
        return {**body, "evidence_sha256": _digest(body)}

    def validate_persisted_manifest_migration(
        self,
        evidence: Mapping[str, Any],
        *,
        intent: Mapping[str, Any],
        persistent: bool,
    ) -> Mapping[str, Any]:
        """Validate stored migration evidence without consuming authority again."""

        value = dict(evidence)
        expected_fields = {
            "schema",
            "intent_sha256",
            "migration",
            "action_digest",
            "scope",
            "persistent",
            "verified_at_ns",
            "capability_expires_at_ns",
            "authority_durable",
            "nonce_consumption",
            "capability",
            "will_receipt",
            "evidence_sha256",
        }
        body = {key: item for key, item in value.items() if key != "evidence_sha256"}
        action_digest, scope, binding = self._expected_manifest_migration(intent)
        verified_at_ns = value.get("verified_at_ns")
        capability_expires_at_ns = value.get("capability_expires_at_ns")
        if (
            not isinstance(persistent, bool)
            or binding["persistent"] is not persistent
            or set(value) != expected_fields
            or value.get("schema") != MANIFEST_MIGRATION_AUTHORITY_EVIDENCE_SCHEMA
            or value.get("intent_sha256") != _digest(dict(intent))
            or value.get("migration") != binding
            or value.get("action_digest") != action_digest
            or value.get("scope") != scope
            or value.get("persistent") is not persistent
            or not isinstance(verified_at_ns, int)
            or isinstance(verified_at_ns, bool)
            or verified_at_ns <= 0
            or verified_at_ns > time.time_ns() + int(self._policy.maximum_clock_skew_s * 1e9)
            or not isinstance(capability_expires_at_ns, int)
            or isinstance(capability_expires_at_ns, bool)
            or capability_expires_at_ns <= verified_at_ns
            or not isinstance(value.get("evidence_sha256"), str)
            or not hmac.compare_digest(str(value["evidence_sha256"]), _digest(body))
        ):
            _fail("manifest_migration_authority_evidence_invalid")

        result = self._capability_verifier.verify(
            value.get("capability"),
            expected_domain=ATTACHMENT_AUTHORITY_DOMAIN,
            expected_action_digest=action_digest,
            consume=False,
            now=float(verified_at_ns) / 1e9,
        )
        if not result.ok or result.capability is None:
            _fail("manifest_migration_authority_persisted_capability_invalid")
        verified = result.capability
        if verified.scope != scope:
            _fail("manifest_migration_authority_persisted_scope_mismatch")
        expected_expires_at_ns = int(verified.expires_at * 1_000_000_000)
        if capability_expires_at_ns != expected_expires_at_ns:
            _fail("manifest_migration_authority_persisted_expiry_mismatch")

        nonce_consumption = value.get("nonce_consumption")
        expected_nonce_fields = {
            "capability_id",
            "nonce_sha256",
            "consumed",
            "ledger_durable",
            "checked_at_ns",
            "expires_at_ns",
        }
        if (
            not isinstance(nonce_consumption, dict)
            or set(nonce_consumption) != expected_nonce_fields
            or nonce_consumption.get("capability_id") != verified.capability_id
            or nonce_consumption.get("nonce_sha256") != _digest({"nonce": verified.nonce})
            or nonce_consumption.get("consumed") is not True
            or nonce_consumption.get("ledger_durable") is not True
            or nonce_consumption.get("checked_at_ns") != verified_at_ns
            or nonce_consumption.get("expires_at_ns") != expected_expires_at_ns
        ):
            _fail("manifest_migration_authority_persisted_nonce_invalid")
        if time.time() < verified.expires_at:
            nonce_ledger = get_nonce_ledger()
            nonce_seen = nonce_ledger.seen(verified.nonce)
            nonce_status = nonce_ledger.status()
            if not nonce_seen or not bool(nonce_status.get("healthy")):
                _fail("manifest_migration_authority_persisted_nonce_missing")

        will_material = value.get("will_receipt")
        if not isinstance(will_material, Mapping):
            _fail("manifest_migration_authority_persisted_will_material_invalid")
        self._validate_will_material(
            will_material,
            capability=verified,
            persistent=persistent,
            stored=True,
            failure_code="manifest_migration_authority_persisted_will_material_invalid",
        )

        if persistent and self._policy.require_asymmetric_persistent_authority:
            authority_status = capability_chain_status()
            if (
                value.get("authority_durable") is not True
                or not bool(authority_status.get("asymmetric"))
                or not bool(authority_status.get("authority_durable"))
                or not verified.key_id.startswith("ed25519-")
            ):
                _fail("manifest_migration_authority_persisted_root_not_durable")
        return value


__all__ = [
    "ATTACHMENT_AUTHORITY_ACTION",
    "ATTACHMENT_AUTHORITY_DOMAIN",
    "ATTACHMENT_AUTHORITY_EVIDENCE_SCHEMA",
    "ATTACHMENT_AUTHORITY_SCHEMA",
    "MANIFEST_MIGRATION_AUTHORITY_ACTION",
    "MANIFEST_MIGRATION_AUTHORITY_EVIDENCE_SCHEMA",
    "MANIFEST_MIGRATION_AUTHORITY_SCHEMA",
    "MANIFEST_MIGRATION_AUTHORITY_SCOPE",
    "AttachmentAuthorityError",
    "AttachmentAuthorityPolicy",
    "AttachmentCapabilityAuthorityVerifier",
    "ManifestMigrationAuthorityVerifier",
    "PhysicalAuthorityVerifier",
    "WillReceiptSource",
    "build_attachment_authority_intent",
    "build_manifest_migration_authority_intent",
]
