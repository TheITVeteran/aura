"""Offline-verifiable external witnesses for Reality Reach acceptance.

The producer may package evidence, but it cannot make that evidence
independent by hashing it.  A physical acceptance promotion therefore needs
two separately pinned Ed25519 roots: one attesting to metrology evidence and
one attesting to the governed action receipt.  This module verifies those
attestations without loading private keys or performing network operations.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import stat
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from core.reality_reach.acceptance import (
    AcceptanceEvidenceClass,
    ConnectorAcceptanceCertificate,
)
from core.reality_reach.acceptance_mandate import AcceptanceVerificationMandate
from core.reality_reach.acceptance_verifier import (
    MandatedAcceptanceVerificationReceipt,
    verify_acceptance_against_mandate,
)
from core.runtime.secure_path_custody import DirectoryCustody, SecurePathCustodyError

if TYPE_CHECKING:
    from core.reality_reach.acceptance_preregistration import (
        PreregisteredAcceptanceReceipt,
    )

ACCEPTANCE_WITNESS_STATEMENT_SCHEMA = (
    "aura.reality_reach.acceptance_witness_statement.v1"
)
ACCEPTANCE_WITNESS_BUNDLE_SCHEMA = "aura.reality_reach.acceptance_witness_bundle.v1"
EXTERNAL_ACCEPTANCE_VERIFICATION_SCHEMA = (
    "aura.reality_reach.external_acceptance_verification.v2"
)
ZERO_SHA256 = "sha256:" + "0" * 64

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_FUTURE_SKEW_NS = 5 * 60 * 1_000_000_000


class AcceptanceWitnessRole(StrEnum):
    METROLOGY = "metrology"
    GOVERNANCE = "governance"


class AcceptanceWitnessError(ValueError):
    """Stable fail-closed external witness error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise AcceptanceWitnessError("acceptance_witness_json_invalid") from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{name} must be a canonical identifier")
    return normalized


def _sha256(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _DIGEST.fullmatch(normalized):
        raise ValueError(f"{name} must be a sha256 digest")
    return normalized


def _decode_b64(value: object, *, role: str, expected_size: int) -> bytes:
    if not isinstance(value, str):
        raise AcceptanceWitnessError(f"acceptance_witness_{role}_invalid")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AcceptanceWitnessError(f"acceptance_witness_{role}_invalid") from exc
    if len(decoded) != expected_size:
        raise AcceptanceWitnessError(f"acceptance_witness_{role}_invalid")
    return decoded


@dataclass(frozen=True, slots=True)
class AcceptanceWitnessStatement:
    role: AcceptanceWitnessRole
    witness_id: str
    campaign_id: str
    mandate_sha256: str
    certificate_sha256: str
    evidence_sha256: str
    sequence: int
    previous_statement_sha256: str
    witnessed_at_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.role, AcceptanceWitnessRole):
            raise TypeError("role must be an AcceptanceWitnessRole")
        for name in ("witness_id", "campaign_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name=name))
        for name in (
            "mandate_sha256",
            "certificate_sha256",
            "evidence_sha256",
            "previous_statement_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence <= 0
        ):
            raise ValueError("sequence must be a positive integer")
        if self.sequence == 1 and self.previous_statement_sha256 != ZERO_SHA256:
            raise ValueError("first witness statement must use the zero predecessor")
        if self.sequence > 1 and self.previous_statement_sha256 == ZERO_SHA256:
            raise ValueError("continued witness statements require a predecessor")
        if (
            isinstance(self.witnessed_at_ns, bool)
            or not isinstance(self.witnessed_at_ns, int)
            or self.witnessed_at_ns <= 0
        ):
            raise ValueError("witnessed_at_ns must be a positive integer")

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        document = {
            "schema": ACCEPTANCE_WITNESS_STATEMENT_SCHEMA,
            "role": self.role.value,
            "witness_id": self.witness_id,
            "campaign_id": self.campaign_id,
            "mandate_sha256": self.mandate_sha256,
            "certificate_sha256": self.certificate_sha256,
            "evidence_sha256": self.evidence_sha256,
            "sequence": self.sequence,
            "previous_statement_sha256": self.previous_statement_sha256,
            "witnessed_at_ns": self.witnessed_at_ns,
        }
        if include_digest:
            document["statement_sha256"] = self.sha256
        return document

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> AcceptanceWitnessStatement:
        expected = {
            "schema",
            "role",
            "witness_id",
            "campaign_id",
            "mandate_sha256",
            "certificate_sha256",
            "evidence_sha256",
            "sequence",
            "previous_statement_sha256",
            "witnessed_at_ns",
            "statement_sha256",
        }
        if not isinstance(document, Mapping) or set(document) != expected:
            raise AcceptanceWitnessError("acceptance_witness_statement_schema_invalid")
        if document.get("schema") != ACCEPTANCE_WITNESS_STATEMENT_SCHEMA:
            raise AcceptanceWitnessError("acceptance_witness_statement_schema_invalid")
        try:
            statement = cls(
                role=AcceptanceWitnessRole(document["role"]),
                witness_id=document["witness_id"],
                campaign_id=document["campaign_id"],
                mandate_sha256=document["mandate_sha256"],
                certificate_sha256=document["certificate_sha256"],
                evidence_sha256=document["evidence_sha256"],
                sequence=document["sequence"],
                previous_statement_sha256=document["previous_statement_sha256"],
                witnessed_at_ns=document["witnessed_at_ns"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AcceptanceWitnessError("acceptance_witness_statement_invalid") from exc
        if not hmac.compare_digest(
            str(document.get("statement_sha256") or ""), statement.sha256
        ):
            raise AcceptanceWitnessError("acceptance_witness_statement_digest_invalid")
        return statement


@dataclass(frozen=True, slots=True)
class AcceptanceWitnessBundle:
    statement: AcceptanceWitnessStatement
    public_key_raw_b64: str
    signature_b64: str

    def __post_init__(self) -> None:
        if not isinstance(self.statement, AcceptanceWitnessStatement):
            raise TypeError("statement must be an AcceptanceWitnessStatement")
        public_key = _decode_b64(
            self.public_key_raw_b64,
            role="public_key",
            expected_size=32,
        )
        _decode_b64(self.signature_b64, role="signature", expected_size=64)
        object.__setattr__(
            self,
            "public_key_raw_b64",
            base64.b64encode(public_key).decode("ascii"),
        )

    @property
    def public_key_sha256(self) -> str:
        return _bytes_digest(
            _decode_b64(
                self.public_key_raw_b64,
                role="public_key",
                expected_size=32,
            )
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict(include_digest=False))

    def signed_payload(self) -> bytes:
        return _canonical_bytes(self.statement.to_dict())

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        document = {
            "schema": ACCEPTANCE_WITNESS_BUNDLE_SCHEMA,
            "statement": self.statement.to_dict(),
            "algorithm": "Ed25519",
            "public_key_raw_b64": self.public_key_raw_b64,
            "public_key_sha256": self.public_key_sha256,
            "signature_b64": self.signature_b64,
        }
        if include_digest:
            document["bundle_sha256"] = self.sha256
        return document

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> AcceptanceWitnessBundle:
        expected = {
            "schema",
            "statement",
            "algorithm",
            "public_key_raw_b64",
            "public_key_sha256",
            "signature_b64",
            "bundle_sha256",
        }
        if not isinstance(document, Mapping) or set(document) != expected:
            raise AcceptanceWitnessError("acceptance_witness_bundle_schema_invalid")
        if (
            document.get("schema") != ACCEPTANCE_WITNESS_BUNDLE_SCHEMA
            or document.get("algorithm") != "Ed25519"
            or not isinstance(document.get("statement"), Mapping)
        ):
            raise AcceptanceWitnessError("acceptance_witness_bundle_schema_invalid")
        try:
            bundle = cls(
                statement=AcceptanceWitnessStatement.from_dict(document["statement"]),
                public_key_raw_b64=document["public_key_raw_b64"],
                signature_b64=document["signature_b64"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AcceptanceWitnessError("acceptance_witness_bundle_invalid") from exc
        if not hmac.compare_digest(
            str(document.get("public_key_sha256") or ""), bundle.public_key_sha256
        ):
            raise AcceptanceWitnessError("acceptance_witness_public_key_digest_invalid")
        if not hmac.compare_digest(
            str(document.get("bundle_sha256") or ""), bundle.sha256
        ):
            raise AcceptanceWitnessError("acceptance_witness_bundle_digest_invalid")
        return bundle


def verify_acceptance_witness_bundle(
    bundle: AcceptanceWitnessBundle | Mapping[str, Any],
    *,
    expected_role: AcceptanceWitnessRole,
    expected_public_key_sha256: str,
    mandate: AcceptanceVerificationMandate,
    certificate: ConnectorAcceptanceCertificate,
    expected_evidence_sha256: str,
    expected_sequence: int,
    expected_previous_statement_sha256: str,
    now_ns: int | None = None,
) -> AcceptanceWitnessBundle:
    """Verify one externally signed role statement against pinned trust input."""

    return verify_acceptance_witness_artifact_bundle(
        bundle,
        expected_role=expected_role,
        expected_public_key_sha256=expected_public_key_sha256,
        expected_campaign_id=mandate.campaign_id,
        expected_mandate_sha256=mandate.sha256,
        expected_artifact_sha256=certificate.sha256,
        expected_evidence_sha256=expected_evidence_sha256,
        expected_sequence=expected_sequence,
        expected_previous_statement_sha256=expected_previous_statement_sha256,
        campaign_completed_at_ns=certificate.completed_at_ns,
        now_ns=now_ns,
    )


def verify_acceptance_witness_artifact_bundle(
    bundle: AcceptanceWitnessBundle | Mapping[str, Any],
    *,
    expected_role: AcceptanceWitnessRole,
    expected_public_key_sha256: str,
    expected_campaign_id: str,
    expected_mandate_sha256: str,
    expected_artifact_sha256: str,
    expected_evidence_sha256: str,
    expected_sequence: int,
    expected_previous_statement_sha256: str,
    campaign_completed_at_ns: int,
    now_ns: int | None = None,
) -> AcceptanceWitnessBundle:
    """Verify a role witness for any mandate-bound acceptance artifact."""

    resolved = (
        bundle
        if isinstance(bundle, AcceptanceWitnessBundle)
        else AcceptanceWitnessBundle.from_dict(bundle)
    )
    trusted_key = _sha256(
        expected_public_key_sha256,
        name="expected_public_key_sha256",
    )
    expected_evidence = _sha256(
        expected_evidence_sha256,
        name="expected_evidence_sha256",
    )
    expected_previous = _sha256(
        expected_previous_statement_sha256,
        name="expected_previous_statement_sha256",
    )
    campaign_id = _identifier(expected_campaign_id, name="expected_campaign_id")
    mandate_sha256 = _sha256(
        expected_mandate_sha256,
        name="expected_mandate_sha256",
    )
    artifact_sha256 = _sha256(
        expected_artifact_sha256,
        name="expected_artifact_sha256",
    )
    if (
        isinstance(campaign_completed_at_ns, bool)
        or not isinstance(campaign_completed_at_ns, int)
        or campaign_completed_at_ns <= 0
    ):
        raise ValueError("campaign_completed_at_ns must be a positive integer")
    statement = resolved.statement
    if statement.role is not expected_role:
        raise AcceptanceWitnessError("acceptance_witness_role_mismatch")
    if not hmac.compare_digest(resolved.public_key_sha256, trusted_key):
        raise AcceptanceWitnessError("acceptance_witness_trust_root_mismatch")
    if statement.campaign_id != campaign_id:
        raise AcceptanceWitnessError("acceptance_witness_campaign_mismatch")
    if not hmac.compare_digest(statement.mandate_sha256, mandate_sha256):
        raise AcceptanceWitnessError("acceptance_witness_mandate_mismatch")
    if not hmac.compare_digest(statement.certificate_sha256, artifact_sha256):
        raise AcceptanceWitnessError("acceptance_witness_certificate_mismatch")
    if not hmac.compare_digest(statement.evidence_sha256, expected_evidence):
        raise AcceptanceWitnessError("acceptance_witness_evidence_mismatch")
    if statement.sequence != expected_sequence:
        raise AcceptanceWitnessError("acceptance_witness_sequence_mismatch")
    if not hmac.compare_digest(
        statement.previous_statement_sha256,
        expected_previous,
    ):
        raise AcceptanceWitnessError("acceptance_witness_predecessor_mismatch")
    current_ns = now_ns if now_ns is not None else time.time_ns()
    if statement.witnessed_at_ns < campaign_completed_at_ns:
        raise AcceptanceWitnessError("acceptance_witness_predates_campaign_completion")
    if statement.witnessed_at_ns > current_ns + _MAX_FUTURE_SKEW_NS:
        raise AcceptanceWitnessError("acceptance_witness_time_in_future")
    public_key = Ed25519PublicKey.from_public_bytes(
        _decode_b64(
            resolved.public_key_raw_b64,
            role="public_key",
            expected_size=32,
        )
    )
    signature = _decode_b64(
        resolved.signature_b64,
        role="signature",
        expected_size=64,
    )
    try:
        public_key.verify(signature, resolved.signed_payload())
    except InvalidSignature as exc:
        raise AcceptanceWitnessError("acceptance_witness_signature_invalid") from exc
    return resolved


@dataclass(frozen=True, slots=True)
class ExternallyWitnessedAcceptanceReceipt:
    mandate_verification: MandatedAcceptanceVerificationReceipt
    preregistration_verification_sha256: str
    metrology_witness_bundle_sha256: str
    governance_witness_bundle_sha256: str
    metrology_witness_key_sha256: str
    governance_witness_key_sha256: str
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.mandate_verification,
            MandatedAcceptanceVerificationReceipt,
        ):
            raise TypeError(
                "mandate_verification must be a MandatedAcceptanceVerificationReceipt"
            )
        for name in (
            "preregistration_verification_sha256",
            "metrology_witness_bundle_sha256",
            "governance_witness_bundle_sha256",
            "metrology_witness_key_sha256",
            "governance_witness_key_sha256",
        ):
            value = str(getattr(self, name))
            if value and not _DIGEST.fullmatch(value):
                raise ValueError(f"{name} must be empty or a sha256 digest")
        if len(self.blockers) != len(set(self.blockers)):
            raise ValueError("blockers must be unique")

    @property
    def accepted(self) -> bool:
        physical = self.mandate_verification.verification.expected_evidence_class in {
            AcceptanceEvidenceClass.HARDWARE_IN_LOOP,
            AcceptanceEvidenceClass.LIVE,
        }
        witness_complete = bool(
            self.metrology_witness_bundle_sha256
            and self.governance_witness_bundle_sha256
        )
        preregistration_complete = bool(self.preregistration_verification_sha256)
        return bool(
            self.mandate_verification.accepted
            and not self.blockers
            and (
                witness_complete and preregistration_complete
                if physical
                else True
            )
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        document = {
            "schema": EXTERNAL_ACCEPTANCE_VERIFICATION_SCHEMA,
            "mandate_verification": self.mandate_verification.to_dict(),
            "preregistration_verification_sha256": (
                self.preregistration_verification_sha256
            ),
            "metrology_witness_bundle_sha256": (
                self.metrology_witness_bundle_sha256
            ),
            "governance_witness_bundle_sha256": (
                self.governance_witness_bundle_sha256
            ),
            "metrology_witness_key_sha256": self.metrology_witness_key_sha256,
            "governance_witness_key_sha256": self.governance_witness_key_sha256,
            "blockers": list(self.blockers),
            "accepted": self.accepted,
        }
        if include_digest:
            document["external_verification_sha256"] = self.sha256
        return document


def verify_acceptance_with_external_witnesses(
    certificate: ConnectorAcceptanceCertificate,
    evidence_document: Mapping[str, Any],
    mandate: AcceptanceVerificationMandate,
    *,
    preregistration_receipt: PreregisteredAcceptanceReceipt | None = None,
    metrology_witness_bundle: AcceptanceWitnessBundle | Mapping[str, Any] | None = None,
    governance_witness_bundle: AcceptanceWitnessBundle | Mapping[str, Any] | None = None,
    metrology_witness_key_sha256: str = "",
    governance_witness_key_sha256: str = "",
    metrology_sequence: int = 1,
    governance_sequence: int = 1,
    metrology_previous_statement_sha256: str = ZERO_SHA256,
    governance_previous_statement_sha256: str = ZERO_SHA256,
    now_ns: int | None = None,
) -> ExternallyWitnessedAcceptanceReceipt:
    """Require two independently pinned witness roots for physical promotion."""

    physical = mandate.expected_evidence_class in {
        AcceptanceEvidenceClass.HARDWARE_IN_LOOP,
        AcceptanceEvidenceClass.LIVE,
    }
    blockers: list[str] = []
    metrology_digest = ""
    governance_digest = ""
    verified_metrology_bundle = ""
    verified_governance_bundle = ""
    if not physical:
        if metrology_witness_bundle is not None or governance_witness_bundle is not None:
            blockers.append("unexpected_external_witness_for_simulation")
        mandate_receipt = verify_acceptance_against_mandate(
            certificate,
            evidence_document,
            mandate,
        )
        return ExternallyWitnessedAcceptanceReceipt(
            mandate_verification=mandate_receipt,
            preregistration_verification_sha256="",
            metrology_witness_bundle_sha256="",
            governance_witness_bundle_sha256="",
            metrology_witness_key_sha256="",
            governance_witness_key_sha256="",
            blockers=tuple(blockers),
        )

    verified_preregistration = ""
    if preregistration_receipt is None:
        blockers.append("acceptance_preregistration_missing")
    else:
        from core.reality_reach.acceptance_preregistration import (
            PreregisteredAcceptanceReceipt,
        )

    if preregistration_receipt is not None and not isinstance(
        preregistration_receipt,
        PreregisteredAcceptanceReceipt,
    ):
        blockers.append("acceptance_preregistration_invalid")
    elif preregistration_receipt is not None and (
        not preregistration_receipt.accepted
        or preregistration_receipt.mandate.sha256 != mandate.sha256
        or preregistration_receipt.campaign_started_at_ns
        != certificate.started_at_ns
        or not preregistration_receipt.strictly_predates_campaign
    ):
        blockers.append("acceptance_preregistration_binding_invalid")
    elif preregistration_receipt is not None:
        verified_preregistration = preregistration_receipt.sha256

    if metrology_witness_bundle is None:
        blockers.append("external_metrology_witness_missing")
    elif not metrology_witness_key_sha256:
        blockers.append("external_metrology_trust_root_missing")
    else:
        try:
            verified = verify_acceptance_witness_bundle(
                metrology_witness_bundle,
                expected_role=AcceptanceWitnessRole.METROLOGY,
                expected_public_key_sha256=metrology_witness_key_sha256,
                mandate=mandate,
                certificate=certificate,
                expected_evidence_sha256=certificate.metrology_evidence_sha256,
                expected_sequence=metrology_sequence,
                expected_previous_statement_sha256=(
                    metrology_previous_statement_sha256
                ),
                now_ns=now_ns,
            )
            metrology_digest = verified.statement.evidence_sha256
            verified_metrology_bundle = verified.sha256
        except (AcceptanceWitnessError, TypeError, ValueError) as exc:
            blockers.append(
                exc.code
                if isinstance(exc, AcceptanceWitnessError)
                else "external_metrology_witness_invalid"
            )

    if governance_witness_bundle is None:
        blockers.append("external_governance_witness_missing")
    elif not governance_witness_key_sha256:
        blockers.append("external_governance_trust_root_missing")
    else:
        try:
            verified = verify_acceptance_witness_bundle(
                governance_witness_bundle,
                expected_role=AcceptanceWitnessRole.GOVERNANCE,
                expected_public_key_sha256=governance_witness_key_sha256,
                mandate=mandate,
                certificate=certificate,
                expected_evidence_sha256=certificate.governance_evidence_sha256,
                expected_sequence=governance_sequence,
                expected_previous_statement_sha256=(
                    governance_previous_statement_sha256
                ),
                now_ns=now_ns,
            )
            governance_digest = verified.statement.evidence_sha256
            verified_governance_bundle = verified.sha256
        except (AcceptanceWitnessError, TypeError, ValueError) as exc:
            blockers.append(
                exc.code
                if isinstance(exc, AcceptanceWitnessError)
                else "external_governance_witness_invalid"
            )

    mandate_receipt = verify_acceptance_against_mandate(
        certificate,
        evidence_document,
        mandate,
        trusted_metrology_evidence_sha256=metrology_digest,
        trusted_governance_evidence_sha256=governance_digest,
    )
    if (
        verified_metrology_bundle
        and verified_governance_bundle
        and hmac.compare_digest(
            metrology_witness_key_sha256,
            governance_witness_key_sha256,
        )
    ):
        blockers.append("external_witness_roots_not_distinct")
    return ExternallyWitnessedAcceptanceReceipt(
        mandate_verification=mandate_receipt,
        preregistration_verification_sha256=verified_preregistration,
        metrology_witness_bundle_sha256=verified_metrology_bundle,
        governance_witness_bundle_sha256=verified_governance_bundle,
        metrology_witness_key_sha256=(
            metrology_witness_key_sha256 if verified_metrology_bundle else ""
        ),
        governance_witness_key_sha256=(
            governance_witness_key_sha256 if verified_governance_bundle else ""
        ),
        blockers=tuple(sorted(set(blockers))),
    )


def persist_externally_witnessed_acceptance_receipt(
    receipt: ExternallyWitnessedAcceptanceReceipt,
    path: str | Path,
) -> bool:
    """Create-once publish the externally witnessed promotion verdict."""

    if not isinstance(receipt, ExternallyWitnessedAcceptanceReceipt):
        raise TypeError("receipt must be an ExternallyWitnessedAcceptanceReceipt")
    target = Path(path).expanduser().absolute()
    if not target.name or target.name in {".", ".."}:
        raise AcceptanceWitnessError("external_acceptance_receipt_path_invalid")
    payload = _canonical_bytes(receipt.to_dict())
    try:
        with DirectoryCustody.acquire(target.parent, create=True, private=True) as custody:
            published = bool(custody.write_bytes_once(target.name, payload, mode=0o600))
            fd = custody.open_file(target.name, os.O_RDONLY)
            try:
                if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                    raise AcceptanceWitnessError(
                        "external_acceptance_receipt_mode_invalid"
                    )
            finally:
                os.close(fd)
            existing = custody.read_bytes(target.name, max_bytes=1024 * 1024)
    except SecurePathCustodyError as exc:
        raise AcceptanceWitnessError(
            "external_acceptance_receipt_custody_invalid"
        ) from exc
    if existing != payload:
        raise AcceptanceWitnessError("external_acceptance_receipt_collision")
    return published


__all__ = [
    "ACCEPTANCE_WITNESS_BUNDLE_SCHEMA",
    "ACCEPTANCE_WITNESS_STATEMENT_SCHEMA",
    "EXTERNAL_ACCEPTANCE_VERIFICATION_SCHEMA",
    "ZERO_SHA256",
    "AcceptanceWitnessBundle",
    "AcceptanceWitnessError",
    "AcceptanceWitnessRole",
    "AcceptanceWitnessStatement",
    "ExternallyWitnessedAcceptanceReceipt",
    "persist_externally_witnessed_acceptance_receipt",
    "verify_acceptance_with_external_witnesses",
    "verify_acceptance_witness_artifact_bundle",
    "verify_acceptance_witness_bundle",
]
