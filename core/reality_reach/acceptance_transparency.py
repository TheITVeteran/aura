"""Transparency-log binding for externally witnessed physical acceptance."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.reality_reach.acceptance import AcceptanceEvidenceClass
from core.reality_reach.acceptance_witness import ExternallyWitnessedAcceptanceReceipt
from core.runtime.secure_path_custody import DirectoryCustody, SecurePathCustodyError
from core.security.rekor_transparency import (
    REKOR_PUBLIC_GOOD_SERVER,
    RekorTransparencyError,
    load_x509_certificate,
    verify_rekord_entry,
    verify_signature,
)

ACCEPTANCE_TRANSPARENCY_STATEMENT_SCHEMA = (
    "aura.reality_reach.acceptance_transparency_statement.v1"
)
ACCEPTANCE_TRANSPARENCY_BUNDLE_SCHEMA = (
    "aura.reality_reach.acceptance_transparency_bundle.v1"
)
TRANSPARENT_ACCEPTANCE_VERIFICATION_SCHEMA = (
    "aura.reality_reach.transparent_acceptance_verification.v1"
)
ZERO_SHA256 = "sha256:" + "0" * 64

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REKOR_UUID = re.compile(r"^[0-9a-f]{80}$")
_MAX_WITNESS_DELAY_S = 60 * 60


class AcceptanceTransparencyError(ValueError):
    """Stable fail-closed acceptance transparency error."""

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
        raise AcceptanceTransparencyError("acceptance_transparency_json_invalid") from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _require_digest(value: object, *, name: str) -> str:
    normalized = str(value or "")
    if not _DIGEST.fullmatch(normalized):
        raise AcceptanceTransparencyError(f"acceptance_transparency_{name}_invalid")
    return normalized


def build_acceptance_transparency_statement(
    receipt: ExternallyWitnessedAcceptanceReceipt,
    *,
    sequence: int,
    previous_statement_sha256: str,
    previous_rekor_uuid: str | None,
    issued_at_unix: int,
) -> dict[str, Any]:
    """Commit one accepted dual-witness verdict for public timestamping."""

    if not isinstance(receipt, ExternallyWitnessedAcceptanceReceipt):
        raise TypeError("receipt must be an ExternallyWitnessedAcceptanceReceipt")
    if not receipt.accepted:
        raise AcceptanceTransparencyError("acceptance_transparency_receipt_not_accepted")
    if type(sequence) is not int or sequence <= 0:
        raise AcceptanceTransparencyError("acceptance_transparency_sequence_invalid")
    previous = _require_digest(
        previous_statement_sha256,
        name="previous_statement_sha256",
    )
    if type(issued_at_unix) is not int or issued_at_unix <= 0:
        raise AcceptanceTransparencyError("acceptance_transparency_issued_at_invalid")
    if sequence == 1:
        if previous != ZERO_SHA256 or previous_rekor_uuid is not None:
            raise AcceptanceTransparencyError("acceptance_transparency_genesis_invalid")
    elif previous == ZERO_SHA256 or not _REKOR_UUID.fullmatch(
        str(previous_rekor_uuid or "")
    ):
        raise AcceptanceTransparencyError("acceptance_transparency_chain_invalid")
    mandate = receipt.mandate_verification
    body = {
        "schema": ACCEPTANCE_TRANSPARENCY_STATEMENT_SCHEMA,
        "domain": "aura.reality-reach.physical-acceptance",
        "campaign_id": mandate.campaign_id,
        "mandate_sha256": mandate.mandate_sha256,
        "external_verification_sha256": receipt.sha256,
        "metrology_witness_bundle_sha256": receipt.metrology_witness_bundle_sha256,
        "governance_witness_bundle_sha256": receipt.governance_witness_bundle_sha256,
        "sequence": sequence,
        "previous_statement_sha256": previous,
        "previous_rekor_uuid": previous_rekor_uuid,
        "issued_at_unix": issued_at_unix,
    }
    return {**body, "statement_sha256": _digest(body)}


def validate_acceptance_transparency_statement_envelope(
    raw: object,
) -> dict[str, Any]:
    """Validate the shared append-only statement envelope without domain policy."""

    if not isinstance(raw, Mapping):
        raise AcceptanceTransparencyError("acceptance_transparency_statement_invalid")
    statement = dict(raw)
    expected_keys = {
        "schema",
        "domain",
        "campaign_id",
        "mandate_sha256",
        "external_verification_sha256",
        "metrology_witness_bundle_sha256",
        "governance_witness_bundle_sha256",
        "sequence",
        "previous_statement_sha256",
        "previous_rekor_uuid",
        "issued_at_unix",
        "statement_sha256",
    }
    if set(statement) != expected_keys:
        raise AcceptanceTransparencyError("acceptance_transparency_statement_invalid")
    digest = statement.pop("statement_sha256", None)
    if not hmac.compare_digest(str(digest or ""), _digest(statement)):
        raise AcceptanceTransparencyError(
            "acceptance_transparency_statement_digest_invalid"
        )
    if statement.get("schema") != ACCEPTANCE_TRANSPARENCY_STATEMENT_SCHEMA:
        raise AcceptanceTransparencyError("acceptance_transparency_statement_invalid")
    domain = statement.get("domain")
    campaign_id = statement.get("campaign_id")
    if not isinstance(domain, str) or not domain or len(domain) > 128:
        raise AcceptanceTransparencyError("acceptance_transparency_domain_invalid")
    if not isinstance(campaign_id, str) or not campaign_id or len(campaign_id) > 128:
        raise AcceptanceTransparencyError("acceptance_transparency_campaign_invalid")
    for name in (
        "mandate_sha256",
        "external_verification_sha256",
        "metrology_witness_bundle_sha256",
        "governance_witness_bundle_sha256",
        "previous_statement_sha256",
    ):
        _require_digest(statement.get(name), name=name)
    sequence = statement.get("sequence")
    issued_at_unix = statement.get("issued_at_unix")
    previous_statement = str(statement.get("previous_statement_sha256"))
    previous_rekor_uuid = statement.get("previous_rekor_uuid")
    if type(sequence) is not int or sequence <= 0:
        raise AcceptanceTransparencyError("acceptance_transparency_sequence_invalid")
    if type(issued_at_unix) is not int or issued_at_unix <= 0:
        raise AcceptanceTransparencyError("acceptance_transparency_issued_at_invalid")
    if sequence == 1:
        if previous_statement != ZERO_SHA256 or previous_rekor_uuid is not None:
            raise AcceptanceTransparencyError("acceptance_transparency_genesis_invalid")
    elif previous_statement == ZERO_SHA256 or not _REKOR_UUID.fullmatch(
        str(previous_rekor_uuid or "")
    ):
        raise AcceptanceTransparencyError("acceptance_transparency_chain_invalid")
    statement["statement_sha256"] = digest
    return statement


def _validate_statement(
    raw: object,
    *,
    receipt: ExternallyWitnessedAcceptanceReceipt,
    expected_sequence: int,
    expected_previous_statement_sha256: str,
    expected_previous_rekor_uuid: str | None,
) -> dict[str, Any]:
    statement = validate_acceptance_transparency_statement_envelope(raw)
    mandate = receipt.mandate_verification
    if (
        statement.get("schema") != ACCEPTANCE_TRANSPARENCY_STATEMENT_SCHEMA
        or statement.get("domain") != "aura.reality-reach.physical-acceptance"
        or statement.get("campaign_id") != mandate.campaign_id
        or statement.get("mandate_sha256") != mandate.mandate_sha256
        or statement.get("external_verification_sha256") != receipt.sha256
        or statement.get("metrology_witness_bundle_sha256")
        != receipt.metrology_witness_bundle_sha256
        or statement.get("governance_witness_bundle_sha256")
        != receipt.governance_witness_bundle_sha256
        or statement.get("sequence") != expected_sequence
        or statement.get("previous_statement_sha256")
        != expected_previous_statement_sha256
        or statement.get("previous_rekor_uuid") != expected_previous_rekor_uuid
        or type(statement.get("issued_at_unix")) is not int
    ):
        raise AcceptanceTransparencyError(
            "acceptance_transparency_statement_binding_invalid"
        )
    build_acceptance_transparency_statement(
        receipt,
        sequence=expected_sequence,
        previous_statement_sha256=expected_previous_statement_sha256,
        previous_rekor_uuid=expected_previous_rekor_uuid,
        issued_at_unix=statement["issued_at_unix"],
    )
    return statement


def build_acceptance_transparency_artifact_bundle(
    *,
    statement: Mapping[str, Any],
    producer_signature: bytes,
    producer_certificate_pem: bytes,
    rekor_uuid: str,
    rekor_entry: Mapping[str, Any],
    trusted_log_public_key_pem: bytes,
) -> dict[str, Any]:
    """Verify and package any accepted statement into a portable Rekor bundle."""

    statement_document = dict(statement)
    issued_at_unix = statement_document.get("issued_at_unix")
    if type(issued_at_unix) is not int or issued_at_unix <= 0:
        raise AcceptanceTransparencyError("acceptance_transparency_issued_at_invalid")
    statement_bytes = _canonical_bytes(statement_document)
    try:
        certificate = load_x509_certificate(
            producer_certificate_pem,
            code="acceptance_transparency_producer_certificate_invalid",
        )
        verify_signature(
            certificate.public_key(),
            producer_signature,
            statement_bytes,
            code="acceptance_transparency_producer_signature",
        )
        verified = verify_rekord_entry(
            entry=rekor_entry,
            artifact_bytes=statement_bytes,
            producer_signature=producer_signature,
            producer_certificate_pem=producer_certificate_pem,
            trusted_log_public_key_pem=trusted_log_public_key_pem,
            issued_at_unix=issued_at_unix,
            rekor_uuid=rekor_uuid,
            code_prefix="acceptance_transparency",
            maximum_witness_delay_s=_MAX_WITNESS_DELAY_S,
        )
    except RekorTransparencyError as exc:
        raise AcceptanceTransparencyError(exc.code) from exc
    body = {
        "schema": ACCEPTANCE_TRANSPARENCY_BUNDLE_SCHEMA,
        "statement": statement_document,
        "producer_signature_b64": base64.b64encode(producer_signature).decode("ascii"),
        "producer_certificate_pem_b64": base64.b64encode(
            producer_certificate_pem
        ).decode("ascii"),
        "producer_certificate_sha256": _bytes_digest(producer_certificate_pem),
        "rekor_server": REKOR_PUBLIC_GOOD_SERVER,
        "rekor_uuid": rekor_uuid,
        "rekor_entry": dict(rekor_entry),
        "trusted_log_key_sha256": "sha256:" + verified.trusted_log_key_sha256,
    }
    return {**body, "bundle_sha256": _digest(body)}


def build_acceptance_transparency_bundle(
    *,
    statement: Mapping[str, Any],
    producer_signature: bytes,
    producer_certificate_pem: bytes,
    rekor_uuid: str,
    rekor_entry: Mapping[str, Any],
    trusted_log_public_key_pem: bytes,
) -> dict[str, Any]:
    """Verify and package a physical-acceptance Rekor bundle."""

    statement_document = validate_acceptance_transparency_statement_envelope(statement)
    if statement_document.get("domain") != "aura.reality-reach.physical-acceptance":
        raise AcceptanceTransparencyError(
            "acceptance_transparency_statement_binding_invalid"
        )
    return build_acceptance_transparency_artifact_bundle(
        statement=statement_document,
        producer_signature=producer_signature,
        producer_certificate_pem=producer_certificate_pem,
        rekor_uuid=rekor_uuid,
        rekor_entry=rekor_entry,
        trusted_log_public_key_pem=trusted_log_public_key_pem,
    )


@dataclass(frozen=True, slots=True)
class TransparentlyLoggedAcceptanceReceipt:
    external_verification: ExternallyWitnessedAcceptanceReceipt
    transparency_bundle_sha256: str
    trusted_log_key_sha256: str
    rekor_uuid: str
    rekor_log_index: int
    rekor_integrated_time: int
    blockers: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        physical = (
            self.external_verification.mandate_verification.verification.expected_evidence_class
            in {AcceptanceEvidenceClass.HARDWARE_IN_LOOP, AcceptanceEvidenceClass.LIVE}
        )
        return bool(
            self.external_verification.accepted
            and not self.blockers
            and (self.transparency_bundle_sha256 if physical else True)
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        document = {
            "schema": TRANSPARENT_ACCEPTANCE_VERIFICATION_SCHEMA,
            "external_verification": self.external_verification.to_dict(),
            "transparency_bundle_sha256": self.transparency_bundle_sha256,
            "trusted_log_key_sha256": self.trusted_log_key_sha256,
            "rekor_uuid": self.rekor_uuid,
            "rekor_log_index": self.rekor_log_index,
            "rekor_integrated_time": self.rekor_integrated_time,
            "blockers": list(self.blockers),
            "accepted": self.accepted,
        }
        if include_digest:
            document["transparent_verification_sha256"] = self.sha256
        return document


@dataclass(frozen=True, slots=True)
class TransparencyArtifactVerification:
    transparency_bundle_sha256: str
    trusted_log_key_sha256: str
    rekor_uuid: str
    rekor_log_index: int
    rekor_integrated_time: int
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("transparency_bundle_sha256", "trusted_log_key_sha256"):
            value = str(getattr(self, name) or "")
            if value and not _DIGEST.fullmatch(value):
                raise ValueError(f"{name} must be empty or a sha256 digest")
        if self.rekor_uuid and not _REKOR_UUID.fullmatch(self.rekor_uuid):
            raise ValueError("rekor_uuid must be empty or a Rekor UUID")
        if type(self.rekor_log_index) is not int or self.rekor_log_index < -1:
            raise ValueError("rekor_log_index must be an integer >= -1")
        if type(self.rekor_integrated_time) is not int or self.rekor_integrated_time < 0:
            raise ValueError("rekor_integrated_time must be a non-negative integer")
        if len(self.blockers) != len(set(self.blockers)):
            raise ValueError("blockers must be unique")

    @property
    def accepted(self) -> bool:
        return bool(self.transparency_bundle_sha256 and not self.blockers)


def verify_acceptance_transparency_artifact(
    *,
    transparency_bundle: Mapping[str, Any] | None,
    trusted_log_public_key_pem: bytes | None,
    statement_validator: Callable[[object], Mapping[str, Any]],
    minimum_log_index: int | None = None,
    minimum_integrated_time: int | None = None,
) -> TransparencyArtifactVerification:
    """Verify one acceptance statement against Rekor and rollback floors."""

    blockers: list[str] = []
    bundle_sha256 = ""
    trusted_key = ""
    rekor_uuid = ""
    log_index = -1
    integrated_time = 0
    if transparency_bundle is None:
        blockers.append("acceptance_transparency_bundle_missing")
    elif not trusted_log_public_key_pem:
        blockers.append("acceptance_transparency_log_key_missing")
    else:
        try:
            bundle = dict(transparency_bundle)
            expected_keys = {
                "schema",
                "statement",
                "producer_signature_b64",
                "producer_certificate_pem_b64",
                "producer_certificate_sha256",
                "rekor_server",
                "rekor_uuid",
                "rekor_entry",
                "trusted_log_key_sha256",
                "bundle_sha256",
            }
            if set(bundle) != expected_keys:
                raise AcceptanceTransparencyError(
                    "acceptance_transparency_bundle_invalid"
                )
            digest = bundle.pop("bundle_sha256")
            if not hmac.compare_digest(str(digest), _digest(bundle)):
                raise AcceptanceTransparencyError(
                    "acceptance_transparency_bundle_digest_invalid"
                )
            bundle["bundle_sha256"] = digest
            if bundle.get("schema") != ACCEPTANCE_TRANSPARENCY_BUNDLE_SCHEMA:
                raise AcceptanceTransparencyError(
                    "acceptance_transparency_bundle_invalid"
                )
            if bundle.get("rekor_server") != REKOR_PUBLIC_GOOD_SERVER:
                raise AcceptanceTransparencyError(
                    "acceptance_transparency_rekor_server_invalid"
                )
            statement = dict(statement_validator(bundle.get("statement")))
            signature = base64.b64decode(
                str(bundle.get("producer_signature_b64") or ""),
                validate=True,
            )
            certificate_pem = base64.b64decode(
                str(bundle.get("producer_certificate_pem_b64") or ""),
                validate=True,
            )
            if bundle.get("producer_certificate_sha256") != _bytes_digest(
                certificate_pem
            ):
                raise AcceptanceTransparencyError(
                    "acceptance_transparency_producer_certificate_commitment_mismatch"
                )
            entry = bundle.get("rekor_entry")
            if not isinstance(entry, Mapping):
                raise AcceptanceTransparencyError(
                    "acceptance_transparency_rekor_entry_invalid"
                )
            verified = verify_rekord_entry(
                entry=entry,
                artifact_bytes=_canonical_bytes(statement),
                producer_signature=signature,
                producer_certificate_pem=certificate_pem,
                trusted_log_public_key_pem=trusted_log_public_key_pem,
                issued_at_unix=statement["issued_at_unix"],
                rekor_uuid=str(bundle.get("rekor_uuid") or ""),
                code_prefix="acceptance_transparency",
                maximum_witness_delay_s=_MAX_WITNESS_DELAY_S,
            )
            expected_key = "sha256:" + verified.trusted_log_key_sha256
            if bundle.get("trusted_log_key_sha256") != expected_key:
                raise AcceptanceTransparencyError(
                    "acceptance_transparency_log_key_commitment_mismatch"
                )
            if minimum_log_index is not None and (
                type(minimum_log_index) is not int
                or verified.log_index <= minimum_log_index
            ):
                raise AcceptanceTransparencyError(
                    "acceptance_transparency_log_index_rollback"
                )
            if minimum_integrated_time is not None and (
                type(minimum_integrated_time) is not int
                or verified.integrated_time < minimum_integrated_time
            ):
                raise AcceptanceTransparencyError(
                    "acceptance_transparency_integrated_time_rollback"
                )
            bundle_sha256 = str(digest)
            trusted_key = expected_key
            rekor_uuid = verified.rekor_uuid
            log_index = verified.log_index
            integrated_time = verified.integrated_time
        except (
            AcceptanceTransparencyError,
            RekorTransparencyError,
            ValueError,
            binascii.Error,
        ) as exc:
            blockers.append(
                exc.code
                if isinstance(exc, (AcceptanceTransparencyError, RekorTransparencyError))
                else "acceptance_transparency_bundle_invalid"
            )
    return TransparencyArtifactVerification(
        transparency_bundle_sha256=bundle_sha256,
        trusted_log_key_sha256=trusted_key,
        rekor_uuid=rekor_uuid,
        rekor_log_index=log_index,
        rekor_integrated_time=integrated_time,
        blockers=tuple(sorted(set(blockers))),
    )


def verify_transparently_logged_acceptance(
    receipt: ExternallyWitnessedAcceptanceReceipt,
    *,
    transparency_bundle: Mapping[str, Any] | None,
    trusted_log_public_key_pem: bytes | None,
    expected_sequence: int,
    expected_previous_statement_sha256: str,
    expected_previous_rekor_uuid: str | None,
    minimum_log_index: int | None = None,
    minimum_integrated_time: int | None = None,
) -> TransparentlyLoggedAcceptanceReceipt:
    """Require public-log inclusion for physical acceptance promotion."""

    physical = receipt.mandate_verification.verification.expected_evidence_class in {
        AcceptanceEvidenceClass.HARDWARE_IN_LOOP,
        AcceptanceEvidenceClass.LIVE,
    }
    if not physical:
        simulation_blockers = (
            ("unexpected_acceptance_transparency_for_simulation",)
            if transparency_bundle is not None
            else ()
        )
        return TransparentlyLoggedAcceptanceReceipt(
            external_verification=receipt,
            transparency_bundle_sha256="",
            trusted_log_key_sha256="",
            rekor_uuid="",
            rekor_log_index=-1,
            rekor_integrated_time=0,
            blockers=simulation_blockers,
        )
    artifact = verify_acceptance_transparency_artifact(
        transparency_bundle=transparency_bundle,
        trusted_log_public_key_pem=trusted_log_public_key_pem,
        statement_validator=lambda raw: _validate_statement(
            raw,
            receipt=receipt,
            expected_sequence=expected_sequence,
            expected_previous_statement_sha256=expected_previous_statement_sha256,
            expected_previous_rekor_uuid=expected_previous_rekor_uuid,
        ),
        minimum_log_index=minimum_log_index,
        minimum_integrated_time=minimum_integrated_time,
    )
    return TransparentlyLoggedAcceptanceReceipt(
        external_verification=receipt,
        transparency_bundle_sha256=artifact.transparency_bundle_sha256,
        trusted_log_key_sha256=artifact.trusted_log_key_sha256,
        rekor_uuid=artifact.rekor_uuid,
        rekor_log_index=artifact.rekor_log_index,
        rekor_integrated_time=artifact.rekor_integrated_time,
        blockers=artifact.blockers,
    )


def persist_transparently_logged_acceptance_receipt(
    receipt: TransparentlyLoggedAcceptanceReceipt,
    path: str | Path,
) -> bool:
    """Create-once publish one transparency-bound acceptance verdict."""

    if not isinstance(receipt, TransparentlyLoggedAcceptanceReceipt):
        raise TypeError("receipt must be a TransparentlyLoggedAcceptanceReceipt")
    target = Path(path).expanduser().absolute()
    if not target.name or target.name in {".", ".."}:
        raise AcceptanceTransparencyError(
            "transparent_acceptance_receipt_path_invalid"
        )
    payload = _canonical_bytes(receipt.to_dict())
    try:
        with DirectoryCustody.acquire(target.parent, create=True, private=True) as custody:
            published = bool(custody.write_bytes_once(target.name, payload, mode=0o600))
            fd = custody.open_file(target.name, os.O_RDONLY)
            try:
                if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                    raise AcceptanceTransparencyError(
                        "transparent_acceptance_receipt_mode_invalid"
                    )
            finally:
                os.close(fd)
            existing = custody.read_bytes(target.name, max_bytes=1024 * 1024)
    except SecurePathCustodyError as exc:
        raise AcceptanceTransparencyError(
            "transparent_acceptance_receipt_custody_invalid"
        ) from exc
    if existing != payload:
        raise AcceptanceTransparencyError(
            "transparent_acceptance_receipt_collision"
        )
    return published


__all__ = [
    "ACCEPTANCE_TRANSPARENCY_BUNDLE_SCHEMA",
    "ACCEPTANCE_TRANSPARENCY_STATEMENT_SCHEMA",
    "TRANSPARENT_ACCEPTANCE_VERIFICATION_SCHEMA",
    "ZERO_SHA256",
    "AcceptanceTransparencyError",
    "TransparentlyLoggedAcceptanceReceipt",
    "TransparencyArtifactVerification",
    "build_acceptance_transparency_artifact_bundle",
    "build_acceptance_transparency_bundle",
    "build_acceptance_transparency_statement",
    "persist_transparently_logged_acceptance_receipt",
    "verify_transparently_logged_acceptance",
    "verify_acceptance_transparency_artifact",
    "validate_acceptance_transparency_statement_envelope",
]
