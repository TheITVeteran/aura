"""Portable, public-log proof that an acceptance mandate predates a campaign."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.reality_reach.acceptance_mandate import (
    AcceptanceMandateProvisionReceipt,
    AcceptanceVerificationMandate,
)
from core.reality_reach.acceptance_transparency import (
    ZERO_SHA256,
    AcceptanceTransparencyError,
    build_acceptance_transparency_artifact_bundle,
    verify_acceptance_transparency_artifact,
)
from core.runtime.secure_path_custody import DirectoryCustody, SecurePathCustodyError

ACCEPTANCE_PREREGISTRATION_STATEMENT_SCHEMA = (
    "aura.reality_reach.acceptance_preregistration_statement.v1"
)
ACCEPTANCE_PREREGISTRATION_VERIFICATION_SCHEMA = (
    "aura.reality_reach.acceptance_preregistration_verification.v1"
)
ACCEPTANCE_PREREGISTRATION_DOMAIN = (
    "aura.reality-reach.acceptance-preregistration"
)

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REKOR_UUID = re.compile(r"^[0-9a-f]{80}$")


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
        raise AcceptanceTransparencyError("acceptance_preregistration_json_invalid") from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_mandate_binding(
    mandate: AcceptanceVerificationMandate,
    provision_receipt: AcceptanceMandateProvisionReceipt,
) -> None:
    if not isinstance(mandate, AcceptanceVerificationMandate):
        raise TypeError("mandate must be an AcceptanceVerificationMandate")
    if not isinstance(provision_receipt, AcceptanceMandateProvisionReceipt):
        raise TypeError(
            "provision_receipt must be an AcceptanceMandateProvisionReceipt"
        )
    if (
        provision_receipt.campaign_id != mandate.campaign_id
        or provision_receipt.mandate_sha256 != mandate.sha256
        or provision_receipt.contract_sha256 != mandate.contract_sha256
        or provision_receipt.provisioned_at_ns != mandate.provisioned_at_ns
        or provision_receipt.custody_sequence != mandate.custody_sequence
    ):
        raise AcceptanceTransparencyError(
            "acceptance_preregistration_mandate_binding_invalid"
        )


def build_acceptance_preregistration_statement(
    mandate: AcceptanceVerificationMandate,
    provision_receipt: AcceptanceMandateProvisionReceipt,
    *,
    sequence: int,
    previous_statement_sha256: str,
    previous_rekor_uuid: str | None,
    issued_at_unix: int,
) -> dict[str, Any]:
    """Commit the exact acceptance question for pre-campaign public timestamping."""

    _validate_mandate_binding(mandate, provision_receipt)
    if type(sequence) is not int or sequence <= 0:
        raise AcceptanceTransparencyError("acceptance_preregistration_sequence_invalid")
    if not _DIGEST.fullmatch(str(previous_statement_sha256 or "")):
        raise AcceptanceTransparencyError(
            "acceptance_preregistration_previous_statement_invalid"
        )
    if type(issued_at_unix) is not int or issued_at_unix <= 0:
        raise AcceptanceTransparencyError("acceptance_preregistration_issued_at_invalid")
    if issued_at_unix < provision_receipt.provisioned_at_ns // 1_000_000_000:
        raise AcceptanceTransparencyError(
            "acceptance_preregistration_predates_local_provision"
        )
    if sequence == 1:
        if previous_statement_sha256 != ZERO_SHA256 or previous_rekor_uuid is not None:
            raise AcceptanceTransparencyError(
                "acceptance_preregistration_genesis_invalid"
            )
    elif previous_statement_sha256 == ZERO_SHA256 or not _REKOR_UUID.fullmatch(
        str(previous_rekor_uuid or "")
    ):
        raise AcceptanceTransparencyError("acceptance_preregistration_chain_invalid")
    body = {
        "schema": ACCEPTANCE_PREREGISTRATION_STATEMENT_SCHEMA,
        "domain": ACCEPTANCE_PREREGISTRATION_DOMAIN,
        "mandate": mandate.to_dict(),
        "provision_receipt": provision_receipt.to_dict(),
        "sequence": sequence,
        "previous_statement_sha256": previous_statement_sha256,
        "previous_rekor_uuid": previous_rekor_uuid,
        "issued_at_unix": issued_at_unix,
    }
    return {**body, "statement_sha256": _digest(body)}


def _validate_preregistration_statement(
    raw: object,
    *,
    mandate: AcceptanceVerificationMandate,
    provision_receipt: AcceptanceMandateProvisionReceipt,
    expected_sequence: int,
    expected_previous_statement_sha256: str,
    expected_previous_rekor_uuid: str | None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise AcceptanceTransparencyError("acceptance_preregistration_statement_invalid")
    statement = dict(raw)
    expected_keys = {
        "schema",
        "domain",
        "mandate",
        "provision_receipt",
        "sequence",
        "previous_statement_sha256",
        "previous_rekor_uuid",
        "issued_at_unix",
        "statement_sha256",
    }
    if set(statement) != expected_keys:
        raise AcceptanceTransparencyError("acceptance_preregistration_statement_invalid")
    issued_at_unix = statement.get("issued_at_unix")
    if type(issued_at_unix) is not int:
        raise AcceptanceTransparencyError("acceptance_preregistration_statement_invalid")
    expected = build_acceptance_preregistration_statement(
        mandate,
        provision_receipt,
        sequence=expected_sequence,
        previous_statement_sha256=expected_previous_statement_sha256,
        previous_rekor_uuid=expected_previous_rekor_uuid,
        issued_at_unix=issued_at_unix,
    )
    if statement != expected:
        raise AcceptanceTransparencyError(
            "acceptance_preregistration_statement_binding_invalid"
        )
    return statement


def build_acceptance_preregistration_bundle(
    *,
    statement: Mapping[str, Any],
    producer_signature: bytes,
    producer_certificate_pem: bytes,
    rekor_uuid: str,
    rekor_entry: Mapping[str, Any],
    trusted_log_public_key_pem: bytes,
) -> dict[str, Any]:
    """Build a portable Rekor bundle for an acceptance preregistration."""

    try:
        mandate = AcceptanceVerificationMandate.from_dict(statement["mandate"])
        provision_receipt = AcceptanceMandateProvisionReceipt.from_dict(
            statement["provision_receipt"]
        )
        sequence = statement["sequence"]
        previous_statement_sha256 = statement["previous_statement_sha256"]
        previous_rekor_uuid = statement["previous_rekor_uuid"]
    except (KeyError, TypeError, ValueError) as exc:
        raise AcceptanceTransparencyError(
            "acceptance_preregistration_statement_binding_invalid"
        ) from exc
    statement_document = _validate_preregistration_statement(
        statement,
        mandate=mandate,
        provision_receipt=provision_receipt,
        expected_sequence=sequence,
        expected_previous_statement_sha256=previous_statement_sha256,
        expected_previous_rekor_uuid=previous_rekor_uuid,
    )
    bundle = build_acceptance_transparency_artifact_bundle(
        statement=statement_document,
        producer_signature=producer_signature,
        producer_certificate_pem=producer_certificate_pem,
        rekor_uuid=rekor_uuid,
        rekor_entry=rekor_entry,
        trusted_log_public_key_pem=trusted_log_public_key_pem,
    )
    if not isinstance(bundle, dict):
        raise AcceptanceTransparencyError("acceptance_transparency_bundle_invalid")
    return dict(bundle)


@dataclass(frozen=True, slots=True)
class PreregisteredAcceptanceReceipt:
    mandate: AcceptanceVerificationMandate
    provision_receipt: AcceptanceMandateProvisionReceipt
    transparency_bundle_sha256: str
    trusted_log_key_sha256: str
    rekor_uuid: str
    rekor_log_index: int
    rekor_integrated_time: int
    campaign_started_at_ns: int
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_mandate_binding(self.mandate, self.provision_receipt)
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
        if (
            type(self.campaign_started_at_ns) is not int
            or self.campaign_started_at_ns <= 0
        ):
            raise ValueError("campaign_started_at_ns must be a positive integer")
        if len(self.blockers) != len(set(self.blockers)):
            raise ValueError("blockers must be unique")

    @property
    def accepted(self) -> bool:
        return bool(self.transparency_bundle_sha256 and not self.blockers)

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        document = {
            "schema": ACCEPTANCE_PREREGISTRATION_VERIFICATION_SCHEMA,
            "mandate": self.mandate.to_dict(),
            "provision_receipt": self.provision_receipt.to_dict(),
            "transparency_bundle_sha256": self.transparency_bundle_sha256,
            "trusted_log_key_sha256": self.trusted_log_key_sha256,
            "rekor_uuid": self.rekor_uuid,
            "rekor_log_index": self.rekor_log_index,
            "rekor_integrated_time": self.rekor_integrated_time,
            "campaign_started_at_ns": self.campaign_started_at_ns,
            "blockers": list(self.blockers),
            "accepted": self.accepted,
        }
        if include_digest:
            document["preregistration_verification_sha256"] = self.sha256
        return document


def verify_acceptance_preregistration(
    mandate: AcceptanceVerificationMandate,
    provision_receipt: AcceptanceMandateProvisionReceipt,
    *,
    transparency_bundle: Mapping[str, Any] | None,
    trusted_log_public_key_pem: bytes | None,
    campaign_started_at_ns: int,
    expected_sequence: int,
    expected_previous_statement_sha256: str,
    expected_previous_rekor_uuid: str | None,
    minimum_log_index: int | None = None,
    minimum_integrated_time: int | None = None,
) -> PreregisteredAcceptanceReceipt:
    """Verify public preregistration and prove it predates physical execution."""

    _validate_mandate_binding(mandate, provision_receipt)
    if type(campaign_started_at_ns) is not int or campaign_started_at_ns <= 0:
        raise ValueError("campaign_started_at_ns must be a positive integer")
    artifact = verify_acceptance_transparency_artifact(
        transparency_bundle=transparency_bundle,
        trusted_log_public_key_pem=trusted_log_public_key_pem,
        statement_validator=lambda raw: _validate_preregistration_statement(
            raw,
            mandate=mandate,
            provision_receipt=provision_receipt,
            expected_sequence=expected_sequence,
            expected_previous_statement_sha256=expected_previous_statement_sha256,
            expected_previous_rekor_uuid=expected_previous_rekor_uuid,
        ),
        minimum_log_index=minimum_log_index,
        minimum_integrated_time=minimum_integrated_time,
    )
    blockers = list(artifact.blockers)
    if artifact.accepted and artifact.rekor_integrated_time >= (
        campaign_started_at_ns // 1_000_000_000
    ):
        blockers.append("acceptance_preregistration_not_strictly_before_campaign")
    return PreregisteredAcceptanceReceipt(
        mandate=mandate,
        provision_receipt=provision_receipt,
        transparency_bundle_sha256=artifact.transparency_bundle_sha256,
        trusted_log_key_sha256=artifact.trusted_log_key_sha256,
        rekor_uuid=artifact.rekor_uuid,
        rekor_log_index=artifact.rekor_log_index,
        rekor_integrated_time=artifact.rekor_integrated_time,
        campaign_started_at_ns=campaign_started_at_ns,
        blockers=tuple(sorted(set(blockers))),
    )


def persist_preregistered_acceptance_receipt(
    receipt: PreregisteredAcceptanceReceipt,
    path: str | Path,
) -> bool:
    """Create-once publish one portable preregistration verdict."""

    if not isinstance(receipt, PreregisteredAcceptanceReceipt):
        raise TypeError("receipt must be a PreregisteredAcceptanceReceipt")
    target = Path(path).expanduser().absolute()
    if not target.name or target.name in {".", ".."}:
        raise AcceptanceTransparencyError(
            "acceptance_preregistration_receipt_path_invalid"
        )
    payload = _canonical_bytes(receipt.to_dict())
    try:
        with DirectoryCustody.acquire(target.parent, create=True, private=True) as custody:
            published = bool(custody.write_bytes_once(target.name, payload, mode=0o600))
            fd = custody.open_file(target.name, os.O_RDONLY)
            try:
                if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                    raise AcceptanceTransparencyError(
                        "acceptance_preregistration_receipt_mode_invalid"
                    )
            finally:
                os.close(fd)
            existing = custody.read_bytes(target.name, max_bytes=4 * 1024 * 1024)
    except SecurePathCustodyError as exc:
        raise AcceptanceTransparencyError(
            "acceptance_preregistration_receipt_custody_invalid"
        ) from exc
    if not hmac.compare_digest(existing, payload):
        raise AcceptanceTransparencyError(
            "acceptance_preregistration_receipt_collision"
        )
    return published


__all__ = [
    "ACCEPTANCE_PREREGISTRATION_DOMAIN",
    "ACCEPTANCE_PREREGISTRATION_STATEMENT_SCHEMA",
    "ACCEPTANCE_PREREGISTRATION_VERIFICATION_SCHEMA",
    "PreregisteredAcceptanceReceipt",
    "build_acceptance_preregistration_bundle",
    "build_acceptance_preregistration_statement",
    "persist_preregistered_acceptance_receipt",
    "verify_acceptance_preregistration",
]
