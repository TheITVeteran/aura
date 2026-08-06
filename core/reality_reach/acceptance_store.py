"""Private create-once custody for Reality Reach acceptance evidence."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.reality_reach.acceptance_contracts import (
    _CERTIFICATE_SCHEMA,
    _CERTIFICATE_VERSION,
    _EVIDENCE_SCHEMA,
    _MAX_CERTIFICATE_BYTES,
    _MAX_EVIDENCE_BYTES,
    AcceptanceError,
    ConnectorAcceptanceCertificate,
    _canonical_json_bytes,
    _digest,
    _identifier,
    _strict_json_loads,
)
from core.runtime.secure_path_custody import DirectoryCustody, SecurePathCustodyError
from core.runtime.state_ownership import state_root

if TYPE_CHECKING:
    from core.reality_reach.metrology import AcquisitionReceipt


class AcceptanceCertificateStore:
    """Private, create-once certificate storage that survives process restart."""

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = (
            Path(root or (state_root() / "data" / "reality_reach" / "acceptance"))
            .expanduser()
            .absolute()
        )

    @property
    def root(self) -> Path:
        return self._root

    @staticmethod
    def _filename(campaign_id: object) -> str:
        canonical_id = _identifier(campaign_id, name="campaign_id")
        digest = _digest({"campaign_id": canonical_id}).removeprefix("sha256:")
        return f"{digest}.json"

    @classmethod
    def _evidence_filename(cls, campaign_id: object) -> str:
        return cls._filename(campaign_id).removesuffix(".json") + ".evidence"

    @staticmethod
    def _envelope(certificate: ConnectorAcceptanceCertificate) -> dict[str, Any]:
        return {
            "schema": _CERTIFICATE_SCHEMA,
            "version": _CERTIFICATE_VERSION,
            "campaign_id": certificate.campaign_id,
            "certificate": certificate.to_dict(),
            "certificate_sha256": certificate.sha256,
        }

    @staticmethod
    def _verify_private_file(custody: DirectoryCustody, filename: str) -> None:
        fd = custody.open_file(filename, os.O_RDONLY)
        try:
            observed = os.fstat(fd)
            if stat.S_IMODE(observed.st_mode) != 0o600:
                raise AcceptanceError("acceptance_certificate_mode_invalid")
        finally:
            os.close(fd)

    def persist(self, certificate: ConnectorAcceptanceCertificate) -> bool:
        if not isinstance(certificate, ConnectorAcceptanceCertificate):
            raise TypeError("certificate must be a ConnectorAcceptanceCertificate")
        filename = self._filename(certificate.campaign_id)
        payload = _canonical_json_bytes(self._envelope(certificate))
        try:
            with DirectoryCustody.acquire(self._root, create=True, private=True) as custody:
                published = bool(custody.write_bytes_once(filename, payload, mode=0o600))
                self._verify_private_file(custody, filename)
                existing = custody.read_bytes(filename, max_bytes=_MAX_CERTIFICATE_BYTES)
        except SecurePathCustodyError as exc:
            raise AcceptanceError("acceptance_certificate_custody_invalid") from exc
        if existing != payload:
            raise AcceptanceError("acceptance_campaign_collision")
        return published

    def load(self, campaign_id: str) -> ConnectorAcceptanceCertificate:
        canonical_id = _identifier(campaign_id, name="campaign_id")
        filename = self._filename(canonical_id)
        try:
            with DirectoryCustody.acquire(self._root, create=False, private=True) as custody:
                if not custody.file_exists(filename):
                    raise AcceptanceError("acceptance_certificate_missing")
                self._verify_private_file(custody, filename)
                payload = custody.read_bytes(filename, max_bytes=_MAX_CERTIFICATE_BYTES)
        except AcceptanceError:
            raise
        except SecurePathCustodyError as exc:
            raise AcceptanceError("acceptance_certificate_custody_invalid") from exc
        document = _strict_json_loads(payload)
        expected = {
            "schema",
            "version",
            "campaign_id",
            "certificate",
            "certificate_sha256",
        }
        if not isinstance(document, dict) or set(document) != expected:
            raise AcceptanceError("acceptance_certificate_envelope_schema_invalid")
        if (
            document["schema"] != _CERTIFICATE_SCHEMA
            or document["version"] != _CERTIFICATE_VERSION
            or document["campaign_id"] != canonical_id
            or not isinstance(document["certificate"], dict)
        ):
            raise AcceptanceError("acceptance_certificate_envelope_invalid")
        certificate = ConnectorAcceptanceCertificate.from_dict(document["certificate"])
        if certificate.campaign_id != canonical_id:
            raise AcceptanceError("acceptance_certificate_campaign_mismatch")
        if document["certificate_sha256"] != certificate.sha256:
            raise AcceptanceError("acceptance_certificate_digest_mismatch")
        if payload != _canonical_json_bytes(document):
            raise AcceptanceError("acceptance_certificate_noncanonical")
        return certificate

    def persist_evidence(
        self,
        certificate: ConnectorAcceptanceCertificate,
        case_evidence: Mapping[str, Any],
        *,
        metrology_receipt: AcquisitionReceipt | None = None,
        governance_evidence: Mapping[str, Any] | None = None,
    ) -> bool:
        """Create-once publish the evidence needed for independent replay."""

        if not isinstance(certificate, ConnectorAcceptanceCertificate):
            raise TypeError("certificate must be a ConnectorAcceptanceCertificate")
        expected_cases = {item.case_id: item for item in certificate.cases}
        if not isinstance(case_evidence, Mapping) or set(case_evidence) != set(expected_cases):
            raise AcceptanceError("acceptance_evidence_case_set_invalid")
        canonical_evidence: dict[str, Any] = {}
        for case_id in sorted(expected_cases):
            evidence = _strict_json_loads(_canonical_json_bytes(case_evidence[case_id]))
            if _digest(evidence) != expected_cases[case_id].evidence_sha256:
                raise AcceptanceError("acceptance_evidence_case_digest_mismatch")
            canonical_evidence[case_id] = evidence
        metrology = metrology_receipt.to_dict() if metrology_receipt is not None else {}
        if certificate.metrology_evidence_sha256:
            if (
                metrology_receipt is None
                or not metrology_receipt.verify_evidence()
                or metrology_receipt.evidence_sha256 != certificate.metrology_evidence_sha256
            ):
                raise AcceptanceError("acceptance_evidence_metrology_mismatch")
        elif metrology_receipt is not None:
            raise AcceptanceError("acceptance_evidence_unbound_metrology")
        governance = (
            _strict_json_loads(_canonical_json_bytes(governance_evidence))
            if governance_evidence is not None
            else {}
        )
        if certificate.governance_evidence_sha256:
            if (
                not isinstance(governance, Mapping)
                or _digest(governance) != certificate.governance_evidence_sha256
            ):
                raise AcceptanceError("acceptance_evidence_governance_mismatch")
        elif governance:
            raise AcceptanceError("acceptance_evidence_unbound_governance")
        document = {
            "schema": _EVIDENCE_SCHEMA,
            "campaign_id": certificate.campaign_id,
            "certificate_sha256": certificate.sha256,
            "case_evidence": canonical_evidence,
            "metrology_receipt": metrology,
            "governance_evidence": governance,
        }
        payload = _canonical_json_bytes(document, max_bytes=_MAX_EVIDENCE_BYTES)
        filename = self._evidence_filename(certificate.campaign_id)
        try:
            with DirectoryCustody.acquire(self._root, create=True, private=True) as custody:
                published = bool(custody.write_bytes_once(filename, payload, mode=0o600))
                self._verify_private_file(custody, filename)
                existing = custody.read_bytes(filename, max_bytes=_MAX_EVIDENCE_BYTES)
        except SecurePathCustodyError as exc:
            raise AcceptanceError("acceptance_evidence_custody_invalid") from exc
        if existing != payload:
            raise AcceptanceError("acceptance_evidence_campaign_collision")
        return published

    def load_evidence(
        self,
        certificate: ConnectorAcceptanceCertificate,
    ) -> dict[str, Any]:
        if not isinstance(certificate, ConnectorAcceptanceCertificate):
            raise TypeError("certificate must be a ConnectorAcceptanceCertificate")
        filename = self._evidence_filename(certificate.campaign_id)
        try:
            with DirectoryCustody.acquire(self._root, create=False, private=True) as custody:
                if not custody.file_exists(filename):
                    raise AcceptanceError("acceptance_evidence_missing")
                self._verify_private_file(custody, filename)
                payload = custody.read_bytes(filename, max_bytes=_MAX_EVIDENCE_BYTES)
        except AcceptanceError:
            raise
        except SecurePathCustodyError as exc:
            raise AcceptanceError("acceptance_evidence_custody_invalid") from exc
        document = _strict_json_loads(payload, max_bytes=_MAX_EVIDENCE_BYTES)
        if (
            not isinstance(document, dict)
            or set(document)
            != {
                "schema",
                "campaign_id",
                "certificate_sha256",
                "case_evidence",
                "metrology_receipt",
                "governance_evidence",
            }
            or document.get("schema") != _EVIDENCE_SCHEMA
            or document.get("campaign_id") != certificate.campaign_id
            or document.get("certificate_sha256") != certificate.sha256
            or not isinstance(document.get("case_evidence"), dict)
            or not isinstance(document.get("metrology_receipt"), dict)
            or not isinstance(document.get("governance_evidence"), dict)
        ):
            raise AcceptanceError("acceptance_evidence_envelope_invalid")
        expected_cases = {item.case_id: item for item in certificate.cases}
        evidence = document["case_evidence"]
        if set(evidence) != set(expected_cases):
            raise AcceptanceError("acceptance_evidence_case_set_invalid")
        for case_id, result in expected_cases.items():
            if _digest(evidence[case_id]) != result.evidence_sha256:
                raise AcceptanceError("acceptance_evidence_case_digest_mismatch")
        governance = document["governance_evidence"]
        if certificate.governance_evidence_sha256:
            if _digest(governance) != certificate.governance_evidence_sha256:
                raise AcceptanceError("acceptance_evidence_governance_mismatch")
        elif governance:
            raise AcceptanceError("acceptance_evidence_unbound_governance")
        if payload != _canonical_json_bytes(document, max_bytes=_MAX_EVIDENCE_BYTES):
            raise AcceptanceError("acceptance_evidence_noncanonical")
        return document

__all__ = ["AcceptanceCertificateStore"]
