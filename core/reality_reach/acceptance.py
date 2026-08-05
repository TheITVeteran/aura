"""Cross-protocol Reality Reach acceptance and fault-injection contracts.

The fault transport is explicit test/HIL infrastructure. It never upgrades
simulation to live evidence, and a post-dispatch fault remains indeterminate
because the wrapped transport may already have changed external state.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import stat
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from core.reality_reach.actuation import ActuationLease, ActuationState
from core.reality_reach.live import ReadingStatus, RealityReachService
from core.reality_reach.metrology import (
    AcquisitionMode,
    AcquisitionReceipt,
    EvidenceSource,
)
from core.reality_reach.scalar_adapter import (
    ScalarProtocolTransport,
    ScalarRealityAdapter,
    ScalarSample,
    ScalarTransportClass,
    ScalarWriteResult,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.lockdep import checked_async_lock
from core.runtime.secure_path_custody import DirectoryCustody, SecurePathCustodyError
from core.runtime.state_ownership import state_root

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_CASES = 128
_MAX_FAULT_RECEIPTS = 4096
_MAX_CERTIFICATE_BYTES = 1024 * 1024
_MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
_CERTIFICATE_SCHEMA = "aura.reality_reach.acceptance_certificate.v3"
_CERTIFICATE_VERSION = 3
_EVIDENCE_SCHEMA = "aura.reality_reach.acceptance_evidence.v3"
ACCEPTANCE_GOVERNANCE_SCHEMA = "aura.reality_reach.acceptance_governance.v1"

AcceptanceExecutor = Callable[..., Awaitable[Mapping[str, Any]]]
AcceptanceOperation = Callable[[], Awaitable["ConnectorAcceptanceCertificate"]]
AcceptanceMetrologyAcquirer = Callable[
    [AcceptanceOperation],
    Awaitable[tuple["ConnectorAcceptanceCertificate", AcquisitionReceipt]],
]

REQUIRED_SCALAR_ACCEPTANCE_CASES = (
    "observation.fresh",
    "cancellation.pre_dispatch",
    "actuation.prepare",
    "actuation.dispatch",
    "effect.independent_readback",
    "restoration.rollback",
)


def acceptance_governance_document(result: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce ActionExecutor output to the independently replayable fields."""

    return {
        "schema": ACCEPTANCE_GOVERNANCE_SCHEMA,
        "action_id": str(result.get("action_id") or ""),
        "request_digest": str(result.get("request_digest") or ""),
        "will_receipt_id": str(result.get("will_receipt_id") or ""),
        "post_action_receipt_id": str(result.get("post_action_receipt_id") or ""),
        "post_action_output_hash": str(result.get("post_action_output_hash") or ""),
        "status": str(result.get("status") or ""),
        "transport_succeeded": result.get("transport_succeeded") is True,
        "effect_verified": result.get("effect_verified") is True,
        "receipt_persisted": result.get("receipt_persisted") is True,
        "welfare_transaction_completed": (
            result.get("welfare_transaction_completed") is True
        ),
    }


def acceptance_governance_accepted(evidence: Mapping[str, Any]) -> bool:
    """Return true only for a complete, persisted, verified governance receipt."""

    return bool(
        evidence.get("schema") == ACCEPTANCE_GOVERNANCE_SCHEMA
        and evidence.get("action_id")
        and evidence.get("request_digest")
        and evidence.get("will_receipt_id")
        and evidence.get("post_action_receipt_id")
        and _DIGEST.fullmatch(str(evidence.get("request_digest") or ""))
        and _DIGEST.fullmatch(str(evidence.get("post_action_output_hash") or ""))
        and evidence.get("status") == "success_verified"
        and evidence.get("transport_succeeded") is True
        and evidence.get("effect_verified") is True
        and evidence.get("receipt_persisted") is True
        and evidence.get("welfare_transaction_completed") is True
    )


class AcceptanceError(RuntimeError):
    """An acceptance evidence or fault-injection contract failed."""


class AcceptanceEvidenceClass(StrEnum):
    SIMULATION = "simulation"
    HARDWARE_IN_LOOP = "hardware_in_loop"
    LIVE = "live"


class AcceptanceVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNMEASURED = "unmeasured"


class ScalarFault(StrEnum):
    READ_PARTITION = "read_partition"
    WRITE_PARTITION = "write_partition"
    WRITE_OUTCOME_UNKNOWN = "write_outcome_unknown"
    STALE_READBACK = "stale_readback"
    DUPLICATE_READBACK = "duplicate_readback"
    REORDERED_READBACK = "reordered_readback"


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


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


def _strict_json_loads(payload: bytes, *, max_bytes: int = _MAX_CERTIFICATE_BYTES) -> Any:
    if not payload or len(payload) > max_bytes:
        raise AcceptanceError("acceptance_certificate_size_invalid")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AcceptanceError("acceptance_certificate_not_utf8") from exc

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AcceptanceError("acceptance_certificate_duplicate_json_key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise AcceptanceError("acceptance_certificate_non_finite_number")

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except AcceptanceError:
        raise
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise AcceptanceError("acceptance_certificate_json_invalid") from exc


def _canonical_json_bytes(value: Any, *, max_bytes: int = _MAX_CERTIFICATE_BYTES) -> bytes:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise AcceptanceError("acceptance_certificate_not_canonical_json") from exc
    if not payload or len(payload) > max_bytes:
        raise AcceptanceError("acceptance_certificate_size_invalid")
    return payload


@dataclass(frozen=True, slots=True)
class AcceptanceCaseResult:
    case_id: str
    verdict: AcceptanceVerdict
    evidence_class: AcceptanceEvidenceClass
    required: bool
    evidence_sha256: str
    duration_ms: float
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _identifier(self.case_id, name="case_id"))
        if not isinstance(self.verdict, AcceptanceVerdict):
            raise TypeError("verdict must be an AcceptanceVerdict")
        if not isinstance(self.evidence_class, AcceptanceEvidenceClass):
            raise TypeError("evidence_class must be an AcceptanceEvidenceClass")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a bool")
        object.__setattr__(
            self,
            "evidence_sha256",
            _sha256(self.evidence_sha256, name="evidence_sha256"),
        )
        duration = float(self.duration_ms)
        if not math.isfinite(duration) or not 0.0 <= duration <= 86_400_000.0:
            raise ValueError("duration_ms must be bounded and non-negative")
        object.__setattr__(self, "duration_ms", duration)
        detail = str(self.detail or "").strip()
        if len(detail.encode("utf-8")) > 1024:
            raise ValueError("acceptance case detail is too large")
        object.__setattr__(self, "detail", detail)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "verdict": self.verdict.value,
            "evidence_class": self.evidence_class.value,
            "required": self.required,
            "evidence_sha256": self.evidence_sha256,
            "duration_ms": self.duration_ms,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> AcceptanceCaseResult:
        expected = {
            "case_id",
            "verdict",
            "evidence_class",
            "required",
            "evidence_sha256",
            "duration_ms",
            "detail",
        }
        if not isinstance(document, Mapping) or set(document) != expected:
            raise AcceptanceError("acceptance_case_schema_invalid")
        try:
            return cls(
                case_id=document["case_id"],
                verdict=AcceptanceVerdict(document["verdict"]),
                evidence_class=AcceptanceEvidenceClass(document["evidence_class"]),
                required=document["required"],
                evidence_sha256=document["evidence_sha256"],
                duration_ms=document["duration_ms"],
                detail=document["detail"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AcceptanceError("acceptance_case_invalid") from exc


@dataclass(frozen=True, slots=True)
class ConnectorAcceptanceCertificate:
    campaign_id: str
    connector_id: str
    adapter_id: str
    physical_identity_sha256: str
    source_commit_sha256: str
    target: float
    target_tolerance: float
    started_at_ns: int
    completed_at_ns: int
    cases: tuple[AcceptanceCaseResult, ...]
    scenario_id: str = ""
    metrology_evidence_sha256: str = ""
    governance_evidence_sha256: str = ""
    governance_accepted: bool = False

    def __post_init__(self) -> None:
        for name in ("campaign_id", "connector_id", "adapter_id"):
            object.__setattr__(
                self,
                name,
                _identifier(getattr(self, name), name=name),
            )
        for name in ("physical_identity_sha256", "source_commit_sha256"):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name=name),
            )
        target = float(self.target)
        tolerance = float(self.target_tolerance)
        if not math.isfinite(target) or not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("acceptance target and tolerance must be finite and non-negative")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "target_tolerance", tolerance)
        for name in ("started_at_ns", "completed_at_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.completed_at_ns < self.started_at_ns:
            raise ValueError("acceptance completion precedes start")
        cases = tuple(self.cases)
        if not 1 <= len(cases) <= _MAX_CASES:
            raise ValueError(f"acceptance requires between 1 and {_MAX_CASES} cases")
        if len({item.case_id for item in cases}) != len(cases):
            raise ValueError("acceptance case ids must be unique")
        object.__setattr__(self, "cases", cases)
        scenario = str(self.scenario_id or "").strip()
        if scenario:
            scenario = _identifier(scenario, name="scenario_id")
        object.__setattr__(self, "scenario_id", scenario)
        metrology = str(self.metrology_evidence_sha256 or "").strip().lower()
        if metrology:
            metrology = _sha256(
                metrology,
                name="metrology_evidence_sha256",
            )
        object.__setattr__(self, "metrology_evidence_sha256", metrology)
        governance = str(self.governance_evidence_sha256 or "").strip().lower()
        if governance:
            governance = _sha256(governance, name="governance_evidence_sha256")
        object.__setattr__(self, "governance_evidence_sha256", governance)
        if not isinstance(self.governance_accepted, bool):
            raise TypeError("governance_accepted must be a bool")
        if self.governance_accepted and not governance:
            raise ValueError("accepted governance requires bound evidence")
        if (
            any(item.evidence_class is AcceptanceEvidenceClass.HARDWARE_IN_LOOP for item in cases)
            and not scenario
        ):
            raise ValueError("HIL acceptance evidence requires a scenario_id")

    @property
    def deterministic_passed(self) -> bool:
        return all(
            not item.required or item.verdict is AcceptanceVerdict.PASS for item in self.cases
        )

    @property
    def live_acceptance_passed(self) -> bool:
        return bool(
            self.physical_evidence_passed
            and self.governance_accepted
            and self.governance_evidence_sha256
        )

    @property
    def physical_evidence_passed(self) -> bool:
        """Producer-side physical/metrology verdict before governance replay."""

        return bool(
            self.deterministic_passed
            and self.metrology_evidence_sha256
            and any(
                item.required
                and item.verdict is AcceptanceVerdict.PASS
                and item.evidence_class
                in {
                    AcceptanceEvidenceClass.HARDWARE_IN_LOOP,
                    AcceptanceEvidenceClass.LIVE,
                }
                for item in self.cases
            )
            and all(
                item.verdict is AcceptanceVerdict.PASS
                for item in self.cases
                if item.required
                and item.evidence_class
                in {
                    AcceptanceEvidenceClass.HARDWARE_IN_LOOP,
                    AcceptanceEvidenceClass.LIVE,
                }
            )
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "connector_id": self.connector_id,
            "adapter_id": self.adapter_id,
            "physical_identity_sha256": self.physical_identity_sha256,
            "source_commit_sha256": self.source_commit_sha256,
            "target": self.target,
            "target_tolerance": self.target_tolerance,
            "started_at_ns": self.started_at_ns,
            "completed_at_ns": self.completed_at_ns,
            "scenario_id": self.scenario_id,
            "metrology_evidence_sha256": self.metrology_evidence_sha256,
            "governance_evidence_sha256": self.governance_evidence_sha256,
            "governance_accepted": self.governance_accepted,
            "cases": [item.to_dict() for item in self.cases],
            "deterministic_passed": self.deterministic_passed,
            "live_acceptance_passed": self.live_acceptance_passed,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> ConnectorAcceptanceCertificate:
        expected = {
            "campaign_id",
            "connector_id",
            "adapter_id",
            "physical_identity_sha256",
            "source_commit_sha256",
            "target",
            "target_tolerance",
            "started_at_ns",
            "completed_at_ns",
            "scenario_id",
            "metrology_evidence_sha256",
            "governance_evidence_sha256",
            "governance_accepted",
            "cases",
            "deterministic_passed",
            "live_acceptance_passed",
        }
        if not isinstance(document, Mapping) or set(document) != expected:
            raise AcceptanceError("acceptance_certificate_body_schema_invalid")
        raw_cases = document.get("cases")
        if not isinstance(raw_cases, list):
            raise AcceptanceError("acceptance_certificate_cases_invalid")
        try:
            certificate = cls(
                campaign_id=document["campaign_id"],
                connector_id=document["connector_id"],
                adapter_id=document["adapter_id"],
                physical_identity_sha256=document["physical_identity_sha256"],
                source_commit_sha256=document["source_commit_sha256"],
                target=document["target"],
                target_tolerance=document["target_tolerance"],
                started_at_ns=document["started_at_ns"],
                completed_at_ns=document["completed_at_ns"],
                cases=tuple(AcceptanceCaseResult.from_dict(item) for item in raw_cases),
                scenario_id=document["scenario_id"],
                metrology_evidence_sha256=document["metrology_evidence_sha256"],
                governance_evidence_sha256=document["governance_evidence_sha256"],
                governance_accepted=document["governance_accepted"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AcceptanceError("acceptance_certificate_body_invalid") from exc
        if (
            document["deterministic_passed"] is not certificate.deterministic_passed
            or document["live_acceptance_passed"] is not certificate.live_acceptance_passed
        ):
            raise AcceptanceError("acceptance_certificate_derived_verdict_invalid")
        return certificate


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


@dataclass(frozen=True, slots=True)
class ScalarAcceptancePlan:
    campaign_id: str
    connector_id: str
    target: float
    source_commit_sha256: str
    authority_receipt_id: str
    evidence_class: AcceptanceEvidenceClass = AcceptanceEvidenceClass.SIMULATION
    scenario_id: str = ""
    deadline_s: float = 30.0
    metrology_effect_hold_s: float = 0.25
    metrology_receipt: AcquisitionReceipt | None = None

    def __post_init__(self) -> None:
        for name in ("campaign_id", "connector_id", "authority_receipt_id"):
            object.__setattr__(
                self,
                name,
                _identifier(getattr(self, name), name=name),
            )
        target = float(self.target)
        if not math.isfinite(target):
            raise ValueError("acceptance target must be finite")
        object.__setattr__(self, "target", target)
        object.__setattr__(
            self,
            "source_commit_sha256",
            _sha256(self.source_commit_sha256, name="source_commit_sha256"),
        )
        if not isinstance(self.evidence_class, AcceptanceEvidenceClass):
            raise TypeError("evidence_class must be an AcceptanceEvidenceClass")
        scenario = str(self.scenario_id or "").strip()
        if scenario:
            scenario = _identifier(scenario, name="scenario_id")
        if self.evidence_class is AcceptanceEvidenceClass.HARDWARE_IN_LOOP and not scenario:
            raise ValueError("HIL acceptance plans require a scenario_id")
        object.__setattr__(self, "scenario_id", scenario)
        receipt = self.metrology_receipt
        if self.evidence_class is AcceptanceEvidenceClass.SIMULATION:
            if receipt is not None and receipt.mode is not AcquisitionMode.SIMULATION:
                raise ValueError("simulation acceptance cannot bind live or HIL metrology")
        elif receipt is not None:
            raise ValueError(
                "physical acceptance metrology must be acquired around the operation"
            )
        deadline = float(self.deadline_s)
        if not math.isfinite(deadline) or not 0.1 <= deadline <= 300.0:
            raise ValueError("deadline_s must lie inside [0.1, 300]")
        object.__setattr__(self, "deadline_s", deadline)
        hold = float(self.metrology_effect_hold_s)
        if not math.isfinite(hold) or not 0.05 <= hold <= min(5.0, deadline):
            raise ValueError(
                "metrology_effect_hold_s must lie inside [0.05, min(5, deadline)]"
            )
        object.__setattr__(self, "metrology_effect_hold_s", hold)


class ScalarAcceptanceRunner:
    """Exercise one scalar adapter under external governance for physical runs."""

    _CASE_IDS = REQUIRED_SCALAR_ACCEPTANCE_CASES

    def __init__(
        self,
        adapter: ScalarRealityAdapter,
        service: RealityReachService,
        plan: ScalarAcceptancePlan,
        *,
        governed_executor: AcceptanceExecutor | None = None,
        metrology_acquirer: AcceptanceMetrologyAcquirer | None = None,
    ) -> None:
        if not isinstance(adapter, ScalarRealityAdapter):
            raise TypeError("adapter must be a ScalarRealityAdapter")
        if not isinstance(service, RealityReachService):
            raise TypeError("service must be a RealityReachService")
        if not isinstance(plan, ScalarAcceptancePlan):
            raise TypeError("plan must be a ScalarAcceptancePlan")
        capabilities = adapter.actuator_capabilities()
        if not any(
            service.adapter_id_for_channel(item.channel_id) == adapter.adapter_id
            for item in capabilities
        ):
            raise AcceptanceError("acceptance_adapter_not_registered")
        if not capabilities:
            raise AcceptanceError("acceptance_adapter_is_read_only")
        self._adapter = adapter
        self._service = service
        self._plan = plan
        self._governed_executor = governed_executor
        self._metrology_acquirer = metrology_acquirer
        self._metrology_receipt = plan.metrology_receipt
        self._observation_channels = tuple(
            dict.fromkeys(
                channel_id
                for capability in capabilities
                for channel_id in capability.observation_channels
            )
        )
        self._case_evidence: dict[str, Any] = {}
        self._governance_evidence: dict[str, Any] = {}

    @property
    def case_evidence(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            _strict_json_loads(_canonical_json_bytes(self._case_evidence)),
        )

    @property
    def metrology_receipt(self) -> AcquisitionReceipt | None:
        return self._metrology_receipt

    @property
    def governance_evidence(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            _strict_json_loads(_canonical_json_bytes(self._governance_evidence)),
        )

    def _lease(
        self,
        command_sha256: str,
        *,
        suffix: str,
        authority_receipt_id: str,
    ) -> ActuationLease:
        now_wall = time.time_ns()
        now_mono = time.monotonic_ns()
        duration_ns = int(self._plan.deadline_s * 1_000_000_000)
        return ActuationLease(
            lease_id=f"lease.{self._plan.campaign_id}.{suffix}"[:128],
            command_sha256=command_sha256,
            adapter_id=self._adapter.adapter_id,
            session_id=self._service.session_id,
            authority_receipt_id=authority_receipt_id,
            issued_at_ns=now_wall,
            expires_at_ns=now_wall + duration_ns,
            issued_monotonic_ns=now_mono,
            expires_monotonic_ns=now_mono + duration_ns,
        )

    def _result(
        self,
        case_id: str,
        verdict: AcceptanceVerdict,
        *,
        started_ns: int,
        evidence: Any,
        detail: str = "",
    ) -> AcceptanceCaseResult:
        canonical_evidence = _strict_json_loads(_canonical_json_bytes(evidence))
        self._case_evidence[case_id] = canonical_evidence
        return AcceptanceCaseResult(
            case_id=case_id,
            verdict=verdict,
            evidence_class=self._plan.evidence_class,
            required=True,
            evidence_sha256=_digest(canonical_evidence),
            duration_ms=max(0.0, (time.monotonic_ns() - started_ns) / 1_000_000),
            detail=detail,
        )

    def _failure(
        self,
        case_id: str,
        *,
        started_ns: int,
        error: BaseException,
    ) -> AcceptanceCaseResult:
        error_type = type(error).__name__
        return self._result(
            case_id,
            AcceptanceVerdict.FAIL,
            started_ns=started_ns,
            evidence={
                "error_type": error_type,
                "error_sha256": _digest(str(error)),
            },
            detail=error_type,
        )

    async def _run_cases(self, *, authority_receipt_id: str) -> ConnectorAcceptanceCertificate:
        self._case_evidence = {}
        started_at_ns = max(1, time.time_ns())
        results: dict[str, AcceptanceCaseResult] = {}
        command = None
        actuation = None

        case_started = time.monotonic_ns()
        try:
            reading = await self._adapter.refresh_readback()
            observation_passed = bool(
                reading.status is ReadingStatus.AVAILABLE
                and reading.value is not None
                and reading.source_event_id
            )
            results["observation.fresh"] = self._result(
                "observation.fresh",
                AcceptanceVerdict.PASS if observation_passed else AcceptanceVerdict.FAIL,
                started_ns=case_started,
                evidence=reading.to_dict(),
                detail="fresh identified readback"
                if observation_passed
                else "readback unavailable",
            )
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            results["observation.fresh"] = self._failure(
                "observation.fresh",
                started_ns=case_started,
                error=exc,
            )

        if results["observation.fresh"].verdict is AcceptanceVerdict.PASS:
            case_started = time.monotonic_ns()
            try:
                cancel_command = await self._adapter.compile_target(
                    self._plan.target,
                    inventory_sha256=self._service.status()["registry_sha256"],
                    deadline_s=self._plan.deadline_s,
                    idempotency_key=f"{self._plan.campaign_id}.cancel",
                    source="reality_reach.acceptance",
                )
                cancel_lease = self._lease(
                    cancel_command.sha256,
                    suffix="cancel",
                    authority_receipt_id=authority_receipt_id,
                )
                cancel_prepared = await self._adapter.prepare(
                    cancel_command,
                    cancel_lease,
                )
                cancellation = await self._adapter.cancel(
                    cancel_command,
                    cancel_prepared,
                )
                passed = bool(
                    cancellation.state is ActuationState.CANCELLED
                    and cancellation.executed is False
                    and cancellation.transport_completed is False
                )
                results["cancellation.pre_dispatch"] = self._result(
                    "cancellation.pre_dispatch",
                    AcceptanceVerdict.PASS if passed else AcceptanceVerdict.FAIL,
                    started_ns=case_started,
                    evidence=cancellation.to_dict(),
                    detail="cancelled before transport"
                    if passed
                    else "cancellation contract failed",
                )
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                results["cancellation.pre_dispatch"] = self._failure(
                    "cancellation.pre_dispatch",
                    started_ns=case_started,
                    error=exc,
                )

        if all(
            results.get(case_id) is not None and results[case_id].verdict is AcceptanceVerdict.PASS
            for case_id in ("observation.fresh", "cancellation.pre_dispatch")
        ):
            case_started = time.monotonic_ns()
            try:
                command = await self._adapter.compile_target(
                    self._plan.target,
                    inventory_sha256=self._service.status()["registry_sha256"],
                    deadline_s=self._plan.deadline_s,
                    idempotency_key=f"{self._plan.campaign_id}.actuate",
                    source="reality_reach.acceptance",
                )
                lease = self._lease(
                    command.sha256,
                    suffix="actuate",
                    authority_receipt_id=authority_receipt_id,
                )
                prepared = await self._adapter.prepare(command, lease)
                results["actuation.prepare"] = self._result(
                    "actuation.prepare",
                    AcceptanceVerdict.PASS,
                    started_ns=case_started,
                    evidence=prepared.to_dict(),
                    detail="preconditions fenced",
                )
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                results["actuation.prepare"] = self._failure(
                    "actuation.prepare",
                    started_ns=case_started,
                    error=exc,
                )
            else:
                case_started = time.monotonic_ns()
                try:
                    actuation = await self._adapter.actuate(command, lease, prepared)
                    passed = actuation.state is ActuationState.EXECUTED
                    results["actuation.dispatch"] = self._result(
                        "actuation.dispatch",
                        AcceptanceVerdict.PASS if passed else AcceptanceVerdict.FAIL,
                        started_ns=case_started,
                        evidence=actuation.to_dict(),
                        detail="transport completed" if passed else "transport not completed",
                    )
                except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                    results["actuation.dispatch"] = self._failure(
                        "actuation.dispatch",
                        started_ns=case_started,
                        error=exc,
                    )

        if (
            command is not None
            and actuation is not None
            and results.get("actuation.dispatch") is not None
            and results["actuation.dispatch"].verdict is AcceptanceVerdict.PASS
        ):
            case_started = time.monotonic_ns()
            try:
                effect = await self._adapter.verify_effect(command, actuation)
                passed = bool(
                    effect.state is ActuationState.EFFECT_VERIFIED and effect.independently_observed
                )
                results["effect.independent_readback"] = self._result(
                    "effect.independent_readback",
                    AcceptanceVerdict.PASS if passed else AcceptanceVerdict.FAIL,
                    started_ns=case_started,
                    evidence=effect.to_dict(),
                    detail="independent fresh effect"
                    if passed
                    else "effect not independently verified",
                )
                if (
                    passed
                    and self._plan.evidence_class
                    is not AcceptanceEvidenceClass.SIMULATION
                ):
                    await asyncio.sleep(self._plan.metrology_effect_hold_s)
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                results["effect.independent_readback"] = self._failure(
                    "effect.independent_readback",
                    started_ns=case_started,
                    error=exc,
                )

        if command is not None:
            case_started = time.monotonic_ns()
            try:
                rollback = (
                    await self._adapter.rollback(command, actuation)
                    if actuation is not None
                    else await self._adapter.safe_state(command, None)
                )
                passed = rollback.state in {
                    ActuationState.ROLLED_BACK,
                    ActuationState.SAFE_STATE,
                }
                results["restoration.rollback"] = self._result(
                    "restoration.rollback",
                    AcceptanceVerdict.PASS if passed else AcceptanceVerdict.FAIL,
                    started_ns=case_started,
                    evidence=rollback.to_dict(),
                    detail=(
                        "initial or safe state restored"
                        if passed
                        else "restoration not independently verified"
                    ),
                )
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
                results["restoration.rollback"] = self._failure(
                    "restoration.rollback",
                    started_ns=case_started,
                    error=exc,
                )

        for case_id in self._CASE_IDS:
            if case_id not in results:
                results[case_id] = self._result(
                    case_id,
                    AcceptanceVerdict.UNMEASURED,
                    started_ns=time.monotonic_ns(),
                    evidence={"blocked_by": [item.to_dict() for item in results.values()]},
                    detail="blocked by an earlier required acceptance failure",
                )
        completed_at_ns = max(started_at_ns, time.time_ns())
        return ConnectorAcceptanceCertificate(
            campaign_id=self._plan.campaign_id,
            connector_id=self._plan.connector_id,
            adapter_id=self._adapter.adapter_id,
            physical_identity_sha256=self._adapter.physical_identity_sha256,
            source_commit_sha256=self._plan.source_commit_sha256,
            target=self._plan.target,
            target_tolerance=self._adapter.effect_tolerance,
            started_at_ns=started_at_ns,
            completed_at_ns=completed_at_ns,
            cases=tuple(results[case_id] for case_id in self._CASE_IDS),
            scenario_id=self._plan.scenario_id,
        )

    def _bind_metrology(
        self,
        certificate: ConnectorAcceptanceCertificate,
        receipt: AcquisitionReceipt,
    ) -> ConnectorAcceptanceCertificate:
        if not isinstance(receipt, AcquisitionReceipt) or not receipt.verify_evidence():
            raise AcceptanceError("acceptance_metrology_evidence_invalid")
        expected_mode = (
            AcquisitionMode.HARDWARE_IN_LOOP
            if self._plan.evidence_class is AcceptanceEvidenceClass.HARDWARE_IN_LOOP
            else AcquisitionMode.LIVE
        )
        if receipt.mode is not expected_mode or receipt.restored_mode is not AcquisitionMode.LIVE:
            raise AcceptanceError("acceptance_metrology_mode_mismatch")
        if not (
            receipt.started_at_ns <= certificate.started_at_ns
            and receipt.completed_at_ns >= certificate.completed_at_ns
        ):
            raise AcceptanceError("acceptance_metrology_does_not_enclose_operation")
        sources = {item.source for item in receipt.measurements}
        expected_sources = (
            {EvidenceSource.LIVE, EvidenceSource.SIMULATED}
            if expected_mode is AcquisitionMode.HARDWARE_IN_LOOP
            else {EvidenceSource.LIVE}
        )
        if sources != expected_sources:
            raise AcceptanceError("acceptance_metrology_source_class_mismatch")
        if (
            expected_mode is AcquisitionMode.HARDWARE_IN_LOOP
            and receipt.scenario_id != self._plan.scenario_id
        ):
            raise AcceptanceError("acceptance_metrology_scenario_mismatch")
        measured_live = {
            item.channel_id
            for item in receipt.measurements
            if item.source is EvidenceSource.LIVE
        }
        if not set(self._observation_channels).issubset(measured_live):
            raise AcceptanceError("acceptance_metrology_readback_channel_missing")
        target_observed = any(
            item.source is EvidenceSource.LIVE
            and item.channel_id in self._observation_channels
            and abs(float(item.value) - certificate.target)
            <= certificate.target_tolerance
            for item in receipt.measurements
        )
        if not target_observed:
            raise AcceptanceError("acceptance_metrology_target_not_observed")
        self._metrology_receipt = receipt
        return replace(
            certificate,
            metrology_evidence_sha256=receipt.evidence_sha256,
        )

    @staticmethod
    def _governance_document(result: Mapping[str, Any]) -> dict[str, Any]:
        return acceptance_governance_document(result)

    @staticmethod
    def _governance_accepted(evidence: Mapping[str, Any]) -> bool:
        return acceptance_governance_accepted(evidence)

    async def _run_governed(self) -> ConnectorAcceptanceCertificate:
        from core.governance.will import ActionDomain
        from core.runtime.action_executor import ActionExecutor
        from core.runtime.skill_contract import ActionExpectation

        executor = self._governed_executor or ActionExecutor.execute
        completed: dict[str, ConnectorAcceptanceCertificate] = {}

        async def effect_handler(context: Mapping[str, Any]) -> Mapping[str, Any]:
            authority_receipt_id = str(context.get("will_receipt_id") or "")
            if not _IDENTIFIER.fullmatch(authority_receipt_id):
                raise AcceptanceError("acceptance_governance_receipt_invalid")
            acquirer = self._metrology_acquirer
            if acquirer is None:
                raise AcceptanceError("acceptance_metrology_acquirer_missing")
            certificate, receipt = await acquirer(
                lambda: self._run_cases(authority_receipt_id=authority_receipt_id)
            )
            certificate = self._bind_metrology(certificate, receipt)
            completed["certificate"] = certificate
            dispatch = next(
                item for item in certificate.cases if item.case_id == "actuation.dispatch"
            )
            return {
                "ok": certificate.deterministic_passed,
                "transport_succeeded": dispatch.verdict is AcceptanceVerdict.PASS,
                "acceptance_certificate_sha256": certificate.sha256,
                "metrology_evidence_sha256": certificate.metrology_evidence_sha256,
            }

        async def effect_verifier(_context: Mapping[str, Any]) -> Mapping[str, Any]:
            certificate = completed.get("certificate")
            return {
                "effect_verified": bool(
                    certificate is not None and certificate.physical_evidence_passed
                ),
                "acceptance_certificate_sha256": (
                    certificate.sha256 if certificate is not None else ""
                ),
            }

        raw_result = await executor(
            domain=ActionDomain.ENVIRONMENT_ACTION,
            action_name=f"reality_reach.acceptance.{self._plan.connector_id}",
            params={
                "campaign_id": self._plan.campaign_id,
                "adapter_id": self._adapter.adapter_id,
                "physical_identity_sha256": self._adapter.physical_identity_sha256,
                "evidence_class": self._plan.evidence_class.value,
                "target": self._plan.target,
            },
            source="reality_reach.acceptance",
            rollback_target="adapter.rollback_or_safe_state",
            expectation=ActionExpectation(
                objective="exercise and restore one declared physical effect",
                required_evidence=[
                    "acceptance_certificate_sha256",
                    "metrology_evidence_sha256",
                ],
                rollback_hint="restore the pre-dispatch value or declared safe state",
                allow_partial=False,
            ),
            effect_handler=effect_handler,
            effect_verifier=effect_verifier,
            execution_timeout_s=self._plan.deadline_s,
            verification_timeout_s=self._plan.deadline_s,
            action_id=f"acceptance.{self._plan.campaign_id}"[:128],
        )
        if not isinstance(raw_result, Mapping):
            raise AcceptanceError("acceptance_governance_result_invalid")
        certificate = completed.get("certificate")
        if certificate is None:
            raise AcceptanceError("acceptance_governance_refused_before_dispatch")
        governance = self._governance_document(raw_result)
        self._governance_evidence = governance
        accepted = self._governance_accepted(governance)
        return replace(
            certificate,
            governance_evidence_sha256=_digest(governance),
            governance_accepted=accepted,
        )

    async def run(self) -> ConnectorAcceptanceCertificate:
        self._governance_evidence = {}
        self._metrology_receipt = self._plan.metrology_receipt
        if self._plan.evidence_class is AcceptanceEvidenceClass.SIMULATION:
            if self._adapter.transport_class is not ScalarTransportClass.SIMULATED:
                raise AcceptanceError(
                    "simulation_acceptance_requires_simulated_adapter"
                )
            return await self._run_cases(
                authority_receipt_id=self._plan.authority_receipt_id,
            )
        if self._adapter.transport_class is not ScalarTransportClass.PHYSICAL:
            raise AcceptanceError("physical_acceptance_requires_physical_adapter")
        if self._metrology_acquirer is None:
            raise AcceptanceError("physical_acceptance_requires_metrology_acquirer")
        return await self._run_governed()

    async def run_and_persist(
        self,
        store: AcceptanceCertificateStore | None = None,
    ) -> ConnectorAcceptanceCertificate:
        """Run once and create-once publish both verdict and replay evidence."""

        target_store = store or AcceptanceCertificateStore()
        if not isinstance(target_store, AcceptanceCertificateStore):
            raise TypeError("store must be an AcceptanceCertificateStore")
        certificate = await self.run()
        target_store.persist(certificate)
        target_store.persist_evidence(
            certificate,
            self.case_evidence,
            metrology_receipt=self.metrology_receipt,
            governance_evidence=(
                self.governance_evidence if certificate.governance_evidence_sha256 else None
            ),
        )
        return certificate


@dataclass(frozen=True, slots=True)
class FaultInjectionReceipt:
    sequence: int
    fault: ScalarFault
    operation: str
    resource_sha256: str
    injected_at_ns: int
    delegate_called: bool
    outcome_indeterminate: bool
    evidence_class: AcceptanceEvidenceClass
    scenario_id: str

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or self.sequence <= 0:
            raise ValueError("fault receipt sequence must be positive")
        if not isinstance(self.fault, ScalarFault):
            raise TypeError("fault must be a ScalarFault")
        object.__setattr__(self, "operation", _identifier(self.operation, name="operation"))
        object.__setattr__(
            self,
            "resource_sha256",
            _sha256(self.resource_sha256, name="resource_sha256"),
        )
        if isinstance(self.injected_at_ns, bool) or self.injected_at_ns <= 0:
            raise ValueError("injected_at_ns must be positive")
        if not isinstance(self.delegate_called, bool) or not isinstance(
            self.outcome_indeterminate,
            bool,
        ):
            raise TypeError("fault receipt booleans must be explicit")
        if self.outcome_indeterminate and not self.delegate_called:
            raise ValueError("indeterminate effect requires delegate dispatch")
        if self.evidence_class not in {
            AcceptanceEvidenceClass.SIMULATION,
            AcceptanceEvidenceClass.HARDWARE_IN_LOOP,
        }:
            raise ValueError("fault injection evidence must be simulation or HIL")
        scenario = _identifier(self.scenario_id, name="scenario_id")
        object.__setattr__(self, "scenario_id", scenario)

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "fault": self.fault.value,
            "operation": self.operation,
            "resource_sha256": self.resource_sha256,
            "injected_at_ns": self.injected_at_ns,
            "delegate_called": self.delegate_called,
            "outcome_indeterminate": self.outcome_indeterminate,
            "evidence_class": self.evidence_class.value,
            "scenario_id": self.scenario_id,
        }


class FaultInjectedReadError(ConnectionError):
    """A deterministic read partition was injected before transport."""


class FaultInjectedWriteError(ConnectionError):
    """A deterministic write partition was injected before transport."""


class FaultInjectedOutcomeUnknownError(TimeoutError):
    """The wrapped write completed but its caller lost the acknowledgement."""


class FaultInjectingScalarTransport:
    """One-shot deterministic faults around a real or simulated scalar transport."""

    def __init__(
        self,
        delegate: ScalarProtocolTransport,
        *,
        evidence_class: AcceptanceEvidenceClass,
        scenario_id: str,
        stale_age_s: float = 3600.0,
    ) -> None:
        if not isinstance(delegate, ScalarProtocolTransport):
            raise TypeError("delegate must satisfy ScalarProtocolTransport")
        if evidence_class not in {
            AcceptanceEvidenceClass.SIMULATION,
            AcceptanceEvidenceClass.HARDWARE_IN_LOOP,
        }:
            raise ValueError("fault transport requires simulation or HIL evidence")
        self._delegate = delegate
        self._evidence_class = evidence_class
        self._scenario_id = _identifier(scenario_id, name="scenario_id")
        stale_age = float(stale_age_s)
        if not math.isfinite(stale_age) or not 1.0 <= stale_age <= 604_800.0:
            raise ValueError("stale_age_s must lie inside [1, 604800]")
        self._stale_age_ns = int(stale_age * 1_000_000_000)
        self._armed: deque[ScalarFault] = deque()
        self._samples: deque[ScalarSample] = deque(maxlen=2)
        self._receipts: deque[FaultInjectionReceipt] = deque(maxlen=_MAX_FAULT_RECEIPTS)
        self._sequence = 0
        self._lock = checked_async_lock("reality_reach.fault_injection")

    @property
    def transport_id(self) -> str:
        return f"acceptance.{self._delegate.transport_id}"

    @property
    def receipts(self) -> tuple[FaultInjectionReceipt, ...]:
        return tuple(self._receipts)

    def arm(self, *faults: ScalarFault) -> None:
        if not faults:
            raise ValueError("at least one fault is required")
        for fault in faults:
            if not isinstance(fault, ScalarFault):
                raise TypeError("faults must be ScalarFault values")
            self._armed.append(fault)

    def clear(self) -> None:
        self._armed.clear()

    def _take(self, operation: str) -> ScalarFault | None:
        if not self._armed:
            return None
        fault = self._armed[0]
        read_faults = {
            ScalarFault.READ_PARTITION,
            ScalarFault.STALE_READBACK,
            ScalarFault.DUPLICATE_READBACK,
            ScalarFault.REORDERED_READBACK,
        }
        write_faults = {
            ScalarFault.WRITE_PARTITION,
            ScalarFault.WRITE_OUTCOME_UNKNOWN,
        }
        allowed = read_faults if operation == "read" else write_faults
        if fault not in allowed:
            return None
        return self._armed.popleft()

    def _record(
        self,
        fault: ScalarFault,
        *,
        operation: str,
        resource_id: str,
        delegate_called: bool,
        outcome_indeterminate: bool = False,
    ) -> None:
        self._sequence += 1
        self._receipts.append(
            FaultInjectionReceipt(
                sequence=self._sequence,
                fault=fault,
                operation=operation,
                resource_sha256=_digest(resource_id),
                injected_at_ns=max(1, time.time_ns()),
                delegate_called=delegate_called,
                outcome_indeterminate=outcome_indeterminate,
                evidence_class=self._evidence_class,
                scenario_id=self._scenario_id,
            )
        )

    async def read_scalar(self, resource_id: str) -> ScalarSample:
        async with self._lock:
            fault = self._take("read")
            if fault is ScalarFault.READ_PARTITION:
                self._record(
                    fault,
                    operation="read",
                    resource_id=resource_id,
                    delegate_called=False,
                )
                raise FaultInjectedReadError("fault_injected_read_partition")
            if fault is ScalarFault.DUPLICATE_READBACK and self._samples:
                self._record(
                    fault,
                    operation="read",
                    resource_id=resource_id,
                    delegate_called=False,
                )
                return self._samples[-1]
            sample = await self._delegate.read_scalar(resource_id)
            if fault is ScalarFault.STALE_READBACK:
                sample = replace(
                    sample,
                    captured_at_ns=max(1, time.time_ns() - self._stale_age_ns),
                    source_event_id=_digest(
                        {
                            "fault": fault.value,
                            "source_event_id": sample.source_event_id,
                        }
                    ),
                    quality="fault_injected_stale",
                )
            elif fault is ScalarFault.REORDERED_READBACK and self._samples:
                current = sample
                sample = self._samples[0]
                self._samples.append(current)
            else:
                self._samples.append(sample)
            if fault is not None:
                self._record(
                    fault,
                    operation="read",
                    resource_id=resource_id,
                    delegate_called=True,
                )
            return sample

    async def write_scalar(
        self,
        resource_id: str,
        value: float,
        *,
        idempotency_key: str,
        recovery: bool = False,
    ) -> ScalarWriteResult:
        async with self._lock:
            fault = self._take("write")
            if fault is ScalarFault.WRITE_PARTITION:
                self._record(
                    fault,
                    operation="write",
                    resource_id=resource_id,
                    delegate_called=False,
                )
                raise FaultInjectedWriteError("fault_injected_write_partition")
            result = await self._delegate.write_scalar(
                resource_id,
                value,
                idempotency_key=idempotency_key,
                recovery=recovery,
            )
            if fault is ScalarFault.WRITE_OUTCOME_UNKNOWN:
                self._record(
                    fault,
                    operation="write",
                    resource_id=resource_id,
                    delegate_called=True,
                    outcome_indeterminate=True,
                )
                raise FaultInjectedOutcomeUnknownError("fault_injected_write_acknowledgement_loss")
            return result


__all__ = [
    "ACCEPTANCE_GOVERNANCE_SCHEMA",
    "AcceptanceCertificateStore",
    "AcceptanceCaseResult",
    "AcceptanceError",
    "AcceptanceEvidenceClass",
    "AcceptanceVerdict",
    "ConnectorAcceptanceCertificate",
    "FaultInjectedOutcomeUnknownError",
    "FaultInjectedReadError",
    "FaultInjectedWriteError",
    "FaultInjectingScalarTransport",
    "FaultInjectionReceipt",
    "ScalarAcceptancePlan",
    "ScalarAcceptanceRunner",
    "ScalarFault",
    "REQUIRED_SCALAR_ACCEPTANCE_CASES",
    "acceptance_governance_accepted",
    "acceptance_governance_document",
]
