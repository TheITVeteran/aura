"""Immutable contracts and canonical validation for Reality Reach acceptance."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Never

if TYPE_CHECKING:
    from core.reality_reach.metrology import AcquisitionReceipt

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
    Awaitable[tuple["ConnectorAcceptanceCertificate", "AcquisitionReceipt"]],
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
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


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

    def reject_constant(_value: str) -> Never:
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
            mode = getattr(getattr(receipt, "mode", None), "value", None)
            if receipt is not None and mode != "simulation":
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


__all__ = [
    "ACCEPTANCE_GOVERNANCE_SCHEMA",
    "REQUIRED_SCALAR_ACCEPTANCE_CASES",
    "AcceptanceCaseResult",
    "AcceptanceError",
    "AcceptanceEvidenceClass",
    "AcceptanceExecutor",
    "AcceptanceMetrologyAcquirer",
    "AcceptanceOperation",
    "AcceptanceVerdict",
    "ConnectorAcceptanceCertificate",
    "ScalarAcceptancePlan",
    "ScalarFault",
    "acceptance_governance_accepted",
    "acceptance_governance_document",
]
