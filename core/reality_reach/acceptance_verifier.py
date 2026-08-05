"""Independent replay of Reality Reach connector acceptance evidence."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.reality_reach.acceptance import (
    ACCEPTANCE_GOVERNANCE_SCHEMA,
    REQUIRED_SCALAR_ACCEPTANCE_CASES,
    AcceptanceError,
    AcceptanceEvidenceClass,
    AcceptanceVerdict,
    ConnectorAcceptanceCertificate,
)
from core.reality_reach.acceptance_mandate import AcceptanceVerificationMandate
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.secure_path_custody import DirectoryCustody, SecurePathCustodyError

VERIFICATION_RECEIPT_SCHEMA = "aura.reality_reach.acceptance_verification.v4"
MANDATED_VERIFICATION_RECEIPT_SCHEMA = (
    "aura.reality_reach.acceptance_mandated_verification.v1"
)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _mapping(value: Any, *, case_id: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AcceptanceError(f"acceptance_replay_{case_id}_evidence_invalid")
    return value


def _is_error_evidence(value: Mapping[str, Any]) -> bool:
    return bool(
        set(value) == {"error_type", "error_sha256"}
        and isinstance(value.get("error_type"), str)
        and str(value.get("error_sha256") or "").startswith("sha256:")
    )


def _predicate(case_id: str, evidence: Mapping[str, Any]) -> bool:
    if case_id == "observation.fresh":
        return bool(
            evidence.get("status") == "available"
            and evidence.get("value") is not None
            and evidence.get("source_event_id")
        )
    if case_id == "cancellation.pre_dispatch":
        return bool(
            evidence.get("state") == "cancelled"
            and evidence.get("executed") is False
            and evidence.get("transport_completed") is False
        )
    if case_id == "actuation.prepare":
        return bool(
            evidence.get("preparation_id")
            and evidence.get("command_sha256")
            and evidence.get("lease_sha256")
            and evidence.get("precondition_sha256")
            and evidence.get("rollback_token_sha256")
        )
    if case_id == "actuation.dispatch":
        return bool(
            evidence.get("state") == "executed"
            and evidence.get("accepted") is True
            and evidence.get("transport_completed") is True
            and evidence.get("executed") is True
        )
    if case_id == "effect.independent_readback":
        return bool(
            evidence.get("state") == "effect_verified"
            and evidence.get("independently_observed") is True
            and evidence.get("observation_sha256")
        )
    if case_id == "restoration.rollback":
        return bool(
            evidence.get("state") in {"rolled_back", "safe_state"}
            and evidence.get("independently_observed") is True
            and evidence.get("safe_state_observation_sha256")
        )
    raise AcceptanceError("acceptance_replay_case_unsupported")


def _cross_case_blockers(evidence: Mapping[str, Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    prepare = evidence.get("actuation.prepare", {})
    dispatch = evidence.get("actuation.dispatch", {})
    effect = evidence.get("effect.independent_readback", {})
    restoration = evidence.get("restoration.rollback", {})
    command_hashes = {
        str(item.get("command_sha256"))
        for item in (prepare, dispatch, effect, restoration)
        if item.get("command_sha256")
    }
    if len(command_hashes) > 1:
        blockers.append("actuation_command_lineage_mismatch")
    prepare_hash = _digest(dict(prepare)) if prepare else ""
    if dispatch and dispatch.get("preparation_sha256") != prepare_hash:
        blockers.append("dispatch_preparation_lineage_mismatch")
    dispatch_hash = _digest(dict(dispatch)) if dispatch else ""
    if effect and effect.get("actuation_receipt_sha256") != dispatch_hash:
        blockers.append("effect_dispatch_lineage_mismatch")
    if restoration and restoration.get("actuation_receipt_sha256") != dispatch_hash:
        blockers.append("restoration_dispatch_lineage_mismatch")
    return blockers


def _metrology_blockers(
    certificate: ConnectorAcceptanceCertificate,
    evidence_document: Mapping[str, Any],
    trusted_digest: str,
    expected_evidence_class: AcceptanceEvidenceClass,
) -> list[str]:
    metrology = evidence_document.get("metrology_receipt")
    if not isinstance(metrology, Mapping):
        return ["metrology_receipt_missing"]
    expected = {
        "run_id",
        "task_sha256",
        "mode",
        "mode_generation",
        "started_at_ns",
        "completed_at_ns",
        "sample_sets",
        "maximum_observed_skew_ns",
        "scenario_id",
        "measurements",
        "summaries",
        "evidence_sha256",
        "restored_mode",
    }
    if set(metrology) != expected:
        return ["metrology_receipt_schema_invalid"]
    evidence_body = {
        key: metrology[key] for key in expected if key not in {"evidence_sha256", "restored_mode"}
    }
    blockers: list[str] = []
    evidence_sha = str(metrology.get("evidence_sha256") or "")
    if _digest(evidence_body) != evidence_sha:
        blockers.append("metrology_evidence_digest_invalid")
    if evidence_sha != certificate.metrology_evidence_sha256 or evidence_sha != trusted_digest:
        blockers.append("trusted_metrology_mismatch")
    if metrology.get("restored_mode") != "live":
        blockers.append("metrology_mode_not_restored")
    expected_mode = (
        "hardware_in_loop"
        if expected_evidence_class is AcceptanceEvidenceClass.HARDWARE_IN_LOOP
        else "live"
    )
    if metrology.get("mode") != expected_mode:
        blockers.append("metrology_mode_mismatch")
    started_at_ns = metrology.get("started_at_ns")
    completed_at_ns = metrology.get("completed_at_ns")
    if (
        isinstance(started_at_ns, bool)
        or not isinstance(started_at_ns, int)
        or isinstance(completed_at_ns, bool)
        or not isinstance(completed_at_ns, int)
        or started_at_ns > certificate.started_at_ns
        or completed_at_ns < certificate.completed_at_ns
    ):
        blockers.append("metrology_does_not_enclose_operation")
    measurements = metrology.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        blockers.append("metrology_measurements_missing")
    else:
        sources = {
            str(item.get("source") or "") for item in measurements if isinstance(item, Mapping)
        }
        expected_sources = (
            {"live", "simulated"} if expected_mode == "hardware_in_loop" else {"live"}
        )
        if sources != expected_sources:
            blockers.append("metrology_source_class_mismatch")
        case_evidence = evidence_document.get("case_evidence")
        observation = (
            case_evidence.get("observation.fresh")
            if isinstance(case_evidence, Mapping)
            else None
        )
        observation_channel = (
            str(observation.get("channel_id") or "")
            if isinstance(observation, Mapping)
            else ""
        )
        live_channels = {
            str(item.get("channel_id") or "")
            for item in measurements
            if isinstance(item, Mapping) and item.get("source") == "live"
        }
        if not observation_channel or observation_channel not in live_channels:
            blockers.append("metrology_readback_channel_missing")
        target_observed = any(
            isinstance(item, Mapping)
            and item.get("source") == "live"
            and item.get("channel_id") == observation_channel
            and isinstance(item.get("value"), (int, float))
            and not isinstance(item.get("value"), bool)
            and abs(float(item["value"]) - certificate.target)
            <= certificate.target_tolerance
            for item in measurements
        )
        if not target_observed:
            blockers.append("metrology_target_not_observed")
    if expected_mode == "hardware_in_loop" and (
        metrology.get("scenario_id") != certificate.scenario_id
    ):
        blockers.append("metrology_scenario_mismatch")
    return blockers


def _governance_blockers(
    certificate: ConnectorAcceptanceCertificate,
    evidence_document: Mapping[str, Any],
    trusted_digest: str,
) -> list[str]:
    governance = evidence_document.get("governance_evidence")
    if not isinstance(governance, Mapping):
        return ["governance_evidence_missing"]
    expected = {
        "schema",
        "action_id",
        "request_digest",
        "will_receipt_id",
        "post_action_receipt_id",
        "post_action_output_hash",
        "status",
        "transport_succeeded",
        "effect_verified",
        "receipt_persisted",
        "welfare_transaction_completed",
    }
    if set(governance) != expected:
        return ["governance_evidence_schema_invalid"]
    blockers: list[str] = []
    if governance.get("schema") != ACCEPTANCE_GOVERNANCE_SCHEMA:
        blockers.append("governance_evidence_schema_invalid")
    for field in (
        "action_id",
        "request_digest",
        "will_receipt_id",
        "post_action_receipt_id",
    ):
        if not str(governance.get(field) or "").strip():
            blockers.append(f"governance_{field}_missing")
    if not _DIGEST.fullmatch(str(governance.get("request_digest") or "")):
        blockers.append("governance_request_digest_invalid")
    if not _DIGEST.fullmatch(str(governance.get("post_action_output_hash") or "")):
        blockers.append("governance_post_action_output_hash_invalid")
    if governance.get("status") != "success_verified":
        blockers.append("governance_status_not_verified")
    for field in (
        "transport_succeeded",
        "effect_verified",
        "receipt_persisted",
        "welfare_transaction_completed",
    ):
        if governance.get(field) is not True:
            blockers.append(f"governance_{field}_false")
    governance_sha = _digest(dict(governance))
    if (
        governance_sha != certificate.governance_evidence_sha256
        or governance_sha != trusted_digest
    ):
        blockers.append("trusted_governance_mismatch")
    if certificate.governance_accepted is not True:
        blockers.append("producer_governance_not_accepted")
    return blockers


@dataclass(frozen=True, slots=True)
class AcceptanceVerificationReceipt:
    campaign_id: str
    certificate_sha256: str
    expected_source_commit_sha256: str
    expected_physical_identity_sha256: str
    expected_evidence_class: AcceptanceEvidenceClass
    trusted_metrology_evidence_sha256: str
    trusted_governance_evidence_sha256: str
    replayed_cases: tuple[str, ...]
    blockers: tuple[str, ...]
    deterministic_accepted: bool
    live_accepted: bool

    def __post_init__(self) -> None:
        if not self.campaign_id:
            raise ValueError("campaign_id must be non-empty")
        for name in (
            "certificate_sha256",
            "expected_source_commit_sha256",
            "expected_physical_identity_sha256",
        ):
            if not _DIGEST.fullmatch(str(getattr(self, name))):
                raise ValueError(f"{name} must be a sha256 digest")
        if not isinstance(self.expected_evidence_class, AcceptanceEvidenceClass):
            raise TypeError("expected_evidence_class must be an AcceptanceEvidenceClass")
        if self.trusted_metrology_evidence_sha256 and not _DIGEST.fullmatch(
            self.trusted_metrology_evidence_sha256
        ):
            raise ValueError("trusted_metrology_evidence_sha256 must be empty or a digest")
        if self.trusted_governance_evidence_sha256 and not _DIGEST.fullmatch(
            self.trusted_governance_evidence_sha256
        ):
            raise ValueError("trusted_governance_evidence_sha256 must be empty or a digest")
        if len(self.replayed_cases) != len(set(self.replayed_cases)):
            raise ValueError("replayed_cases must be unique")
        if len(self.blockers) != len(set(self.blockers)):
            raise ValueError("blockers must be unique")
        if not isinstance(self.deterministic_accepted, bool) or not isinstance(
            self.live_accepted, bool
        ):
            raise TypeError("verification verdicts must be bools")

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict(include_digest=False))

    @property
    def accepted(self) -> bool:
        """Return the verdict for the externally declared evidence burden."""

        if self.expected_evidence_class is AcceptanceEvidenceClass.SIMULATION:
            return self.deterministic_accepted
        return self.live_accepted

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        document = {
            "schema": VERIFICATION_RECEIPT_SCHEMA,
            "campaign_id": self.campaign_id,
            "certificate_sha256": self.certificate_sha256,
            "expected_source_commit_sha256": self.expected_source_commit_sha256,
            "expected_physical_identity_sha256": self.expected_physical_identity_sha256,
            "expected_evidence_class": self.expected_evidence_class.value,
            "trusted_metrology_evidence_sha256": self.trusted_metrology_evidence_sha256,
            "trusted_governance_evidence_sha256": self.trusted_governance_evidence_sha256,
            "replayed_cases": list(self.replayed_cases),
            "blockers": list(self.blockers),
            "deterministic_accepted": self.deterministic_accepted,
            "live_accepted": self.live_accepted,
            "accepted": self.accepted,
        }
        if include_digest:
            document["verification_sha256"] = self.sha256
        return document


@dataclass(frozen=True, slots=True)
class MandatedAcceptanceVerificationReceipt:
    """A replay verdict bound to the question fixed before execution."""

    campaign_id: str
    mandate_sha256: str
    mandate_contract_sha256: str
    verification: AcceptanceVerificationReceipt
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.campaign_id:
            raise ValueError("campaign_id must be non-empty")
        for name in ("mandate_sha256", "mandate_contract_sha256"):
            if not _DIGEST.fullmatch(str(getattr(self, name))):
                raise ValueError(f"{name} must be a sha256 digest")
        if not isinstance(self.verification, AcceptanceVerificationReceipt):
            raise TypeError("verification must be an AcceptanceVerificationReceipt")
        if len(self.blockers) != len(set(self.blockers)):
            raise ValueError("blockers must be unique")
        if (
            self.verification.campaign_id != self.campaign_id
            and "mandate_campaign_mismatch" not in self.blockers
        ):
            raise ValueError(
                "campaign mismatch must be represented by a mandate blocker"
            )

    @property
    def accepted(self) -> bool:
        return bool(not self.blockers and self.verification.accepted)

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        document = {
            "schema": MANDATED_VERIFICATION_RECEIPT_SCHEMA,
            "campaign_id": self.campaign_id,
            "mandate_sha256": self.mandate_sha256,
            "mandate_contract_sha256": self.mandate_contract_sha256,
            "verification": self.verification.to_dict(),
            "blockers": list(self.blockers),
            "accepted": self.accepted,
        }
        if include_digest:
            document["mandated_verification_sha256"] = self.sha256
        return document


def verify_acceptance_evidence(
    certificate: ConnectorAcceptanceCertificate,
    evidence_document: Mapping[str, Any],
    *,
    expected_source_commit_sha256: str,
    expected_physical_identity_sha256: str,
    expected_evidence_class: AcceptanceEvidenceClass = AcceptanceEvidenceClass.SIMULATION,
    trusted_metrology_evidence_sha256: str = "",
    trusted_governance_evidence_sha256: str = "",
    required_cases: Sequence[str] = REQUIRED_SCALAR_ACCEPTANCE_CASES,
) -> AcceptanceVerificationReceipt:
    """Recompute case verdicts without trusting producer-derived booleans."""

    if not isinstance(certificate, ConnectorAcceptanceCertificate):
        raise TypeError("certificate must be a ConnectorAcceptanceCertificate")
    if not _DIGEST.fullmatch(str(expected_source_commit_sha256)):
        raise ValueError("expected_source_commit_sha256 must be a sha256 digest")
    if not _DIGEST.fullmatch(str(expected_physical_identity_sha256)):
        raise ValueError("expected_physical_identity_sha256 must be a sha256 digest")
    if not isinstance(expected_evidence_class, AcceptanceEvidenceClass):
        raise TypeError("expected_evidence_class must be an AcceptanceEvidenceClass")
    if trusted_metrology_evidence_sha256 and not _DIGEST.fullmatch(
        trusted_metrology_evidence_sha256
    ):
        raise ValueError("trusted_metrology_evidence_sha256 must be empty or a digest")
    if trusted_governance_evidence_sha256 and not _DIGEST.fullmatch(
        trusted_governance_evidence_sha256
    ):
        raise ValueError("trusted_governance_evidence_sha256 must be empty or a digest")
    raw_evidence = evidence_document.get("case_evidence")
    if not isinstance(raw_evidence, Mapping):
        raise AcceptanceError("acceptance_replay_evidence_missing")
    cases = {item.case_id: item for item in certificate.cases}
    required = tuple(str(item) for item in required_cases)
    blockers: list[str] = []
    if tuple(cases) != required or set(raw_evidence) != set(cases):
        blockers.append("required_case_set_mismatch")
    if certificate.source_commit_sha256 != expected_source_commit_sha256:
        blockers.append("source_commit_mismatch")
    if certificate.physical_identity_sha256 != expected_physical_identity_sha256:
        blockers.append("physical_identity_mismatch")
    observed_evidence_classes = {item.evidence_class for item in certificate.cases if item.required}
    if observed_evidence_classes != {expected_evidence_class}:
        blockers.append("evidence_class_mismatch")
    if expected_evidence_class is AcceptanceEvidenceClass.SIMULATION:
        if trusted_metrology_evidence_sha256:
            blockers.append("unexpected_trusted_metrology")
        if trusted_governance_evidence_sha256:
            blockers.append("unexpected_trusted_governance")
        if (
            certificate.governance_evidence_sha256
            or certificate.governance_accepted
            or evidence_document.get("governance_evidence")
        ):
            blockers.append("unexpected_governance_evidence")
    elif not trusted_metrology_evidence_sha256:
        blockers.append("trusted_metrology_missing")
    if (
        expected_evidence_class is not AcceptanceEvidenceClass.SIMULATION
        and not trusted_governance_evidence_sha256
    ):
        blockers.append("trusted_governance_missing")

    replayed: list[str] = []
    normalized: dict[str, Mapping[str, Any]] = {}
    prior_nonpass = False
    for case_id in required:
        result = cases.get(case_id)
        raw = raw_evidence.get(case_id)
        if result is None or raw is None:
            continue
        evidence = _mapping(raw, case_id=case_id)
        normalized[case_id] = evidence
        if _digest(dict(evidence)) != result.evidence_sha256:
            blockers.append(f"{case_id}:evidence_digest_mismatch")
            continue
        predicate = _predicate(case_id, evidence) if not _is_error_evidence(evidence) else False
        if result.verdict is AcceptanceVerdict.PASS and not predicate:
            blockers.append(f"{case_id}:pass_not_reproduced")
        elif result.verdict is AcceptanceVerdict.FAIL and predicate:
            blockers.append(f"{case_id}:failure_not_reproduced")
        elif result.verdict is AcceptanceVerdict.UNMEASURED and not prior_nonpass:
            blockers.append(f"{case_id}:unmeasured_without_blocker")
        else:
            replayed.append(case_id)
        prior_nonpass = prior_nonpass or result.verdict is not AcceptanceVerdict.PASS
    blockers.extend(_cross_case_blockers(normalized))

    live_evidence = expected_evidence_class in {
        AcceptanceEvidenceClass.HARDWARE_IN_LOOP,
        AcceptanceEvidenceClass.LIVE,
    }
    if live_evidence:
        blockers.extend(
            _metrology_blockers(
                certificate,
                evidence_document,
                trusted_metrology_evidence_sha256,
                expected_evidence_class,
            )
        )
        blockers.extend(
            _governance_blockers(
                certificate,
                evidence_document,
                trusted_governance_evidence_sha256,
            )
        )
    deterministic = not blockers and certificate.deterministic_passed
    return AcceptanceVerificationReceipt(
        campaign_id=certificate.campaign_id,
        certificate_sha256=certificate.sha256,
        expected_source_commit_sha256=expected_source_commit_sha256,
        expected_physical_identity_sha256=expected_physical_identity_sha256,
        expected_evidence_class=expected_evidence_class,
        trusted_metrology_evidence_sha256=trusted_metrology_evidence_sha256,
        trusted_governance_evidence_sha256=trusted_governance_evidence_sha256,
        replayed_cases=tuple(replayed),
        blockers=tuple(sorted(set(blockers))),
        deterministic_accepted=deterministic,
        live_accepted=bool(deterministic and live_evidence and certificate.live_acceptance_passed),
    )


def verify_acceptance_against_mandate(
    certificate: ConnectorAcceptanceCertificate,
    evidence_document: Mapping[str, Any],
    mandate: AcceptanceVerificationMandate,
    *,
    trusted_metrology_evidence_sha256: str = "",
    trusted_governance_evidence_sha256: str = "",
) -> MandatedAcceptanceVerificationReceipt:
    """Replay evidence against a create-once pre-execution mandate."""

    if not isinstance(mandate, AcceptanceVerificationMandate):
        raise TypeError("mandate must be an AcceptanceVerificationMandate")
    blockers: list[str] = []
    if certificate.campaign_id != mandate.campaign_id:
        blockers.append("mandate_campaign_mismatch")
    if certificate.connector_id != mandate.connector_id:
        blockers.append("mandate_connector_mismatch")
    if certificate.adapter_id != mandate.adapter_id:
        blockers.append("mandate_adapter_mismatch")
    if certificate.source_commit_sha256 != mandate.expected_source_commit_sha256:
        blockers.append("mandate_source_commit_mismatch")
    if (
        certificate.physical_identity_sha256
        != mandate.expected_physical_identity_sha256
    ):
        blockers.append("mandate_physical_identity_mismatch")
    observed_classes = {
        item.evidence_class for item in certificate.cases if item.required
    }
    if observed_classes != {mandate.expected_evidence_class}:
        blockers.append("mandate_evidence_class_mismatch")
    if certificate.target != mandate.target:
        blockers.append("mandate_target_mismatch")
    if certificate.target_tolerance != mandate.target_tolerance:
        blockers.append("mandate_target_tolerance_mismatch")
    if certificate.scenario_id != mandate.scenario_id:
        blockers.append("mandate_scenario_mismatch")
    if tuple(item.case_id for item in certificate.cases) != mandate.required_cases:
        blockers.append("mandate_required_case_set_mismatch")
    if certificate.started_at_ns < mandate.provisioned_at_ns:
        blockers.append("campaign_predates_mandate")
    verification = verify_acceptance_evidence(
        certificate,
        evidence_document,
        expected_source_commit_sha256=mandate.expected_source_commit_sha256,
        expected_physical_identity_sha256=mandate.expected_physical_identity_sha256,
        expected_evidence_class=mandate.expected_evidence_class,
        trusted_metrology_evidence_sha256=trusted_metrology_evidence_sha256,
        trusted_governance_evidence_sha256=trusted_governance_evidence_sha256,
        required_cases=mandate.required_cases,
    )
    return MandatedAcceptanceVerificationReceipt(
        campaign_id=mandate.campaign_id,
        mandate_sha256=mandate.sha256,
        mandate_contract_sha256=mandate.contract_sha256,
        verification=verification,
        blockers=tuple(sorted(set(blockers))),
    )


def _persist_receipt_document(
    document: Mapping[str, Any],
    path: str | Path,
    *,
    collision_code: str,
) -> bool:
    target = Path(path).expanduser().absolute()
    if not target.name or target.name in {".", ".."}:
        raise AcceptanceError("acceptance_verification_path_invalid")
    payload = canonical_json(dict(document))
    try:
        with DirectoryCustody.acquire(target.parent, create=True, private=True) as custody:
            published = bool(custody.write_bytes_once(target.name, payload, mode=0o600))
            fd = custody.open_file(target.name, os.O_RDONLY)
            try:
                if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                    raise AcceptanceError("acceptance_verification_mode_invalid")
            finally:
                os.close(fd)
            existing = custody.read_bytes(target.name, max_bytes=1024 * 1024)
    except SecurePathCustodyError as exc:
        raise AcceptanceError("acceptance_verification_custody_invalid") from exc
    if existing != payload:
        raise AcceptanceError(collision_code)
    return published


def persist_verification_receipt(
    receipt: AcceptanceVerificationReceipt,
    path: str | Path,
) -> bool:
    """Create-once publish one independently reconstructed verdict."""

    if not isinstance(receipt, AcceptanceVerificationReceipt):
        raise TypeError("receipt must be an AcceptanceVerificationReceipt")
    return _persist_receipt_document(
        receipt.to_dict(),
        path,
        collision_code="acceptance_verification_receipt_collision",
    )


def persist_mandated_verification_receipt(
    receipt: MandatedAcceptanceVerificationReceipt,
    path: str | Path,
) -> bool:
    """Create-once publish a mandate-bound independently replayed verdict."""

    if not isinstance(receipt, MandatedAcceptanceVerificationReceipt):
        raise TypeError("receipt must be a MandatedAcceptanceVerificationReceipt")
    return _persist_receipt_document(
        receipt.to_dict(),
        path,
        collision_code="acceptance_mandated_verification_receipt_collision",
    )


__all__ = [
    "AcceptanceVerificationReceipt",
    "MANDATED_VERIFICATION_RECEIPT_SCHEMA",
    "MandatedAcceptanceVerificationReceipt",
    "VERIFICATION_RECEIPT_SCHEMA",
    "persist_mandated_verification_receipt",
    "persist_verification_receipt",
    "verify_acceptance_against_mandate",
    "verify_acceptance_evidence",
]
