"""Preregistered A1 acoustic calibration and held-out physical scoring."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.reality_reach.acceptance import (
    AcceptanceEvidenceClass,
    acceptance_governance_accepted,
    acceptance_governance_document,
)
from core.reality_reach.acceptance_mandate import AcceptanceVerificationMandate
from core.reality_reach.acceptance_preregistration import (
    PreregisteredAcceptanceReceipt,
)
from core.reality_reach.acceptance_transparency import (
    ACCEPTANCE_TRANSPARENCY_STATEMENT_SCHEMA,
    AcceptanceTransparencyError,
    build_acceptance_transparency_artifact_bundle,
    validate_acceptance_transparency_statement_envelope,
    verify_acceptance_transparency_artifact,
)
from core.reality_reach.acceptance_witness import (
    AcceptanceWitnessBundle,
    AcceptanceWitnessError,
    AcceptanceWitnessRole,
    verify_acceptance_witness_artifact_bundle,
)
from core.reality_reach.scalar_adapter import ScalarSample
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.secure_path_custody import DirectoryCustody, SecurePathCustodyError
from core.runtime.state_ownership import state_root

ACOUSTIC_A1_RECEIPT_SCHEMA = "aura.reality_reach.acoustic_a1_acceptance.v1"
ACOUSTIC_A1_CAMPAIGN_SCHEMA = "aura.reality_reach.acoustic_a1_campaign.v2"
EXTERNAL_ACOUSTIC_A1_VERIFICATION_SCHEMA = (
    "aura.reality_reach.external_acoustic_a1_verification.v3"
)
TRANSPARENT_ACOUSTIC_A1_VERIFICATION_SCHEMA = (
    "aura.reality_reach.transparent_acoustic_a1_verification.v2"
)
_REKOR_UUID = re.compile(r"^[0-9a-f]{80}$")
ACOUSTIC_A1_CONNECTOR_ID = "macos.acoustic.a1"
ACOUSTIC_A1_REQUIRED_CASES = (
    "calibration.monotone_transfer",
    "heldout.equal_work_control",
    "heldout.error_reduction",
    "restoration.silence",
)
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class AcousticAcceptanceError(RuntimeError):
    """Stable fail-closed A1 experiment error."""


class AcousticTrialArm(StrEnum):
    CALIBRATION = "calibration"
    OPEN_LOOP = "open_loop"
    CLOSED_LOOP = "closed_loop"
    RESTORATION = "restoration"


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@runtime_checkable
class AcousticTrialDriver(Protocol):
    async def measure_stimulus(
        self,
        amplitude: float,
        *,
        trial_id: str,
    ) -> ScalarSample: ...


@dataclass(frozen=True, slots=True)
class AcousticAcceptanceConfig:
    campaign_id: str
    calibration_amplitudes: tuple[float, ...] = (
        0.0,
        0.0025,
        0.005,
        0.01,
        0.02,
        0.04,
        0.08,
    )
    calibration_repeats: int = 2
    heldout_repeats: int = 3
    heldout_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
    minimum_signal_span_db: float = 12.0
    baseline_margin_db: float = 6.0
    ceiling_margin_db: float = 3.0
    required_error_reduction: float = 0.5
    restoration_tolerance_db: float = 3.0
    schedule_seed: str = "aura.reality-reach.a1.v1"

    def __post_init__(self) -> None:
        if not self.campaign_id or len(self.campaign_id) > 128:
            raise ValueError("campaign_id must be a bounded non-empty string")
        amplitudes = tuple(
            _finite(value, name="calibration_amplitude")
            for value in self.calibration_amplitudes
        )
        if (
            len(amplitudes) < 4
            or amplitudes != tuple(sorted(set(amplitudes)))
            or amplitudes[0] != 0.0
            or amplitudes[-1] > 0.2
        ):
            raise ValueError("calibration amplitudes must be unique, sorted, and bounded")
        object.__setattr__(self, "calibration_amplitudes", amplitudes)
        if not 1 <= self.calibration_repeats <= 8 or not 1 <= self.heldout_repeats <= 8:
            raise ValueError("trial repeats must lie inside [1, 8]")
        quantiles = tuple(_finite(value, name="heldout_quantile") for value in self.heldout_quantiles)
        if (
            not quantiles
            or quantiles != tuple(sorted(set(quantiles)))
            or any(not 0.0 < value < 1.0 for value in quantiles)
        ):
            raise ValueError("heldout quantiles must be unique, sorted, and interior")
        object.__setattr__(self, "heldout_quantiles", quantiles)
        for name in (
            "minimum_signal_span_db",
            "baseline_margin_db",
            "ceiling_margin_db",
            "restoration_tolerance_db",
        ):
            value = _finite(getattr(self, name), name=name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        reduction = _finite(self.required_error_reduction, name="required_error_reduction")
        if not 0.0 < reduction < 1.0:
            raise ValueError("required_error_reduction must lie inside (0, 1)")
        object.__setattr__(self, "required_error_reduction", reduction)
        if not self.schedule_seed or len(self.schedule_seed) > 256:
            raise ValueError("schedule_seed must be bounded and non-empty")

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "campaign_id": self.campaign_id,
                "calibration_amplitudes": list(self.calibration_amplitudes),
                "calibration_repeats": self.calibration_repeats,
                "heldout_repeats": self.heldout_repeats,
                "heldout_quantiles": list(self.heldout_quantiles),
                "minimum_signal_span_db": self.minimum_signal_span_db,
                "baseline_margin_db": self.baseline_margin_db,
                "ceiling_margin_db": self.ceiling_margin_db,
                "required_error_reduction": self.required_error_reduction,
                "restoration_tolerance_db": self.restoration_tolerance_db,
                "schedule_seed": self.schedule_seed,
            }
        )


@dataclass(frozen=True, slots=True)
class AcousticTrial:
    trial_id: str
    arm: AcousticTrialArm
    order: int
    repeat: int
    amplitude: float
    target_dbfs: float | None
    observed_dbfs: float
    source_event_id: str
    captured_at_ns: int

    def __post_init__(self) -> None:
        if not self.trial_id or len(self.trial_id) > 160:
            raise ValueError("trial_id must be a bounded non-empty string")
        if not isinstance(self.arm, AcousticTrialArm):
            raise TypeError("arm must be an AcousticTrialArm")
        if isinstance(self.order, bool) or self.order <= 0:
            raise ValueError("order must be positive")
        if isinstance(self.repeat, bool) or self.repeat < 0:
            raise ValueError("repeat must be non-negative")
        for name in ("amplitude", "observed_dbfs"):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        if not 0.0 <= self.amplitude <= 0.2:
            raise ValueError("amplitude must lie inside [0, 0.2]")
        if self.target_dbfs is not None:
            object.__setattr__(
                self,
                "target_dbfs",
                _finite(self.target_dbfs, name="target_dbfs"),
            )
        digest = str(self.source_event_id or "")
        if not _SHA256.fullmatch(digest):
            raise ValueError("source_event_id must be a sha256 digest")
        if isinstance(self.captured_at_ns, bool) or self.captured_at_ns <= 0:
            raise ValueError("captured_at_ns must be positive")

    @property
    def absolute_error_db(self) -> float | None:
        return (
            abs(self.observed_dbfs - self.target_dbfs)
            if self.target_dbfs is not None
            else None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "arm": self.arm.value,
            "order": self.order,
            "repeat": self.repeat,
            "amplitude": self.amplitude,
            "target_dbfs": self.target_dbfs,
            "observed_dbfs": self.observed_dbfs,
            "absolute_error_db": self.absolute_error_db,
            "source_event_id": self.source_event_id,
            "captured_at_ns": self.captured_at_ns,
        }

    @classmethod
    def from_dict(cls, document: Any) -> AcousticTrial:
        expected = {
            "trial_id",
            "arm",
            "order",
            "repeat",
            "amplitude",
            "target_dbfs",
            "observed_dbfs",
            "absolute_error_db",
            "source_event_id",
            "captured_at_ns",
        }
        if not isinstance(document, dict) or set(document) != expected:
            raise AcousticAcceptanceError("acoustic_a1_trial_schema_invalid")
        try:
            trial = cls(
                trial_id=document["trial_id"],
                arm=AcousticTrialArm(document["arm"]),
                order=document["order"],
                repeat=document["repeat"],
                amplitude=document["amplitude"],
                target_dbfs=document["target_dbfs"],
                observed_dbfs=document["observed_dbfs"],
                source_event_id=document["source_event_id"],
                captured_at_ns=document["captured_at_ns"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AcousticAcceptanceError("acoustic_a1_trial_invalid") from exc
        if document["absolute_error_db"] != trial.absolute_error_db:
            raise AcousticAcceptanceError("acoustic_a1_trial_error_invalid")
        return trial


@dataclass(frozen=True, slots=True)
class AcousticA1AcceptanceReceipt:
    campaign_id: str
    config_sha256: str
    heldout_targets_dbfs: tuple[float, ...]
    trials: tuple[AcousticTrial, ...]
    open_loop_mae_db: float
    closed_loop_mae_db: float
    error_reduction: float
    restoration_error_db: float
    calibration_span_db: float
    blockers: tuple[str, ...]
    completed_at_ns: int

    def __post_init__(self) -> None:
        if not self.campaign_id or len(self.campaign_id) > 128:
            raise ValueError("campaign_id must be a bounded non-empty string")
        if not _SHA256.fullmatch(self.config_sha256):
            raise ValueError("config_sha256 must be a canonical sha256 digest")
        targets = tuple(
            _finite(value, name="heldout_target_dbfs")
            for value in self.heldout_targets_dbfs
        )
        if not targets or targets != tuple(sorted(set(targets))):
            raise ValueError("heldout targets must be unique and sorted")
        object.__setattr__(self, "heldout_targets_dbfs", targets)
        if not self.trials:
            raise ValueError("receipt requires measured trials")
        if tuple(trial.order for trial in self.trials) != tuple(
            range(1, len(self.trials) + 1)
        ):
            raise ValueError("trial order must be contiguous and one-indexed")
        if self.trials[-1].arm is not AcousticTrialArm.RESTORATION:
            raise ValueError("receipt must end with a restoration trial")
        for name in (
            "open_loop_mae_db",
            "closed_loop_mae_db",
            "error_reduction",
            "restoration_error_db",
            "calibration_span_db",
        ):
            value = _finite(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        if self.open_loop_mae_db < 0.0 or self.closed_loop_mae_db < 0.0:
            raise ValueError("mean absolute errors must be non-negative")
        if self.restoration_error_db < 0.0 or self.calibration_span_db < 0.0:
            raise ValueError("restoration error and calibration span must be non-negative")
        blockers = tuple(str(blocker).strip() for blocker in self.blockers)
        if any(not blocker for blocker in blockers) or len(set(blockers)) != len(blockers):
            raise ValueError("blockers must be unique non-empty strings")
        object.__setattr__(self, "blockers", blockers)
        if isinstance(self.completed_at_ns, bool) or self.completed_at_ns <= 0:
            raise ValueError("completed_at_ns must be positive")

    @property
    def accepted(self) -> bool:
        return not self.blockers

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        document = {
            "schema": ACOUSTIC_A1_RECEIPT_SCHEMA,
            "campaign_id": self.campaign_id,
            "config_sha256": self.config_sha256,
            "heldout_targets_dbfs": list(self.heldout_targets_dbfs),
            "trials": [trial.to_dict() for trial in self.trials],
            "open_loop_mae_db": self.open_loop_mae_db,
            "closed_loop_mae_db": self.closed_loop_mae_db,
            "error_reduction": self.error_reduction,
            "restoration_error_db": self.restoration_error_db,
            "calibration_span_db": self.calibration_span_db,
            "blockers": list(self.blockers),
            "completed_at_ns": self.completed_at_ns,
            "raw_audio_retained": False,
            "accepted": self.accepted,
        }
        if include_digest:
            document["receipt_sha256"] = self.sha256
        return document

    @classmethod
    def from_dict(cls, document: Any) -> AcousticA1AcceptanceReceipt:
        expected = {
            "schema",
            "campaign_id",
            "config_sha256",
            "heldout_targets_dbfs",
            "trials",
            "open_loop_mae_db",
            "closed_loop_mae_db",
            "error_reduction",
            "restoration_error_db",
            "calibration_span_db",
            "blockers",
            "completed_at_ns",
            "raw_audio_retained",
            "accepted",
            "receipt_sha256",
        }
        if not isinstance(document, dict) or set(document) != expected:
            raise AcousticAcceptanceError("acoustic_a1_receipt_schema_invalid")
        if (
            document.get("schema") != ACOUSTIC_A1_RECEIPT_SCHEMA
            or document.get("raw_audio_retained") is not False
        ):
            raise AcousticAcceptanceError("acoustic_a1_receipt_schema_invalid")
        try:
            receipt = cls(
                campaign_id=document["campaign_id"],
                config_sha256=document["config_sha256"],
                heldout_targets_dbfs=tuple(document["heldout_targets_dbfs"]),
                trials=tuple(AcousticTrial.from_dict(item) for item in document["trials"]),
                open_loop_mae_db=document["open_loop_mae_db"],
                closed_loop_mae_db=document["closed_loop_mae_db"],
                error_reduction=document["error_reduction"],
                restoration_error_db=document["restoration_error_db"],
                calibration_span_db=document["calibration_span_db"],
                blockers=tuple(document["blockers"]),
                completed_at_ns=document["completed_at_ns"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AcousticAcceptanceError("acoustic_a1_receipt_invalid") from exc
        if document["accepted"] is not receipt.accepted or document["receipt_sha256"] != receipt.sha256:
            raise AcousticAcceptanceError("acoustic_a1_receipt_derived_field_invalid")
        return receipt


@dataclass(frozen=True, slots=True)
class AcousticA1CampaignRecord:
    campaign_id: str
    adapter_id: str
    source_commit_sha256: str
    workspace_state_sha256: str
    physical_identity_sha256: str
    mandate_sha256: str
    receipt: AcousticA1AcceptanceReceipt
    governance_evidence: dict[str, Any]
    started_at_ns: int
    completed_at_ns: int

    def __post_init__(self) -> None:
        if self.campaign_id != self.receipt.campaign_id:
            raise ValueError("campaign record and receipt identity must match")
        if not self.adapter_id or len(self.adapter_id) > 256:
            raise ValueError("adapter_id must be a bounded non-empty string")
        for name in (
            "source_commit_sha256",
            "workspace_state_sha256",
            "physical_identity_sha256",
            "mandate_sha256",
        ):
            digest = str(getattr(self, name) or "")
            if not _SHA256.fullmatch(digest):
                raise ValueError(f"{name} must be a sha256 digest")
        if not isinstance(self.governance_evidence, dict):
            raise TypeError("governance_evidence must be a dict")
        governance = acceptance_governance_document(self.governance_evidence)
        if set(self.governance_evidence) != set(governance):
            raise ValueError("governance_evidence schema is invalid")
        object.__setattr__(
            self,
            "governance_evidence",
            governance,
        )
        if (
            isinstance(self.started_at_ns, bool)
            or isinstance(self.completed_at_ns, bool)
            or self.started_at_ns <= 0
            or self.completed_at_ns < self.started_at_ns
            or self.receipt.completed_at_ns > self.completed_at_ns
        ):
            raise ValueError("campaign record timestamps are invalid")

    @property
    def accepted(self) -> bool:
        return bool(
            self.receipt.accepted
            and acceptance_governance_accepted(self.governance_evidence)
        )

    @property
    def governance_evidence_sha256(self) -> str:
        return _digest(self.governance_evidence)

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        document = {
            "schema": ACOUSTIC_A1_CAMPAIGN_SCHEMA,
            "campaign_id": self.campaign_id,
            "adapter_id": self.adapter_id,
            "source_commit_sha256": self.source_commit_sha256,
            "workspace_state_sha256": self.workspace_state_sha256,
            "physical_identity_sha256": self.physical_identity_sha256,
            "mandate_sha256": self.mandate_sha256,
            "receipt": self.receipt.to_dict(),
            "governance_evidence": dict(self.governance_evidence),
            "started_at_ns": self.started_at_ns,
            "completed_at_ns": self.completed_at_ns,
            "accepted": self.accepted,
        }
        if include_digest:
            document["campaign_record_sha256"] = self.sha256
        return document

    @classmethod
    def from_dict(cls, document: Any) -> AcousticA1CampaignRecord:
        expected = {
            "schema",
            "campaign_id",
            "adapter_id",
            "source_commit_sha256",
            "workspace_state_sha256",
            "physical_identity_sha256",
            "mandate_sha256",
            "receipt",
            "governance_evidence",
            "started_at_ns",
            "completed_at_ns",
            "accepted",
            "campaign_record_sha256",
        }
        if (
            not isinstance(document, dict)
            or set(document) != expected
            or document.get("schema") != ACOUSTIC_A1_CAMPAIGN_SCHEMA
        ):
            raise AcousticAcceptanceError("acoustic_a1_campaign_schema_invalid")
        try:
            record = cls(
                campaign_id=document["campaign_id"],
                adapter_id=document["adapter_id"],
                source_commit_sha256=document["source_commit_sha256"],
                workspace_state_sha256=document["workspace_state_sha256"],
                physical_identity_sha256=document["physical_identity_sha256"],
                mandate_sha256=document["mandate_sha256"],
                receipt=AcousticA1AcceptanceReceipt.from_dict(document["receipt"]),
                governance_evidence=dict(document["governance_evidence"]),
                started_at_ns=document["started_at_ns"],
                completed_at_ns=document["completed_at_ns"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AcousticAcceptanceError("acoustic_a1_campaign_invalid") from exc
        if document["accepted"] is not record.accepted or document["campaign_record_sha256"] != record.sha256:
            raise AcousticAcceptanceError("acoustic_a1_campaign_derived_field_invalid")
        return record


@dataclass(frozen=True, slots=True)
class ExternallyWitnessedAcousticA1Receipt:
    campaign_id: str
    campaign_record_sha256: str
    mandate_sha256: str
    preregistration_verification_sha256: str
    metrology_witness_bundle_sha256: str
    governance_witness_bundle_sha256: str
    metrology_witness_key_sha256: str
    governance_witness_key_sha256: str
    acceptance_log_key_sha256: str
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.campaign_id or len(self.campaign_id) > 128:
            raise ValueError("campaign_id must be a bounded non-empty string")
        for name in (
            "campaign_record_sha256",
            "mandate_sha256",
            "preregistration_verification_sha256",
            "metrology_witness_bundle_sha256",
            "governance_witness_bundle_sha256",
            "metrology_witness_key_sha256",
            "governance_witness_key_sha256",
            "acceptance_log_key_sha256",
        ):
            value = str(getattr(self, name) or "")
            if name in {"campaign_record_sha256", "mandate_sha256"}:
                if not _SHA256.fullmatch(value):
                    raise ValueError(f"{name} must be a sha256 digest")
            elif value and not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be empty or a sha256 digest")
        if len(self.blockers) != len(set(self.blockers)):
            raise ValueError("blockers must be unique")

    @property
    def accepted(self) -> bool:
        return bool(
            not self.blockers
            and self.preregistration_verification_sha256
            and self.metrology_witness_bundle_sha256
            and self.governance_witness_bundle_sha256
            and self.metrology_witness_key_sha256
            and self.governance_witness_key_sha256
            and self.acceptance_log_key_sha256
            and self.metrology_witness_key_sha256
            != self.governance_witness_key_sha256
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        document = {
            "schema": EXTERNAL_ACOUSTIC_A1_VERIFICATION_SCHEMA,
            "campaign_id": self.campaign_id,
            "campaign_record_sha256": self.campaign_record_sha256,
            "mandate_sha256": self.mandate_sha256,
            "preregistration_verification_sha256": (
                self.preregistration_verification_sha256
            ),
            "metrology_witness_bundle_sha256": self.metrology_witness_bundle_sha256,
            "governance_witness_bundle_sha256": self.governance_witness_bundle_sha256,
            "metrology_witness_key_sha256": self.metrology_witness_key_sha256,
            "governance_witness_key_sha256": self.governance_witness_key_sha256,
            "acceptance_log_key_sha256": self.acceptance_log_key_sha256,
            "blockers": list(self.blockers),
            "accepted": self.accepted,
        }
        if include_digest:
            document["verification_sha256"] = self.sha256
        return document


def acoustic_a1_campaign_binding_blockers(
    record: AcousticA1CampaignRecord,
    mandate: AcceptanceVerificationMandate,
) -> tuple[str, ...]:
    """Replay the immutable A1 campaign question before external promotion."""

    if not isinstance(record, AcousticA1CampaignRecord):
        raise TypeError("record must be an AcousticA1CampaignRecord")
    if not isinstance(mandate, AcceptanceVerificationMandate):
        raise TypeError("mandate must be an AcceptanceVerificationMandate")
    config = AcousticAcceptanceConfig(campaign_id=record.campaign_id)
    blockers: list[str] = []
    expected_mandate = {
        "campaign_id": record.campaign_id,
        "connector_id": ACOUSTIC_A1_CONNECTOR_ID,
        "adapter_id": record.adapter_id,
        "expected_source_commit_sha256": record.source_commit_sha256,
        "expected_physical_identity_sha256": record.physical_identity_sha256,
        "expected_evidence_class": AcceptanceEvidenceClass.LIVE,
        "target": config.required_error_reduction,
        "target_tolerance": 0.0,
        "scenario_id": "",
        "expected_simulated_channel_ids": (),
        "required_cases": ACOUSTIC_A1_REQUIRED_CASES,
    }
    if mandate.sha256 != record.mandate_sha256:
        blockers.append("acoustic_a1_mandate_digest_mismatch")
    for field, value in expected_mandate.items():
        if getattr(mandate, field) != value:
            blockers.append(f"acoustic_a1_mandate_{field}_mismatch")
    if record.receipt.config_sha256 != config.sha256:
        blockers.append("acoustic_a1_config_digest_mismatch")
    if not record.accepted:
        blockers.append("acoustic_a1_producer_record_not_accepted")
    return tuple(sorted(set(blockers)))


def verify_acoustic_a1_with_external_witnesses(
    record: AcousticA1CampaignRecord,
    mandate: AcceptanceVerificationMandate,
    *,
    preregistration_receipt: PreregisteredAcceptanceReceipt | None,
    metrology_witness_bundle: AcceptanceWitnessBundle | Mapping[str, Any] | None,
    governance_witness_bundle: AcceptanceWitnessBundle | Mapping[str, Any] | None,
    metrology_witness_key_sha256: str,
    governance_witness_key_sha256: str,
    metrology_sequence: int = 1,
    governance_sequence: int = 1,
    metrology_previous_statement_sha256: str = "sha256:" + "0" * 64,
    governance_previous_statement_sha256: str = "sha256:" + "0" * 64,
    now_ns: int | None = None,
) -> ExternallyWitnessedAcousticA1Receipt:
    """Independently bind A1 physical and governance evidence to two roots."""

    blockers = list(acoustic_a1_campaign_binding_blockers(record, mandate))
    verified_preregistration = ""
    expected_acceptance_log_key_sha256 = ""
    if preregistration_receipt is None:
        blockers.append("acceptance_preregistration_missing")
    elif not isinstance(preregistration_receipt, PreregisteredAcceptanceReceipt):
        blockers.append("acceptance_preregistration_invalid")
    elif (
        not preregistration_receipt.accepted
        or preregistration_receipt.mandate.sha256 != mandate.sha256
        or preregistration_receipt.campaign_started_at_ns != record.started_at_ns
        or not preregistration_receipt.strictly_predates_campaign
    ):
        blockers.append("acceptance_preregistration_binding_invalid")
    else:
        verified_preregistration = preregistration_receipt.sha256
        if preregistration_receipt.trust_policy is None:
            blockers.append("acceptance_trust_policy_missing_or_invalid")
        else:
            policy = preregistration_receipt.trust_policy
            expected_acceptance_log_key_sha256 = policy.acceptance_log_key_sha256
            if (
                metrology_witness_key_sha256
                != policy.metrology_witness_key_sha256
            ):
                blockers.append("external_metrology_trust_root_not_preregistered")
            if (
                governance_witness_key_sha256
                != policy.governance_witness_key_sha256
            ):
                blockers.append("external_governance_trust_root_not_preregistered")

    verified_metrology_bundle = ""
    verified_governance_bundle = ""
    if metrology_witness_bundle is None:
        blockers.append("external_metrology_witness_missing")
    elif not metrology_witness_key_sha256:
        blockers.append("external_metrology_trust_root_missing")
    else:
        try:
            verified = verify_acceptance_witness_artifact_bundle(
                metrology_witness_bundle,
                expected_role=AcceptanceWitnessRole.METROLOGY,
                expected_public_key_sha256=metrology_witness_key_sha256,
                expected_campaign_id=record.campaign_id,
                expected_mandate_sha256=mandate.sha256,
                expected_artifact_sha256=record.sha256,
                expected_evidence_sha256=record.receipt.sha256,
                expected_sequence=metrology_sequence,
                expected_previous_statement_sha256=(
                    metrology_previous_statement_sha256
                ),
                campaign_completed_at_ns=record.completed_at_ns,
                now_ns=now_ns,
            )
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
            verified = verify_acceptance_witness_artifact_bundle(
                governance_witness_bundle,
                expected_role=AcceptanceWitnessRole.GOVERNANCE,
                expected_public_key_sha256=governance_witness_key_sha256,
                expected_campaign_id=record.campaign_id,
                expected_mandate_sha256=mandate.sha256,
                expected_artifact_sha256=record.sha256,
                expected_evidence_sha256=_digest(record.governance_evidence),
                expected_sequence=governance_sequence,
                expected_previous_statement_sha256=(
                    governance_previous_statement_sha256
                ),
                campaign_completed_at_ns=record.completed_at_ns,
                now_ns=now_ns,
            )
            verified_governance_bundle = verified.sha256
        except (AcceptanceWitnessError, TypeError, ValueError) as exc:
            blockers.append(
                exc.code
                if isinstance(exc, AcceptanceWitnessError)
                else "external_governance_witness_invalid"
            )
    if (
        verified_metrology_bundle
        and verified_governance_bundle
        and metrology_witness_key_sha256 == governance_witness_key_sha256
    ):
        blockers.append("external_witness_roots_not_distinct")
    return ExternallyWitnessedAcousticA1Receipt(
        campaign_id=record.campaign_id,
        campaign_record_sha256=record.sha256,
        mandate_sha256=mandate.sha256,
        preregistration_verification_sha256=verified_preregistration,
        metrology_witness_bundle_sha256=verified_metrology_bundle,
        governance_witness_bundle_sha256=verified_governance_bundle,
        metrology_witness_key_sha256=(
            metrology_witness_key_sha256 if verified_metrology_bundle else ""
        ),
        governance_witness_key_sha256=(
            governance_witness_key_sha256 if verified_governance_bundle else ""
        ),
        acceptance_log_key_sha256=expected_acceptance_log_key_sha256,
        blockers=tuple(sorted(set(blockers))),
    )


def build_acoustic_a1_transparency_statement(
    receipt: ExternallyWitnessedAcousticA1Receipt,
    *,
    sequence: int,
    previous_statement_sha256: str,
    previous_rekor_uuid: str | None,
    issued_at_unix: int,
) -> dict[str, Any]:
    """Commit one accepted A1 dual-witness verdict for public timestamping."""

    if not isinstance(receipt, ExternallyWitnessedAcousticA1Receipt):
        raise TypeError("receipt must be an ExternallyWitnessedAcousticA1Receipt")
    if not receipt.accepted:
        raise AcceptanceTransparencyError("acceptance_transparency_receipt_not_accepted")
    if type(sequence) is not int or sequence <= 0:
        raise AcceptanceTransparencyError("acceptance_transparency_sequence_invalid")
    if not _SHA256.fullmatch(previous_statement_sha256):
        raise AcceptanceTransparencyError(
            "acceptance_transparency_previous_statement_sha256_invalid"
        )
    if type(issued_at_unix) is not int or issued_at_unix <= 0:
        raise AcceptanceTransparencyError("acceptance_transparency_issued_at_invalid")
    zero = "sha256:" + "0" * 64
    if sequence == 1:
        if previous_statement_sha256 != zero or previous_rekor_uuid is not None:
            raise AcceptanceTransparencyError("acceptance_transparency_genesis_invalid")
    elif previous_statement_sha256 == zero or not _REKOR_UUID.fullmatch(
        str(previous_rekor_uuid or "")
    ):
        raise AcceptanceTransparencyError("acceptance_transparency_chain_invalid")
    body = {
        "schema": ACCEPTANCE_TRANSPARENCY_STATEMENT_SCHEMA,
        "domain": "aura.reality-reach.acoustic-a1",
        "campaign_id": receipt.campaign_id,
        "mandate_sha256": receipt.mandate_sha256,
        "external_verification_sha256": receipt.sha256,
        "acceptance_log_key_sha256": receipt.acceptance_log_key_sha256,
        "metrology_witness_bundle_sha256": receipt.metrology_witness_bundle_sha256,
        "governance_witness_bundle_sha256": receipt.governance_witness_bundle_sha256,
        "sequence": sequence,
        "previous_statement_sha256": previous_statement_sha256,
        "previous_rekor_uuid": previous_rekor_uuid,
        "issued_at_unix": issued_at_unix,
    }
    return {**body, "statement_sha256": _digest(body)}


def _validate_acoustic_a1_transparency_statement(
    raw: object,
    *,
    receipt: ExternallyWitnessedAcousticA1Receipt,
    expected_sequence: int,
    expected_previous_statement_sha256: str,
    expected_previous_rekor_uuid: str | None,
) -> dict[str, Any]:
    statement = validate_acceptance_transparency_statement_envelope(raw)
    issued_at = statement.get("issued_at_unix")
    if type(issued_at) is not int:
        raise AcceptanceTransparencyError(
            "acceptance_transparency_statement_binding_invalid"
        )
    expected = build_acoustic_a1_transparency_statement(
        receipt,
        sequence=expected_sequence,
        previous_statement_sha256=expected_previous_statement_sha256,
        previous_rekor_uuid=expected_previous_rekor_uuid,
        issued_at_unix=issued_at,
    )
    if statement != expected:
        raise AcceptanceTransparencyError(
            "acceptance_transparency_statement_binding_invalid"
        )
    return dict(statement)


def build_acoustic_a1_transparency_bundle(
    *,
    statement: Mapping[str, Any],
    producer_signature: bytes,
    producer_certificate_pem: bytes,
    rekor_uuid: str,
    rekor_entry: Mapping[str, Any],
    trusted_log_public_key_pem: bytes,
) -> dict[str, Any]:
    """Build a portable Rekor bundle for one accepted A1 statement."""

    statement_document = validate_acceptance_transparency_statement_envelope(statement)
    if statement_document.get("domain") != "aura.reality-reach.acoustic-a1":
        raise AcceptanceTransparencyError(
            "acceptance_transparency_statement_binding_invalid"
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
class TransparentlyLoggedAcousticA1Receipt:
    external_verification: ExternallyWitnessedAcousticA1Receipt
    transparency_bundle_sha256: str
    trusted_log_key_sha256: str
    rekor_uuid: str
    rekor_log_index: int
    rekor_integrated_time: int
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(
            self.external_verification,
            ExternallyWitnessedAcousticA1Receipt,
        ):
            raise TypeError(
                "external_verification must be an "
                "ExternallyWitnessedAcousticA1Receipt"
            )
        for name in ("transparency_bundle_sha256", "trusted_log_key_sha256"):
            value = str(getattr(self, name) or "")
            if value and not _SHA256.fullmatch(value):
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
        return bool(
            self.external_verification.accepted
            and self.transparency_bundle_sha256
            and not self.blockers
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        document = {
            "schema": TRANSPARENT_ACOUSTIC_A1_VERIFICATION_SCHEMA,
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


def verify_transparently_logged_acoustic_a1(
    receipt: ExternallyWitnessedAcousticA1Receipt,
    *,
    transparency_bundle: Mapping[str, Any] | None,
    trusted_log_public_key_pem: bytes | None,
    expected_sequence: int,
    expected_previous_statement_sha256: str,
    expected_previous_rekor_uuid: str | None,
    minimum_log_index: int | None = None,
    minimum_integrated_time: int | None = None,
) -> TransparentlyLoggedAcousticA1Receipt:
    """Require append-only public-log inclusion for A1 promotion."""

    if not isinstance(receipt, ExternallyWitnessedAcousticA1Receipt):
        raise TypeError("receipt must be an ExternallyWitnessedAcousticA1Receipt")
    artifact = verify_acceptance_transparency_artifact(
        transparency_bundle=transparency_bundle,
        trusted_log_public_key_pem=trusted_log_public_key_pem,
        statement_validator=lambda raw: _validate_acoustic_a1_transparency_statement(
            raw,
            receipt=receipt,
            expected_sequence=expected_sequence,
            expected_previous_statement_sha256=expected_previous_statement_sha256,
            expected_previous_rekor_uuid=expected_previous_rekor_uuid,
        ),
        minimum_log_index=minimum_log_index,
        minimum_integrated_time=minimum_integrated_time,
    )
    blockers = list(artifact.blockers)
    if (
        artifact.accepted
        and artifact.trusted_log_key_sha256 != receipt.acceptance_log_key_sha256
    ):
        blockers.append("acceptance_transparency_log_root_not_preregistered")
    return TransparentlyLoggedAcousticA1Receipt(
        external_verification=receipt,
        transparency_bundle_sha256=artifact.transparency_bundle_sha256,
        trusted_log_key_sha256=artifact.trusted_log_key_sha256,
        rekor_uuid=artifact.rekor_uuid,
        rekor_log_index=artifact.rekor_log_index,
        rekor_integrated_time=artifact.rekor_integrated_time,
        blockers=tuple(sorted(set(blockers))),
    )


def persist_transparently_logged_acoustic_a1_receipt(
    receipt: TransparentlyLoggedAcousticA1Receipt,
    path: str | Path,
) -> bool:
    """Create-once publish one transparency-bound A1 promotion verdict."""

    if not isinstance(receipt, TransparentlyLoggedAcousticA1Receipt):
        raise TypeError("receipt must be a TransparentlyLoggedAcousticA1Receipt")
    target = Path(path).expanduser().absolute()
    if not target.name or target.name in {".", ".."}:
        raise AcousticAcceptanceError("transparent_acoustic_a1_receipt_path_invalid")
    payload = canonical_json(receipt.to_dict())
    try:
        with DirectoryCustody.acquire(target.parent, create=True, private=True) as custody:
            published = bool(custody.write_bytes_once(target.name, payload, mode=0o600))
            fd = custody.open_file(target.name, os.O_RDONLY)
            try:
                if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                    raise AcousticAcceptanceError(
                        "transparent_acoustic_a1_receipt_mode_invalid"
                    )
            finally:
                os.close(fd)
            existing = custody.read_bytes(target.name, max_bytes=1024 * 1024)
    except SecurePathCustodyError as exc:
        raise AcousticAcceptanceError(
            "transparent_acoustic_a1_receipt_custody_invalid"
        ) from exc
    if existing != payload:
        raise AcousticAcceptanceError("transparent_acoustic_a1_receipt_collision")
    return published


def persist_externally_witnessed_acoustic_a1_receipt(
    receipt: ExternallyWitnessedAcousticA1Receipt,
    path: str | Path,
) -> bool:
    """Create-once publish one dual-root A1 promotion verdict."""

    if not isinstance(receipt, ExternallyWitnessedAcousticA1Receipt):
        raise TypeError("receipt must be an ExternallyWitnessedAcousticA1Receipt")
    target = Path(path).expanduser().absolute()
    if not target.name or target.name in {".", ".."}:
        raise AcousticAcceptanceError("external_acoustic_a1_receipt_path_invalid")
    payload = canonical_json(receipt.to_dict())
    try:
        with DirectoryCustody.acquire(target.parent, create=True, private=True) as custody:
            published = bool(custody.write_bytes_once(target.name, payload, mode=0o600))
            fd = custody.open_file(target.name, os.O_RDONLY)
            try:
                if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                    raise AcousticAcceptanceError(
                        "external_acoustic_a1_receipt_mode_invalid"
                    )
            finally:
                os.close(fd)
            existing = custody.read_bytes(target.name, max_bytes=1024 * 1024)
    except SecurePathCustodyError as exc:
        raise AcousticAcceptanceError(
            "external_acoustic_a1_receipt_custody_invalid"
        ) from exc
    if existing != payload:
        raise AcousticAcceptanceError("external_acoustic_a1_receipt_collision")
    return published


class AcousticA1CampaignStore:
    """Private create-once storage for positive and negative A1 outcomes."""

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(
            root
            or (state_root() / "data" / "reality_reach" / "acoustic_a1_acceptance")
        ).expanduser().absolute()

    @property
    def root(self) -> Path:
        return self._root

    @staticmethod
    def _filename(campaign_id: str) -> str:
        if not campaign_id or len(campaign_id) > 128:
            raise ValueError("campaign_id must be a bounded non-empty string")
        return _digest({"campaign_id": campaign_id}).removeprefix("sha256:") + ".json"

    @staticmethod
    def _verify_mode(custody: DirectoryCustody, filename: str) -> None:
        fd = custody.open_file(filename, os.O_RDONLY)
        try:
            if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                raise AcousticAcceptanceError("acoustic_a1_campaign_mode_invalid")
        finally:
            os.close(fd)

    def persist(self, record: AcousticA1CampaignRecord) -> bool:
        if not isinstance(record, AcousticA1CampaignRecord):
            raise TypeError("record must be an AcousticA1CampaignRecord")
        filename = self._filename(record.campaign_id)
        payload = canonical_json(record.to_dict())
        try:
            with DirectoryCustody.acquire(self._root, create=True, private=True) as custody:
                published = bool(custody.write_bytes_once(filename, payload, mode=0o600))
                self._verify_mode(custody, filename)
                existing = custody.read_bytes(filename, max_bytes=2 * 1024 * 1024)
        except SecurePathCustodyError as exc:
            raise AcousticAcceptanceError("acoustic_a1_campaign_custody_invalid") from exc
        if existing != payload:
            raise AcousticAcceptanceError("acoustic_a1_campaign_collision")
        return published

    def load(self, campaign_id: str) -> AcousticA1CampaignRecord:
        filename = self._filename(campaign_id)
        try:
            os.lstat(self._root)
        except FileNotFoundError as exc:
            raise AcousticAcceptanceError("acoustic_a1_campaign_unavailable") from exc
        except OSError as exc:
            raise AcousticAcceptanceError("acoustic_a1_campaign_custody_invalid") from exc
        try:
            with DirectoryCustody.acquire(self._root, create=False, private=True) as custody:
                self._verify_mode(custody, filename)
                payload = custody.read_bytes(filename, max_bytes=2 * 1024 * 1024)
        except FileNotFoundError as exc:
            raise AcousticAcceptanceError("acoustic_a1_campaign_unavailable") from exc
        except SecurePathCustodyError as exc:
            raise AcousticAcceptanceError("acoustic_a1_campaign_custody_invalid") from exc
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, ValueError) as exc:
            raise AcousticAcceptanceError("acoustic_a1_campaign_json_invalid") from exc
        record = AcousticA1CampaignRecord.from_dict(document)
        if record.campaign_id != campaign_id:
            raise AcousticAcceptanceError("acoustic_a1_campaign_identity_mismatch")
        return record


def _pava(values: Sequence[float]) -> tuple[float, ...]:
    blocks: list[tuple[float, int]] = []
    for value in values:
        blocks.append((_finite(value, name="pava_value"), 1))
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right_mean, right_count = blocks.pop()
            left_mean, left_count = blocks.pop()
            count = left_count + right_count
            blocks.append(
                (
                    (left_mean * left_count + right_mean * right_count) / count,
                    count,
                )
            )
    fitted: list[float] = []
    for mean, count in blocks:
        fitted.extend([mean] * count)
    return tuple(fitted)


def _inverse_monotone(
    amplitudes: Sequence[float],
    observed: Sequence[float],
    target: float,
) -> float:
    if target <= observed[0]:
        return amplitudes[0]
    if target >= observed[-1]:
        return amplitudes[-1]
    for index in range(1, len(observed)):
        if target <= observed[index]:
            low_y = observed[index - 1]
            high_y = observed[index]
            if high_y <= low_y:
                return amplitudes[index]
            fraction = (target - low_y) / (high_y - low_y)
            return amplitudes[index - 1] + fraction * (
                amplitudes[index] - amplitudes[index - 1]
            )
    return amplitudes[-1]


def _nominal_open_loop_amplitude(target_dbfs: float, maximum: float) -> float:
    return float(
        min(maximum, max(0.0, math.sqrt(2.0) * 10.0 ** (target_dbfs / 20.0)))
    )


def _trial_order(seed: str, labels: Sequence[tuple[str, float, int]]) -> list[tuple[str, float, int]]:
    return sorted(
        labels,
        key=lambda item: hashlib.sha256(
            f"{seed}|{item[0]}|{item[1]:.9f}|{item[2]}".encode()
        ).digest(),
    )


async def run_acoustic_a1_acceptance(
    driver: AcousticTrialDriver,
    config: AcousticAcceptanceConfig,
) -> AcousticA1AcceptanceReceipt:
    """Run equal-work calibration/control arms and restore silence in ``finally``."""

    if not isinstance(driver, AcousticTrialDriver):
        raise TypeError("driver must satisfy AcousticTrialDriver")
    if not isinstance(config, AcousticAcceptanceConfig):
        raise TypeError("config must be an AcousticAcceptanceConfig")
    trials: list[AcousticTrial] = []
    order = 0

    async def capture(
        arm: AcousticTrialArm,
        amplitude: float,
        *,
        target: float | None,
        repeat: int,
    ) -> AcousticTrial:
        nonlocal order
        order += 1
        raw_trial_id = f"{config.campaign_id}.{arm.value}.{order:04d}"
        trial_id = raw_trial_id
        if len(trial_id) > 128:
            suffix = hashlib.sha256(raw_trial_id.encode()).hexdigest()[:24]
            trial_id = f"{config.campaign_id[:80]}.{arm.value}.{order:04d}.{suffix}"
        sample = await driver.measure_stimulus(amplitude, trial_id=trial_id)
        trial = AcousticTrial(
            trial_id=trial_id,
            arm=arm,
            order=order,
            repeat=repeat,
            amplitude=amplitude,
            target_dbfs=target,
            observed_dbfs=sample.value,
            source_event_id=sample.source_event_id,
            captured_at_ns=sample.captured_at_ns,
        )
        trials.append(trial)
        return trial

    baseline = math.nan
    restoration_error = math.inf
    experiment_error: BaseException | None = None
    try:
        calibration_groups: list[list[float]] = []
        for amplitude in config.calibration_amplitudes:
            group: list[float] = []
            for repeat in range(config.calibration_repeats):
                trial = await capture(
                    AcousticTrialArm.CALIBRATION,
                    amplitude,
                    target=None,
                    repeat=repeat,
                )
                group.append(trial.observed_dbfs)
            calibration_groups.append(group)
        medians = tuple(statistics.median(group) for group in calibration_groups)
        fitted = _pava(medians)
        baseline = fitted[0]
        calibration_span = fitted[-1] - baseline
        lower = baseline + config.baseline_margin_db
        upper = fitted[-1] - config.ceiling_margin_db
        if upper <= lower:
            raise AcousticAcceptanceError("acoustic_a1_calibration_span_insufficient")
        targets = tuple(
            lower + quantile * (upper - lower)
            for quantile in config.heldout_quantiles
        )
        schedule = _trial_order(
            f"{config.schedule_seed}|{config.sha256}",
            tuple(
                (arm.value, target, repeat)
                for target in targets
                for repeat in range(config.heldout_repeats)
                for arm in (AcousticTrialArm.OPEN_LOOP, AcousticTrialArm.CLOSED_LOOP)
            ),
        )
        maximum = config.calibration_amplitudes[-1]
        for arm_value, target, repeat in schedule:
            arm = AcousticTrialArm(arm_value)
            amplitude = (
                _nominal_open_loop_amplitude(target, maximum)
                if arm is AcousticTrialArm.OPEN_LOOP
                else _inverse_monotone(config.calibration_amplitudes, fitted, target)
            )
            await capture(arm, amplitude, target=target, repeat=repeat)
    except BaseException as exc:
        experiment_error = exc

    try:
        restored = await capture(
            AcousticTrialArm.RESTORATION,
            0.0,
            target=baseline if math.isfinite(baseline) else None,
            repeat=0,
        )
        if math.isfinite(baseline):
            restoration_error = abs(restored.observed_dbfs - baseline)
    except BaseException as restoration_exc:
        if experiment_error is not None:
            experiment_error.add_note(
                "acoustic silence restoration also failed: "
                f"{type(restoration_exc).__name__}: {restoration_exc}"
            )
            raise experiment_error from restoration_exc
        raise AcousticAcceptanceError("acoustic_a1_restoration_failed") from restoration_exc

    if experiment_error is not None:
        raise experiment_error

    open_errors = tuple(
        trial.absolute_error_db
        for trial in trials
        if trial.arm is AcousticTrialArm.OPEN_LOOP
        and trial.absolute_error_db is not None
    )
    closed_errors = tuple(
        trial.absolute_error_db
        for trial in trials
        if trial.arm is AcousticTrialArm.CLOSED_LOOP
        and trial.absolute_error_db is not None
    )
    if not open_errors or len(open_errors) != len(closed_errors):
        raise AcousticAcceptanceError("acoustic_a1_equal_work_contract_failed")
    open_mae = statistics.fmean(open_errors)
    closed_mae = statistics.fmean(closed_errors)
    reduction = 1.0 - closed_mae / max(open_mae, 1e-9)
    calibration_trials = tuple(
        trial for trial in trials if trial.arm is AcousticTrialArm.CALIBRATION
    )
    calibration_values = tuple(trial.observed_dbfs for trial in calibration_trials)
    calibration_span = max(calibration_values) - min(calibration_values)
    blockers: list[str] = []
    if calibration_span < config.minimum_signal_span_db:
        blockers.append("acoustic_a1_signal_span_below_threshold")
    if reduction < config.required_error_reduction:
        blockers.append("acoustic_a1_error_reduction_below_threshold")
    if restoration_error > config.restoration_tolerance_db:
        blockers.append("acoustic_a1_restoration_out_of_tolerance")
    heldout_targets = tuple(
        sorted(
            {
                float(trial.target_dbfs)
                for trial in trials
                if trial.target_dbfs is not None
                and trial.arm in {AcousticTrialArm.OPEN_LOOP, AcousticTrialArm.CLOSED_LOOP}
            }
        )
    )
    return AcousticA1AcceptanceReceipt(
        campaign_id=config.campaign_id,
        config_sha256=config.sha256,
        heldout_targets_dbfs=heldout_targets,
        trials=tuple(trials),
        open_loop_mae_db=open_mae,
        closed_loop_mae_db=closed_mae,
        error_reduction=reduction,
        restoration_error_db=restoration_error,
        calibration_span_db=calibration_span,
        blockers=tuple(blockers),
        completed_at_ns=time.time_ns(),
    )


__all__ = [
    "ACOUSTIC_A1_CONNECTOR_ID",
    "ACOUSTIC_A1_CAMPAIGN_SCHEMA",
    "ACOUSTIC_A1_RECEIPT_SCHEMA",
    "ACOUSTIC_A1_REQUIRED_CASES",
    "EXTERNAL_ACOUSTIC_A1_VERIFICATION_SCHEMA",
    "TRANSPARENT_ACOUSTIC_A1_VERIFICATION_SCHEMA",
    "AcousticA1AcceptanceReceipt",
    "AcousticA1CampaignRecord",
    "AcousticA1CampaignStore",
    "AcousticAcceptanceConfig",
    "AcousticAcceptanceError",
    "AcousticTrial",
    "AcousticTrialArm",
    "AcousticTrialDriver",
    "ExternallyWitnessedAcousticA1Receipt",
    "TransparentlyLoggedAcousticA1Receipt",
    "acoustic_a1_campaign_binding_blockers",
    "build_acoustic_a1_transparency_bundle",
    "build_acoustic_a1_transparency_statement",
    "persist_externally_witnessed_acoustic_a1_receipt",
    "persist_transparently_logged_acoustic_a1_receipt",
    "run_acoustic_a1_acceptance",
    "verify_acoustic_a1_with_external_witnesses",
    "verify_transparently_logged_acoustic_a1",
]
