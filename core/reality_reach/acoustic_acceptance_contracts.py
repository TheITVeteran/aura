"""Immutable evidence contracts for acoustic A1 acceptance."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from core.reality_reach.acceptance_contracts import (
    acceptance_governance_accepted,
    acceptance_governance_document,
)

if TYPE_CHECKING:
    from core.reality_reach.scalar_adapter import ScalarSample

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
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


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


__all__ = [
    "ACOUSTIC_A1_CONNECTOR_ID",
    "ACOUSTIC_A1_CAMPAIGN_SCHEMA",
    "ACOUSTIC_A1_RECEIPT_SCHEMA",
    "ACOUSTIC_A1_REQUIRED_CASES",
    "EXTERNAL_ACOUSTIC_A1_VERIFICATION_SCHEMA",
    "TRANSPARENT_ACOUSTIC_A1_VERIFICATION_SCHEMA",
    "AcousticA1AcceptanceReceipt",
    "AcousticA1CampaignRecord",
    "AcousticAcceptanceConfig",
    "AcousticAcceptanceError",
    "AcousticTrial",
    "AcousticTrialArm",
    "AcousticTrialDriver",
    "ExternallyWitnessedAcousticA1Receipt",
]
