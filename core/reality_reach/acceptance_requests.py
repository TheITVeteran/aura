"""Immutable request contracts for Reality Reach acceptance campaigns."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from core.reality_reach.acceptance import AcceptanceEvidenceClass

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_MAX_HIL_COMPANION_CHANNELS = 63


@dataclass(frozen=True, slots=True)
class ScalarAcceptanceRequest:
    campaign_id: str
    connector_id: str
    adapter_id: str
    target: float
    expected_source_commit_sha256: str
    evidence_class: AcceptanceEvidenceClass
    mandate_sha256: str = ""
    scenario_id: str = ""
    simulated_channel_ids: tuple[str, ...] = ()
    deadline_s: float = 5.0
    sample_interval_s: float = 0.1
    effect_hold_s: float = 0.25

    def __post_init__(self) -> None:
        mandate_sha256 = str(self.mandate_sha256 or "").strip().lower()
        if mandate_sha256 and not _SHA256.fullmatch(mandate_sha256):
            raise ValueError("mandate_sha256 must be empty or a sha256 digest")
        object.__setattr__(self, "mandate_sha256", mandate_sha256)
        channels = tuple(self.simulated_channel_ids)
        if len(channels) > _MAX_HIL_COMPANION_CHANNELS:
            raise ValueError(
                "simulated_channel_ids exceeds the HIL companion-channel bound"
            )
        if len(set(channels)) != len(channels):
            raise ValueError("simulated_channel_ids must be unique")
        object.__setattr__(self, "simulated_channel_ids", channels)
        if self.evidence_class is AcceptanceEvidenceClass.SIMULATION:
            if self.simulated_channel_ids:
                raise ValueError("simulation acceptance does not use HIL companion channels")
            return
        if self.evidence_class is AcceptanceEvidenceClass.HARDWARE_IN_LOOP:
            if not self.scenario_id or not self.simulated_channel_ids:
                raise ValueError("HIL acceptance requires a scenario and simulated channels")
        elif self.simulated_channel_ids:
            raise ValueError("live acceptance cannot include simulated channels")
        interval = float(self.sample_interval_s)
        deadline = float(self.deadline_s)
        hold = float(self.effect_hold_s)
        if not math.isfinite(interval) or not 0.01 <= interval <= 0.5:
            raise ValueError("sample_interval_s must lie inside [0.01, 0.5]")
        if not math.isfinite(deadline) or not 0.5 <= deadline <= 60.0:
            raise ValueError("deadline_s must lie inside [0.5, 60]")
        if not math.isfinite(hold) or not max(0.05, interval * 2.0) <= hold <= 5.0:
            raise ValueError("effect_hold_s must span at least two sample intervals")


@dataclass(frozen=True, slots=True)
class ScalarAcceptanceMandateRequest:
    campaign_id: str
    connector_id: str
    adapter_id: str
    target: float
    expected_source_commit_sha256: str
    evidence_class: AcceptanceEvidenceClass
    scenario_id: str = ""
    simulated_channel_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_class, AcceptanceEvidenceClass):
            raise TypeError("evidence_class must be an AcceptanceEvidenceClass")
        if not _SHA256.fullmatch(str(self.expected_source_commit_sha256 or "")):
            raise ValueError("expected_source_commit_sha256 must be a sha256 digest")
        channels = tuple(self.simulated_channel_ids)
        if len(channels) > _MAX_HIL_COMPANION_CHANNELS:
            raise ValueError(
                "simulated_channel_ids exceeds the HIL companion-channel bound"
            )
        if len(channels) != len(set(channels)):
            raise ValueError("simulated_channel_ids must be unique")
        object.__setattr__(self, "simulated_channel_ids", channels)
        if self.evidence_class is AcceptanceEvidenceClass.HARDWARE_IN_LOOP:
            if not self.scenario_id or not channels:
                raise ValueError("HIL mandate requires a scenario and simulated channels")
        elif channels:
            raise ValueError("only HIL mandates can bind simulated channels")
        target = float(self.target)
        if not math.isfinite(target):
            raise ValueError("acceptance mandate target must be finite")
        object.__setattr__(self, "target", target)


@dataclass(frozen=True, slots=True)
class AcousticA1MandateRequest:
    campaign_id: str
    expected_source_commit_sha256: str

    def __post_init__(self) -> None:
        campaign_id = str(self.campaign_id or "").strip().lower()
        if not _IDENTIFIER.fullmatch(campaign_id):
            raise ValueError("campaign_id must be a canonical identifier")
        object.__setattr__(self, "campaign_id", campaign_id)
        source = str(self.expected_source_commit_sha256 or "").strip().lower()
        if not _SHA256.fullmatch(source):
            raise ValueError("expected_source_commit_sha256 must be a sha256 digest")
        object.__setattr__(self, "expected_source_commit_sha256", source)


@dataclass(frozen=True, slots=True)
class AcousticA1Request:
    campaign_id: str
    expected_source_commit_sha256: str
    mandate_sha256: str

    def __post_init__(self) -> None:
        mandate_request = AcousticA1MandateRequest(
            campaign_id=self.campaign_id,
            expected_source_commit_sha256=self.expected_source_commit_sha256,
        )
        object.__setattr__(self, "campaign_id", mandate_request.campaign_id)
        object.__setattr__(
            self,
            "expected_source_commit_sha256",
            mandate_request.expected_source_commit_sha256,
        )
        mandate = str(self.mandate_sha256 or "").strip().lower()
        if not _SHA256.fullmatch(mandate):
            raise ValueError("mandate_sha256 must be a sha256 digest")
        object.__setattr__(self, "mandate_sha256", mandate)


__all__ = [
    "AcousticA1MandateRequest",
    "AcousticA1Request",
    "ScalarAcceptanceMandateRequest",
    "ScalarAcceptanceRequest",
]
