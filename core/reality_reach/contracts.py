"""Immutable contracts for Aura's physical-reality reachability boundary.

The types in this module distinguish an intended physical outcome from the
channels that can cause and independently observe it. They are deliberately
strict: unsupported units, undeclared coupling, and insufficient metrology are
certified as limitations instead of being converted into optimistic prose.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.audit_chain import canonical_json, sha256_hex

REALITY_IR_SCHEMA = "aura.reality-ir.v1"
REACHABILITY_CERTIFICATE_SCHEMA = "aura.reachability-certificate.v1"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")


def _finite(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_identifier(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a canonical identifier")


def _validate_unique_identifiers(values: tuple[str, ...], *, name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    for value in values:
        _validate_identifier(value, name=name)


class ObjectiveKind(StrEnum):
    OBSERVE = "observe"
    CONTROL = "control"
    IDENTIFY = "identify"


class RealityLayer(StrEnum):
    INTERNAL = "internal"
    EFFECTIVE = "effective"
    DIRECT = "direct"
    AMBIENT = "ambient"


class ChannelKind(StrEnum):
    ACTUATOR = "actuator"
    SENSOR = "sensor"


class CouplingClass(StrEnum):
    SOFTWARE = "software"
    NETWORK = "network"
    AUDIO = "audio"
    OPTICAL = "optical"
    THERMAL = "thermal"
    RADIO = "radio"
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    UNKNOWN = "unknown"


class ConstraintKind(StrEnum):
    SAFETY = "safety"
    LEGAL = "legal"
    PRIVACY = "privacy"
    RESOURCE = "resource"
    TEMPORAL = "temporal"
    OPERATIONAL = "operational"


class EvidenceLevel(StrEnum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"
    P5 = "P5"
    P6 = "P6"

    @property
    def rank(self) -> int:
        return int(self.value[1:])


class FailureCode(StrEnum):
    NO_CHANNEL = "NO_CHANNEL"
    BELOW_SENSOR_FLOOR = "BELOW_SENSOR_FLOOR"
    SHARED_REFERENCE = "SHARED_REFERENCE"
    ORDINARY_MODEL = "ORDINARY_MODEL"
    NOT_REPRODUCIBLE = "NOT_REPRODUCIBLE"
    NOT_CONTROLLABLE = "NOT_CONTROLLABLE"
    SEARCH_INCOMPLETE = "SEARCH_INCOMPLETE"
    AMBIENT_IDENTITY_UNRESOLVED = "AMBIENT_IDENTITY_UNRESOLVED"
    TARGET_OUT_OF_RANGE = "TARGET_OUT_OF_RANGE"
    CONSTRAINT_UNSATISFIED = "CONSTRAINT_UNSATISFIED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ReachabilityStatus(StrEnum):
    REACHABLE = "reachable"
    PARTIAL = "partial"
    UNREACHABLE = "unreachable"


@dataclass(frozen=True, slots=True)
class NumericDomain:
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        minimum = _finite(self.minimum, name="domain.minimum")
        maximum = _finite(self.maximum, name="domain.maximum")
        if minimum > maximum:
            raise ValueError("domain.minimum must not exceed domain.maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def contains(self, value: float, *, tolerance: float = 0.0) -> bool:
        number = _finite(value, name="value")
        margin = _finite(tolerance, name="tolerance")
        if margin < 0.0:
            raise ValueError("tolerance must be non-negative")
        return self.minimum <= number - margin and number + margin <= self.maximum

    def to_dict(self) -> dict[str, float]:
        return {"minimum": self.minimum, "maximum": self.maximum}


@dataclass(frozen=True, slots=True)
class Constraint:
    constraint_id: str
    kind: ConstraintKind
    description: str
    applies_to_channels: tuple[str, ...] = ()
    required: bool = True

    def __post_init__(self) -> None:
        _validate_identifier(self.constraint_id, name="constraint_id")
        if not isinstance(self.kind, ConstraintKind):
            raise TypeError("kind must be a ConstraintKind")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("constraint description must be non-empty")
        _validate_unique_identifiers(
            self.applies_to_channels,
            name="applies_to_channels",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind.value,
            "description": self.description,
            "applies_to_channels": list(self.applies_to_channels),
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class ProofRequirement:
    minimum_evidence: EvidenceLevel = EvidenceLevel.P2
    minimum_independent_sensors: int = 1
    blinded: bool = False
    dose_response: bool = False
    frequency_or_time_response: bool = False
    cross_sensor: bool = False
    heldout_prediction: bool = False
    inverse_control: bool = False
    reboot_robustness: bool = False
    ordinary_model_challenge: bool = True
    external_metrology: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.minimum_evidence, EvidenceLevel):
            raise TypeError("minimum_evidence must be an EvidenceLevel")
        if isinstance(self.minimum_independent_sensors, bool) or not isinstance(
            self.minimum_independent_sensors,
            int,
        ):
            raise TypeError("minimum_independent_sensors must be an integer")
        if self.minimum_independent_sensors < 1:
            raise ValueError("minimum_independent_sensors must be at least one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_evidence": self.minimum_evidence.value,
            "minimum_independent_sensors": self.minimum_independent_sensors,
            "blinded": self.blinded,
            "dose_response": self.dose_response,
            "frequency_or_time_response": self.frequency_or_time_response,
            "cross_sensor": self.cross_sensor,
            "heldout_prediction": self.heldout_prediction,
            "inverse_control": self.inverse_control,
            "reboot_robustness": self.reboot_robustness,
            "ordinary_model_challenge": self.ordinary_model_challenge,
            "external_metrology": self.external_metrology,
        }


@dataclass(frozen=True, slots=True)
class RealityIR:
    request_id: str
    objective: str
    objective_kind: ObjectiveKind
    observable: str
    unit: str
    target: float
    tolerance: float
    domain: NumericDomain
    allowed_actuators: tuple[str, ...]
    allowed_sensors: tuple[str, ...]
    constraints: tuple[Constraint, ...] = ()
    required_proof: ProofRequirement = field(default_factory=ProofRequirement)
    reality_layer: RealityLayer = RealityLayer.EFFECTIVE
    horizon_s: float = 60.0
    schema: str = REALITY_IR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REALITY_IR_SCHEMA:
            raise ValueError(f"unsupported RealityIR schema: {self.schema}")
        _validate_identifier(self.request_id, name="request_id")
        if not isinstance(self.objective, str) or not self.objective.strip():
            raise ValueError("objective must be non-empty")
        if not isinstance(self.objective_kind, ObjectiveKind):
            raise TypeError("objective_kind must be an ObjectiveKind")
        _validate_identifier(self.observable, name="observable")
        _validate_identifier(self.unit, name="unit")
        target = _finite(self.target, name="target")
        tolerance = _finite(self.tolerance, name="tolerance")
        horizon_s = _finite(self.horizon_s, name="horizon_s")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        if horizon_s <= 0.0:
            raise ValueError("horizon_s must be positive")
        if not self.domain.contains(target):
            raise ValueError("target must lie inside the declared domain")
        if self.objective_kind != ObjectiveKind.OBSERVE and not self.allowed_actuators:
            raise ValueError("control and identification objectives require an actuator")
        if not self.allowed_sensors:
            raise ValueError("RealityIR requires at least one observation channel")
        _validate_unique_identifiers(self.allowed_actuators, name="allowed_actuators")
        _validate_unique_identifiers(self.allowed_sensors, name="allowed_sensors")
        constraint_ids = tuple(item.constraint_id for item in self.constraints)
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraints must have unique identifiers")
        if not isinstance(self.required_proof, ProofRequirement):
            raise TypeError("required_proof must be a ProofRequirement")
        if not isinstance(self.reality_layer, RealityLayer):
            raise TypeError("reality_layer must be a RealityLayer")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "horizon_s", horizon_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request_id": self.request_id,
            "objective": self.objective,
            "objective_kind": self.objective_kind.value,
            "observable": self.observable,
            "unit": self.unit,
            "target": self.target,
            "tolerance": self.tolerance,
            "domain": self.domain.to_dict(),
            "allowed_actuators": list(self.allowed_actuators),
            "allowed_sensors": list(self.allowed_sensors),
            "constraints": [item.to_dict() for item in self.constraints],
            "required_proof": self.required_proof.to_dict(),
            "reality_layer": self.reality_layer.value,
            "horizon_s": self.horizon_s,
        }

    @property
    def sha256(self) -> str:
        return sha256_hex(canonical_json(self.to_dict()))


@dataclass(frozen=True, slots=True)
class ChannelDeclaration:
    channel_id: str
    kind: ChannelKind
    observable: str
    unit: str
    domain: NumericDomain
    coupling: CouplingClass
    reality_layers: tuple[RealityLayer, ...]
    evidence_level: EvidenceLevel
    owner: str
    resolution: float = 0.0
    sample_rate_hz: float = 0.0
    max_latency_s: float = 0.0
    stale_after_s: float = 30.0
    reference_id: str = ""
    calibration_id: str = ""
    calibration_valid_until_ns: int | None = None
    compliance_tags: tuple[str, ...] = ()
    external_metrology: bool = False
    coupling_validated: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        _validate_identifier(self.channel_id, name="channel_id")
        _validate_identifier(self.observable, name="observable")
        _validate_identifier(self.unit, name="unit")
        if not isinstance(self.kind, ChannelKind):
            raise TypeError("kind must be a ChannelKind")
        if not isinstance(self.coupling, CouplingClass):
            raise TypeError("coupling must be a CouplingClass")
        if not self.reality_layers or len(self.reality_layers) != len(
            set(self.reality_layers)
        ):
            raise ValueError("reality_layers must be non-empty and unique")
        if any(not isinstance(layer, RealityLayer) for layer in self.reality_layers):
            raise TypeError("reality_layers must contain RealityLayer members")
        if not isinstance(self.evidence_level, EvidenceLevel):
            raise TypeError("evidence_level must be an EvidenceLevel")
        if not isinstance(self.owner, str) or not self.owner.strip():
            raise ValueError("owner must be non-empty")
        resolution = _finite(self.resolution, name="resolution")
        sample_rate_hz = _finite(self.sample_rate_hz, name="sample_rate_hz")
        max_latency_s = _finite(self.max_latency_s, name="max_latency_s")
        stale_after_s = _finite(self.stale_after_s, name="stale_after_s")
        if min(resolution, sample_rate_hz, max_latency_s) < 0.0:
            raise ValueError("channel metrology fields must be non-negative")
        if stale_after_s <= 0.0:
            raise ValueError("stale_after_s must be positive")
        if self.kind == ChannelKind.SENSOR:
            _validate_identifier(self.reference_id, name="reference_id")
            if sample_rate_hz <= 0.0:
                raise ValueError("sensor sample_rate_hz must be positive")
        elif self.reference_id:
            _validate_identifier(self.reference_id, name="reference_id")
        if self.calibration_id:
            _validate_identifier(self.calibration_id, name="calibration_id")
        if self.external_metrology and not self.calibration_id:
            raise ValueError("external metrology requires a calibration identifier")
        if self.calibration_valid_until_ns is not None:
            if (
                isinstance(self.calibration_valid_until_ns, bool)
                or not isinstance(self.calibration_valid_until_ns, int)
                or self.calibration_valid_until_ns <= 0
            ):
                raise ValueError("calibration_valid_until_ns must be a positive integer")
            if not self.calibration_id:
                raise ValueError("calibration expiry requires a calibration identifier")
        _validate_unique_identifiers(self.compliance_tags, name="compliance_tags")
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        object.__setattr__(self, "max_latency_s", max_latency_s)
        object.__setattr__(self, "stale_after_s", stale_after_s)

    def supports(self, contract: RealityIR) -> bool:
        return (
            self.enabled
            and self.observable == contract.observable
            and self.unit == contract.unit
            and contract.reality_layer in self.reality_layers
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "kind": self.kind.value,
            "observable": self.observable,
            "unit": self.unit,
            "domain": self.domain.to_dict(),
            "coupling": self.coupling.value,
            "reality_layers": [layer.value for layer in self.reality_layers],
            "evidence_level": self.evidence_level.value,
            "owner": self.owner,
            "resolution": self.resolution,
            "sample_rate_hz": self.sample_rate_hz,
            "max_latency_s": self.max_latency_s,
            "stale_after_s": self.stale_after_s,
            "reference_id": self.reference_id,
            "calibration_id": self.calibration_id,
            "calibration_valid_until_ns": self.calibration_valid_until_ns,
            "compliance_tags": list(self.compliance_tags),
            "external_metrology": self.external_metrology,
            "coupling_validated": self.coupling_validated,
            "enabled": self.enabled,
        }

    @property
    def sha256(self) -> str:
        return sha256_hex(canonical_json(self.to_dict()))


@dataclass(frozen=True, slots=True)
class ReachabilityFailure:
    code: FailureCode
    message: str
    channels: tuple[str, ...] = ()
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.code, FailureCode):
            raise TypeError("code must be a FailureCode")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("failure message must be non-empty")
        _validate_unique_identifiers(self.channels, name="failure channels")
        detail_keys = tuple(key for key, _value in self.details)
        if len(detail_keys) != len(set(detail_keys)):
            raise ValueError("failure detail keys must be unique")

    @classmethod
    def from_mapping(
        cls,
        code: FailureCode,
        message: str,
        *,
        channels: tuple[str, ...] = (),
        details: Mapping[str, Any] | None = None,
    ) -> ReachabilityFailure:
        canonical_details = tuple(
            sorted((str(key), str(value)) for key, value in (details or {}).items())
        )
        return cls(
            code=code,
            message=message,
            channels=channels,
            details=canonical_details,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "channels": list(self.channels),
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class ReachabilityCertificate:
    contract_sha256: str
    registry_sha256: str
    status: ReachabilityStatus
    selected_actuators: tuple[str, ...]
    selected_sensors: tuple[str, ...]
    failures: tuple[ReachabilityFailure, ...]
    evidence_ceiling: EvidenceLevel
    claim_boundary: RealityLayer
    issued_at_ns: int
    certificate_sha256: str = ""
    schema: str = REACHABILITY_CERTIFICATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != REACHABILITY_CERTIFICATE_SCHEMA:
            raise ValueError(f"unsupported certificate schema: {self.schema}")
        if not isinstance(self.status, ReachabilityStatus):
            raise TypeError("status must be a ReachabilityStatus")
        if not isinstance(self.evidence_ceiling, EvidenceLevel):
            raise TypeError("evidence_ceiling must be an EvidenceLevel")
        if not isinstance(self.claim_boundary, RealityLayer):
            raise TypeError("claim_boundary must be a RealityLayer")
        _validate_unique_identifiers(
            self.selected_actuators,
            name="selected_actuators",
        )
        _validate_unique_identifiers(self.selected_sensors, name="selected_sensors")
        if isinstance(self.issued_at_ns, bool) or self.issued_at_ns <= 0:
            raise ValueError("issued_at_ns must be a positive integer")
        expected = self.compute_sha256()
        if self.certificate_sha256 and self.certificate_sha256 != expected:
            raise ValueError("certificate_sha256 does not match certificate payload")
        object.__setattr__(self, "certificate_sha256", expected)

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "contract_sha256": self.contract_sha256,
            "registry_sha256": self.registry_sha256,
            "status": self.status.value,
            "selected_actuators": list(self.selected_actuators),
            "selected_sensors": list(self.selected_sensors),
            "failures": [failure.to_dict() for failure in self.failures],
            "evidence_ceiling": self.evidence_ceiling.value,
            "claim_boundary": self.claim_boundary.value,
            "issued_at_ns": self.issued_at_ns,
        }
        if include_digest:
            payload["certificate_sha256"] = self.certificate_sha256
        return payload

    def compute_sha256(self) -> str:
        return sha256_hex(canonical_json(self.to_dict(include_digest=False)))

    def verify_integrity(self) -> bool:
        return self.certificate_sha256 == self.compute_sha256()


__all__ = [
    "ChannelDeclaration",
    "ChannelKind",
    "Constraint",
    "ConstraintKind",
    "CouplingClass",
    "EvidenceLevel",
    "FailureCode",
    "NumericDomain",
    "ObjectiveKind",
    "ProofRequirement",
    "REALITY_IR_SCHEMA",
    "REACHABILITY_CERTIFICATE_SCHEMA",
    "RealityIR",
    "RealityLayer",
    "ReachabilityCertificate",
    "ReachabilityFailure",
    "ReachabilityStatus",
]
