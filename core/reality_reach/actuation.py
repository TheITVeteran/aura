"""Typed bidirectional adapter contracts for Reality Reach.

Declaring an actuator is not enough to make it executable. An adapter must
publish bounded capabilities and implement the complete prepare, effect,
cancellation, safe-state, and rollback protocol defined here.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from core.reality_reach.contracts import ChannelDeclaration, NumericDomain
from core.runtime.audit_chain import canonical_json, sha256_hex

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_PARAMETERS_BYTES = 256 * 1024


def _identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a canonical identifier")
    return value


def _digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a canonical sha256:<hex> digest")
    return value


def _positive_int(value: int, *, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _finite(value: float, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return number


def _boolean(value: bool, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _identifiers(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or len(values) != len(set(values)):
        raise ValueError(f"{name} must be a duplicate-free tuple")
    for value in values:
        _identifier(value, name=name)
    return values


def _frozen_json_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain canonical JSON values") from exc
    if len(encoded) > _MAX_PARAMETERS_BYTES:
        raise ValueError(f"{name} exceeds {_MAX_PARAMETERS_BYTES} bytes")
    decoded = json.loads(encoded.decode("utf-8"))
    if not isinstance(decoded, dict) or any(not isinstance(key, str) for key in decoded):
        raise ValueError(f"{name} must have string keys")
    return MappingProxyType(decoded)


class Reversibility(StrEnum):
    REVERSIBLE = "reversible"
    COMPENSATABLE = "compensatable"
    IRREVERSIBLE = "irreversible"


class ActuationState(StrEnum):
    PLANNED = "planned"
    PREPARED = "prepared"
    ADMITTED = "admitted"
    DISPATCHED = "dispatched"
    EXECUTED = "executed"
    EFFECT_VERIFIED = "effect_verified"
    CANCELLED = "cancelled"
    SAFE_STATE = "safe_state"
    COMPENSATED = "compensated"
    ROLLED_BACK = "rolled_back"
    TIMED_OUT = "timed_out"
    INDETERMINATE = "indeterminate"
    FAILED = "failed"
    MANUALLY_RECONCILED = "manually_reconciled"


@dataclass(frozen=True, slots=True)
class ActuatorCapability:
    adapter_id: str
    channel_id: str
    reversibility: Reversibility
    magnitude_domain: NumericDomain
    max_commands_per_minute: int
    observation_channels: tuple[str, ...]
    required_permissions: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    warmup_s: float = 0.0
    cooldown_s: float = 0.0
    watchdog_timeout_s: float = 10.0
    exclusive: bool = True
    supports_cancel: bool = True
    supports_safe_state: bool = True
    supports_rollback: bool = True
    compensation_action: str = ""
    non_reversibility_reason: str = ""

    def __post_init__(self) -> None:
        _identifier(self.adapter_id, name="adapter_id")
        _identifier(self.channel_id, name="channel_id")
        if not isinstance(self.reversibility, Reversibility):
            raise TypeError("reversibility must be a Reversibility")
        if not isinstance(self.magnitude_domain, NumericDomain):
            raise TypeError("magnitude_domain must be a NumericDomain")
        _positive_int(
            self.max_commands_per_minute,
            name="max_commands_per_minute",
        )
        if not self.observation_channels:
            raise ValueError("observation_channels must not be empty")
        _identifiers(self.observation_channels, name="observation_channels")
        _identifiers(self.required_permissions, name="required_permissions")
        _identifiers(self.failure_modes, name="failure_modes")
        for name in (
            "exclusive",
            "supports_cancel",
            "supports_safe_state",
            "supports_rollback",
        ):
            _boolean(getattr(self, name), name=name)
        object.__setattr__(self, "warmup_s", _finite(self.warmup_s, name="warmup_s"))
        object.__setattr__(self, "cooldown_s", _finite(self.cooldown_s, name="cooldown_s"))
        object.__setattr__(
            self,
            "watchdog_timeout_s",
            _finite(self.watchdog_timeout_s, name="watchdog_timeout_s", minimum=0.001),
        )
        if self.reversibility == Reversibility.IRREVERSIBLE and self.supports_rollback:
            raise ValueError("irreversible capabilities cannot claim rollback support")
        if self.reversibility == Reversibility.REVERSIBLE and not self.supports_rollback:
            raise ValueError("reversible capabilities must support rollback")
        if self.compensation_action:
            _identifier(self.compensation_action, name="compensation_action")
        if self.reversibility == Reversibility.COMPENSATABLE and not self.compensation_action:
            raise ValueError("compensatable capabilities require a compensation action")
        if (
            self.reversibility == Reversibility.IRREVERSIBLE
            and not self.non_reversibility_reason.strip()
        ):
            raise ValueError("irreversible capabilities require a reason certificate")
        if len(self.non_reversibility_reason) > 500:
            raise ValueError("non_reversibility_reason exceeds 500 characters")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "channel_id": self.channel_id,
            "reversibility": self.reversibility.value,
            "magnitude_domain": self.magnitude_domain.to_dict(),
            "max_commands_per_minute": self.max_commands_per_minute,
            "observation_channels": list(self.observation_channels),
            "required_permissions": list(self.required_permissions),
            "failure_modes": list(self.failure_modes),
            "warmup_s": self.warmup_s,
            "cooldown_s": self.cooldown_s,
            "watchdog_timeout_s": self.watchdog_timeout_s,
            "exclusive": self.exclusive,
            "supports_cancel": self.supports_cancel,
            "supports_safe_state": self.supports_safe_state,
            "supports_rollback": self.supports_rollback,
            "compensation_action": self.compensation_action,
            "non_reversibility_reason": self.non_reversibility_reason,
        }

    @property
    def sha256(self) -> str:
        return str(sha256_hex(canonical_json(self.to_dict())))


@dataclass(frozen=True, slots=True)
class ActuationCommand:
    command_id: str
    request_id: str
    adapter_id: str
    channel_id: str
    observable: str
    unit: str
    target: float
    tolerance: float
    magnitude: float
    idempotency_key: str
    inventory_sha256: str
    deadline_ns: int
    safe_envelope: NumericDomain
    parameters: Mapping[str, Any] = field(default_factory=dict)
    preconditions: tuple[str, ...] = ()
    expected_effects: tuple[str, ...] = ()
    abort_predicates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "command_id",
            "request_id",
            "adapter_id",
            "channel_id",
            "observable",
            "unit",
            "idempotency_key",
        ):
            _identifier(getattr(self, name), name=name)
        _digest(self.inventory_sha256, name="inventory_sha256")
        _positive_int(self.deadline_ns, name="deadline_ns")
        if not isinstance(self.safe_envelope, NumericDomain):
            raise TypeError("safe_envelope must be a NumericDomain")
        object.__setattr__(self, "target", _finite(self.target, name="target", minimum=-math.inf))
        object.__setattr__(self, "tolerance", _finite(self.tolerance, name="tolerance"))
        object.__setattr__(self, "magnitude", _finite(self.magnitude, name="magnitude", minimum=-math.inf))
        if not self.safe_envelope.contains(self.magnitude):
            raise ValueError("magnitude lies outside the command safe envelope")
        object.__setattr__(
            self,
            "parameters",
            _frozen_json_mapping(self.parameters, name="parameters"),
        )
        _identifiers(self.preconditions, name="preconditions")
        _identifiers(self.expected_effects, name="expected_effects")
        _identifiers(self.abort_predicates, name="abort_predicates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "request_id": self.request_id,
            "adapter_id": self.adapter_id,
            "channel_id": self.channel_id,
            "observable": self.observable,
            "unit": self.unit,
            "target": self.target,
            "tolerance": self.tolerance,
            "magnitude": self.magnitude,
            "idempotency_key": self.idempotency_key,
            "inventory_sha256": self.inventory_sha256,
            "deadline_ns": self.deadline_ns,
            "safe_envelope": self.safe_envelope.to_dict(),
            "parameters": dict(self.parameters),
            "preconditions": list(self.preconditions),
            "expected_effects": list(self.expected_effects),
            "abort_predicates": list(self.abort_predicates),
        }

    @property
    def sha256(self) -> str:
        return str(sha256_hex(canonical_json(self.to_dict())))


@dataclass(frozen=True, slots=True)
class ActuationLease:
    lease_id: str
    command_sha256: str
    adapter_id: str
    session_id: str
    authority_receipt_id: str
    issued_at_ns: int
    expires_at_ns: int
    issued_monotonic_ns: int
    expires_monotonic_ns: int

    def __post_init__(self) -> None:
        _identifier(self.lease_id, name="lease_id")
        _digest(self.command_sha256, name="command_sha256")
        _identifier(self.adapter_id, name="adapter_id")
        _identifier(self.session_id, name="session_id")
        _identifier(self.authority_receipt_id, name="authority_receipt_id")
        for name in (
            "issued_at_ns",
            "expires_at_ns",
            "issued_monotonic_ns",
            "expires_monotonic_ns",
        ):
            _positive_int(getattr(self, name), name=name)
        if self.expires_at_ns <= self.issued_at_ns:
            raise ValueError("lease wall expiry must follow issuance")
        if self.expires_monotonic_ns <= self.issued_monotonic_ns:
            raise ValueError("lease monotonic expiry must follow issuance")

    def is_valid(self, *, now_ns: int, monotonic_now_ns: int, session_id: str) -> bool:
        return (
            session_id == self.session_id
            and self.issued_at_ns <= now_ns < self.expires_at_ns
            and self.issued_monotonic_ns <= monotonic_now_ns < self.expires_monotonic_ns
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "command_sha256": self.command_sha256,
            "adapter_id": self.adapter_id,
            "session_id": self.session_id,
            "authority_receipt_id": self.authority_receipt_id,
            "issued_at_ns": self.issued_at_ns,
            "expires_at_ns": self.expires_at_ns,
            "issued_monotonic_ns": self.issued_monotonic_ns,
            "expires_monotonic_ns": self.expires_monotonic_ns,
        }

    @property
    def sha256(self) -> str:
        return str(sha256_hex(canonical_json(self.to_dict())))


@dataclass(frozen=True, slots=True)
class PreparedActuation:
    preparation_id: str
    command_sha256: str
    lease_sha256: str
    adapter_id: str
    capability_sha256: str
    precondition_sha256: str
    rollback_token_sha256: str
    prepared_at_ns: int

    def __post_init__(self) -> None:
        _identifier(self.preparation_id, name="preparation_id")
        _identifier(self.adapter_id, name="adapter_id")
        for name in (
            "command_sha256",
            "lease_sha256",
            "capability_sha256",
            "precondition_sha256",
            "rollback_token_sha256",
        ):
            _digest(getattr(self, name), name=name)
        _positive_int(self.prepared_at_ns, name="prepared_at_ns")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def sha256(self) -> str:
        return str(sha256_hex(canonical_json(self.to_dict())))


@dataclass(frozen=True, slots=True)
class ActuationReceipt:
    receipt_id: str
    command_sha256: str
    preparation_sha256: str
    adapter_id: str
    state: ActuationState
    accepted: bool
    transport_completed: bool
    executed: bool
    recorded_at_ns: int
    detail_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.receipt_id, name="receipt_id")
        _identifier(self.adapter_id, name="adapter_id")
        for name in ("command_sha256", "preparation_sha256", "detail_sha256"):
            _digest(getattr(self, name), name=name)
        if not isinstance(self.state, ActuationState):
            raise TypeError("state must be an ActuationState")
        for name in ("accepted", "transport_completed", "executed"):
            _boolean(getattr(self, name), name=name)
        _positive_int(self.recorded_at_ns, name="recorded_at_ns")
        if self.executed and not self.transport_completed:
            raise ValueError("execution requires completed transport")
        if self.transport_completed and not self.accepted:
            raise ValueError("transport completion requires command acceptance")
        if self.state == ActuationState.EXECUTED and not self.executed:
            raise ValueError("executed state requires executed=True")
        if self.executed and self.state != ActuationState.EXECUTED:
            raise ValueError("executed=True requires the executed state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "command_sha256": self.command_sha256,
            "preparation_sha256": self.preparation_sha256,
            "adapter_id": self.adapter_id,
            "state": self.state.value,
            "accepted": self.accepted,
            "transport_completed": self.transport_completed,
            "executed": self.executed,
            "recorded_at_ns": self.recorded_at_ns,
            "detail_sha256": self.detail_sha256,
        }

    @property
    def sha256(self) -> str:
        return str(sha256_hex(canonical_json(self.to_dict())))


@dataclass(frozen=True, slots=True)
class EffectReceipt:
    receipt_id: str
    command_sha256: str
    actuation_receipt_sha256: str
    observation_channel_id: str
    observation_sha256: str
    state: ActuationState
    target_error: float | None
    independently_observed: bool
    recorded_at_ns: int

    def __post_init__(self) -> None:
        _identifier(self.receipt_id, name="receipt_id")
        _identifier(self.observation_channel_id, name="observation_channel_id")
        _digest(self.command_sha256, name="command_sha256")
        _digest(self.actuation_receipt_sha256, name="actuation_receipt_sha256")
        _digest(self.observation_sha256, name="observation_sha256")
        if not isinstance(self.state, ActuationState):
            raise TypeError("state must be an ActuationState")
        _boolean(self.independently_observed, name="independently_observed")
        if self.target_error is not None:
            object.__setattr__(
                self,
                "target_error",
                _finite(self.target_error, name="target_error"),
            )
        _positive_int(self.recorded_at_ns, name="recorded_at_ns")
        if self.state == ActuationState.EFFECT_VERIFIED and not self.independently_observed:
            raise ValueError("effect verification requires independent observation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "command_sha256": self.command_sha256,
            "actuation_receipt_sha256": self.actuation_receipt_sha256,
            "observation_channel_id": self.observation_channel_id,
            "observation_sha256": self.observation_sha256,
            "state": self.state.value,
            "target_error": self.target_error,
            "independently_observed": self.independently_observed,
            "recorded_at_ns": self.recorded_at_ns,
        }

    @property
    def sha256(self) -> str:
        return str(sha256_hex(canonical_json(self.to_dict())))


@dataclass(frozen=True, slots=True)
class RollbackReceipt:
    receipt_id: str
    command_sha256: str
    actuation_receipt_sha256: str
    adapter_id: str
    state: ActuationState
    safe_state_observation_sha256: str
    independently_observed: bool
    recorded_at_ns: int

    def __post_init__(self) -> None:
        _identifier(self.receipt_id, name="receipt_id")
        _identifier(self.adapter_id, name="adapter_id")
        for name in (
            "command_sha256",
            "actuation_receipt_sha256",
            "safe_state_observation_sha256",
        ):
            _digest(getattr(self, name), name=name)
        if self.state not in {
            ActuationState.COMPENSATED,
            ActuationState.ROLLED_BACK,
            ActuationState.SAFE_STATE,
            ActuationState.INDETERMINATE,
            ActuationState.FAILED,
        }:
            raise ValueError("rollback receipt state is invalid")
        _boolean(self.independently_observed, name="independently_observed")
        _positive_int(self.recorded_at_ns, name="recorded_at_ns")
        if self.state in {ActuationState.ROLLED_BACK, ActuationState.SAFE_STATE} and not self.independently_observed:
            raise ValueError("safe or rolled-back state requires independent observation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "command_sha256": self.command_sha256,
            "actuation_receipt_sha256": self.actuation_receipt_sha256,
            "adapter_id": self.adapter_id,
            "state": self.state.value,
            "safe_state_observation_sha256": self.safe_state_observation_sha256,
            "independently_observed": self.independently_observed,
            "recorded_at_ns": self.recorded_at_ns,
        }

    @property
    def sha256(self) -> str:
        return str(sha256_hex(canonical_json(self.to_dict())))


@runtime_checkable
class RealityAdapter(Protocol):
    @property
    def adapter_id(self) -> str: ...

    def declarations(self) -> tuple[ChannelDeclaration, ...]: ...

    def actuator_capabilities(self) -> tuple[ActuatorCapability, ...]: ...

    def read(self) -> tuple[Any, ...]: ...

    async def prepare(
        self,
        command: ActuationCommand,
        lease: ActuationLease,
    ) -> PreparedActuation: ...

    async def actuate(
        self,
        command: ActuationCommand,
        lease: ActuationLease,
        prepared: PreparedActuation,
    ) -> ActuationReceipt: ...

    async def verify_effect(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt,
    ) -> EffectReceipt: ...

    async def cancel(
        self,
        command: ActuationCommand,
        prepared: PreparedActuation | None,
    ) -> ActuationReceipt: ...

    async def safe_state(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt | None,
    ) -> RollbackReceipt: ...

    async def rollback(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt,
    ) -> RollbackReceipt: ...


__all__ = [
    "ActuationCommand",
    "ActuationLease",
    "ActuationReceipt",
    "ActuationState",
    "ActuatorCapability",
    "EffectReceipt",
    "PreparedActuation",
    "RealityAdapter",
    "Reversibility",
    "RollbackReceipt",
]
