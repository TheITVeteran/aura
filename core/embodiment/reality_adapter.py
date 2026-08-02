"""Explicit physical-device contracts for the Reality Reach effect boundary."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from core.embodiment.base_device import BaseHardwareDevice
from core.reality_reach.actuation import (
    ActuationCommand,
    ActuationLease,
    ActuationReceipt,
    ActuationState,
    ActuatorCapability,
    EffectReceipt,
    PreparedActuation,
    Reversibility,
    RollbackReceipt,
)
from core.reality_reach.contracts import ChannelDeclaration, ChannelKind, NumericDomain
from core.reality_reach.live import ChannelReading, ReadingStatus
from core.runtime.audit_chain import canonical_json, sha256_hex

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_MAX_DEVICE_PARAMETERS_BYTES = 64 * 1024


class HardwareRealityError(RuntimeError):
    """Stable refusal raised by a physical-device Reality Reach adapter."""


def _identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a canonical identifier")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    number = float(cast(Any, value))
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _frozen_parameters(value: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        raw = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("device parameters must contain canonical JSON values") from exc
    if len(raw) > _MAX_DEVICE_PARAMETERS_BYTES:
        raise ValueError("device parameters exceed the bounded command envelope")
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("device parameters must be a mapping")
    return MappingProxyType(decoded)


@dataclass(frozen=True, slots=True)
class HardwareCommandContract:
    """One manifest-authorized hardware command and its scalar effect binding."""

    command: str
    safe_envelope: NumericDomain
    target: float | None = None
    target_parameter: str = ""
    magnitude: float | None = None
    magnitude_parameter: str = ""
    tolerance: float = 0.0
    allowed_parameters: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    expected_effects: tuple[str, ...] = ()
    abort_predicates: tuple[str, ...] = ()
    rollback_command: str = ""

    def __post_init__(self) -> None:
        _identifier(self.command, name="command")
        if not isinstance(self.safe_envelope, NumericDomain):
            raise TypeError("safe_envelope must be a NumericDomain")
        if (self.target is None) == (not self.target_parameter):
            raise ValueError("exactly one fixed or parameter-bound target is required")
        if (self.magnitude is None) == (not self.magnitude_parameter):
            raise ValueError("exactly one fixed or parameter-bound magnitude is required")
        if self.target is not None:
            object.__setattr__(self, "target", _finite(self.target, name="target"))
        if self.magnitude is not None:
            magnitude = _finite(self.magnitude, name="magnitude")
            if not self.safe_envelope.contains(magnitude):
                raise ValueError("fixed magnitude lies outside the safe envelope")
            object.__setattr__(self, "magnitude", magnitude)
        for name in ("target_parameter", "magnitude_parameter", "rollback_command"):
            value = getattr(self, name)
            if value:
                _identifier(value, name=name)
        for name in (
            "allowed_parameters",
            "preconditions",
            "expected_effects",
            "abort_predicates",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != len(set(values)):
                raise ValueError(f"{name} must be a duplicate-free tuple")
            for value in values:
                _identifier(value, name=name)
        required_parameters = {
            value
            for value in (self.target_parameter, self.magnitude_parameter)
            if value
        }
        if not required_parameters.issubset(set(self.allowed_parameters)):
            raise ValueError("numeric binding parameters must be explicitly allowed")
        tolerance = _finite(self.tolerance, name="tolerance")
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "safe_envelope": self.safe_envelope.to_dict(),
            "target": self.target,
            "target_parameter": self.target_parameter,
            "magnitude": self.magnitude,
            "magnitude_parameter": self.magnitude_parameter,
            "tolerance": self.tolerance,
            "allowed_parameters": list(self.allowed_parameters),
            "preconditions": list(self.preconditions),
            "expected_effects": list(self.expected_effects),
            "abort_predicates": list(self.abort_predicates),
            "rollback_command": self.rollback_command,
        }

    def resolve(self, parameters: Mapping[str, Any]) -> tuple[float, float, Mapping[str, Any]]:
        frozen = _frozen_parameters(parameters)
        unknown = set(frozen) - set(self.allowed_parameters)
        if unknown:
            raise HardwareRealityError(
                f"hardware_command_parameters_undeclared:{','.join(sorted(unknown))}"
            )
        missing = {
            value
            for value in (self.target_parameter, self.magnitude_parameter)
            if value and value not in frozen
        }
        if missing:
            raise HardwareRealityError(
                f"hardware_command_parameters_missing:{','.join(sorted(missing))}"
            )
        if self.target_parameter:
            target = _finite(frozen[self.target_parameter], name="target_parameter")
        elif self.target is not None:
            target = float(self.target)
        else:  # Constructor validation makes this unreachable.
            raise HardwareRealityError("hardware_command_target_contract_invalid")
        if self.magnitude_parameter:
            magnitude = _finite(
                frozen[self.magnitude_parameter],
                name="magnitude_parameter",
            )
        elif self.magnitude is not None:
            magnitude = float(self.magnitude)
        else:  # Constructor validation makes this unreachable.
            raise HardwareRealityError("hardware_command_magnitude_contract_invalid")
        if not self.safe_envelope.contains(magnitude):
            raise HardwareRealityError("hardware_command_magnitude_outside_safe_envelope")
        return target, magnitude, frozen


@dataclass(frozen=True, slots=True)
class HardwareRealityManifest:
    """Complete opt-in contract required before a hardware device can execute."""

    adapter_id: str
    actuator: ChannelDeclaration
    observation: ChannelDeclaration
    capability: ActuatorCapability
    observation_field: str
    command_transport_id: str
    readback_transport_id: str
    commands: tuple[HardwareCommandContract, ...]
    safe_state_command: str
    safe_state_target: float

    def __post_init__(self) -> None:
        _identifier(self.adapter_id, name="adapter_id")
        _identifier(self.observation_field, name="observation_field")
        _identifier(self.command_transport_id, name="command_transport_id")
        _identifier(self.readback_transport_id, name="readback_transport_id")
        _identifier(self.safe_state_command, name="safe_state_command")
        if self.command_transport_id == self.readback_transport_id:
            raise ValueError("effect readback must use a transport-distinct route")
        if self.actuator.kind != ChannelKind.ACTUATOR:
            raise ValueError("actuator declaration must have actuator kind")
        if self.observation.kind != ChannelKind.SENSOR:
            raise ValueError("observation declaration must have sensor kind")
        if (
            self.capability.adapter_id != self.adapter_id
            or self.capability.channel_id != self.actuator.channel_id
            or self.capability.observation_channels != (self.observation.channel_id,)
        ):
            raise ValueError("manifest capability identity does not match declarations")
        if (
            self.actuator.observable != self.observation.observable
            or self.actuator.unit != self.observation.unit
        ):
            raise ValueError("actuator and observation must share an observable and unit")
        if not self.commands:
            raise ValueError("hardware manifest must declare at least one command")
        command_names = tuple(item.command for item in self.commands)
        if len(command_names) != len(set(command_names)):
            raise ValueError("hardware manifest commands must be unique")
        command_by_name = {item.command: item for item in self.commands}
        safe_contract = command_by_name.get(self.safe_state_command)
        if safe_contract is None:
            raise ValueError("safe_state_command must name a declared command")
        if safe_contract.target_parameter or safe_contract.magnitude_parameter:
            raise ValueError("safe-state command must have fixed target and magnitude")
        safe_target = _finite(self.safe_state_target, name="safe_state_target")
        if safe_contract.target is None or float(safe_contract.target) != safe_target:
            raise ValueError("safe-state target differs from its command contract")
        object.__setattr__(self, "safe_state_target", safe_target)
        for contract in self.commands:
            if not self.capability.magnitude_domain.contains(
                contract.safe_envelope.minimum
            ) or not self.capability.magnitude_domain.contains(
                contract.safe_envelope.maximum
            ):
                raise ValueError("command safe envelope exceeds actuator capability")
            if contract.target is not None and not self.observation.domain.contains(
                contract.target,
                tolerance=contract.tolerance,
            ):
                raise ValueError("fixed command target exceeds observation domain")
            if contract.rollback_command:
                rollback = command_by_name.get(contract.rollback_command)
                if rollback is None:
                    raise ValueError("rollback command must name a declared command")
                if rollback.target_parameter or rollback.magnitude_parameter:
                    raise ValueError("rollback command must have fixed target and magnitude")
        if self.capability.reversibility == Reversibility.REVERSIBLE and any(
            not item.rollback_command for item in self.commands
        ):
            raise ValueError("every reversible hardware command requires rollback")

    def command(self, name: str) -> HardwareCommandContract:
        for contract in self.commands:
            if contract.command == name:
                return contract
        raise HardwareRealityError(f"hardware_command_not_declared:{name}")


class HardwareRealityAdapter:
    """RealityAdapter that delegates only manifest-authorized effects to a device."""

    def __init__(self, device: BaseHardwareDevice, manifest: HardwareRealityManifest) -> None:
        if not isinstance(device, BaseHardwareDevice):
            raise TypeError("device must be a BaseHardwareDevice")
        if not isinstance(manifest, HardwareRealityManifest):
            raise TypeError("manifest must be a HardwareRealityManifest")
        self._device = device
        self._manifest = manifest
        self._last_observation = ChannelReading(
            channel_id=manifest.observation.channel_id,
            value=None,
            unit=manifest.observation.unit,
            captured_at_ns=max(1, time.time_ns()),
            status=ReadingStatus.UNAVAILABLE,
            source=f"hardware.{device.device_id}.readback",
            error="device_readback_not_primed",
        )
        self._interlock_by_command: dict[str, str] = {}

    @property
    def adapter_id(self) -> str:
        return self._manifest.adapter_id

    @property
    def device_id(self) -> str:
        return str(self._device.device_id)

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        return (self._manifest.actuator, self._manifest.observation)

    def actuator_capabilities(self) -> tuple[ActuatorCapability, ...]:
        return (self._manifest.capability,)

    def read(self) -> tuple[ChannelReading, ...]:
        observation = self._last_observation
        if not self._device.is_connected:
            observation = ChannelReading(
                channel_id=self._manifest.observation.channel_id,
                value=None,
                unit=self._manifest.observation.unit,
                captured_at_ns=max(1, time.time_ns()),
                status=ReadingStatus.UNAVAILABLE,
                source=self._manifest.readback_transport_id,
                error="device_not_connected",
            )
        actuator = ChannelReading(
            channel_id=self._manifest.actuator.channel_id,
            value=None,
            unit=self._manifest.actuator.unit,
            captured_at_ns=max(1, time.time_ns()),
            status=ReadingStatus.UNAVAILABLE,
            source=f"hardware.{self.device_id}.actuator",
            error="actuator_channels_do_not_self-report_effects",
        )
        return (actuator, observation)

    async def refresh_readback(self) -> ChannelReading:
        captured_at_ns = max(1, time.time_ns())
        try:
            status = await self._device.get_status()
            if not isinstance(status, Mapping):
                raise HardwareRealityError("hardware_status_not_mapping")
            if status.get("ok") is not True:
                raise HardwareRealityError(
                    str(status.get("error") or "hardware_status_not_available")
                )
            value = _finite(
                status.get(self._manifest.observation_field),
                name="hardware_observation",
            )
            if not self._manifest.observation.domain.contains(value):
                raise HardwareRealityError("hardware_observation_outside_declared_domain")
            reading = ChannelReading(
                channel_id=self._manifest.observation.channel_id,
                value=value,
                unit=self._manifest.observation.unit,
                captured_at_ns=captured_at_ns,
                status=ReadingStatus.AVAILABLE,
                source=self._manifest.readback_transport_id,
                uncertainty=self._manifest.observation.resolution,
            )
        except (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
            reading = ChannelReading(
                channel_id=self._manifest.observation.channel_id,
                value=None,
                unit=self._manifest.observation.unit,
                captured_at_ns=captured_at_ns,
                status=ReadingStatus.DEGRADED,
                source=self._manifest.readback_transport_id,
                error=f"{type(exc).__name__}:{exc}"[:300],
            )
        self._last_observation = reading
        return reading

    async def compile_command(
        self,
        command_name: str,
        parameters: Mapping[str, Any],
        *,
        inventory_sha256: str,
        deadline_s: float,
        idempotency_key: str,
        source: str,
    ) -> ActuationCommand:
        contract = self._manifest.command(command_name)
        target, magnitude, frozen = contract.resolve(parameters)
        if not self._manifest.observation.domain.contains(
            target,
            tolerance=contract.tolerance,
        ):
            raise HardwareRealityError("hardware_command_target_outside_observation_domain")
        current = await self.refresh_readback()
        if current.status != ReadingStatus.AVAILABLE:
            raise HardwareRealityError("hardware_command_readback_unavailable")
        identifier = str(idempotency_key or uuid.uuid4().hex)
        if not _IDENTIFIER.fullmatch(identifier):
            identifier = f"hardware.idem.{_digest(identifier).removeprefix('sha256:')[:32]}"
        deadline_ns = time.time_ns() + max(1, int(float(deadline_s) * 1_000_000_000))
        rollback_target: float | None = None
        if contract.rollback_command:
            rollback_target = self._manifest.command(contract.rollback_command).target
        return ActuationCommand(
            command_id=f"hardware.command.{uuid.uuid4().hex}",
            request_id=f"hardware.request.{uuid.uuid4().hex}",
            adapter_id=self.adapter_id,
            channel_id=self._manifest.actuator.channel_id,
            observable=self._manifest.actuator.observable,
            unit=self._manifest.actuator.unit,
            target=target,
            tolerance=contract.tolerance,
            magnitude=magnitude,
            idempotency_key=identifier,
            inventory_sha256=inventory_sha256,
            deadline_ns=deadline_ns,
            safe_envelope=contract.safe_envelope,
            parameters={
                "device_id": self.device_id,
                "device_command": contract.command,
                "device_parameters": dict(frozen),
                "command_contract_sha256": contract.sha256,
                "rollback_command": contract.rollback_command,
                "rollback_target": rollback_target,
                "source": str(source or "robotics_actuator")[:128],
            },
            preconditions=("device_connected", "device_interlocks_clear", *contract.preconditions),
            expected_effects=contract.expected_effects,
            abort_predicates=("device_interlock_changed", *contract.abort_predicates),
        )

    def _contract_for_command(self, command: ActuationCommand) -> HardwareCommandContract:
        if (
            command.adapter_id != self.adapter_id
            or command.channel_id != self._manifest.actuator.channel_id
            or command.parameters.get("device_id") != self.device_id
        ):
            raise HardwareRealityError("hardware_command_identity_mismatch")
        contract = self._manifest.command(str(command.parameters.get("device_command") or ""))
        if command.parameters.get("command_contract_sha256") != contract.sha256:
            raise HardwareRealityError("hardware_command_contract_drift")
        target, magnitude, _ = contract.resolve(
            command.parameters.get("device_parameters")
            if isinstance(command.parameters.get("device_parameters"), Mapping)
            else {}
        )
        if target != command.target or magnitude != command.magnitude:
            raise HardwareRealityError("hardware_command_compilation_mismatch")
        return contract

    async def prepare(
        self,
        command: ActuationCommand,
        lease: ActuationLease,
    ) -> PreparedActuation:
        contract = self._contract_for_command(command)
        if lease.command_sha256 != command.sha256 or lease.adapter_id != self.adapter_id:
            raise HardwareRealityError("hardware_actuation_lease_identity_mismatch")
        if not lease.is_valid(
            now_ns=time.time_ns(),
            monotonic_now_ns=time.monotonic_ns(),
            session_id=lease.session_id,
        ):
            raise HardwareRealityError("hardware_actuation_lease_expired")
        status = await self._device.get_status()
        if not isinstance(status, Mapping) or status.get("ok") is not True:
            raise HardwareRealityError("hardware_precondition_readback_unavailable")
        parameters = command.parameters.get("device_parameters")
        if not isinstance(parameters, Mapping):
            raise HardwareRealityError("hardware_command_parameters_invalid")
        interlock = await self._device.check_interlocks(contract.command, parameters, status)
        if not isinstance(interlock, Mapping) or interlock.get("ok") is not True:
            reason = "device_interlock_refused"
            if isinstance(interlock, Mapping):
                reason = str(interlock.get("reason") or reason)[:200]
            raise HardwareRealityError(reason)
        interlock_sha256 = str(interlock.get("interlock_sha256") or _digest(interlock))
        self._interlock_by_command[command.sha256] = interlock_sha256
        return PreparedActuation(
            preparation_id=f"hardware.prepare.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            lease_sha256=lease.sha256,
            adapter_id=self.adapter_id,
            capability_sha256=self._manifest.capability.sha256,
            precondition_sha256=interlock_sha256,
            rollback_token_sha256=_digest(
                {
                    "rollback_command": command.parameters.get("rollback_command"),
                    "rollback_target": command.parameters.get("rollback_target"),
                }
            ),
            prepared_at_ns=time.time_ns(),
        )

    async def actuate(
        self,
        command: ActuationCommand,
        lease: ActuationLease,
        prepared: PreparedActuation,
    ) -> ActuationReceipt:
        contract = self._contract_for_command(command)
        if prepared.command_sha256 != command.sha256 or prepared.lease_sha256 != lease.sha256:
            raise HardwareRealityError("hardware_preparation_identity_mismatch")
        if not lease.is_valid(
            now_ns=time.time_ns(),
            monotonic_now_ns=time.monotonic_ns(),
            session_id=lease.session_id,
        ):
            raise HardwareRealityError("hardware_actuation_lease_expired_before_transport")
        parameters = command.parameters.get("device_parameters")
        if not isinstance(parameters, Mapping):
            raise HardwareRealityError("hardware_command_parameters_invalid")
        result = await self._device.safe_execute(
            contract.command,
            timeout_s=self._manifest.capability.watchdog_timeout_s,
            expected_interlock_sha256=self._interlock_by_command.get(command.sha256, ""),
            **dict(parameters),
        )
        if not isinstance(result, Mapping):
            raise HardwareRealityError("hardware_transport_result_not_mapping")
        executed = result.get("ok") is True
        transport_completed = executed or result.get("transport_completed") is True
        return ActuationReceipt(
            receipt_id=f"hardware.actuate.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            preparation_sha256=prepared.sha256,
            adapter_id=self.adapter_id,
            state=ActuationState.EXECUTED if executed else ActuationState.FAILED,
            accepted=True,
            transport_completed=transport_completed,
            executed=executed,
            recorded_at_ns=time.time_ns(),
            detail_sha256=_digest(dict(result)),
        )

    async def verify_effect(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt,
    ) -> EffectReceipt:
        self._contract_for_command(command)
        reading = await self.refresh_readback()
        target_error = (
            abs(float(reading.value) - command.target)
            if reading.value is not None
            else None
        )
        independently_observed = (
            reading.status == ReadingStatus.AVAILABLE
            and self._manifest.command_transport_id != self._manifest.readback_transport_id
        )
        verified = (
            actuation.executed
            and independently_observed
            and target_error is not None
            and target_error <= command.tolerance
        )
        return EffectReceipt(
            receipt_id=f"hardware.effect.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            actuation_receipt_sha256=actuation.sha256,
            observation_channel_id=self._manifest.observation.channel_id,
            observation_sha256=reading.sha256,
            state=ActuationState.EFFECT_VERIFIED if verified else ActuationState.FAILED,
            target_error=target_error,
            independently_observed=verified,
            recorded_at_ns=time.time_ns(),
        )

    async def cancel(
        self,
        command: ActuationCommand,
        prepared: PreparedActuation | None,
    ) -> ActuationReceipt:
        self._contract_for_command(command)
        return ActuationReceipt(
            receipt_id=f"hardware.cancel.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            preparation_sha256=prepared.sha256 if prepared is not None else command.sha256,
            adapter_id=self.adapter_id,
            state=ActuationState.CANCELLED,
            accepted=False,
            transport_completed=False,
            executed=False,
            recorded_at_ns=time.time_ns(),
            detail_sha256=_digest({"cancelled_before_execution": True}),
        )

    async def safe_state(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt | None,
    ) -> RollbackReceipt:
        return await self._recover(
            command,
            actuation,
            recovery_command=self._manifest.safe_state_command,
            target=self._manifest.safe_state_target,
            success_state=ActuationState.SAFE_STATE,
        )

    async def rollback(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt,
    ) -> RollbackReceipt:
        contract = self._contract_for_command(command)
        rollback_command = contract.rollback_command or self._manifest.safe_state_command
        target_value = command.parameters.get("rollback_target")
        target = (
            _finite(target_value, name="rollback_target")
            if target_value is not None
            else self._manifest.safe_state_target
        )
        return await self._recover(
            command,
            actuation,
            recovery_command=rollback_command,
            target=target,
            success_state=ActuationState.ROLLED_BACK,
        )

    async def _recover(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt | None,
        *,
        recovery_command: str,
        target: float,
        success_state: ActuationState,
    ) -> RollbackReceipt:
        recovery = self._manifest.command(recovery_command)
        _target, _magnitude, parameters = recovery.resolve({})
        result = await self._device.safe_execute(
            recovery.command,
            timeout_s=self._manifest.capability.watchdog_timeout_s,
            **dict(parameters),
        )
        reading = await self.refresh_readback()
        observed = (
            isinstance(result, Mapping)
            and result.get("ok") is True
            and reading.status == ReadingStatus.AVAILABLE
            and reading.value is not None
            and abs(float(reading.value) - target) <= recovery.tolerance
        )
        return RollbackReceipt(
            receipt_id=(
                f"hardware.{success_state.value}."
                f"{command.sha256.removeprefix('sha256:')[:32]}"
            ),
            command_sha256=command.sha256,
            actuation_receipt_sha256=(
                actuation.sha256 if actuation is not None else command.sha256
            ),
            adapter_id=self.adapter_id,
            state=success_state if observed else ActuationState.INDETERMINATE,
            safe_state_observation_sha256=reading.sha256,
            independently_observed=observed,
            recorded_at_ns=time.time_ns(),
        )


__all__ = [
    "HardwareCommandContract",
    "HardwareRealityAdapter",
    "HardwareRealityError",
    "HardwareRealityManifest",
]
