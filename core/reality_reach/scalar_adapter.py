"""Complete bidirectional Reality Reach adapter for scalar protocol resources.

Protocol connectors supply transport-specific discovery, read, and write
operations.  This module owns the shared physical contract: declarations,
bounded command compilation, precondition fencing, rate limits, fresh
readback, effect receipts, cancellation, safe state, and rollback.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

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
from core.reality_reach.contracts import (
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
    NumericDomain,
    RealityLayer,
)
from core.reality_reach.live import ChannelReading, ReadingStatus
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.lockdep import checked_async_lock

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class ScalarAdapterError(RuntimeError):
    """A scalar resource violated its manifest or transaction contract."""


class ScalarTransportClass(StrEnum):
    """Whether adapter writes target a simulated or physical transport."""

    SIMULATED = "simulated"
    PHYSICAL = "physical"


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{name} must be a canonical identifier")
    return normalized


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class ScalarSample:
    value: float
    captured_at_ns: int
    source_event_id: str
    quality: str = "good"
    uncertainty: float | None = None
    wall_clock_source: str = "system.time_ns"
    source_epoch: str = ""
    source_sequence: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite(self.value, name="value"))
        if isinstance(self.captured_at_ns, bool) or self.captured_at_ns <= 0:
            raise ValueError("captured_at_ns must be positive")
        if not _DIGEST.fullmatch(str(self.source_event_id or "")):
            raise ValueError("source_event_id must be a sha256 digest")
        quality = _identifier(self.quality, name="quality")
        object.__setattr__(self, "quality", quality)
        if not isinstance(self.wall_clock_source, str) or not self.wall_clock_source:
            raise ValueError("wall_clock_source must be non-empty")
        if not isinstance(self.source_epoch, str) or len(self.source_epoch) > 256:
            raise ValueError("source_epoch must be a bounded string")
        if (
            isinstance(self.source_sequence, bool)
            or not isinstance(self.source_sequence, int)
            or self.source_sequence < 0
        ):
            raise ValueError("source_sequence must be a non-negative integer")
        if self.source_sequence and not self.source_epoch:
            raise ValueError("source_sequence requires source_epoch")
        if self.uncertainty is not None:
            uncertainty = _finite(self.uncertainty, name="uncertainty")
            if uncertainty < 0.0:
                raise ValueError("uncertainty must be non-negative")
            object.__setattr__(self, "uncertainty", uncertainty)


@dataclass(frozen=True, slots=True)
class ScalarWriteResult:
    accepted: bool
    transport_completed: bool
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool) or not isinstance(self.transport_completed, bool):
            raise TypeError("write result booleans must be explicit")
        if self.accepted and not self.transport_completed:
            raise ValueError("accepted write requires completed transport")
        canonical_json(dict(self.receipt))


@runtime_checkable
class ScalarProtocolTransport(Protocol):
    @property
    def transport_id(self) -> str: ...

    async def read_scalar(self, resource_id: str) -> ScalarSample: ...

    async def write_scalar(
        self,
        resource_id: str,
        value: float,
        *,
        idempotency_key: str,
        recovery: bool = False,
    ) -> ScalarWriteResult: ...


@dataclass(frozen=True, slots=True)
class ScalarResourceProfile:
    resource_id: str
    observable: str
    unit: str
    domain: NumericDomain
    resolution: float
    writable: bool
    physical_identity_sha256: str
    owner: str
    protocol: str
    safe_value: float | None = None
    tolerance: float | None = None
    max_commands_per_minute: int = 12
    cooldown_s: float = 0.0
    stale_after_s: float = 30.0
    readback_distinct_from_command: bool = True

    def __post_init__(self) -> None:
        for name in ("resource_id", "observable", "unit", "owner", "protocol"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name=name))
        if not isinstance(self.domain, NumericDomain):
            raise TypeError("domain must be a NumericDomain")
        if not _DIGEST.fullmatch(str(self.physical_identity_sha256 or "")):
            raise ValueError("physical_identity_sha256 must be a sha256 digest")
        resolution = _finite(self.resolution, name="resolution")
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")
        object.__setattr__(self, "resolution", resolution)
        tolerance = self.tolerance if self.tolerance is not None else resolution
        tolerance = _finite(tolerance, name="tolerance")
        if tolerance < resolution:
            raise ValueError("tolerance must not be smaller than resolution")
        object.__setattr__(self, "tolerance", tolerance)
        if self.safe_value is not None:
            safe_value = _finite(self.safe_value, name="safe_value")
            if not self.domain.contains(safe_value):
                raise ValueError("safe_value lies outside the resource domain")
            object.__setattr__(self, "safe_value", safe_value)
        if not isinstance(self.writable, bool) or not isinstance(
            self.readback_distinct_from_command, bool
        ):
            raise TypeError("profile boolean claims must be explicit")
        if not 1 <= int(self.max_commands_per_minute) <= 600:
            raise ValueError("max_commands_per_minute must lie inside [1, 600]")
        cooldown = _finite(self.cooldown_s, name="cooldown_s")
        stale = _finite(self.stale_after_s, name="stale_after_s")
        if cooldown < 0.0 or not 0.1 <= stale <= 86_400.0:
            raise ValueError("profile timing bounds are invalid")
        object.__setattr__(self, "cooldown_s", cooldown)
        object.__setattr__(self, "stale_after_s", stale)

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "resource_id": self.resource_id,
                "observable": self.observable,
                "unit": self.unit,
                "domain": self.domain.to_dict(),
                "resolution": self.resolution,
                "writable": self.writable,
                "physical_identity_sha256": self.physical_identity_sha256,
                "owner": self.owner,
                "protocol": self.protocol,
                "safe_value": self.safe_value,
                "tolerance": self.tolerance,
                "max_commands_per_minute": self.max_commands_per_minute,
                "cooldown_s": self.cooldown_s,
                "stale_after_s": self.stale_after_s,
                "readback_distinct_from_command": (self.readback_distinct_from_command),
            }
        )


class ScalarRealityAdapter:
    """A manifest-bound scalar sensor or complete reversible actuator."""

    def __init__(
        self,
        transport: ScalarProtocolTransport,
        profile: ScalarResourceProfile,
        *,
        initial_sample: ScalarSample,
        transport_class: ScalarTransportClass = ScalarTransportClass.PHYSICAL,
    ) -> None:
        if not isinstance(transport, ScalarProtocolTransport):
            raise TypeError("transport must satisfy ScalarProtocolTransport")
        if not isinstance(profile, ScalarResourceProfile):
            raise TypeError("profile must be a ScalarResourceProfile")
        if not profile.domain.contains(initial_sample.value):
            raise ValueError("initial sample lies outside the resource domain")
        if not isinstance(transport_class, ScalarTransportClass):
            raise TypeError("transport_class must be a ScalarTransportClass")
        self._transport = transport
        self._profile = profile
        self._transport_class = transport_class
        prefix = f"{profile.protocol}.{profile.resource_id}"
        self._adapter_id = f"{prefix}.adapter"
        self._observation = ChannelDeclaration(
            channel_id=f"{prefix}.readback",
            kind=ChannelKind.SENSOR,
            observable=profile.observable,
            unit=profile.unit,
            domain=profile.domain,
            coupling=CouplingClass.NETWORK,
            reality_layers=(RealityLayer.EFFECTIVE,),
            evidence_level=EvidenceLevel.P1,
            owner=profile.owner,
            resolution=profile.resolution,
            sample_rate_hz=min(10.0, 1.0 / max(0.1, profile.cooldown_s or 1.0)),
            max_latency_s=min(profile.stale_after_s, 30.0),
            stale_after_s=profile.stale_after_s,
            reference_id=f"{prefix}.read_api",
            compliance_tags=(profile.protocol, "typed_scalar", "fresh_readback"),
            coupling_validated=True,
        )
        self._actuator = (
            ChannelDeclaration(
                channel_id=f"{prefix}.command",
                kind=ChannelKind.ACTUATOR,
                observable=profile.observable,
                unit=profile.unit,
                domain=profile.domain,
                coupling=CouplingClass.NETWORK,
                reality_layers=(RealityLayer.EFFECTIVE,),
                evidence_level=EvidenceLevel.P1,
                owner=profile.owner,
                stale_after_s=profile.stale_after_s,
                compliance_tags=(profile.protocol, "typed_scalar_actuation"),
                coupling_validated=True,
            )
            if profile.writable
            else None
        )
        self._capability = (
            ActuatorCapability(
                adapter_id=self._adapter_id,
                channel_id=self._actuator.channel_id,
                reversibility=Reversibility.REVERSIBLE,
                magnitude_domain=profile.domain,
                max_commands_per_minute=profile.max_commands_per_minute,
                observation_channels=(self._observation.channel_id,),
                required_permissions=("environment.physical", "network.local"),
                failure_modes=(
                    "transport_failure",
                    "readback_mismatch",
                    "resource_changed_before_dispatch",
                    "common_driver_false_confirmation",
                ),
                cooldown_s=profile.cooldown_s,
                watchdog_timeout_s=min(profile.stale_after_s, 30.0),
                exclusive=True,
                supports_cancel=False,
                supports_safe_state=profile.safe_value is not None,
                supports_rollback=True,
                compensation_action=f"restore_previous_{profile.protocol}_value",
            )
            if self._actuator is not None
            else None
        )
        self._last_observation = self._reading(initial_sample)
        self._prepared: dict[str, dict[str, Any]] = {}
        self._dispatch_times: deque[float] = deque()
        self._last_dispatch_at = 0.0
        self._lock = checked_async_lock(f"scalar_reality.{self._adapter_id}")

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def physical_identity_sha256(self) -> str:
        return self._profile.physical_identity_sha256

    @property
    def transport_class(self) -> ScalarTransportClass:
        return self._transport_class

    @property
    def profile_sha256(self) -> str:
        return self._profile.sha256

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        if self._actuator is None:
            return (self._observation,)
        return (self._actuator, self._observation)

    def actuator_capabilities(self) -> tuple[ActuatorCapability, ...]:
        return (self._capability,) if self._capability is not None else ()

    def read(self) -> tuple[ChannelReading, ...]:
        if self._actuator is None:
            return (self._last_observation,)
        return (
            ChannelReading(
                channel_id=self._actuator.channel_id,
                value=None,
                unit=self._actuator.unit,
                captured_at_ns=max(1, time.time_ns()),
                status=ReadingStatus.UNAVAILABLE,
                source=f"{self.adapter_id}.command",
                error="actuator_channel_has_no_observation_semantics",
            ),
            self._last_observation,
        )

    def _reading(self, sample: ScalarSample) -> ChannelReading:
        if not self._profile.domain.contains(sample.value):
            raise ScalarAdapterError("scalar_readback_outside_manifest_domain")
        now_ns = time.time_ns()
        if sample.captured_at_ns > now_ns + 300_000_000_000:
            raise ScalarAdapterError("scalar_readback_clock_is_too_far_in_future")
        age_ns = max(0, now_ns - sample.captured_at_ns)
        status = (
            ReadingStatus.STALE
            if age_ns > int(self._profile.stale_after_s * 1_000_000_000)
            else ReadingStatus.AVAILABLE
        )
        return ChannelReading(
            channel_id=self._observation.channel_id,
            value=sample.value,
            unit=self._observation.unit,
            captured_at_ns=sample.captured_at_ns,
            status=status,
            source=f"{self.adapter_id}.readback",
            uncertainty=(
                sample.uncertainty if sample.uncertainty is not None else self._profile.resolution
            ),
            wall_clock_source=sample.wall_clock_source,
            source_epoch=sample.source_epoch,
            source_sequence=sample.source_sequence,
            source_event_id=sample.source_event_id,
            source_quality=sample.quality,
        )

    async def refresh_readback(self) -> ChannelReading:
        sample = await self._transport.read_scalar(self._profile.resource_id)
        self._last_observation = self._reading(sample)
        return self._last_observation

    async def compile_target(
        self,
        target: float,
        *,
        inventory_sha256: str,
        deadline_s: float,
        idempotency_key: str,
        source: str,
    ) -> ActuationCommand:
        if self._actuator is None:
            raise ScalarAdapterError("scalar_resource_is_read_only")
        target_value = _finite(target, name="target")
        if not self._profile.domain.contains(target_value):
            raise ScalarAdapterError("scalar_target_outside_manifest_domain")
        if not _DIGEST.fullmatch(str(inventory_sha256 or "")):
            raise ValueError("inventory_sha256 must be a sha256 digest")
        key = str(idempotency_key or "").strip().lower()
        if not _IDENTIFIER.fullmatch(key):
            key = f"scalar.idem.{_digest(key).removeprefix('sha256:')[:32]}"
        bounded_deadline = max(0.1, min(float(deadline_s), 300.0))
        tolerance = self._profile.tolerance
        if tolerance is None:
            raise ScalarAdapterError("scalar_profile_tolerance_missing")
        return ActuationCommand(
            command_id=f"scalar.command.{uuid.uuid4().hex}",
            request_id=f"scalar.request.{uuid.uuid4().hex}",
            adapter_id=self.adapter_id,
            channel_id=self._actuator.channel_id,
            observable=self._actuator.observable,
            unit=self._actuator.unit,
            target=target_value,
            tolerance=float(tolerance),
            magnitude=target_value,
            idempotency_key=key,
            inventory_sha256=inventory_sha256,
            deadline_ns=time.time_ns() + int(bounded_deadline * 1_000_000_000),
            safe_envelope=self._profile.domain,
            parameters={
                "resource_id": self._profile.resource_id,
                "profile_sha256": self._profile.sha256,
                "protocol": self._profile.protocol,
                "source": str(source or "reality_reach")[:128],
            },
            preconditions=("fresh_readback", "resource_unchanged_before_dispatch"),
            expected_effects=("fresh_scalar_readback_matches",),
            abort_predicates=("resource_changed_before_dispatch",),
        )

    async def compile_command(
        self,
        target: float,
        *,
        inventory_sha256: str,
        deadline_s: float,
        idempotency_key: str,
        source: str,
    ) -> ActuationCommand:
        """Compatibility alias for callers predating TargetCommandCompiler."""

        return await self.compile_target(
            target,
            inventory_sha256=inventory_sha256,
            deadline_s=deadline_s,
            idempotency_key=idempotency_key,
            source=source,
        )

    def _validate_command(self, command: ActuationCommand) -> None:
        if self._actuator is None or (
            command.adapter_id != self.adapter_id
            or command.channel_id != self._actuator.channel_id
            or command.observable != self._profile.observable
            or command.unit != self._profile.unit
            or command.parameters.get("resource_id") != self._profile.resource_id
            or command.parameters.get("profile_sha256") != self._profile.sha256
            or command.parameters.get("protocol") != self._profile.protocol
            or not self._profile.domain.contains(command.target)
            or command.target != command.magnitude
        ):
            raise ScalarAdapterError("scalar_command_manifest_mismatch")

    @staticmethod
    def _lease_valid(command: ActuationCommand, lease: ActuationLease) -> bool:
        return bool(
            lease.command_sha256 == command.sha256
            and lease.adapter_id == command.adapter_id
            and lease.is_valid(
                now_ns=time.time_ns(),
                monotonic_now_ns=time.monotonic_ns(),
                session_id=lease.session_id,
            )
        )

    def _check_rate_limit(self, now: float) -> None:
        assert self._capability is not None
        while self._dispatch_times and now - self._dispatch_times[0] >= 60.0:
            self._dispatch_times.popleft()
        if len(self._dispatch_times) >= self._capability.max_commands_per_minute:
            raise ScalarAdapterError("scalar_command_rate_limited")
        if now - self._last_dispatch_at < self._capability.cooldown_s:
            raise ScalarAdapterError("scalar_command_cooldown_active")

    @staticmethod
    def _value_fence(reading: ChannelReading) -> str:
        return _digest({"channel_id": reading.channel_id, "value": reading.value})

    @staticmethod
    def _reading_is_newer(
        reading: ChannelReading,
        baseline: ChannelReading,
    ) -> bool:
        if (
            not reading.source_event_id
            or reading.source_event_id == baseline.source_event_id
            or reading.captured_at_ns <= baseline.captured_at_ns
        ):
            return False
        if (
            reading.source_epoch
            and reading.source_epoch == baseline.source_epoch
            and reading.source_sequence
            and baseline.source_sequence
            and reading.source_sequence <= baseline.source_sequence
        ):
            return False
        return True

    def _target_was_independently_effected(
        self,
        *,
        reading: ChannelReading,
        baseline: ChannelReading | None,
        target: float,
        tolerance: float,
    ) -> bool:
        return bool(
            baseline is not None
            and self._profile.readback_distinct_from_command
            and reading.status is ReadingStatus.AVAILABLE
            and reading.value is not None
            and baseline.value is not None
            and abs(float(baseline.value) - target) > tolerance
            and abs(float(reading.value) - target) <= tolerance
            and self._reading_is_newer(reading, baseline)
        )

    async def prepare(
        self,
        command: ActuationCommand,
        lease: ActuationLease,
    ) -> PreparedActuation:
        self._validate_command(command)
        if self._capability is None or not self._lease_valid(command, lease):
            raise ScalarAdapterError("scalar_actuation_lease_invalid")
        async with self._lock:
            self._check_rate_limit(time.monotonic())
            reading = await self.refresh_readback()
            if reading.status is not ReadingStatus.AVAILABLE or reading.value is None:
                raise ScalarAdapterError("scalar_prepare_readback_unavailable")
            precondition = self._value_fence(reading)
            rollback = float(reading.value)
            rollback_digest = _digest({"resource_id": self._profile.resource_id, "value": rollback})
            self._prepared[command.sha256] = {
                "precondition_sha256": precondition,
                "rollback_value": rollback,
                "rollback_sha256": rollback_digest,
            }
            while len(self._prepared) > 256:
                self._prepared.pop(next(iter(self._prepared)))
        return PreparedActuation(
            preparation_id=f"scalar.prepare.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            lease_sha256=lease.sha256,
            adapter_id=self.adapter_id,
            capability_sha256=self._capability.sha256,
            precondition_sha256=precondition,
            rollback_token_sha256=rollback_digest,
            prepared_at_ns=time.time_ns(),
        )

    async def actuate(
        self,
        command: ActuationCommand,
        lease: ActuationLease,
        prepared: PreparedActuation,
    ) -> ActuationReceipt:
        self._validate_command(command)
        if (
            prepared.command_sha256 != command.sha256
            or prepared.lease_sha256 != lease.sha256
            or prepared.adapter_id != self.adapter_id
            or not self._lease_valid(command, lease)
        ):
            raise ScalarAdapterError("scalar_preparation_identity_invalid")
        async with self._lock:
            context = self._prepared.get(command.sha256)
            if context is None:
                raise ScalarAdapterError("scalar_preparation_context_missing")
            reading = await self.refresh_readback()
            if self._value_fence(reading) != prepared.precondition_sha256:
                raise ScalarAdapterError("scalar_resource_changed_before_dispatch")
            context["dispatch_readback"] = reading
            now = time.monotonic()
            self._check_rate_limit(now)
            result = await self._transport.write_scalar(
                self._profile.resource_id,
                command.target,
                idempotency_key=command.idempotency_key,
            )
            executed = result.accepted and result.transport_completed
            if executed:
                self._dispatch_times.append(now)
                self._last_dispatch_at = now
        return ActuationReceipt(
            receipt_id=f"scalar.actuate.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            preparation_sha256=prepared.sha256,
            adapter_id=self.adapter_id,
            state=ActuationState.EXECUTED if executed else ActuationState.FAILED,
            accepted=executed,
            transport_completed=executed,
            executed=executed,
            recorded_at_ns=time.time_ns(),
            detail_sha256=_digest(dict(result.receipt)),
        )

    async def verify_effect(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt,
    ) -> EffectReceipt:
        self._validate_command(command)
        context = self._prepared.get(command.sha256)
        baseline = context.get("dispatch_readback") if context is not None else None
        if not isinstance(baseline, ChannelReading):
            baseline = None
        reading = self._last_observation
        for attempt in range(3):
            try:
                reading = await self.refresh_readback()
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
                if attempt < 2:
                    await asyncio.sleep(0.2)
                continue
            if self._target_was_independently_effected(
                reading=reading,
                baseline=baseline,
                target=command.target,
                tolerance=command.tolerance,
            ):
                break
            if attempt < 2:
                await asyncio.sleep(0.2)
        target_error = (
            abs(float(reading.value) - command.target) if reading.value is not None else None
        )
        independently_observed = self._target_was_independently_effected(
            reading=reading,
            baseline=baseline,
            target=command.target,
            tolerance=command.tolerance,
        )
        verified = bool(
            actuation.executed
            and independently_observed
            and target_error is not None
            and target_error <= command.tolerance
        )
        return EffectReceipt(
            receipt_id=f"scalar.effect.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            actuation_receipt_sha256=actuation.sha256,
            observation_channel_id=self._observation.channel_id,
            observation_sha256=reading.sha256,
            state=ActuationState.EFFECT_VERIFIED if verified else ActuationState.FAILED,
            target_error=target_error,
            independently_observed=independently_observed,
            recorded_at_ns=time.time_ns(),
        )

    async def cancel(
        self,
        command: ActuationCommand,
        prepared: PreparedActuation | None,
    ) -> ActuationReceipt:
        self._validate_command(command)
        self._prepared.pop(command.sha256, None)
        return ActuationReceipt(
            receipt_id=f"scalar.cancel.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            preparation_sha256=(prepared.sha256 if prepared is not None else command.sha256),
            adapter_id=self.adapter_id,
            state=ActuationState.CANCELLED,
            accepted=False,
            transport_completed=False,
            executed=False,
            recorded_at_ns=time.time_ns(),
            detail_sha256=_digest({"cancelled_before_dispatch": True}),
        )

    async def safe_state(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt | None,
    ) -> RollbackReceipt:
        self._validate_command(command)
        if self._profile.safe_value is None:
            return self._indeterminate_recovery(command, actuation)
        return await self._recover(
            command,
            actuation,
            value=self._profile.safe_value,
            success_state=ActuationState.SAFE_STATE,
        )

    async def rollback(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt,
    ) -> RollbackReceipt:
        self._validate_command(command)
        context = self._prepared.get(command.sha256)
        value = context.get("rollback_value") if context is not None else None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return self._indeterminate_recovery(command, actuation)
        return await self._recover(
            command,
            actuation,
            value=float(value),
            success_state=ActuationState.ROLLED_BACK,
        )

    def _indeterminate_recovery(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt | None,
    ) -> RollbackReceipt:
        return RollbackReceipt(
            receipt_id=f"scalar.indeterminate.{command.sha256.removeprefix('sha256:')[:32]}",
            command_sha256=command.sha256,
            actuation_receipt_sha256=(
                actuation.sha256 if actuation is not None else command.sha256
            ),
            adapter_id=self.adapter_id,
            state=ActuationState.INDETERMINATE,
            safe_state_observation_sha256=self._last_observation.sha256,
            independently_observed=False,
            recorded_at_ns=time.time_ns(),
        )

    async def _recover(
        self,
        command: ActuationCommand,
        actuation: ActuationReceipt | None,
        *,
        value: float,
        success_state: ActuationState,
    ) -> RollbackReceipt:
        async with self._lock:
            try:
                baseline = await self.refresh_readback()
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
                baseline = self._last_observation
            result = await self._transport.write_scalar(
                self._profile.resource_id,
                value,
                idempotency_key=f"recovery.{command.idempotency_key}"[:128],
                recovery=True,
            )
            try:
                tolerance = self._profile.tolerance
                if tolerance is None:
                    raise ScalarAdapterError("scalar_profile_tolerance_missing")
                reading = await self.refresh_readback()
                observed = bool(
                    result.accepted
                    and result.transport_completed
                    and self._target_was_independently_effected(
                        reading=reading,
                        baseline=baseline,
                        target=value,
                        tolerance=float(tolerance),
                    )
                )
            except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
                reading = self._last_observation
                observed = False
            self._prepared.pop(command.sha256, None)
        return RollbackReceipt(
            receipt_id=(
                f"scalar.{success_state.value}.{command.sha256.removeprefix('sha256:')[:32]}"
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
    "ScalarAdapterError",
    "ScalarProtocolTransport",
    "ScalarRealityAdapter",
    "ScalarResourceProfile",
    "ScalarSample",
    "ScalarTransportClass",
    "ScalarWriteResult",
]
