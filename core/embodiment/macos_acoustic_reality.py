"""Reversible built-in speaker-to-microphone physical scalar adapter.

The actuator emits a short, bounded reference tone.  The observation path is
the built-in microphone and a lock-in estimator at the exact reference
frequency, so ambient broadband sound does not masquerade as the commanded
effect.  Raw audio is never retained in receipts.
"""

from __future__ import annotations

import asyncio
import math
import platform
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from core.reality_reach.contracts import NumericDomain
from core.reality_reach.scalar_adapter import (
    ScalarRealityAdapter,
    ScalarResourceProfile,
    ScalarSample,
    ScalarTransportClass,
    ScalarWriteResult,
)
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.lockdep import checked_async_lock

ACOUSTIC_RESOURCE_ID = "macos.acoustic.reference_tone_dbfs"
ACOUSTIC_FREQUENCY_HZ = 997.0
ACOUSTIC_ADAPTER_ID = f"macos_acoustic.{ACOUSTIC_RESOURCE_ID}.adapter"


class MacOSAcousticRealityError(RuntimeError):
    """Stable acoustic adapter failure."""


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class AcousticDeviceIdentity:
    input_name: str
    output_name: str
    input_channels: int
    output_channels: int
    sample_rate_hz: int

    def __post_init__(self) -> None:
        if not self.input_name or not self.output_name:
            raise ValueError("acoustic device names must be non-empty")
        if self.input_channels < 1 or self.output_channels < 1:
            raise ValueError("acoustic devices must expose input and output channels")
        if not 8_000 <= self.sample_rate_hz <= 384_000:
            raise ValueError("acoustic sample rate is outside the supported range")

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "input_name": self.input_name,
                "output_name": self.output_name,
                "input_channels": self.input_channels,
                "output_channels": self.output_channels,
                "sample_rate_hz": self.sample_rate_hz,
                "host": platform.node(),
            }
        )


@runtime_checkable
class AcousticMeasurementBackend(Protocol):
    def identity(self) -> AcousticDeviceIdentity: ...

    def play_and_record(
        self,
        signal: Sequence[float],
        *,
        sample_rate_hz: int,
    ) -> Sequence[float]: ...


class SoundDeviceAcousticBackend:
    """Lazy sounddevice backend; importing the module does not touch hardware."""

    def __init__(
        self,
        *,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        sample_rate_hz: int = 48_000,
    ) -> None:
        self._input_device = input_device
        self._output_device = output_device
        self._sample_rate_hz = int(sample_rate_hz)

    @staticmethod
    def _module() -> Any:
        try:
            import sounddevice
        except ImportError as exc:
            raise MacOSAcousticRealityError("sounddevice_unavailable") from exc
        return sounddevice

    def identity(self) -> AcousticDeviceIdentity:
        sounddevice = self._module()
        try:
            input_info = sounddevice.query_devices(self._input_device, "input")
            output_info = sounddevice.query_devices(self._output_device, "output")
        except Exception as exc:
            raise MacOSAcousticRealityError("acoustic_device_query_failed") from exc
        return AcousticDeviceIdentity(
            input_name=str(input_info.get("name") or ""),
            output_name=str(output_info.get("name") or ""),
            input_channels=int(input_info.get("max_input_channels") or 0),
            output_channels=int(output_info.get("max_output_channels") or 0),
            sample_rate_hz=self._sample_rate_hz,
        )

    def play_and_record(
        self,
        signal: Sequence[float],
        *,
        sample_rate_hz: int,
    ) -> Sequence[float]:
        sounddevice = self._module()
        try:
            import numpy as np

            output = np.asarray(tuple(signal), dtype=np.float32)
            recorded = sounddevice.playrec(
                output,
                samplerate=sample_rate_hz,
                channels=1,
                dtype="float32",
                input_mapping=[1],
                device=(self._input_device, self._output_device),
                blocking=True,
            )
            return tuple(float(value) for value in recorded[:, 0])
        except Exception as exc:
            raise MacOSAcousticRealityError("acoustic_playrec_failed") from exc


class MacOSAcousticScalarTransport:
    """Bounded adaptive control of measured reference-tone level."""

    transport_id = "macos.acoustic.loopback"

    def __init__(
        self,
        backend: AcousticMeasurementBackend,
        *,
        duration_s: float = 0.35,
        frequency_hz: float = ACOUSTIC_FREQUENCY_HZ,
        maximum_amplitude: float = 0.08,
        tolerance_db: float = 2.0,
        maximum_attempts: int = 5,
    ) -> None:
        if not isinstance(backend, AcousticMeasurementBackend):
            raise TypeError("backend must satisfy AcousticMeasurementBackend")
        self._backend = backend
        self._identity = backend.identity()
        self._duration_s = _finite(duration_s, name="duration_s")
        self._frequency_hz = _finite(frequency_hz, name="frequency_hz")
        self._maximum_amplitude = _finite(
            maximum_amplitude,
            name="maximum_amplitude",
        )
        self._tolerance_db = _finite(tolerance_db, name="tolerance_db")
        self._maximum_attempts = int(maximum_attempts)
        if not 0.1 <= self._duration_s <= 2.0:
            raise ValueError("duration_s must lie inside [0.1, 2.0]")
        if not 100.0 <= self._frequency_hz <= 8_000.0:
            raise ValueError("frequency_hz must lie inside [100, 8000]")
        if not 0.0 < self._maximum_amplitude <= 0.2:
            raise ValueError("maximum_amplitude must lie inside (0, 0.2]")
        if not 0.25 <= self._tolerance_db <= 6.0:
            raise ValueError("tolerance_db must lie inside [0.25, 6]")
        if not 1 <= self._maximum_attempts <= 8:
            raise ValueError("maximum_attempts must lie inside [1, 8]")
        self._lock = checked_async_lock("macos_acoustic_reality")
        self._current_amplitude = 0.0
        self._sequence = 0
        self._epoch = str(uuid.uuid4())
        self._idempotency: dict[str, ScalarWriteResult] = {}

    @property
    def physical_identity_sha256(self) -> str:
        return self._identity.sha256

    @property
    def tolerance_db(self) -> float:
        return self._tolerance_db

    def _signal(self, amplitude: float) -> tuple[float, ...]:
        count = max(1, int(self._duration_s * self._identity.sample_rate_hz))
        fade = max(1, min(count // 8, int(0.02 * self._identity.sample_rate_hz)))
        signal: list[float] = []
        for index in range(count):
            envelope = 1.0
            if index < fade:
                envelope = index / fade
            elif index >= count - fade:
                envelope = (count - index - 1) / fade
            signal.append(
                amplitude
                * max(0.0, envelope)
                * math.sin(
                    2.0
                    * math.pi
                    * self._frequency_hz
                    * index
                    / self._identity.sample_rate_hz
                )
            )
        return tuple(signal)

    def _measure(self, amplitude: float) -> float:
        samples = tuple(
            float(value)
            for value in self._backend.play_and_record(
                self._signal(amplitude),
                sample_rate_hz=self._identity.sample_rate_hz,
            )
        )
        if not samples:
            raise MacOSAcousticRealityError("acoustic_measurement_empty")
        start = min(len(samples) // 4, int(0.05 * self._identity.sample_rate_hz))
        usable = samples[start:]
        if not usable:
            raise MacOSAcousticRealityError("acoustic_measurement_too_short")
        omega = 2.0 * math.pi * self._frequency_hz / self._identity.sample_rate_hz
        sine = 0.0
        cosine = 0.0
        for index, value in enumerate(usable):
            sine += value * math.sin(omega * (index + start))
            cosine += value * math.cos(omega * (index + start))
        component_rms = math.sqrt(2.0) * math.hypot(sine, cosine) / len(usable)
        return 20.0 * math.log10(max(component_rms, 1e-8))

    def _sample(self, value: float, *, amplitude: float) -> ScalarSample:
        self._sequence += 1
        captured_at_ns = time.time_ns()
        return ScalarSample(
            value=value,
            captured_at_ns=captured_at_ns,
            source_event_id=_digest(
                {
                    "transport_id": self.transport_id,
                    "physical_identity_sha256": self.physical_identity_sha256,
                    "captured_at_ns": captured_at_ns,
                    "sequence": self._sequence,
                    "amplitude": round(amplitude, 8),
                    "measured_dbfs": round(value, 6),
                }
            ),
            quality="good",
            uncertainty=self._tolerance_db / 2.0,
            wall_clock_source="time.time_ns",
            source_epoch=self._epoch,
            source_sequence=self._sequence,
        )

    async def read_scalar(self, resource_id: str) -> ScalarSample:
        if resource_id != ACOUSTIC_RESOURCE_ID:
            raise MacOSAcousticRealityError("acoustic_resource_unknown")
        async with self._lock:
            observed = await asyncio.to_thread(self._measure, self._current_amplitude)
            return self._sample(observed, amplitude=self._current_amplitude)

    async def write_scalar(
        self,
        resource_id: str,
        value: float,
        *,
        idempotency_key: str,
        recovery: bool = False,
    ) -> ScalarWriteResult:
        if resource_id != ACOUSTIC_RESOURCE_ID:
            raise MacOSAcousticRealityError("acoustic_resource_unknown")
        target = _finite(value, name="value")
        if not -100.0 <= target <= -10.0:
            raise MacOSAcousticRealityError("acoustic_target_out_of_domain")
        if not idempotency_key or len(idempotency_key) > 128:
            raise ValueError("idempotency_key must be a bounded non-empty string")
        async with self._lock:
            existing = self._idempotency.get(idempotency_key)
            if existing is not None:
                return existing
            baseline = await asyncio.to_thread(self._measure, 0.0)
            amplitude = 0.0 if recovery or target <= baseline + self._tolerance_db else 0.01
            observed = baseline
            attempts = 1
            if amplitude > 0.0:
                for attempts in range(1, self._maximum_attempts + 1):
                    observed = await asyncio.to_thread(self._measure, amplitude)
                    error_db = target - observed
                    if abs(error_db) <= self._tolerance_db:
                        break
                    if attempts < self._maximum_attempts:
                        amplitude = min(
                            self._maximum_amplitude,
                            max(0.0001, amplitude * 10.0 ** (error_db / 20.0)),
                        )
            self._current_amplitude = amplitude
            accepted = abs(observed - target) <= self._tolerance_db
            receipt = ScalarWriteResult(
                accepted=accepted,
                transport_completed=True,
                receipt={
                    "schema": "aura.reality_reach.macos_acoustic_write.v1",
                    "physical_identity_sha256": self.physical_identity_sha256,
                    "target_dbfs": target,
                    "observed_dbfs": observed,
                    "tolerance_db": self._tolerance_db,
                    "amplitude": amplitude,
                    "attempts": attempts,
                    "recovery": bool(recovery),
                    "raw_audio_retained": False,
                    "input_name_sha256": _digest(self._identity.input_name),
                    "output_name_sha256": _digest(self._identity.output_name),
                },
            )
            self._idempotency[idempotency_key] = receipt
            return receipt


async def build_macos_acoustic_reality_adapter(
    backend: AcousticMeasurementBackend | None = None,
) -> ScalarRealityAdapter:
    """Build a physical adapter after a real microphone baseline is measured."""

    transport = MacOSAcousticScalarTransport(
        backend or SoundDeviceAcousticBackend()
    )
    initial = await transport.read_scalar(ACOUSTIC_RESOURCE_ID)
    safe_value = max(-100.0, min(-10.0, round(initial.value * 2.0) / 2.0))
    profile = ScalarResourceProfile(
        resource_id=ACOUSTIC_RESOURCE_ID,
        observable="reference_tone_level",
        unit="dbfs",
        domain=NumericDomain(-100.0, -10.0),
        resolution=0.5,
        writable=True,
        physical_identity_sha256=transport.physical_identity_sha256,
        owner="core.embodiment.macos_acoustic_reality",
        protocol="macos_acoustic",
        safe_value=safe_value,
        tolerance=transport.tolerance_db,
        max_commands_per_minute=8,
        cooldown_s=0.25,
        stale_after_s=5.0,
        readback_distinct_from_command=True,
    )
    return ScalarRealityAdapter(
        transport,
        profile,
        initial_sample=initial,
        transport_class=ScalarTransportClass.PHYSICAL,
    )


__all__ = [
    "ACOUSTIC_FREQUENCY_HZ",
    "ACOUSTIC_ADAPTER_ID",
    "ACOUSTIC_RESOURCE_ID",
    "AcousticDeviceIdentity",
    "AcousticMeasurementBackend",
    "MacOSAcousticRealityError",
    "MacOSAcousticScalarTransport",
    "SoundDeviceAcousticBackend",
    "build_macos_acoustic_reality_adapter",
]
