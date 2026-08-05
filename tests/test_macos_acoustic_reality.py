from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from core.embodiment.macos_acoustic_reality import (
    ACOUSTIC_FREQUENCY_HZ,
    ACOUSTIC_RESOURCE_ID,
    AcousticDeviceIdentity,
    MacOSAcousticScalarTransport,
    build_macos_acoustic_reality_adapter,
)
from core.reality_reach.scalar_adapter import ScalarTransportClass


class _AcousticBackend:
    def __init__(self, *, coupling: float = 1.0, floor_peak: float = 0.0001) -> None:
        self.coupling = coupling
        self.floor_peak = floor_peak
        self.calls = 0

    def identity(self) -> AcousticDeviceIdentity:
        return AcousticDeviceIdentity(
            input_name="Fixture microphone",
            output_name="Fixture speaker",
            input_channels=1,
            output_channels=2,
            sample_rate_hz=48_000,
        )

    def play_and_record(
        self,
        signal: Sequence[float],
        *,
        sample_rate_hz: int,
    ) -> Sequence[float]:
        self.calls += 1
        peak = max((abs(float(value)) for value in signal), default=0.0)
        observed_peak = max(self.floor_peak, peak * self.coupling)
        return tuple(
            observed_peak
            * math.sin(2.0 * math.pi * ACOUSTIC_FREQUENCY_HZ * index / sample_rate_hz)
            for index in range(len(signal))
        )


@pytest.mark.asyncio
async def test_transport_adapts_stimulus_and_measures_independent_tone() -> None:
    backend = _AcousticBackend()
    transport = MacOSAcousticScalarTransport(backend)

    result = await transport.write_scalar(
        ACOUSTIC_RESOURCE_ID,
        -30.0,
        idempotency_key="campaign.fixture.write",
    )
    sample = await transport.read_scalar(ACOUSTIC_RESOURCE_ID)

    assert result.accepted is True
    assert result.transport_completed is True
    assert abs(float(result.receipt["observed_dbfs"]) - -30.0) <= 2.0
    assert abs(sample.value - -30.0) <= 2.0
    assert result.receipt["raw_audio_retained"] is False
    assert result.receipt["amplitude"] <= 0.08
    assert backend.calls >= 3


@pytest.mark.asyncio
async def test_transport_is_idempotent_and_recovery_returns_to_noise_floor() -> None:
    backend = _AcousticBackend()
    transport = MacOSAcousticScalarTransport(backend)
    baseline = await transport.read_scalar(ACOUSTIC_RESOURCE_ID)
    first = await transport.write_scalar(
        ACOUSTIC_RESOURCE_ID,
        -34.0,
        idempotency_key="campaign.fixture.once",
    )
    calls = backend.calls
    repeated = await transport.write_scalar(
        ACOUSTIC_RESOURCE_ID,
        -34.0,
        idempotency_key="campaign.fixture.once",
    )
    assert backend.calls == calls
    restored = await transport.write_scalar(
        ACOUSTIC_RESOURCE_ID,
        baseline.value,
        idempotency_key="campaign.fixture.restore",
        recovery=True,
    )
    observed = await transport.read_scalar(ACOUSTIC_RESOURCE_ID)

    assert repeated == first
    assert restored.accepted is True
    assert abs(observed.value - baseline.value) <= 2.0
    assert restored.receipt["amplitude"] == 0.0


@pytest.mark.asyncio
async def test_adapter_is_physical_reversible_and_identity_bound() -> None:
    backend = _AcousticBackend()
    adapter = await build_macos_acoustic_reality_adapter(backend)

    declarations = adapter.declarations()
    capabilities = adapter.actuator_capabilities()

    assert adapter.transport_class is ScalarTransportClass.PHYSICAL
    assert adapter.physical_identity_sha256.startswith("sha256:")
    assert len(declarations) == 2
    assert len(capabilities) == 1
    assert capabilities[0].observation_channels == (
        "macos_acoustic.macos.acoustic.reference_tone_dbfs.readback",
    )
    assert adapter.effect_tolerance == 2.0


@pytest.mark.asyncio
async def test_unreachable_target_fails_honestly_without_overdriving() -> None:
    backend = _AcousticBackend(coupling=0.00001)
    transport = MacOSAcousticScalarTransport(backend, maximum_attempts=3)

    result = await transport.write_scalar(
        ACOUSTIC_RESOURCE_ID,
        -20.0,
        idempotency_key="campaign.fixture.unreachable",
    )

    assert result.accepted is False
    assert result.transport_completed is True
    assert result.receipt["amplitude"] <= 0.08
    assert result.receipt["attempts"] == 3
