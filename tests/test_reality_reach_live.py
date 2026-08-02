from __future__ import annotations

from dataclasses import replace

import pytest

from core.reality_reach import (
    ChannelDeclaration,
    ChannelKind,
    ChannelReading,
    CouplingClass,
    EvidenceLevel,
    FailureCode,
    HostResourceAdapter,
    NumericDomain,
    ObjectiveKind,
    ProofRequirement,
    ReachabilityStatus,
    ReadingStatus,
    RealityIR,
    RealityLayer,
    RealityReachService,
)
from core.runtime.resource_observation import SimulatedResourceObserver

NOW_NS = 2_000_000_000_000


def _declaration(
    *,
    channel_id: str = "test.cpu",
    stale_after_s: float = 10.0,
    calibration_id: str = "",
    calibration_valid_until_ns: int | None = None,
) -> ChannelDeclaration:
    return ChannelDeclaration(
        channel_id=channel_id,
        kind=ChannelKind.SENSOR,
        observable="cpu_usage_percent",
        unit="percent",
        domain=NumericDomain(0.0, 100.0),
        coupling=CouplingClass.SOFTWARE,
        reality_layers=(RealityLayer.INTERNAL, RealityLayer.EFFECTIVE),
        evidence_level=EvidenceLevel.P1,
        owner="tests",
        resolution=0.1,
        sample_rate_hz=1.0,
        max_latency_s=1.0,
        stale_after_s=stale_after_s,
        reference_id="test.kernel.cpu",
        calibration_id=calibration_id,
        calibration_valid_until_ns=calibration_valid_until_ns,
        coupling_validated=True,
    )


def _reading(
    *,
    channel_id: str = "test.cpu",
    value: float | None = 25.0,
    captured_at_ns: int = NOW_NS,
    status: ReadingStatus = ReadingStatus.AVAILABLE,
    source: str = "host",
) -> ChannelReading:
    return ChannelReading(
        channel_id=channel_id,
        value=value,
        unit="percent",
        captured_at_ns=captured_at_ns,
        status=status,
        source=source,
        uncertainty=0.1,
    )


class StaticAdapter:
    def __init__(
        self,
        declaration: ChannelDeclaration,
        readings: tuple[ChannelReading, ...],
        *,
        adapter_id: str = "test.static",
    ) -> None:
        self.adapter_id = adapter_id
        self._declaration = declaration
        self._readings = readings

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        return (self._declaration,)

    def read(self) -> tuple[ChannelReading, ...]:
        return self._readings


def _contract(*, layer: RealityLayer) -> RealityIR:
    return RealityIR(
        request_id=f"test.observe.{layer.value}",
        objective="Observe current CPU utilization",
        objective_kind=ObjectiveKind.OBSERVE,
        observable="cpu_usage_percent",
        unit="percent",
        target=25.0,
        tolerance=1.0,
        domain=NumericDomain(0.0, 100.0),
        allowed_actuators=(),
        allowed_sensors=("test.cpu",),
        required_proof=ProofRequirement(minimum_evidence=EvidenceLevel.P1),
        reality_layer=layer,
    )


def _codes(certificate: object) -> set[FailureCode]:
    return {failure.code for failure in certificate.failures}


def test_host_resource_adapter_preserves_simulated_provenance() -> None:
    adapter = HostResourceAdapter(
        SimulatedResourceObserver(
            scenario_id="reality-reach-test",
            cpu_percent=37.5,
            memory_percent=41.0,
            battery_percent=72.0,
        )
    )

    readings = adapter.read()

    assert len(readings) == 5
    assert all(reading.status == ReadingStatus.SIMULATED for reading in readings)
    assert all(reading.source == "simulated" for reading in readings)
    assert {reading.channel_id: reading.value for reading in readings}[
        "host.compute.cpu_percent"
    ] == pytest.approx(37.5)


def test_simulated_reading_only_satisfies_internal_contract() -> None:
    adapter = StaticAdapter(
        _declaration(),
        (_reading(status=ReadingStatus.SIMULATED, source="simulated"),),
    )
    service = RealityReachService((adapter,), clock_ns=lambda: NOW_NS)

    internal = service.analyze(_contract(layer=RealityLayer.INTERNAL))
    effective = service.analyze(_contract(layer=RealityLayer.EFFECTIVE))

    assert internal.status == ReachabilityStatus.REACHABLE
    assert effective.status == ReachabilityStatus.UNREACHABLE
    assert FailureCode.NO_CHANNEL in _codes(effective)


def test_stale_reading_is_removed_from_effective_inventory() -> None:
    declaration = _declaration(stale_after_s=1.0)
    adapter = StaticAdapter(
        declaration,
        (_reading(captured_at_ns=NOW_NS - 2_000_000_000),),
    )
    service = RealityReachService((adapter,), clock_ns=lambda: NOW_NS)

    service.refresh()
    reading = service.reading("test.cpu")
    certificate = service.analyze(
        _contract(layer=RealityLayer.EFFECTIVE),
        refresh=False,
    )

    assert reading is not None
    assert reading.status == ReadingStatus.STALE
    assert certificate.status == ReachabilityStatus.UNREACHABLE
    assert FailureCode.NO_CHANNEL in _codes(certificate)


def test_freshness_uses_monotonic_lineage_after_wall_clock_rollback() -> None:
    clocks = {"wall": NOW_NS, "monotonic": 10_000_000_000}
    service = RealityReachService(
        (StaticAdapter(_declaration(stale_after_s=1.0), (_reading(),)),),
        clock_ns=lambda: clocks["wall"],
        monotonic_clock_ns=lambda: clocks["monotonic"],
        session_id="test-session",
    )

    first = service.refresh()["test.cpu"]
    clocks["wall"] -= 3_600_000_000_000
    clocks["monotonic"] += 2_000_000_000
    later = service.reading("test.cpu")

    assert first.session_id == "test-session"
    assert first.sequence == 1
    assert first.ingested_monotonic_ns == 10_000_000_000
    assert later is not None
    assert later.status == ReadingStatus.STALE
    assert later.error == "reading_stale:age_ns=2000000000"


def test_prior_service_session_reading_cannot_be_replayed_as_live() -> None:
    replayed = replace(
        _reading(),
        ingested_at_ns=NOW_NS,
        ingested_monotonic_ns=9_000_000_000,
        session_id="prior-session",
        sequence=8,
    )
    service = RealityReachService(
        (StaticAdapter(_declaration(), (replayed,)),),
        clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: 10_000_000_000,
        session_id="current-session",
    )

    reading = service.refresh()["test.cpu"]

    assert reading.status == ReadingStatus.STALE
    assert reading.session_id == "current-session"
    assert reading.sequence == 1
    assert reading.error == "reading_session_mismatch"


def test_refresh_sequence_and_operational_readiness_require_live_evidence() -> None:
    service = RealityReachService(
        (StaticAdapter(_declaration(), (_reading(),)),),
        clock_ns=lambda: NOW_NS,
        monotonic_clock_ns=lambda: 10_000_000_000,
        session_id="test-session",
    )

    assert service.is_ready() is False
    first = service.refresh()["test.cpu"]
    second = service.refresh()["test.cpu"]

    assert first.sequence == 1
    assert second.sequence == 2
    assert first.sha256 != second.sha256
    assert service.is_ready() is True
    status = service.status()
    assert status["session_id"] == "test-session"
    assert status["refresh_generation"] == 2
    assert status["last_refresh_monotonic_ns"] == 10_000_000_000


def test_source_event_identity_is_independent_of_ingestion_lineage() -> None:
    source = replace(
        _reading(),
        source_epoch="sensor.boot.a",
        source_sequence=7,
        source_event_id="sensor.event.7",
        source_quality="good",
    )
    first = replace(
        source,
        ingested_at_ns=NOW_NS + 1,
        ingested_monotonic_ns=10_000,
        session_id="aura.session.a",
        sequence=1,
    )
    second = replace(
        source,
        ingested_at_ns=NOW_NS + 2,
        ingested_monotonic_ns=20_000,
        session_id="aura.session.b",
        sequence=9,
    )

    assert first.sha256 != second.sha256
    assert first.event_sha256 == second.event_sha256


def test_out_of_domain_and_expired_calibration_are_not_live_evidence() -> None:
    out_of_domain = RealityReachService(
        (StaticAdapter(_declaration(), (_reading(value=101.0),)),),
        clock_ns=lambda: NOW_NS,
    )
    out_of_domain_reading = out_of_domain.refresh()["test.cpu"]
    assert out_of_domain_reading.status == ReadingStatus.DEGRADED
    assert out_of_domain_reading.sequence == 1
    assert out_of_domain_reading.ingested_at_ns == NOW_NS
    assert out_of_domain_reading.ingested_monotonic_ns > 0
    assert out_of_domain_reading.session_id

    calibrated = _declaration(
        calibration_id="test.calibration.1",
        calibration_valid_until_ns=NOW_NS - 1,
    )
    expired = RealityReachService(
        (StaticAdapter(calibrated, (_reading(),)),),
        clock_ns=lambda: NOW_NS,
    )
    expired_reading = expired.refresh()["test.cpu"]
    assert expired_reading.status == ReadingStatus.UNCALIBRATED
    assert expired_reading.sequence == 1
    assert expired_reading.ingested_at_ns == NOW_NS
    assert expired_reading.ingested_monotonic_ns > 0
    assert expired_reading.session_id


def test_missing_or_failed_adapter_reading_becomes_explicitly_unavailable() -> None:
    missing = RealityReachService(
        (StaticAdapter(_declaration(), ()),),
        clock_ns=lambda: NOW_NS,
    )
    reading = missing.refresh()["test.cpu"]

    assert reading.status == ReadingStatus.UNAVAILABLE
    assert reading.value is None
    assert "adapter_missing_reading" in reading.error
    assert missing.is_ready() is False


def test_adapter_registration_collision_rolls_back_atomically() -> None:
    first = StaticAdapter(_declaration(), (_reading(),), adapter_id="test.first")
    service = RealityReachService((first,), clock_ns=lambda: NOW_NS)
    conflicting = StaticAdapter(
        replace(_declaration(), owner="other"),
        (_reading(),),
        adapter_id="test.second",
    )

    with pytest.raises(ValueError, match="already registered"):
        service.register_adapter(conflicting)

    status = service.status()
    assert status["adapter_count"] == 1
    assert status["channel_count"] == 1
