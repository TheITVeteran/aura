from __future__ import annotations

import asyncio
import json
import math
import threading
from pathlib import Path

import pytest

from core.reality_reach.contracts import (
    ChannelDeclaration,
    ChannelKind,
    CouplingClass,
    EvidenceLevel,
    NumericDomain,
    RealityLayer,
)
from core.reality_reach.live import ChannelReading, ReadingStatus, RealityReachService
from core.reality_reach.metrology import (
    AcquisitionChannel,
    AcquisitionMode,
    AcquisitionTask,
    CalibrationCertificate,
    EvidenceSource,
    MetrologyError,
    RealityMetrologyService,
)

NOW_NS = 2_000_000_000_000_000_000


def _declaration(
    channel_id: str,
    *,
    calibration_id: str = "",
    valid_until_ns: int | None = None,
) -> ChannelDeclaration:
    return ChannelDeclaration(
        channel_id=channel_id,
        kind=ChannelKind.SENSOR,
        observable="temperature",
        unit="celsius",
        domain=NumericDomain(-100.0, 200.0),
        coupling=CouplingClass.THERMAL,
        reality_layers=(RealityLayer.DIRECT,),
        evidence_level=EvidenceLevel.P3,
        owner="tests.metrology",
        resolution=0.2,
        sample_rate_hz=10.0,
        max_latency_s=0.2,
        stale_after_s=5.0,
        reference_id=f"reference.{channel_id}",
        calibration_id=calibration_id,
        calibration_valid_until_ns=valid_until_ns,
        coupling_validated=True,
    )


def _reading(
    channel_id: str,
    value: float,
    *,
    status: ReadingStatus = ReadingStatus.AVAILABLE,
    captured_at_ns: int = NOW_NS - 1_000_000,
    scenario_id: str = "",
    uncertainty: float = 0.1,
    wall_clock_source: str = "test.synchronized_clock",
) -> ChannelReading:
    return ChannelReading(
        channel_id=channel_id,
        value=value,
        unit="celsius",
        captured_at_ns=captured_at_ns,
        status=status,
        source="test.simulator" if status is ReadingStatus.SIMULATED else "test.sensor",
        scenario_id=scenario_id,
        uncertainty=uncertainty,
        wall_clock_source=wall_clock_source,
    )


class SequenceAdapter:
    adapter_id = "test.metrology.adapter"
    physical_identity_sha256 = "sha256:" + "b" * 64

    def __init__(
        self,
        declarations: tuple[ChannelDeclaration, ...],
        samples: list[tuple[ChannelReading, ...]],
    ) -> None:
        self._declarations = declarations
        self._samples = list(samples)
        self._index = 0

    def declarations(self) -> tuple[ChannelDeclaration, ...]:
        return self._declarations

    def read(self) -> tuple[ChannelReading, ...]:
        sample = self._samples[min(self._index, len(self._samples) - 1)]
        self._index += 1
        return sample


def _service(
    tmp_path: Path,
    declarations: tuple[ChannelDeclaration, ...],
    samples: list[tuple[ChannelReading, ...]],
) -> RealityMetrologyService:
    reality = RealityReachService(
        (SequenceAdapter(declarations, samples),),
        clock_ns=lambda: NOW_NS,
    )
    return RealityMetrologyService(
        reality,
        state_path=tmp_path / "metrology.json",
        wall_clock_ns=lambda: NOW_NS,
    )


def _certificate(channel_id: str, calibration_id: str) -> CalibrationCertificate:
    return CalibrationCertificate(
        calibration_id=calibration_id,
        channel_id=channel_id,
        reference_standard_id="nist.thermometry.reference-1",
        traceability_sha256="sha256:" + "a" * 64,
        issued_at_ns=NOW_NS - 10_000_000_000,
        valid_until_ns=NOW_NS + 10_000_000_000,
        scale=2.0,
        offset=1.0,
        standard_uncertainty=0.3,
        issuer="NIST-traceable test laboratory",
    )


@pytest.mark.asyncio
async def test_calibrated_acquisition_propagates_uncertainty_and_restores_live(
    tmp_path: Path,
) -> None:
    declaration = _declaration(
        "lab.temperature",
        calibration_id="lab.temperature.cal-1",
        valid_until_ns=NOW_NS + 10_000_000_000,
    )
    service = _service(
        tmp_path,
        (declaration,),
        [
            (_reading("lab.temperature", 10.0),),
            (_reading("lab.temperature", 12.0),),
        ],
    )
    await service.start()
    certificate = _certificate("lab.temperature", "lab.temperature.cal-1")
    assert await service.register_calibration(certificate) == certificate.sha256

    receipt = await service.acquire(
        AcquisitionTask(
            task_id="lab.temperature.mean",
            channels=(AcquisitionChannel("lab.temperature"),),
            sample_count=2,
            require_calibration=True,
        )
    )

    summary = receipt.summaries[0]
    assert summary.mean == 23.0
    assert summary.minimum == 21.0
    assert summary.maximum == 25.0
    assert summary.calibration_sha256 == certificate.sha256
    assert summary.standard_uncertainty > 1.4
    assert summary.coverage_factor == 2.0
    assert summary.expanded_uncertainty_k2 == pytest.approx(
        2.0 * summary.standard_uncertainty
    )
    assert receipt.evidence_sha256.startswith("sha256:")
    assert receipt.verify_evidence() is True
    assert len(receipt.measurements) == 2
    assert receipt.measurements[0].wall_clock_source == "test.synchronized_clock"
    assert receipt.maximum_observed_skew_ns == 0
    assert service.status()["mode"] == "live"
    assert service.status()["active_run"] is None
    envelope = json.loads((tmp_path / "metrology.json").read_text())
    persisted = envelope["payload"]["receipts"][-1]
    assert len(persisted["measurements"]) == 2
    assert persisted["evidence_sha256"] == receipt.evidence_sha256


@pytest.mark.asyncio
async def test_acquire_around_encloses_operation_and_restores_live(tmp_path: Path) -> None:
    declaration = _declaration("lab.temperature")
    service = _service(
        tmp_path,
        (declaration,),
        [
            (_reading("lab.temperature", 10.0),),
            (_reading("lab.temperature", 11.0),),
        ],
    )
    await service.start()

    async def operation() -> str:
        active = service.status()["active_run"]
        assert active is not None
        assert active["task_sha256"] == task.sha256
        await asyncio.sleep(0.01)
        return "completed"

    task = AcquisitionTask(
        task_id="lab.enclosing-acquisition",
        channels=(AcquisitionChannel("lab.temperature"),),
        sample_count=2,
        sample_interval_s=0.05,
        timeout_s=1.0,
    )
    result, receipt = await service.acquire_around(task, operation)

    assert result == "completed"
    assert receipt.started_at_ns <= receipt.completed_at_ns
    assert receipt.sample_sets == 2
    assert service.status()["mode"] == "live"
    assert service.status()["active_run"] is None


@pytest.mark.asyncio
async def test_acquire_around_rejects_non_enclosing_task_before_operation(
    tmp_path: Path,
) -> None:
    declaration = _declaration("lab.temperature")
    service = _service(
        tmp_path,
        (declaration,),
        [(_reading("lab.temperature", 10.0),)],
    )
    await service.start()
    called = False

    async def operation() -> None:
        nonlocal called
        called = True

    with pytest.raises(MetrologyError, match="temporally separated"):
        await service.acquire_around(
            AcquisitionTask(
                task_id="lab.not-enclosing",
                channels=(AcquisitionChannel("lab.temperature"),),
            ),
            operation,
        )

    assert called is False


@pytest.mark.asyncio
async def test_live_simulation_and_hil_sources_cannot_cross_contaminate(
    tmp_path: Path,
) -> None:
    live = _declaration("rig.live.temperature")
    simulated = _declaration("rig.sim.temperature")
    service = _service(
        tmp_path,
        (live, simulated),
        [
            (
                _reading("rig.live.temperature", 20.0),
                _reading(
                    "rig.sim.temperature",
                    20.5,
                    status=ReadingStatus.SIMULATED,
                    scenario_id="thermal-rig-1",
                ),
            )
        ],
    )
    await service.start()

    with pytest.raises(MetrologyError, match="does not satisfy live"):
        await service.acquire(
            AcquisitionTask(
                task_id="wrong.live.claim",
                channels=(AcquisitionChannel("rig.sim.temperature"),),
            )
        )

    hil = await service.acquire(
        AcquisitionTask(
            task_id="rig.hil.compare",
            channels=(
                AcquisitionChannel("rig.live.temperature", EvidenceSource.LIVE),
                AcquisitionChannel("rig.sim.temperature", EvidenceSource.SIMULATED),
            ),
            mode=AcquisitionMode.HARDWARE_IN_LOOP,
            scenario_id="thermal-rig-1",
        )
    )
    assert {item.source for item in hil.summaries} == {
        EvidenceSource.LIVE,
        EvidenceSource.SIMULATED,
    }
    assert hil.scenario_id == "thermal-rig-1"
    assert service.status()["mode"] == "live"


def test_task_contract_requires_explicit_hil_partition_and_scenario() -> None:
    with pytest.raises(ValueError, match="explicit live and simulated"):
        AcquisitionTask(
            task_id="invalid.hil",
            channels=(AcquisitionChannel("rig.live.temperature"),),
            mode=AcquisitionMode.HARDWARE_IN_LOOP,
            scenario_id="scenario-1",
        )
    with pytest.raises(ValueError, match="scenario_id"):
        AcquisitionTask(
            task_id="invalid.sim",
            channels=(
                AcquisitionChannel("rig.sim.temperature", EvidenceSource.SIMULATED),
            ),
            mode=AcquisitionMode.SIMULATION,
        )


@pytest.mark.asyncio
async def test_capture_skew_failure_restores_live_and_records_no_success(
    tmp_path: Path,
) -> None:
    first = _declaration("rig.temperature.a")
    second = _declaration("rig.temperature.b")
    service = _service(
        tmp_path,
        (first, second),
        [
            (
                _reading("rig.temperature.a", 1.0, captured_at_ns=NOW_NS - 200_000_000),
                _reading("rig.temperature.b", 1.0, captured_at_ns=NOW_NS - 1_000_000),
            )
        ],
    )
    await service.start()

    with pytest.raises(MetrologyError, match="capture skew"):
        await service.acquire(
            AcquisitionTask(
                task_id="rig.synchronized",
                channels=(
                    AcquisitionChannel("rig.temperature.a"),
                    AcquisitionChannel("rig.temperature.b"),
                ),
                max_capture_skew_ns=10_000_000,
            )
        )

    status = service.status()
    assert status["mode"] == "live"
    assert status["active_run"] is None
    assert status["receipt_count"] == 0
    assert status["failure_count"] == 1
    assert status["last_failure"]["failure_sha256"].startswith("sha256:")
    assert status["live_restoration_required"] is False


@pytest.mark.asyncio
async def test_capture_skew_refuses_incomparable_clock_sources(tmp_path: Path) -> None:
    first = _declaration("rig.temperature.a")
    second = _declaration("rig.temperature.b")
    service = _service(
        tmp_path,
        (first, second),
        [
            (
                _reading(
                    "rig.temperature.a",
                    1.0,
                    wall_clock_source="clock.domain.a",
                ),
                _reading(
                    "rig.temperature.b",
                    1.0,
                    wall_clock_source="clock.domain.b",
                ),
            )
        ],
    )
    await service.start()

    with pytest.raises(MetrologyError, match="different wall-clock sources"):
        await service.acquire(
            AcquisitionTask(
                task_id="rig.incomparable-clocks",
                channels=(
                    AcquisitionChannel("rig.temperature.a"),
                    AcquisitionChannel("rig.temperature.b"),
                ),
            )
        )


def test_task_bounds_retained_measurement_evidence() -> None:
    with pytest.raises(ValueError, match="retained evidence bound"):
        AcquisitionTask(
            task_id="rig.unbounded-evidence",
            channels=tuple(
                AcquisitionChannel(f"rig.channel.{index}") for index in range(32)
            ),
            sample_count=33,
        )


@pytest.mark.asyncio
async def test_expired_or_unbound_calibration_cannot_support_acquisition(
    tmp_path: Path,
) -> None:
    declaration = _declaration(
        "lab.temperature",
        calibration_id="lab.temperature.cal-1",
        valid_until_ns=NOW_NS + 10_000_000_000,
    )
    service = _service(
        tmp_path,
        (declaration,),
        [(_reading("lab.temperature", 1.0),)],
    )
    await service.start()

    with pytest.raises(MetrologyError, match="no registered certificate"):
        await service.acquire(
            AcquisitionTask(
                task_id="missing.calibration",
                channels=(AcquisitionChannel("lab.temperature"),),
            )
        )

    with pytest.raises(MetrologyError, match="identity differs"):
        await service.register_calibration(
            _certificate("lab.temperature", "lab.temperature.wrong-cal")
        )


@pytest.mark.asyncio
async def test_calibrated_evidence_rejects_unstable_adapter_identity(
    tmp_path: Path,
) -> None:
    declaration = _declaration(
        "lab.temperature",
        calibration_id="lab.temperature.cal-1",
        valid_until_ns=NOW_NS + 10_000_000_000,
    )

    class EphemeralAdapter(SequenceAdapter):
        physical_identity_sha256 = ""

    reality = RealityReachService(
        (
            EphemeralAdapter(
                (declaration,),
                [(_reading("lab.temperature", 1.0),)],
            ),
        ),
        clock_ns=lambda: NOW_NS,
    )
    service = RealityMetrologyService(
        reality,
        state_path=tmp_path / "metrology.json",
        wall_clock_ns=lambda: NOW_NS,
    )
    await service.start()
    await service.register_calibration(
        _certificate("lab.temperature", "lab.temperature.cal-1")
    )

    with pytest.raises(MetrologyError, match="stable adapter identity"):
        await service.acquire(
            AcquisitionTask(
                task_id="lab.unstable-adapter",
                channels=(AcquisitionChannel("lab.temperature"),),
                require_calibration=True,
            )
        )


@pytest.mark.asyncio
async def test_restart_recovers_interrupted_mode_to_live(tmp_path: Path) -> None:
    declaration = _declaration("lab.temperature")
    state_path = tmp_path / "metrology.json"
    service = _service(
        tmp_path,
        (declaration,),
        [(_reading("lab.temperature", 1.0),)],
    )
    await service.start()
    envelope = json.loads(state_path.read_text())
    payload = envelope["payload"]
    payload["mode"] = "simulation"
    payload["active_run"] = {
        "run_id": "abandoned",
        "task_sha256": "sha256:" + "b" * 64,
        "mode": "simulation",
        "mode_generation": 1,
        "started_at_ns": NOW_NS - 1,
    }
    unsigned = dict(payload)
    unsigned.pop("state_sha256")
    from core.runtime.audit_chain import canonical_json, sha256_hex

    payload["state_sha256"] = str(sha256_hex(canonical_json(unsigned)))
    state_path.write_text(json.dumps(envelope))

    recovered = RealityMetrologyService(
        service._reality,  # noqa: SLF001 - explicit restart fixture
        state_path=state_path,
        wall_clock_ns=lambda: NOW_NS,
    )
    await recovered.start()
    assert recovered.status()["mode"] == "live"
    assert recovered.status()["active_run"] is None
    assert recovered.status()["recovered_interrupted_count"] == 1


def test_state_tampering_fails_closed(tmp_path: Path) -> None:
    declaration = _declaration("lab.temperature")
    state_path = tmp_path / "metrology.json"
    reality = RealityReachService(
        (SequenceAdapter((declaration,), [(_reading("lab.temperature", 1.0),)]),),
        clock_ns=lambda: NOW_NS,
    )
    state_path.write_text(
        json.dumps(
            {
                "schema": "aura.reality_reach.metrology",
                "schema_name": "aura.reality_reach.metrology",
                "schema_version": 1,
                "payload": {
                    "mode": "live",
                    "state_sha256": "sha256:" + "0" * 64,
                },
            }
        )
    )
    with pytest.raises(MetrologyError, match="integrity"):
        RealityMetrologyService(reality, state_path=state_path)


def test_uncertainty_math_does_not_reduce_systematic_error_with_repetition() -> None:
    # A fixed calibration uncertainty remains fixed while independent random
    # uncertainty shrinks with sqrt(n).
    systematic = 0.4
    random = 0.2
    one = math.sqrt(random**2 + systematic**2)
    four = math.sqrt((random / 2.0) ** 2 + systematic**2)
    assert four < one
    assert four > systematic


@pytest.mark.asyncio
async def test_timed_out_refresh_fences_new_runs_until_worker_reconciles(
    tmp_path: Path,
) -> None:
    declaration = _declaration("lab.temperature")
    release = threading.Event()

    class BlockingAdapter(SequenceAdapter):
        def read(self) -> tuple[ChannelReading, ...]:
            release.wait(timeout=2.0)
            return (_reading("lab.temperature", 1.0),)

    reality = RealityReachService(
        (BlockingAdapter((declaration,), []),),
        clock_ns=lambda: NOW_NS,
    )
    service = RealityMetrologyService(
        reality,
        state_path=tmp_path / "metrology.json",
        wall_clock_ns=lambda: NOW_NS,
    )
    await service.start()
    task = AcquisitionTask(
        task_id="lab.timeout",
        channels=(AcquisitionChannel("lab.temperature"),),
        timeout_s=0.05,
    )

    with pytest.raises(MetrologyError, match="is reconciling"):
        await service.acquire(task)
    assert service.status()["refresh_reconciliation_required"] is True
    with pytest.raises(MetrologyError, match="still reconciling"):
        await service.acquire(task)

    release.set()
    for _ in range(100):
        if not service.status()["refresh_reconciliation_required"]:
            break
        await asyncio.sleep(0.01)
    completed = await service.acquire(
        AcquisitionTask(
            task_id="lab.after-reconciliation",
            channels=(AcquisitionChannel("lab.temperature"),),
        )
    )
    assert completed.sample_sets == 1


@pytest.mark.asyncio
async def test_cancellation_durably_restores_live_mode(tmp_path: Path) -> None:
    declaration = _declaration("lab.temperature")
    service = _service(
        tmp_path,
        (declaration,),
        [(_reading("lab.temperature", 1.0),)],
    )
    await service.start()
    acquisition = asyncio.create_task(
        service.acquire(
            AcquisitionTask(
                task_id="lab.cancelled",
                channels=(AcquisitionChannel("lab.temperature"),),
                sample_count=2,
                sample_interval_s=5.0,
                timeout_s=10.0,
            )
        )
    )
    await asyncio.sleep(0.05)
    acquisition.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquisition

    assert service.status()["mode"] == "live"
    envelope = json.loads((tmp_path / "metrology.json").read_text())
    assert envelope["payload"]["mode"] == "live"
    assert envelope["payload"]["active_run"] is None
    assert envelope["payload"]["failures"][-1]["error_type"] == "CancelledError"
