from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import time
from dataclasses import replace
from typing import Any

import pytest

from core.reality_reach.acceptance import (
    REQUIRED_SCALAR_ACCEPTANCE_CASES,
    AcceptanceCaseResult,
    AcceptanceCertificateStore,
    AcceptanceError,
    AcceptanceEvidenceClass,
    AcceptanceVerdict,
    ConnectorAcceptanceCertificate,
    FaultInjectedOutcomeUnknownError,
    FaultInjectedReadError,
    FaultInjectedWriteError,
    FaultInjectingScalarTransport,
    ScalarAcceptancePlan,
    ScalarAcceptanceRunner,
    ScalarFault,
)
from core.reality_reach.acceptance_verifier import (
    persist_verification_receipt,
    verify_acceptance_evidence,
)
from core.reality_reach.contracts import NumericDomain
from core.reality_reach.live import RealityReachService
from core.reality_reach.metrology import (
    AcquisitionMode,
    AcquisitionReceipt,
    EvidenceSource,
    Measurement,
    MeasurementSummary,
)
from core.reality_reach.scalar_adapter import (
    ScalarRealityAdapter,
    ScalarResourceProfile,
    ScalarSample,
    ScalarTransportClass,
    ScalarWriteResult,
)
from core.runtime.audit_chain import canonical_json, sha256_hex


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


class _Transport:
    transport_id = "test.acceptance"

    def __init__(self) -> None:
        self.value = 1.0
        self.reads = 0
        self.writes = 0
        self.base_ns = time.time_ns()

    async def read_scalar(self, resource_id: str) -> ScalarSample:
        self.reads += 1
        return ScalarSample(
            value=self.value,
            captured_at_ns=self.base_ns + self.reads,
            source_event_id=_digest(
                {"resource_id": resource_id, "reads": self.reads, "value": self.value}
            ),
            quality="fixture_reported",
            source_epoch="fixture.epoch.1",
            source_sequence=self.reads,
        )

    async def write_scalar(
        self,
        resource_id: str,
        value: float,
        *,
        idempotency_key: str,
        recovery: bool = False,
    ) -> ScalarWriteResult:
        self.writes += 1
        self.value = value
        return ScalarWriteResult(
            accepted=True,
            transport_completed=True,
            receipt={
                "resource_sha256": _digest(resource_id),
                "idempotency_sha256": _digest(idempotency_key),
                "recovery": recovery,
            },
        )


def _proxy(delegate: _Transport | None = None) -> FaultInjectingScalarTransport:
    return FaultInjectingScalarTransport(
        delegate or _Transport(),
        evidence_class=AcceptanceEvidenceClass.SIMULATION,
        scenario_id="connector-fault-matrix-1",
        stale_age_s=3600.0,
    )


@pytest.mark.asyncio
async def test_read_and_write_partitions_stop_before_delegate() -> None:
    delegate = _Transport()
    proxy = _proxy(delegate)
    proxy.arm(ScalarFault.READ_PARTITION, ScalarFault.WRITE_PARTITION)

    with pytest.raises(FaultInjectedReadError, match="read_partition"):
        await proxy.read_scalar("fixture.level")
    with pytest.raises(FaultInjectedWriteError, match="write_partition"):
        await proxy.write_scalar(
            "fixture.level",
            2.0,
            idempotency_key="partitioned-write",
        )

    assert delegate.reads == 0
    assert delegate.writes == 0
    assert [item.delegate_called for item in proxy.receipts] == [False, False]
    assert all(item.outcome_indeterminate is False for item in proxy.receipts)


@pytest.mark.asyncio
async def test_transport_recovers_after_one_shot_network_partitions() -> None:
    delegate = _Transport()
    proxy = _proxy(delegate)
    proxy.arm(ScalarFault.READ_PARTITION)
    with pytest.raises(FaultInjectedReadError):
        await proxy.read_scalar("fixture.level")

    recovered_read = await proxy.read_scalar("fixture.level")
    proxy.arm(ScalarFault.WRITE_PARTITION)
    with pytest.raises(FaultInjectedWriteError):
        await proxy.write_scalar(
            "fixture.level",
            2.0,
            idempotency_key="partitioned-write",
        )
    recovered_write = await proxy.write_scalar(
        "fixture.level",
        3.0,
        idempotency_key="restored-write",
    )

    assert recovered_read.value == 1.0
    assert recovered_write.accepted is True
    assert delegate.reads == 1
    assert delegate.writes == 1
    assert delegate.value == 3.0


@pytest.mark.asyncio
async def test_post_dispatch_ack_loss_is_explicitly_indeterminate() -> None:
    delegate = _Transport()
    proxy = _proxy(delegate)
    proxy.arm(ScalarFault.WRITE_OUTCOME_UNKNOWN)

    with pytest.raises(FaultInjectedOutcomeUnknownError, match="acknowledgement_loss"):
        await proxy.write_scalar(
            "fixture.level",
            7.0,
            idempotency_key="lost-ack",
        )

    assert delegate.writes == 1
    assert delegate.value == 7.0
    receipt = proxy.receipts[-1]
    assert receipt.delegate_called is True
    assert receipt.outcome_indeterminate is True
    assert receipt.sha256.startswith("sha256:")
    assert "fixture.level" not in str(receipt.to_dict())


@pytest.mark.asyncio
async def test_stale_duplicate_and_reordered_readbacks_are_deterministic() -> None:
    delegate = _Transport()
    proxy = _proxy(delegate)
    first = await proxy.read_scalar("fixture.level")
    delegate.value = 2.0
    second = await proxy.read_scalar("fixture.level")

    proxy.arm(ScalarFault.DUPLICATE_READBACK)
    duplicate = await proxy.read_scalar("fixture.level")
    assert duplicate == second
    assert delegate.reads == 2

    delegate.value = 3.0
    proxy.arm(ScalarFault.REORDERED_READBACK)
    reordered = await proxy.read_scalar("fixture.level")
    assert reordered == first
    assert delegate.reads == 3

    proxy.arm(ScalarFault.STALE_READBACK)
    stale = await proxy.read_scalar("fixture.level")
    assert stale.quality == "fault_injected_stale"
    assert stale.captured_at_ns < time.time_ns() - 3_500_000_000_000
    assert stale.source_event_id != first.source_event_id
    assert [item.fault for item in proxy.receipts] == [
        ScalarFault.DUPLICATE_READBACK,
        ScalarFault.REORDERED_READBACK,
        ScalarFault.STALE_READBACK,
    ]


def _case(
    case_id: str,
    evidence_class: AcceptanceEvidenceClass,
    verdict: AcceptanceVerdict = AcceptanceVerdict.PASS,
    *,
    required: bool = True,
) -> AcceptanceCaseResult:
    return AcceptanceCaseResult(
        case_id=case_id,
        verdict=verdict,
        evidence_class=evidence_class,
        required=required,
        evidence_sha256=_digest({"case": case_id, "verdict": verdict.value}),
        duration_ms=1.25,
    )


def _certificate(
    cases: tuple[AcceptanceCaseResult, ...],
    *,
    scenario_id: str = "",
    metrology_evidence_sha256: str = "",
    governance_evidence_sha256: str = "",
    governance_accepted: bool = False,
) -> ConnectorAcceptanceCertificate:
    return ConnectorAcceptanceCertificate(
        campaign_id="cp810.connector.acceptance",
        connector_id="mqtt.manifest",
        adapter_id="mqtt.tank.level.adapter",
        physical_identity_sha256=_digest("physical.fixture"),
        source_commit_sha256=_digest("dae896754"),
        started_at_ns=1_000,
        completed_at_ns=2_000,
        cases=cases,
        scenario_id=scenario_id,
        metrology_evidence_sha256=metrology_evidence_sha256,
        governance_evidence_sha256=governance_evidence_sha256,
        governance_accepted=governance_accepted,
    )


def _governance_evidence() -> dict[str, Any]:
    return {
        "schema": "aura.reality_reach.acceptance_governance.v1",
        "action_id": "acceptance.cp810.fixture",
        "request_digest": _digest("governed-request"),
        "will_receipt_id": "will.cp810.fixture",
        "post_action_receipt_id": "post.cp810.fixture",
        "post_action_output_hash": _digest("governed-output"),
        "status": "success_verified",
        "transport_succeeded": True,
        "effect_verified": True,
        "receipt_persisted": True,
        "welfare_transaction_completed": True,
    }


def test_simulation_certificate_can_pass_deterministically_but_not_live() -> None:
    certificate = _certificate(
        (
            _case("read.partition", AcceptanceEvidenceClass.SIMULATION),
            _case("stale.readback", AcceptanceEvidenceClass.SIMULATION),
        )
    )

    assert certificate.deterministic_passed is True
    assert certificate.live_acceptance_passed is False
    assert certificate.to_dict()["live_acceptance_passed"] is False
    assert certificate.sha256.startswith("sha256:")


def test_hil_certificate_requires_scenario_and_every_required_case() -> None:
    cases = (
        _case("command.effect", AcceptanceEvidenceClass.HARDWARE_IN_LOOP),
        _case("safe.state", AcceptanceEvidenceClass.HARDWARE_IN_LOOP),
        _case(
            "credential.rotation",
            AcceptanceEvidenceClass.HARDWARE_IN_LOOP,
            AcceptanceVerdict.UNMEASURED,
        ),
    )
    with pytest.raises(ValueError, match="scenario_id"):
        _certificate(cases)

    incomplete = _certificate(cases, scenario_id="physical-rig-1")
    assert incomplete.deterministic_passed is False
    assert incomplete.live_acceptance_passed is False

    complete = _certificate(
        tuple(_case(item.case_id, item.evidence_class) for item in cases),
        scenario_id="physical-rig-1",
        metrology_evidence_sha256=_digest("verified-hil-metrology"),
        governance_evidence_sha256=_digest(_governance_evidence()),
        governance_accepted=True,
    )
    assert complete.deterministic_passed is True
    assert complete.live_acceptance_passed is True


def test_required_unmeasured_case_cannot_be_hidden_by_optional_passes() -> None:
    certificate = _certificate(
        (
            _case(
                "restart.recovery",
                AcceptanceEvidenceClass.SIMULATION,
                AcceptanceVerdict.UNMEASURED,
            ),
            _case(
                "cosmetic.status",
                AcceptanceEvidenceClass.SIMULATION,
                required=False,
            ),
        )
    )
    assert certificate.deterministic_passed is False
    assert certificate.live_acceptance_passed is False


def _metrology_receipt(
    mode: AcquisitionMode,
    sources: tuple[EvidenceSource, ...],
    *,
    scenario_id: str = "",
) -> AcquisitionReceipt:
    measurements = tuple(
        Measurement(
            channel_id=f"fixture.{source.value}",
            value=1.0,
            unit="percent",
            captured_at_ns=1_100 + index,
            source=source,
            scenario_id=scenario_id if source is EvidenceSource.SIMULATED else "",
            wall_clock_source="fixture.clock",
            random_uncertainty=0.1,
            resolution_uncertainty=0.1,
            systematic_uncertainty=0.1,
            calibration_sha256="",
            reading_sha256=_digest({"source": source.value}),
        )
        for index, source in enumerate(sources)
    )
    summaries = tuple(
        MeasurementSummary(
            channel_id=item.channel_id,
            unit=item.unit,
            sample_count=1,
            mean=item.value,
            minimum=item.value,
            maximum=item.value,
            standard_uncertainty=item.standard_uncertainty,
            coverage_factor=2.0,
            expanded_uncertainty_k2=2.0 * item.standard_uncertainty,
            source=item.source,
            wall_clock_source=item.wall_clock_source,
            calibration_sha256="",
        )
        for item in measurements
    )
    evidence = {
        "run_id": "metrology.acceptance.1",
        "task_sha256": _digest("metrology-task"),
        "mode": mode.value,
        "mode_generation": 1,
        "started_at_ns": 1_000,
        "completed_at_ns": 2_000,
        "sample_sets": 1,
        "maximum_observed_skew_ns": 1,
        "scenario_id": scenario_id,
        "measurements": [item.to_dict() for item in measurements],
        "summaries": [item.to_dict() for item in summaries],
    }
    return AcquisitionReceipt(
        run_id=str(evidence["run_id"]),
        task_sha256=str(evidence["task_sha256"]),
        mode=mode,
        mode_generation=1,
        started_at_ns=1_000,
        completed_at_ns=2_000,
        sample_sets=1,
        maximum_observed_skew_ns=1,
        scenario_id=scenario_id,
        measurements=measurements,
        summaries=summaries,
        evidence_sha256=_digest(evidence),
    )


def test_live_acceptance_plan_requires_matching_verified_metrology() -> None:
    with pytest.raises(ValueError, match="verified metrology"):
        ScalarAcceptancePlan(
            campaign_id="cp810.live",
            connector_id="fixture.connector",
            target=7.0,
            source_commit_sha256=_digest("dae896754"),
            authority_receipt_id="authority.cp810.live",
            evidence_class=AcceptanceEvidenceClass.LIVE,
        )

    live = _metrology_receipt(AcquisitionMode.LIVE, (EvidenceSource.LIVE,))
    plan = ScalarAcceptancePlan(
        campaign_id="cp810.live",
        connector_id="fixture.connector",
        target=7.0,
        source_commit_sha256=_digest("dae896754"),
        authority_receipt_id="authority.cp810.live",
        evidence_class=AcceptanceEvidenceClass.LIVE,
        metrology_receipt=live,
    )
    assert plan.metrology_receipt is live

    simulated = _metrology_receipt(
        AcquisitionMode.SIMULATION,
        (EvidenceSource.SIMULATED,),
        scenario_id="sim-rig-1",
    )
    with pytest.raises(ValueError, match="matching verified metrology"):
        ScalarAcceptancePlan(
            campaign_id="cp810.false-live",
            connector_id="fixture.connector",
            target=7.0,
            source_commit_sha256=_digest("dae896754"),
            authority_receipt_id="authority.cp810.false-live",
            evidence_class=AcceptanceEvidenceClass.LIVE,
            metrology_receipt=simulated,
        )


def _runner(
    transport: _Transport | FaultInjectingScalarTransport,
    *,
    initial: ScalarSample,
    target: float = 7.0,
    evidence_class: AcceptanceEvidenceClass = AcceptanceEvidenceClass.SIMULATION,
    metrology_receipt: AcquisitionReceipt | None = None,
    governed_executor: Any = None,
    transport_class: ScalarTransportClass | None = None,
) -> ScalarAcceptanceRunner:
    adapter = ScalarRealityAdapter(
        transport,
        ScalarResourceProfile(
            resource_id="fixture.level",
            observable="fixture_level",
            unit="percent",
            domain=NumericDomain(0.0, 100.0),
            resolution=0.1,
            tolerance=0.2,
            writable=True,
            physical_identity_sha256=_digest("physical.fixture"),
            owner="tests.reality_reach_acceptance",
            protocol="acceptance_fixture",
            safe_value=1.0,
            readback_distinct_from_command=True,
        ),
        initial_sample=initial,
        transport_class=(
            transport_class
            if transport_class is not None
            else (
                ScalarTransportClass.SIMULATED
                if evidence_class is AcceptanceEvidenceClass.SIMULATION
                else ScalarTransportClass.PHYSICAL
            )
        ),
    )
    service = RealityReachService((adapter,), session_id="acceptance.runner.session")
    return ScalarAcceptanceRunner(
        adapter,
        service,
        ScalarAcceptancePlan(
            campaign_id="cp810.scalar.lifecycle",
            connector_id="fixture.connector",
            target=target,
            source_commit_sha256=_digest("dae896754"),
            authority_receipt_id="authority.cp810.fixture",
            evidence_class=evidence_class,
            metrology_receipt=metrology_receipt,
        ),
        governed_executor=governed_executor,
    )


async def _governed_executor(**kwargs: Any) -> dict[str, Any]:
    context = {"will_receipt_id": "will.cp810.actual"}
    dispatch = dict(await kwargs["effect_handler"](context))
    verification = dict(await kwargs["effect_verifier"](context))
    return {
        **dispatch,
        **verification,
        "action_id": kwargs["action_id"],
        "request_digest": _digest("governed-request"),
        "will_receipt_id": context["will_receipt_id"],
        "post_action_receipt_id": "post.cp810.actual",
        "post_action_output_hash": _digest("governed-output"),
        "status": "success_verified",
        "transport_succeeded": True,
        "effect_verified": True,
        "receipt_persisted": True,
        "welfare_transaction_completed": True,
    }


@pytest.mark.asyncio
async def test_scalar_acceptance_runner_proves_complete_reversible_lifecycle() -> None:
    transport = _Transport()
    runner = _runner(
        transport,
        initial=await transport.read_scalar("fixture.level"),
    )

    certificate = await runner.run()

    assert [item.case_id for item in certificate.cases] == list(REQUIRED_SCALAR_ACCEPTANCE_CASES)
    assert all(item.verdict is AcceptanceVerdict.PASS for item in certificate.cases)
    assert certificate.deterministic_passed is True
    assert certificate.live_acceptance_passed is False
    assert transport.value == 1.0
    assert transport.writes == 2


@pytest.mark.asyncio
async def test_live_acceptance_is_governed_and_independently_replayed(tmp_path) -> None:
    transport = _Transport()
    metrology = _metrology_receipt(AcquisitionMode.LIVE, (EvidenceSource.LIVE,))
    runner = _runner(
        transport,
        initial=await transport.read_scalar("fixture.level"),
        evidence_class=AcceptanceEvidenceClass.LIVE,
        metrology_receipt=metrology,
        governed_executor=_governed_executor,
    )
    store = AcceptanceCertificateStore(tmp_path / "acceptance")

    certificate = await runner.run_and_persist(store)
    evidence = store.load_evidence(certificate)
    receipt = verify_acceptance_evidence(
        certificate,
        evidence,
        expected_source_commit_sha256=certificate.source_commit_sha256,
        expected_physical_identity_sha256=certificate.physical_identity_sha256,
        expected_evidence_class=AcceptanceEvidenceClass.LIVE,
        trusted_metrology_evidence_sha256=metrology.evidence_sha256,
        trusted_governance_evidence_sha256=certificate.governance_evidence_sha256,
    )

    assert runner.governance_evidence["will_receipt_id"] == "will.cp810.actual"
    assert certificate.governance_accepted is True
    assert certificate.live_acceptance_passed is True
    assert transport.value == 1.0
    assert transport.writes == 2
    assert receipt.accepted is True
    assert receipt.blockers == ()

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "tools/reality_reach/verify_acceptance.py",
        "--root",
        str(store.root),
        "--campaign-id",
        certificate.campaign_id,
        "--source-commit-sha256",
        certificate.source_commit_sha256,
        "--physical-identity-sha256",
        certificate.physical_identity_sha256,
        "--expected-evidence-class",
        "live",
        "--metrology-evidence-sha256",
        metrology.evidence_sha256,
        "--governance-evidence-sha256",
        certificate.governance_evidence_sha256,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)

    assert process.returncode == 0, stderr.decode()
    replay = json.loads(stdout)
    assert replay["accepted"] is True
    assert replay["expected_evidence_class"] == "live"
    assert replay["trusted_governance_evidence_sha256"] == (
        certificate.governance_evidence_sha256
    )

    substituted = verify_acceptance_evidence(
        certificate,
        evidence,
        expected_source_commit_sha256=certificate.source_commit_sha256,
        expected_physical_identity_sha256=certificate.physical_identity_sha256,
        expected_evidence_class=AcceptanceEvidenceClass.LIVE,
        trusted_metrology_evidence_sha256=metrology.evidence_sha256,
        trusted_governance_evidence_sha256=_digest("substituted-governance"),
    )
    assert substituted.accepted is False
    assert "trusted_governance_mismatch" in substituted.blockers


@pytest.mark.asyncio
async def test_live_acceptance_refused_before_dispatch_performs_no_writes() -> None:
    async def refused_executor(**_kwargs: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "status": "blocked_by_policy",
            "will_receipt_id": "will.cp810.refused",
        }

    transport = _Transport()
    runner = _runner(
        transport,
        initial=await transport.read_scalar("fixture.level"),
        evidence_class=AcceptanceEvidenceClass.LIVE,
        metrology_receipt=_metrology_receipt(
            AcquisitionMode.LIVE,
            (EvidenceSource.LIVE,),
        ),
        governed_executor=refused_executor,
    )

    with pytest.raises(AcceptanceError, match="refused_before_dispatch"):
        await runner.run()

    assert transport.writes == 0


@pytest.mark.asyncio
async def test_simulation_acceptance_refuses_a_physical_adapter_before_writes() -> None:
    transport = _Transport()
    runner = _runner(
        transport,
        initial=await transport.read_scalar("fixture.level"),
        transport_class=ScalarTransportClass.PHYSICAL,
    )

    with pytest.raises(AcceptanceError, match="requires_simulated_adapter"):
        await runner.run()

    assert transport.writes == 0


@pytest.mark.asyncio
async def test_live_acceptance_refuses_a_simulated_adapter_before_writes() -> None:
    transport = _Transport()
    runner = _runner(
        transport,
        initial=await transport.read_scalar("fixture.level"),
        evidence_class=AcceptanceEvidenceClass.LIVE,
        metrology_receipt=_metrology_receipt(
            AcquisitionMode.LIVE,
            (EvidenceSource.LIVE,),
        ),
        governed_executor=_governed_executor,
        transport_class=ScalarTransportClass.SIMULATED,
    )

    with pytest.raises(AcceptanceError, match="requires_physical_adapter"):
        await runner.run()

    assert transport.writes == 0


@pytest.mark.asyncio
async def test_independent_replay_recomputes_every_scalar_acceptance_predicate(
    tmp_path,
) -> None:
    transport = _Transport()
    runner = _runner(transport, initial=await transport.read_scalar("fixture.level"))
    store = AcceptanceCertificateStore(tmp_path / "acceptance")
    certificate = await runner.run_and_persist(store)

    restarted_store = AcceptanceCertificateStore(store.root)
    restored = restarted_store.load(certificate.campaign_id)
    evidence = restarted_store.load_evidence(restored)
    receipt = verify_acceptance_evidence(
        restored,
        evidence,
        expected_source_commit_sha256=restored.source_commit_sha256,
        expected_physical_identity_sha256=restored.physical_identity_sha256,
    )

    assert receipt.deterministic_accepted is True
    assert receipt.live_accepted is False
    assert receipt.blockers == ()
    assert receipt.replayed_cases == REQUIRED_SCALAR_ACCEPTANCE_CASES
    assert receipt.sha256.startswith("sha256:")

    wrong_source = verify_acceptance_evidence(
        restored,
        evidence,
        expected_source_commit_sha256=_digest("different-source"),
        expected_physical_identity_sha256=restored.physical_identity_sha256,
    )
    wrong_identity = verify_acceptance_evidence(
        restored,
        evidence,
        expected_source_commit_sha256=restored.source_commit_sha256,
        expected_physical_identity_sha256=_digest("different-device"),
    )
    assert wrong_source.deterministic_accepted is False
    assert "source_commit_mismatch" in wrong_source.blockers
    assert wrong_identity.deterministic_accepted is False
    assert "physical_identity_mismatch" in wrong_identity.blockers


@pytest.mark.asyncio
async def test_independent_replay_rejects_producer_pass_not_supported_by_evidence() -> None:
    transport = _Transport()
    runner = _runner(transport, initial=await transport.read_scalar("fixture.level"))
    certificate = await runner.run()
    evidence = runner.case_evidence
    forged_observation = dict(evidence["observation.fresh"])
    forged_observation["status"] = "stale"
    evidence["observation.fresh"] = forged_observation
    forged_case = replace(
        certificate.cases[0],
        evidence_sha256=_digest(forged_observation),
    )
    forged = replace(certificate, cases=(forged_case, *certificate.cases[1:]))

    receipt = verify_acceptance_evidence(
        forged,
        {"case_evidence": evidence},
        expected_source_commit_sha256=forged.source_commit_sha256,
        expected_physical_identity_sha256=forged.physical_identity_sha256,
    )

    assert receipt.deterministic_accepted is False
    assert "observation.fresh:pass_not_reproduced" in receipt.blockers


@pytest.mark.asyncio
async def test_evidence_store_rejects_case_tampering_after_restart(tmp_path) -> None:
    transport = _Transport()
    runner = _runner(transport, initial=await transport.read_scalar("fixture.level"))
    certificate = await runner.run()
    store = AcceptanceCertificateStore(tmp_path / "acceptance")
    store.persist(certificate)
    store.persist_evidence(certificate, runner.case_evidence)
    evidence_path = next(store.root.glob("*.evidence"))
    document = json.loads(evidence_path.read_bytes())
    document["case_evidence"]["actuation.dispatch"]["executed"] = False
    evidence_path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))
    evidence_path.chmod(0o600)

    with pytest.raises(AcceptanceError, match="case_digest_mismatch"):
        store.load_evidence(certificate)


@pytest.mark.asyncio
async def test_acceptance_cli_replays_in_fresh_process(tmp_path) -> None:
    transport = _Transport()
    runner = _runner(transport, initial=await transport.read_scalar("fixture.level"))
    certificate = await runner.run()
    store = AcceptanceCertificateStore(tmp_path / "acceptance")
    store.persist(certificate)
    store.persist_evidence(certificate, runner.case_evidence)
    receipt_path = tmp_path / "verifier" / "receipt.json"

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "tools/reality_reach/verify_acceptance.py",
        "--root",
        str(store.root),
        "--campaign-id",
        certificate.campaign_id,
        "--source-commit-sha256",
        certificate.source_commit_sha256,
        "--physical-identity-sha256",
        certificate.physical_identity_sha256,
        "--expected-evidence-class",
        "simulation",
        "--receipt-output",
        str(receipt_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)

    assert process.returncode == 0, stderr.decode()
    receipt = json.loads(stdout)
    assert receipt["deterministic_accepted"] is True
    assert receipt["live_accepted"] is False
    assert receipt["expected_evidence_class"] == "simulation"
    assert receipt["accepted"] is True
    assert json.loads(receipt_path.read_bytes()) == receipt
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600

    verification = verify_acceptance_evidence(
        certificate,
        store.load_evidence(certificate),
        expected_source_commit_sha256=certificate.source_commit_sha256,
        expected_physical_identity_sha256=certificate.physical_identity_sha256,
    )
    assert persist_verification_receipt(verification, receipt_path) is False
    attacked = replace(
        verification,
        expected_source_commit_sha256=_digest("attacker-source"),
    )
    with pytest.raises(AcceptanceError, match="receipt_collision"):
        persist_verification_receipt(attacked, receipt_path)
    receipt_path.chmod(0o644)
    with pytest.raises(AcceptanceError, match="mode_invalid"):
        persist_verification_receipt(verification, receipt_path)


def test_live_replay_requires_recomputed_trusted_metrology() -> None:
    metrology = _metrology_receipt(AcquisitionMode.LIVE, (EvidenceSource.LIVE,))
    governance = _governance_evidence()
    cases = tuple(
        _case(case_id, AcceptanceEvidenceClass.LIVE) for case_id in REQUIRED_SCALAR_ACCEPTANCE_CASES
    )
    certificate = _certificate(
        cases,
        metrology_evidence_sha256=metrology.evidence_sha256,
        governance_evidence_sha256=_digest(governance),
        governance_accepted=True,
    )
    evidence = {
        "case_evidence": {
            case.case_id: {"error_type": "fixture", "error_sha256": case.evidence_sha256}
            for case in cases
        },
        "metrology_receipt": metrology.to_dict(),
        "governance_evidence": governance,
    }

    receipt = verify_acceptance_evidence(
        certificate,
        evidence,
        expected_source_commit_sha256=certificate.source_commit_sha256,
        expected_physical_identity_sha256=certificate.physical_identity_sha256,
        expected_evidence_class=AcceptanceEvidenceClass.LIVE,
        trusted_metrology_evidence_sha256=_digest("wrong-metrology"),
        trusted_governance_evidence_sha256=_digest(governance),
    )

    assert receipt.live_accepted is False
    assert "trusted_metrology_mismatch" in receipt.blockers


@pytest.mark.asyncio
async def test_independent_replay_binds_exact_external_evidence_class() -> None:
    transport = _Transport()
    runner = _runner(transport, initial=await transport.read_scalar("fixture.level"))
    simulated = await runner.run()
    metrology = _metrology_receipt(AcquisitionMode.LIVE, (EvidenceSource.LIVE,))
    governance = _governance_evidence()
    live_certificate = replace(
        simulated,
        cases=tuple(
            replace(item, evidence_class=AcceptanceEvidenceClass.LIVE) for item in simulated.cases
        ),
        metrology_evidence_sha256=metrology.evidence_sha256,
        governance_evidence_sha256=_digest(governance),
        governance_accepted=True,
    )
    evidence = {
        "case_evidence": runner.case_evidence,
        "metrology_receipt": metrology.to_dict(),
        "governance_evidence": governance,
    }

    accepted = verify_acceptance_evidence(
        live_certificate,
        evidence,
        expected_source_commit_sha256=live_certificate.source_commit_sha256,
        expected_physical_identity_sha256=live_certificate.physical_identity_sha256,
        expected_evidence_class=AcceptanceEvidenceClass.LIVE,
        trusted_metrology_evidence_sha256=metrology.evidence_sha256,
        trusted_governance_evidence_sha256=_digest(governance),
    )
    substituted = verify_acceptance_evidence(
        live_certificate,
        evidence,
        expected_source_commit_sha256=live_certificate.source_commit_sha256,
        expected_physical_identity_sha256=live_certificate.physical_identity_sha256,
        expected_evidence_class=AcceptanceEvidenceClass.HARDWARE_IN_LOOP,
        trusted_metrology_evidence_sha256=metrology.evidence_sha256,
        trusted_governance_evidence_sha256=_digest(governance),
    )

    assert accepted.accepted is True
    assert accepted.live_accepted is True
    assert accepted.blockers == ()
    assert substituted.accepted is False
    assert "evidence_class_mismatch" in substituted.blockers
    assert "metrology_mode_mismatch" in substituted.blockers


@pytest.mark.asyncio
async def test_acceptance_cli_refuses_simulation_when_live_was_requested(tmp_path) -> None:
    transport = _Transport()
    runner = _runner(transport, initial=await transport.read_scalar("fixture.level"))
    store = AcceptanceCertificateStore(tmp_path / "acceptance")
    certificate = await runner.run_and_persist(store)

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "tools/reality_reach/verify_acceptance.py",
        "--root",
        str(store.root),
        "--campaign-id",
        certificate.campaign_id,
        "--source-commit-sha256",
        certificate.source_commit_sha256,
        "--physical-identity-sha256",
        certificate.physical_identity_sha256,
        "--expected-evidence-class",
        "live",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)

    assert process.returncode == 2, stderr.decode()
    receipt = json.loads(stdout)
    assert receipt["deterministic_accepted"] is False
    assert receipt["accepted"] is False
    assert "evidence_class_mismatch" in receipt["blockers"]
    assert "trusted_metrology_missing" in receipt["blockers"]


@pytest.mark.asyncio
async def test_scalar_acceptance_runner_refuses_preexisting_target_as_effect() -> None:
    transport = _Transport()
    transport.value = 7.0
    runner = _runner(
        transport,
        initial=await transport.read_scalar("fixture.level"),
        target=7.0,
    )

    certificate = await runner.run()
    cases = {item.case_id: item for item in certificate.cases}

    assert cases["actuation.dispatch"].verdict is AcceptanceVerdict.PASS
    assert cases["effect.independent_readback"].verdict is AcceptanceVerdict.FAIL
    assert certificate.deterministic_passed is False


@pytest.mark.asyncio
async def test_scalar_acceptance_runner_records_blocked_stages_as_unmeasured() -> None:
    delegate = _Transport()
    initial = await delegate.read_scalar("fixture.level")
    proxy = _proxy(delegate)
    proxy.arm(ScalarFault.READ_PARTITION)
    runner = _runner(proxy, initial=initial)

    certificate = await runner.run()
    cases = {item.case_id: item for item in certificate.cases}

    assert cases["observation.fresh"].verdict is AcceptanceVerdict.FAIL
    assert cases["cancellation.pre_dispatch"].verdict is AcceptanceVerdict.UNMEASURED
    assert cases["actuation.dispatch"].verdict is AcceptanceVerdict.UNMEASURED
    assert certificate.deterministic_passed is False


@pytest.mark.asyncio
async def test_ack_loss_campaign_restores_safe_state_from_fresh_baseline() -> None:
    delegate = _Transport()
    initial = await delegate.read_scalar("fixture.level")
    proxy = _proxy(delegate)
    proxy.arm(ScalarFault.WRITE_OUTCOME_UNKNOWN)
    runner = _runner(proxy, initial=initial)

    certificate = await runner.run()
    cases = {item.case_id: item for item in certificate.cases}

    assert cases["actuation.dispatch"].verdict is AcceptanceVerdict.FAIL
    assert cases["effect.independent_readback"].verdict is AcceptanceVerdict.UNMEASURED
    assert cases["restoration.rollback"].verdict is AcceptanceVerdict.PASS
    assert delegate.value == 1.0
    assert delegate.writes == 2
    assert proxy.receipts[-1].outcome_indeterminate is True


def test_acceptance_certificate_store_survives_restart_and_is_idempotent(
    tmp_path,
) -> None:
    certificate = _certificate((_case("restart.recovery", AcceptanceEvidenceClass.SIMULATION),))
    root = tmp_path / "acceptance"

    assert AcceptanceCertificateStore(root).persist(certificate) is True
    assert AcceptanceCertificateStore(root).persist(certificate) is False
    restored = AcceptanceCertificateStore(root).load(certificate.campaign_id)

    assert restored == certificate
    assert restored.sha256 == certificate.sha256
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    files = list(root.iterdir())
    assert len(files) == 1
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o600


def test_acceptance_certificate_store_rejects_campaign_collision(tmp_path) -> None:
    first = _certificate((_case("restart.recovery", AcceptanceEvidenceClass.SIMULATION),))
    second = replace(first, completed_at_ns=3_000)
    store = AcceptanceCertificateStore(tmp_path / "acceptance")
    store.persist(first)

    with pytest.raises(AcceptanceError, match="acceptance_campaign_collision"):
        store.persist(second)


@pytest.mark.parametrize("mutation", ["truncate", "digest", "duplicate_key"])
def test_acceptance_certificate_store_rejects_tampering(tmp_path, mutation) -> None:
    certificate = _certificate((_case("restart.recovery", AcceptanceEvidenceClass.SIMULATION),))
    store = AcceptanceCertificateStore(tmp_path / "acceptance")
    store.persist(certificate)
    path = next(store.root.iterdir())
    payload = path.read_bytes()

    if mutation == "truncate":
        path.write_bytes(payload[: len(payload) // 2])
    elif mutation == "digest":
        document = json.loads(payload)
        document["certificate_sha256"] = _digest("tampered")
        path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))
    else:
        path.write_bytes(payload[:-1] + b',"schema":"duplicate"}')

    with pytest.raises(AcceptanceError):
        AcceptanceCertificateStore(store.root).load(certificate.campaign_id)


def test_acceptance_certificate_store_rejects_unsafe_mode(tmp_path) -> None:
    certificate = _certificate((_case("restart.recovery", AcceptanceEvidenceClass.SIMULATION),))
    store = AcceptanceCertificateStore(tmp_path / "acceptance")
    store.persist(certificate)
    path = next(store.root.iterdir())
    os.chmod(path, 0o644)

    with pytest.raises(AcceptanceError, match="acceptance_certificate_mode_invalid"):
        AcceptanceCertificateStore(store.root).load(certificate.campaign_id)


def test_acceptance_certificate_store_rejects_symlink_replacement(tmp_path) -> None:
    certificate = _certificate((_case("restart.recovery", AcceptanceEvidenceClass.SIMULATION),))
    store = AcceptanceCertificateStore(tmp_path / "acceptance")
    store.persist(certificate)
    path = next(store.root.iterdir())
    outside = tmp_path / "outside.json"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)

    with pytest.raises(AcceptanceError, match="acceptance_certificate_custody_invalid"):
        AcceptanceCertificateStore(store.root).load(certificate.campaign_id)


@pytest.mark.asyncio
async def test_acceptance_failure_evidence_digests_provider_message() -> None:
    secret = "provider-secret-that-must-not-be-persisted"

    class _SecretFailureTransport(_Transport):
        async def read_scalar(self, resource_id: str) -> ScalarSample:
            raise RuntimeError(secret)

    transport = _SecretFailureTransport()
    runner = _runner(
        transport,
        initial=ScalarSample(
            value=1.0,
            captured_at_ns=time.time_ns(),
            source_event_id=_digest("initial"),
            quality="fixture_reported",
        ),
    )

    certificate = await runner.run()
    document = json.dumps(certificate.to_dict(), sort_keys=True)

    assert secret not in document
    assert certificate.cases[0].detail == "RuntimeError"
    assert certificate.cases[0].evidence_sha256 == _digest(
        {"error_type": "RuntimeError", "error_sha256": _digest(secret)}
    )
