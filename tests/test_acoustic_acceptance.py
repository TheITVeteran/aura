from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import NameOID

from core.reality_reach.acceptance import AcceptanceEvidenceClass
from core.reality_reach.acceptance_mandate import AcceptanceVerificationMandate
from core.reality_reach.acceptance_transparency import (
    ZERO_SHA256 as TRANSPARENCY_ZERO_SHA256,
)
from core.reality_reach.acceptance_witness import (
    ZERO_SHA256,
    AcceptanceWitnessBundle,
    AcceptanceWitnessRole,
    AcceptanceWitnessStatement,
)
from core.reality_reach.acoustic_acceptance import (
    ACOUSTIC_A1_CONNECTOR_ID,
    ACOUSTIC_A1_REQUIRED_CASES,
    AcousticA1AcceptanceReceipt,
    AcousticA1CampaignRecord,
    AcousticAcceptanceConfig,
    AcousticAcceptanceError,
    AcousticTrialArm,
    build_acoustic_a1_transparency_bundle,
    build_acoustic_a1_transparency_statement,
    persist_externally_witnessed_acoustic_a1_receipt,
    persist_transparently_logged_acoustic_a1_receipt,
    run_acoustic_a1_acceptance,
    verify_acoustic_a1_with_external_witnesses,
    verify_transparently_logged_acoustic_a1,
)
from core.reality_reach.scalar_adapter import ScalarSample
from core.runtime.audit_chain import canonical_json, sha256_hex


class _TransferDriver:
    def __init__(
        self,
        *,
        transfer: str = "nonlinear",
        fail_at_call: int | None = None,
    ) -> None:
        self.transfer = transfer
        self.fail_at_call = fail_at_call
        self.calls: list[tuple[float, str]] = []

    async def measure_stimulus(
        self,
        amplitude: float,
        *,
        trial_id: str,
    ) -> ScalarSample:
        self.calls.append((amplitude, trial_id))
        if self.fail_at_call == len(self.calls):
            raise RuntimeError("fixture_measurement_failed")
        if amplitude == 0.0:
            observed = -80.0
        elif self.transfer == "nominal":
            observed = 20.0 * math.log10(amplitude / math.sqrt(2.0))
        else:
            observed = -80.0 + 60.0 * (amplitude / 0.08) ** 0.35
        jitter = (
            int(hashlib.sha256(trial_id.encode("utf-8")).hexdigest()[:4], 16) % 21
            - 10
        ) / 100.0
        captured_at_ns = time.time_ns()
        return ScalarSample(
            value=observed + jitter,
            captured_at_ns=captured_at_ns,
            source_event_id="sha256:"
            + hashlib.sha256(
                f"{trial_id}|{amplitude:.9f}|{observed:.6f}".encode()
            ).hexdigest(),
            uncertainty=0.1,
            source_epoch="fixture",
            source_sequence=len(self.calls),
        )


def _witness_bundle(
    role: AcceptanceWitnessRole,
    private_key: Ed25519PrivateKey,
    *,
    record: AcousticA1CampaignRecord,
    mandate: AcceptanceVerificationMandate,
    evidence_sha256: str,
) -> AcceptanceWitnessBundle:
    statement = AcceptanceWitnessStatement(
        role=role,
        witness_id=f"fixture.{role.value}",
        campaign_id=record.campaign_id,
        mandate_sha256=mandate.sha256,
        certificate_sha256=record.sha256,
        evidence_sha256=evidence_sha256,
        sequence=1,
        previous_statement_sha256=ZERO_SHA256,
        witnessed_at_ns=record.completed_at_ns + 1,
    )
    payload = json.dumps(
        statement.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return AcceptanceWitnessBundle(
        statement=statement,
        public_key_raw_b64=base64.b64encode(public_key).decode("ascii"),
        signature_b64=base64.b64encode(private_key.sign(payload)).decode("ascii"),
    )


def _transparency_bundle(
    statement: dict[str, object],
) -> tuple[dict[str, object], bytes]:
    issued_at = int(statement["issued_at_unix"])
    statement_bytes = json.dumps(
        statement,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    producer_key = Ed25519PrivateKey.generate()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "A1 fixture")])
    start = datetime.fromtimestamp(issued_at - 60, tz=UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(producer_key.public_key())
        .serial_number(1)
        .not_valid_before(start)
        .not_valid_after(start + timedelta(days=1))
        .sign(producer_key, algorithm=None)
        .public_bytes(serialization.Encoding.PEM)
    )
    signature = producer_key.sign(statement_bytes)
    body = {
        "apiVersion": "0.0.1",
        "kind": "rekord",
        "spec": {
            "data": {
                "hash": {
                    "algorithm": "sha256",
                    "value": hashlib.sha256(statement_bytes).hexdigest(),
                }
            },
            "signature": {
                "content": base64.b64encode(signature).decode("ascii"),
                "format": "x509",
                "publicKey": {
                    "content": base64.b64encode(certificate).decode("ascii")
                },
            },
        },
    }
    body_bytes = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    body_b64 = base64.b64encode(body_bytes).decode("ascii")
    root = hashlib.sha256(b"\x00" + body_bytes).digest()
    log_key = ec.generate_private_key(ec.SECP256R1())
    log_public_pem = log_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    log_der = log_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    log_id = hashlib.sha256(log_der).hexdigest()
    checkpoint_text = (
        "rekor.sigstore.dev - 123\n"
        "1\n"
        f"{base64.b64encode(root).decode('ascii')}\n"
    )
    checkpoint_signature = log_key.sign(
        checkpoint_text.encode("utf-8"),
        ec.ECDSA(hashes.SHA256()),
    )
    checkpoint = (
        checkpoint_text
        + "\n— rekor.sigstore.dev "
        + base64.b64encode(bytes.fromhex(log_id[:8]) + checkpoint_signature).decode(
            "ascii"
        )
        + "\n"
    )
    entry: dict[str, object] = {
        "body": body_b64,
        "integratedTime": issued_at + 1,
        "logID": log_id,
        "logIndex": 0,
        "verification": {
            "inclusionProof": {
                "checkpoint": checkpoint,
                "hashes": [],
                "logIndex": 0,
                "rootHash": root.hex(),
                "treeSize": 1,
            },
            "signedEntryTimestamp": "",
        },
    }
    set_payload = json.dumps(
        {
            "body": body_b64,
            "integratedTime": issued_at + 1,
            "logID": log_id,
            "logIndex": 0,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    verification = entry["verification"]
    assert isinstance(verification, dict)
    verification["signedEntryTimestamp"] = base64.b64encode(
        log_key.sign(set_payload, ec.ECDSA(hashes.SHA256()))
    ).decode("ascii")
    rekor_uuid = f"{123:016x}{root.hex()}"
    bundle = build_acoustic_a1_transparency_bundle(
        statement=statement,
        producer_signature=signature,
        producer_certificate_pem=certificate,
        rekor_uuid=rekor_uuid,
        rekor_entry=entry,
        trusted_log_public_key_pem=log_public_pem,
    )
    return bundle, log_public_pem


@pytest.mark.asyncio
async def test_a1_accepts_calibrated_nonlinear_physical_transfer() -> None:
    driver = _TransferDriver()
    config = AcousticAcceptanceConfig(campaign_id="fixture-positive")

    receipt = await run_acoustic_a1_acceptance(driver, config)

    open_trials = tuple(
        trial for trial in receipt.trials if trial.arm is AcousticTrialArm.OPEN_LOOP
    )
    closed_trials = tuple(
        trial for trial in receipt.trials if trial.arm is AcousticTrialArm.CLOSED_LOOP
    )
    assert receipt.accepted is True
    assert receipt.error_reduction >= 0.5
    assert receipt.closed_loop_mae_db < receipt.open_loop_mae_db
    assert len(open_trials) == len(closed_trials) == 9
    assert receipt.trials[-1].arm is AcousticTrialArm.RESTORATION
    assert driver.calls[-1][0] == 0.0
    assert receipt.to_dict()["raw_audio_retained"] is False
    assert receipt.sha256.startswith("sha256:")
    assert len(receipt.sha256) == 71


@pytest.mark.asyncio
async def test_a1_receipt_and_campaign_record_round_trip_derived_verdicts() -> None:
    receipt = await run_acoustic_a1_acceptance(
        _TransferDriver(),
        AcousticAcceptanceConfig(campaign_id="fixture-round-trip"),
    )
    restored = AcousticA1AcceptanceReceipt.from_dict(receipt.to_dict())
    governance = {
        "schema": "aura.reality_reach.acceptance_governance.v1",
        "action_id": "acceptance.fixture-round-trip",
        "request_digest": "sha256:" + "1" * 64,
        "will_receipt_id": "will.fixture",
        "post_action_receipt_id": "post.fixture",
        "post_action_output_hash": "sha256:" + "2" * 64,
        "status": "success_verified",
        "transport_succeeded": True,
        "effect_verified": True,
        "receipt_persisted": True,
        "welfare_transaction_completed": True,
    }
    record = AcousticA1CampaignRecord(
        campaign_id=receipt.campaign_id,
        adapter_id="macos_acoustic.macos.acoustic.reference_tone_dbfs.adapter",
        source_commit_sha256="sha256:" + "3" * 64,
        workspace_state_sha256="sha256:" + "4" * 64,
        physical_identity_sha256="sha256:" + "5" * 64,
        mandate_sha256="sha256:" + "6" * 64,
        receipt=receipt,
        governance_evidence=governance,
        started_at_ns=receipt.trials[0].captured_at_ns,
        completed_at_ns=receipt.completed_at_ns,
    )

    restored_record = AcousticA1CampaignRecord.from_dict(record.to_dict())

    assert restored == receipt
    assert restored_record == record
    assert restored_record.accepted is True
    tampered = receipt.to_dict()
    tampered["accepted"] = False
    with pytest.raises(AcousticAcceptanceError, match="derived_field_invalid"):
        AcousticA1AcceptanceReceipt.from_dict(tampered)


@pytest.mark.asyncio
async def test_a1_requires_distinct_valid_external_witness_roots(tmp_path) -> None:
    receipt = await run_acoustic_a1_acceptance(
        _TransferDriver(),
        AcousticAcceptanceConfig(campaign_id="fixture-external-witness"),
    )
    governance = {
        "schema": "aura.reality_reach.acceptance_governance.v1",
        "action_id": "acceptance.fixture-external-witness",
        "request_digest": "sha256:" + "1" * 64,
        "will_receipt_id": "will.fixture",
        "post_action_receipt_id": "post.fixture",
        "post_action_output_hash": "sha256:" + "2" * 64,
        "status": "success_verified",
        "transport_succeeded": True,
        "effect_verified": True,
        "receipt_persisted": True,
        "welfare_transaction_completed": True,
    }
    adapter_id = "macos_acoustic.macos.acoustic.reference_tone_dbfs.adapter"
    source_sha256 = "sha256:" + "3" * 64
    physical_sha256 = "sha256:" + "5" * 64
    mandate = AcceptanceVerificationMandate(
        campaign_id=receipt.campaign_id,
        connector_id=ACOUSTIC_A1_CONNECTOR_ID,
        adapter_id=adapter_id,
        expected_source_commit_sha256=source_sha256,
        expected_physical_identity_sha256=physical_sha256,
        expected_evidence_class=AcceptanceEvidenceClass.LIVE,
        target=0.5,
        target_tolerance=0.0,
        scenario_id="",
        expected_live_channel_ids=(
            "macos_acoustic.macos.acoustic.reference_tone_dbfs.readback",
        ),
        expected_simulated_channel_ids=(),
        required_cases=ACOUSTIC_A1_REQUIRED_CASES,
        provisioned_at_ns=receipt.trials[0].captured_at_ns - 1,
        custody_sequence=1,
    )
    record = AcousticA1CampaignRecord(
        campaign_id=receipt.campaign_id,
        adapter_id=adapter_id,
        source_commit_sha256=source_sha256,
        workspace_state_sha256="sha256:" + "4" * 64,
        physical_identity_sha256=physical_sha256,
        mandate_sha256=mandate.sha256,
        receipt=receipt,
        governance_evidence=governance,
        started_at_ns=receipt.trials[0].captured_at_ns,
        completed_at_ns=receipt.completed_at_ns,
    )
    metrology_key = Ed25519PrivateKey.generate()
    governance_key = Ed25519PrivateKey.generate()
    metrology = _witness_bundle(
        AcceptanceWitnessRole.METROLOGY,
        metrology_key,
        record=record,
        mandate=mandate,
        evidence_sha256=receipt.sha256,
    )
    governed = _witness_bundle(
        AcceptanceWitnessRole.GOVERNANCE,
        governance_key,
        record=record,
        mandate=mandate,
        evidence_sha256=str(sha256_hex(canonical_json(governance))),
    )

    verified = verify_acoustic_a1_with_external_witnesses(
        record,
        mandate,
        metrology_witness_bundle=metrology,
        governance_witness_bundle=governed,
        metrology_witness_key_sha256=metrology.public_key_sha256,
        governance_witness_key_sha256=governed.public_key_sha256,
        now_ns=record.completed_at_ns + 2,
    )
    shared_root = verify_acoustic_a1_with_external_witnesses(
        record,
        mandate,
        metrology_witness_bundle=metrology,
        governance_witness_bundle=_witness_bundle(
            AcceptanceWitnessRole.GOVERNANCE,
            metrology_key,
            record=record,
            mandate=mandate,
            evidence_sha256=str(sha256_hex(canonical_json(governance))),
        ),
        metrology_witness_key_sha256=metrology.public_key_sha256,
        governance_witness_key_sha256=metrology.public_key_sha256,
        now_ns=record.completed_at_ns + 2,
    )

    assert verified.accepted is True
    assert verified.blockers == ()
    assert verified.campaign_id == record.campaign_id
    assert shared_root.accepted is False
    assert "external_witness_roots_not_distinct" in shared_root.blockers
    receipt_path = tmp_path / "external-a1.json"
    assert persist_externally_witnessed_acoustic_a1_receipt(
        verified,
        receipt_path,
    ) is True
    assert persist_externally_witnessed_acoustic_a1_receipt(
        verified,
        receipt_path,
    ) is False
    with pytest.raises(AcousticAcceptanceError, match="receipt_collision"):
        persist_externally_witnessed_acoustic_a1_receipt(
            replace(verified, blockers=("tampered",)),
            receipt_path,
        )

    issued_at = 1_785_082_400
    statement = build_acoustic_a1_transparency_statement(
        verified,
        sequence=1,
        previous_statement_sha256=TRANSPARENCY_ZERO_SHA256,
        previous_rekor_uuid=None,
        issued_at_unix=issued_at,
    )
    bundle, log_public_pem = _transparency_bundle(statement)
    transparent = verify_transparently_logged_acoustic_a1(
        verified,
        transparency_bundle=bundle,
        trusted_log_public_key_pem=log_public_pem,
        expected_sequence=1,
        expected_previous_statement_sha256=TRANSPARENCY_ZERO_SHA256,
        expected_previous_rekor_uuid=None,
    )
    assert transparent.accepted is True
    assert transparent.rekor_log_index == 0
    assert transparent.rekor_integrated_time == issued_at + 1

    missing = verify_transparently_logged_acoustic_a1(
        verified,
        transparency_bundle=None,
        trusted_log_public_key_pem=log_public_pem,
        expected_sequence=1,
        expected_previous_statement_sha256=TRANSPARENCY_ZERO_SHA256,
        expected_previous_rekor_uuid=None,
    )
    rollback = verify_transparently_logged_acoustic_a1(
        verified,
        transparency_bundle=bundle,
        trusted_log_public_key_pem=log_public_pem,
        expected_sequence=1,
        expected_previous_statement_sha256=TRANSPARENCY_ZERO_SHA256,
        expected_previous_rekor_uuid=None,
        minimum_log_index=0,
    )
    rebound = verify_transparently_logged_acoustic_a1(
        replace(verified, campaign_id="fixture-rebound"),
        transparency_bundle=bundle,
        trusted_log_public_key_pem=log_public_pem,
        expected_sequence=1,
        expected_previous_statement_sha256=TRANSPARENCY_ZERO_SHA256,
        expected_previous_rekor_uuid=None,
    )
    assert missing.accepted is False
    assert missing.blockers == ("acceptance_transparency_bundle_missing",)
    assert rollback.accepted is False
    assert rollback.blockers == ("acceptance_transparency_log_index_rollback",)
    assert rebound.accepted is False
    assert rebound.blockers == (
        "acceptance_transparency_statement_binding_invalid",
    )

    transparent_path = tmp_path / "transparent-a1.json"
    assert persist_transparently_logged_acoustic_a1_receipt(
        transparent,
        transparent_path,
    ) is True
    assert persist_transparently_logged_acoustic_a1_receipt(
        transparent,
        transparent_path,
    ) is False
    with pytest.raises(AcousticAcceptanceError, match="receipt_collision"):
        persist_transparently_logged_acoustic_a1_receipt(
            replace(transparent, blockers=("tampered",)),
            transparent_path,
        )


@pytest.mark.asyncio
async def test_a1_rejects_when_calibration_does_not_beat_equal_work_control() -> None:
    driver = _TransferDriver(transfer="nominal")
    config = AcousticAcceptanceConfig(campaign_id="fixture-null")

    receipt = await run_acoustic_a1_acceptance(driver, config)

    assert receipt.accepted is False
    assert "acoustic_a1_error_reduction_below_threshold" in receipt.blockers
    assert driver.calls[-1][0] == 0.0


@pytest.mark.asyncio
async def test_a1_schedule_is_deterministic_and_blind_between_arms() -> None:
    config = AcousticAcceptanceConfig(campaign_id="fixture-schedule")
    first = await run_acoustic_a1_acceptance(_TransferDriver(), config)
    second = await run_acoustic_a1_acceptance(_TransferDriver(), config)

    first_schedule = tuple(
        (trial.arm, trial.target_dbfs, trial.repeat)
        for trial in first.trials
        if trial.arm in {AcousticTrialArm.OPEN_LOOP, AcousticTrialArm.CLOSED_LOOP}
    )
    second_schedule = tuple(
        (trial.arm, trial.target_dbfs, trial.repeat)
        for trial in second.trials
        if trial.arm in {AcousticTrialArm.OPEN_LOOP, AcousticTrialArm.CLOSED_LOOP}
    )
    assert first_schedule == second_schedule
    assert first_schedule != tuple(sorted(first_schedule, key=lambda item: item[0].value))


@pytest.mark.asyncio
async def test_a1_bounds_trial_identity_for_maximum_campaign_id() -> None:
    driver = _TransferDriver()

    receipt = await run_acoustic_a1_acceptance(
        driver,
        AcousticAcceptanceConfig(campaign_id="a" * 128),
    )

    assert receipt.accepted is True
    assert all(len(trial_id) <= 128 for _amplitude, trial_id in driver.calls)


@pytest.mark.asyncio
async def test_a1_restores_silence_after_measurement_failure() -> None:
    driver = _TransferDriver(fail_at_call=16)
    config = AcousticAcceptanceConfig(campaign_id="fixture-failure")

    with pytest.raises(RuntimeError, match="fixture_measurement_failed"):
        await run_acoustic_a1_acceptance(driver, config)

    assert driver.calls[-1][0] == 0.0
    assert driver.calls[-1][1].endswith("restoration.0017")


@pytest.mark.asyncio
async def test_a1_reports_restoration_failure_without_claiming_acceptance() -> None:
    calibration_calls = 7 * 2
    heldout_calls = 3 * 3 * 2
    driver = _TransferDriver(fail_at_call=calibration_calls + heldout_calls + 1)
    config = AcousticAcceptanceConfig(campaign_id="fixture-restore-failure")

    with pytest.raises(AcousticAcceptanceError, match="acoustic_a1_restoration_failed"):
        await run_acoustic_a1_acceptance(driver, config)


def test_a1_config_and_receipt_validation_fail_closed() -> None:
    with pytest.raises(ValueError, match="calibration amplitudes"):
        AcousticAcceptanceConfig(
            campaign_id="bad-amplitudes",
            calibration_amplitudes=(0.0, 0.02, 0.01, 0.08),
        )
    with pytest.raises(ValueError, match="trial repeats"):
        AcousticAcceptanceConfig(campaign_id="bad-repeats", heldout_repeats=0)

    config = AcousticAcceptanceConfig(campaign_id="valid")
    assert config.sha256.startswith("sha256:")
    assert len(config.sha256) == 71
    assert replace(config, required_error_reduction=0.75).required_error_reduction == 0.75
