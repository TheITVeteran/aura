from __future__ import annotations

import base64
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.reality_reach.acceptance import (
    ACCEPTANCE_GOVERNANCE_SCHEMA,
    REQUIRED_SCALAR_ACCEPTANCE_CASES,
    AcceptanceCaseResult,
    AcceptanceEvidenceClass,
    AcceptanceVerdict,
    ConnectorAcceptanceCertificate,
)
from core.reality_reach.acceptance_mandate import AcceptanceVerificationMandate
from core.reality_reach.acceptance_witness import (
    ZERO_SHA256,
    AcceptanceWitnessBundle,
    AcceptanceWitnessRole,
    AcceptanceWitnessStatement,
    persist_externally_witnessed_acceptance_receipt,
    verify_acceptance_with_external_witnesses,
    verify_acceptance_witness_bundle,
)
from core.reality_reach.metrology import (
    AcquisitionMode,
    AcquisitionReceipt,
    EvidenceSource,
    Measurement,
    MeasurementSummary,
)
from core.runtime.audit_chain import canonical_json, sha256_hex


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _case_evidence() -> dict[str, dict[str, Any]]:
    prepare = {
        "preparation_id": "prep-1",
        "command_sha256": _digest("command"),
        "lease_sha256": _digest("lease"),
        "precondition_sha256": _digest("precondition"),
        "rollback_token_sha256": _digest("rollback-token"),
    }
    dispatch = {
        "state": "executed",
        "accepted": True,
        "transport_completed": True,
        "executed": True,
        "command_sha256": prepare["command_sha256"],
        "preparation_sha256": _digest(prepare),
    }
    return {
        "observation.fresh": {
            "status": "available",
            "value": 7.0,
            "source_event_id": "event-1",
            "channel_id": "fixture.live",
        },
        "cancellation.pre_dispatch": {
            "state": "cancelled",
            "executed": False,
            "transport_completed": False,
        },
        "actuation.prepare": prepare,
        "actuation.dispatch": dispatch,
        "effect.independent_readback": {
            "state": "effect_verified",
            "independently_observed": True,
            "observation_sha256": _digest("effect"),
            "command_sha256": prepare["command_sha256"],
            "actuation_receipt_sha256": _digest(dispatch),
        },
        "restoration.rollback": {
            "state": "rolled_back",
            "independently_observed": True,
            "safe_state_observation_sha256": _digest("safe-state"),
            "command_sha256": prepare["command_sha256"],
            "actuation_receipt_sha256": _digest(dispatch),
        },
    }


def _metrology() -> AcquisitionReceipt:
    measurement = Measurement(
        channel_id="fixture.live",
        value=7.0,
        unit="percent",
        captured_at_ns=2_500,
        source=EvidenceSource.LIVE,
        scenario_id="",
        wall_clock_source="fixture.clock",
        random_uncertainty=0.1,
        resolution_uncertainty=0.1,
        systematic_uncertainty=0.1,
        calibration_sha256="",
        reading_sha256=_digest("reading"),
    )
    summary = MeasurementSummary(
        channel_id=measurement.channel_id,
        unit=measurement.unit,
        sample_count=1,
        mean=measurement.value,
        minimum=measurement.value,
        maximum=measurement.value,
        standard_uncertainty=measurement.standard_uncertainty,
        coverage_factor=2.0,
        expanded_uncertainty_k2=2.0 * measurement.standard_uncertainty,
        source=measurement.source,
        wall_clock_source=measurement.wall_clock_source,
        calibration_sha256="",
    )
    body = {
        "run_id": "metrology.witness.1",
        "task_sha256": _digest("task"),
        "mode": AcquisitionMode.LIVE.value,
        "mode_generation": 1,
        "started_at_ns": 1_500,
        "completed_at_ns": 3_500,
        "sample_sets": 1,
        "maximum_observed_skew_ns": 1,
        "scenario_id": "",
        "measurements": [measurement.to_dict()],
        "summaries": [summary.to_dict()],
    }
    return AcquisitionReceipt(
        run_id=body["run_id"],
        task_sha256=body["task_sha256"],
        mode=AcquisitionMode.LIVE,
        mode_generation=1,
        started_at_ns=1_500,
        completed_at_ns=3_500,
        sample_sets=1,
        maximum_observed_skew_ns=1,
        scenario_id="",
        measurements=(measurement,),
        summaries=(summary,),
        evidence_sha256=_digest(body),
    )


def _governance() -> dict[str, Any]:
    return {
        "schema": ACCEPTANCE_GOVERNANCE_SCHEMA,
        "action_id": "acceptance.witness.fixture",
        "request_digest": _digest("request"),
        "will_receipt_id": "will.witness.fixture",
        "post_action_receipt_id": "post.witness.fixture",
        "post_action_output_hash": _digest("output"),
        "status": "success_verified",
        "transport_succeeded": True,
        "effect_verified": True,
        "receipt_persisted": True,
        "welfare_transaction_completed": True,
    }


def _campaign() -> tuple[
    AcceptanceVerificationMandate,
    ConnectorAcceptanceCertificate,
    dict[str, Any],
]:
    mandate = AcceptanceVerificationMandate(
        campaign_id="campaign.external-witness",
        connector_id="connector.fixture",
        adapter_id="adapter.fixture",
        expected_source_commit_sha256=_digest("source"),
        expected_physical_identity_sha256=_digest("device"),
        expected_evidence_class=AcceptanceEvidenceClass.LIVE,
        target=7.0,
        target_tolerance=0.2,
        scenario_id="",
        required_cases=REQUIRED_SCALAR_ACCEPTANCE_CASES,
        provisioned_at_ns=1_000,
        custody_sequence=1,
    )
    case_evidence = _case_evidence()
    cases = tuple(
        AcceptanceCaseResult(
            case_id=case_id,
            verdict=AcceptanceVerdict.PASS,
            evidence_class=AcceptanceEvidenceClass.LIVE,
            required=True,
            evidence_sha256=_digest(case_evidence[case_id]),
            duration_ms=1.0,
        )
        for case_id in REQUIRED_SCALAR_ACCEPTANCE_CASES
    )
    metrology = _metrology()
    governance = _governance()
    certificate = ConnectorAcceptanceCertificate(
        campaign_id=mandate.campaign_id,
        connector_id=mandate.connector_id,
        adapter_id=mandate.adapter_id,
        physical_identity_sha256=mandate.expected_physical_identity_sha256,
        source_commit_sha256=mandate.expected_source_commit_sha256,
        target=mandate.target,
        target_tolerance=mandate.target_tolerance,
        started_at_ns=2_000,
        completed_at_ns=3_000,
        cases=cases,
        metrology_evidence_sha256=metrology.evidence_sha256,
        governance_evidence_sha256=_digest(governance),
        governance_accepted=True,
    )
    evidence = {
        "case_evidence": case_evidence,
        "metrology_receipt": metrology.to_dict(),
        "governance_evidence": governance,
    }
    return mandate, certificate, evidence


def _bundle(
    key: Ed25519PrivateKey,
    *,
    role: AcceptanceWitnessRole,
    mandate: AcceptanceVerificationMandate,
    certificate: ConnectorAcceptanceCertificate,
    evidence_sha256: str,
    witnessed_at_ns: int = 4_000,
    sequence: int = 1,
    previous: str = ZERO_SHA256,
) -> AcceptanceWitnessBundle:
    statement = AcceptanceWitnessStatement(
        role=role,
        witness_id=f"external.{role.value}.fixture",
        campaign_id=mandate.campaign_id,
        mandate_sha256=mandate.sha256,
        certificate_sha256=certificate.sha256,
        evidence_sha256=evidence_sha256,
        sequence=sequence,
        previous_statement_sha256=previous,
        witnessed_at_ns=witnessed_at_ns,
    )
    public_raw = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    unsigned = AcceptanceWitnessBundle(
        statement=statement,
        public_key_raw_b64=base64.b64encode(public_raw).decode("ascii"),
        signature_b64=base64.b64encode(b"\x00" * 64).decode("ascii"),
    )
    return replace(
        unsigned,
        signature_b64=base64.b64encode(key.sign(unsigned.signed_payload())).decode(
            "ascii"
        ),
    )


def test_two_distinct_external_roots_promote_live_acceptance() -> None:
    mandate, certificate, evidence = _campaign()
    metrology_key = Ed25519PrivateKey.generate()
    governance_key = Ed25519PrivateKey.generate()
    metrology = _bundle(
        metrology_key,
        role=AcceptanceWitnessRole.METROLOGY,
        mandate=mandate,
        certificate=certificate,
        evidence_sha256=certificate.metrology_evidence_sha256,
    )
    governance = _bundle(
        governance_key,
        role=AcceptanceWitnessRole.GOVERNANCE,
        mandate=mandate,
        certificate=certificate,
        evidence_sha256=certificate.governance_evidence_sha256,
    )

    receipt = verify_acceptance_with_external_witnesses(
        certificate,
        evidence,
        mandate,
        metrology_witness_bundle=metrology.to_dict(),
        governance_witness_bundle=governance.to_dict(),
        metrology_witness_key_sha256=metrology.public_key_sha256,
        governance_witness_key_sha256=governance.public_key_sha256,
        now_ns=5_000,
    )

    assert receipt.accepted is True
    assert receipt.blockers == ()
    assert receipt.mandate_verification.accepted is True
    assert receipt.metrology_witness_bundle_sha256 == metrology.sha256
    assert receipt.governance_witness_bundle_sha256 == governance.sha256


def test_physical_acceptance_cannot_promote_from_raw_producer_digests() -> None:
    mandate, certificate, evidence = _campaign()

    receipt = verify_acceptance_with_external_witnesses(
        certificate,
        evidence,
        mandate,
        now_ns=5_000,
    )

    assert receipt.accepted is False
    assert set(receipt.blockers) == {
        "external_governance_witness_missing",
        "external_metrology_witness_missing",
    }
    assert "trusted_metrology_missing" in receipt.mandate_verification.verification.blockers
    assert "trusted_governance_missing" in receipt.mandate_verification.verification.blockers


def test_witness_signature_tamper_and_wrong_pinned_root_fail_closed() -> None:
    mandate, certificate, _ = _campaign()
    signer = Ed25519PrivateKey.generate()
    other = Ed25519PrivateKey.generate()
    bundle = _bundle(
        signer,
        role=AcceptanceWitnessRole.METROLOGY,
        mandate=mandate,
        certificate=certificate,
        evidence_sha256=certificate.metrology_evidence_sha256,
    )
    wrong_root = _bundle(
        other,
        role=AcceptanceWitnessRole.METROLOGY,
        mandate=mandate,
        certificate=certificate,
        evidence_sha256=certificate.metrology_evidence_sha256,
    )
    tampered = replace(
        bundle,
        signature_b64=base64.b64encode(b"\x01" * 64).decode("ascii"),
    )

    for candidate, trusted, expected in (
        (tampered, bundle.public_key_sha256, "acceptance_witness_signature_invalid"),
        (bundle, wrong_root.public_key_sha256, "acceptance_witness_trust_root_mismatch"),
    ):
        try:
            verify_acceptance_witness_bundle(
                candidate,
                expected_role=AcceptanceWitnessRole.METROLOGY,
                expected_public_key_sha256=trusted,
                mandate=mandate,
                certificate=certificate,
                expected_evidence_sha256=certificate.metrology_evidence_sha256,
                expected_sequence=1,
                expected_previous_statement_sha256=ZERO_SHA256,
                now_ns=5_000,
            )
        except ValueError as exc:
            assert str(exc) == expected
        else:
            raise AssertionError("untrusted witness was accepted")


def test_witness_role_sequence_predecessor_and_time_are_bound() -> None:
    mandate, certificate, _ = _campaign()
    key = Ed25519PrivateKey.generate()
    predecessor = _digest("prior-witness")
    bundle = _bundle(
        key,
        role=AcceptanceWitnessRole.METROLOGY,
        mandate=mandate,
        certificate=certificate,
        evidence_sha256=certificate.metrology_evidence_sha256,
        witnessed_at_ns=certificate.completed_at_ns - 1,
        sequence=2,
        previous=predecessor,
    )

    for role, sequence, previous, expected in (
        (
            AcceptanceWitnessRole.GOVERNANCE,
            2,
            predecessor,
            "acceptance_witness_role_mismatch",
        ),
        (
            AcceptanceWitnessRole.METROLOGY,
            3,
            predecessor,
            "acceptance_witness_sequence_mismatch",
        ),
        (
            AcceptanceWitnessRole.METROLOGY,
            2,
            _digest("wrong-prior"),
            "acceptance_witness_predecessor_mismatch",
        ),
        (
            AcceptanceWitnessRole.METROLOGY,
            2,
            predecessor,
            "acceptance_witness_predates_campaign_completion",
        ),
    ):
        try:
            verify_acceptance_witness_bundle(
                bundle,
                expected_role=role,
                expected_public_key_sha256=bundle.public_key_sha256,
                mandate=mandate,
                certificate=certificate,
                expected_evidence_sha256=certificate.metrology_evidence_sha256,
                expected_sequence=sequence,
                expected_previous_statement_sha256=previous,
                now_ns=5_000,
            )
        except ValueError as exc:
            assert str(exc) == expected
        else:
            raise AssertionError("invalid witness lineage was accepted")


def test_metrology_and_governance_must_use_distinct_roots() -> None:
    mandate, certificate, evidence = _campaign()
    shared_key = Ed25519PrivateKey.generate()
    metrology = _bundle(
        shared_key,
        role=AcceptanceWitnessRole.METROLOGY,
        mandate=mandate,
        certificate=certificate,
        evidence_sha256=certificate.metrology_evidence_sha256,
    )
    governance = _bundle(
        shared_key,
        role=AcceptanceWitnessRole.GOVERNANCE,
        mandate=mandate,
        certificate=certificate,
        evidence_sha256=certificate.governance_evidence_sha256,
    )

    receipt = verify_acceptance_with_external_witnesses(
        certificate,
        evidence,
        mandate,
        metrology_witness_bundle=metrology,
        governance_witness_bundle=governance,
        metrology_witness_key_sha256=metrology.public_key_sha256,
        governance_witness_key_sha256=governance.public_key_sha256,
        now_ns=5_000,
    )

    assert receipt.accepted is False
    assert receipt.blockers == ("external_witness_roots_not_distinct",)


def test_external_receipt_is_private_create_once_and_collision_safe(
    tmp_path: Path,
) -> None:
    mandate, certificate, evidence = _campaign()
    metrology_key = Ed25519PrivateKey.generate()
    governance_key = Ed25519PrivateKey.generate()
    metrology = _bundle(
        metrology_key,
        role=AcceptanceWitnessRole.METROLOGY,
        mandate=mandate,
        certificate=certificate,
        evidence_sha256=certificate.metrology_evidence_sha256,
    )
    governance = _bundle(
        governance_key,
        role=AcceptanceWitnessRole.GOVERNANCE,
        mandate=mandate,
        certificate=certificate,
        evidence_sha256=certificate.governance_evidence_sha256,
    )
    receipt = verify_acceptance_with_external_witnesses(
        certificate,
        evidence,
        mandate,
        metrology_witness_bundle=metrology,
        governance_witness_bundle=governance,
        metrology_witness_key_sha256=metrology.public_key_sha256,
        governance_witness_key_sha256=governance.public_key_sha256,
        now_ns=5_000,
    )
    path = tmp_path / "receipts" / "external.json"

    assert persist_externally_witnessed_acceptance_receipt(receipt, path) is True
    assert persist_externally_witnessed_acceptance_receipt(receipt, path) is False
    assert path.stat().st_mode & 0o077 == 0

    attacked = replace(receipt, blockers=("post_result_substitution",))
    with pytest.raises(ValueError, match="external_acceptance_receipt_collision"):
        persist_externally_witnessed_acceptance_receipt(attacked, path)


def test_operator_assembler_verifies_detached_signature(tmp_path: Path) -> None:
    mandate, certificate, _ = _campaign()
    key = Ed25519PrivateKey.generate()
    bundle = _bundle(
        key,
        role=AcceptanceWitnessRole.METROLOGY,
        mandate=mandate,
        certificate=certificate,
        evidence_sha256=certificate.metrology_evidence_sha256,
    )
    statement_path = tmp_path / "statement.json"
    public_key_path = tmp_path / "public.raw"
    signature_path = tmp_path / "signature.raw"
    output_path = tmp_path / "bundle.json"
    statement_path.write_text(
        json.dumps(bundle.statement.to_dict()),
        encoding="utf-8",
    )
    public_key_path.write_bytes(
        base64.b64decode(bundle.public_key_raw_b64, validate=True)
    )
    signature_path.write_bytes(base64.b64decode(bundle.signature_b64, validate=True))

    command = [
        sys.executable,
        "tools/reality_reach/manage_acceptance_witness.py",
        "assemble",
        "--statement",
        str(statement_path),
        "--public-key-raw",
        str(public_key_path),
        "--signature",
        str(signature_path),
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=20)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output_path.read_text(encoding="utf-8")) == bundle.to_dict()

    signature_path.write_bytes(b"\x01" * 64)
    failed = subprocess.run(
        [*command[:-1], str(tmp_path / "tampered.json")],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert failed.returncode != 0
    assert "acceptance_witness_signature_invalid" in failed.stderr
