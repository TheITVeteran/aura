from __future__ import annotations

import base64
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import NameOID

from core.reality_reach.acceptance import (
    ACCEPTANCE_GOVERNANCE_SCHEMA,
    REQUIRED_SCALAR_ACCEPTANCE_CASES,
    AcceptanceCaseResult,
    AcceptanceCertificateStore,
    AcceptanceEvidenceClass,
    AcceptanceVerdict,
    ConnectorAcceptanceCertificate,
)
from core.reality_reach.acceptance_mandate import (
    AcceptanceMandateProvisionReceipt,
    AcceptanceVerificationMandate,
)
from core.reality_reach.acceptance_preregistration import (
    PreregisteredAcceptanceReceipt,
)
from core.reality_reach.acceptance_transparency import (
    ZERO_SHA256 as TRANSPARENCY_ZERO_SHA256,
)
from core.reality_reach.acceptance_transparency import (
    build_acceptance_transparency_bundle,
    build_acceptance_transparency_statement,
    verify_transparently_logged_acceptance,
)
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
from tools.reality_reach import manage_acceptance_witness as witness_tool

PROVISIONED_AT_NS = 1_000_000_000
METROLOGY_STARTED_AT_NS = 1_500_000_000
CAMPAIGN_STARTED_AT_NS = 2_000_000_000
MEASUREMENT_CAPTURED_AT_NS = 2_500_000_000
CAMPAIGN_COMPLETED_AT_NS = 3_000_000_000
METROLOGY_COMPLETED_AT_NS = 3_500_000_000
WITNESSED_AT_NS = 4_000_000_000
VERIFIED_AT_NS = 5_000_000_000


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _transparency_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


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
        captured_at_ns=MEASUREMENT_CAPTURED_AT_NS,
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
        "started_at_ns": METROLOGY_STARTED_AT_NS,
        "completed_at_ns": METROLOGY_COMPLETED_AT_NS,
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
        started_at_ns=METROLOGY_STARTED_AT_NS,
        completed_at_ns=METROLOGY_COMPLETED_AT_NS,
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
        expected_live_channel_ids=("fixture.live",),
        expected_simulated_channel_ids=(),
        required_cases=REQUIRED_SCALAR_ACCEPTANCE_CASES,
        provisioned_at_ns=PROVISIONED_AT_NS,
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
        started_at_ns=CAMPAIGN_STARTED_AT_NS,
        completed_at_ns=CAMPAIGN_COMPLETED_AT_NS,
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


def _preregistration(
    mandate: AcceptanceVerificationMandate,
    certificate: ConnectorAcceptanceCertificate,
) -> PreregisteredAcceptanceReceipt:
    provision_receipt = AcceptanceMandateProvisionReceipt(
        campaign_id=mandate.campaign_id,
        mandate_sha256=mandate.sha256,
        contract_sha256=mandate.contract_sha256,
        custody_identity_sha256=_digest("mandate-custody"),
        provisioned_at_ns=mandate.provisioned_at_ns,
        created=True,
        custody_sequence=mandate.custody_sequence,
    )
    return PreregisteredAcceptanceReceipt(
        mandate=mandate,
        provision_receipt=provision_receipt,
        transparency_bundle_sha256=_digest("preregistration-bundle"),
        trusted_log_key_sha256=_digest("preregistration-log-key"),
        rekor_uuid="0" * 80,
        rekor_log_index=0,
        rekor_integrated_time=0,
        campaign_started_at_ns=certificate.started_at_ns,
        blockers=(),
    )


def _bundle(
    key: Ed25519PrivateKey,
    *,
    role: AcceptanceWitnessRole,
    mandate: AcceptanceVerificationMandate,
    certificate: ConnectorAcceptanceCertificate,
    evidence_sha256: str,
    witnessed_at_ns: int = WITNESSED_AT_NS,
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


def _external_receipt():
    mandate, certificate, evidence = _campaign()
    metrology = _bundle(
        Ed25519PrivateKey.generate(),
        role=AcceptanceWitnessRole.METROLOGY,
        mandate=mandate,
        certificate=certificate,
        evidence_sha256=certificate.metrology_evidence_sha256,
    )
    governance = _bundle(
        Ed25519PrivateKey.generate(),
        role=AcceptanceWitnessRole.GOVERNANCE,
        mandate=mandate,
        certificate=certificate,
        evidence_sha256=certificate.governance_evidence_sha256,
    )
    receipt = verify_acceptance_with_external_witnesses(
        certificate,
        evidence,
        mandate,
        preregistration_receipt=_preregistration(mandate, certificate),
        metrology_witness_bundle=metrology,
        governance_witness_bundle=governance,
        metrology_witness_key_sha256=metrology.public_key_sha256,
        governance_witness_key_sha256=governance.public_key_sha256,
        now_ns=VERIFIED_AT_NS,
    )
    assert receipt.accepted is True
    return receipt


def test_physical_external_receipt_cannot_omit_preregistration_digest() -> None:
    receipt = _external_receipt()

    assert replace(receipt, preregistration_verification_sha256="").accepted is False


def _transparency_fixture():
    receipt = _external_receipt()
    issued_at = 1_785_082_400
    statement = build_acceptance_transparency_statement(
        receipt,
        sequence=1,
        previous_statement_sha256=TRANSPARENCY_ZERO_SHA256,
        previous_rekor_uuid=None,
        issued_at_unix=issued_at,
    )
    statement_bytes = json.dumps(
        statement,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    producer_key = Ed25519PrivateKey.generate()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Acceptance fixture")])
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
                "publicKey": {"content": base64.b64encode(certificate).decode("ascii")},
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
    entry = {
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
    entry["verification"]["signedEntryTimestamp"] = base64.b64encode(
        log_key.sign(set_payload, ec.ECDSA(hashes.SHA256()))
    ).decode("ascii")
    rekor_uuid = f"{123:016x}{root.hex()}"
    bundle = build_acceptance_transparency_bundle(
        statement=statement,
        producer_signature=signature,
        producer_certificate_pem=certificate,
        rekor_uuid=rekor_uuid,
        rekor_entry=entry,
        trusted_log_public_key_pem=log_public_pem,
    )
    return receipt, bundle, log_public_pem


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
        preregistration_receipt=_preregistration(mandate, certificate),
        metrology_witness_bundle=metrology.to_dict(),
        governance_witness_bundle=governance.to_dict(),
        metrology_witness_key_sha256=metrology.public_key_sha256,
        governance_witness_key_sha256=governance.public_key_sha256,
        now_ns=VERIFIED_AT_NS,
    )

    assert receipt.accepted is True
    assert receipt.blockers == ()
    assert receipt.mandate_verification.accepted is True
    assert receipt.preregistration_verification_sha256
    assert receipt.metrology_witness_bundle_sha256 == metrology.sha256
    assert receipt.governance_witness_bundle_sha256 == governance.sha256


def test_physical_acceptance_requires_verified_transparency_log_inclusion() -> None:
    receipt, bundle, log_public_pem = _transparency_fixture()

    missing = verify_transparently_logged_acceptance(
        receipt,
        transparency_bundle=None,
        trusted_log_public_key_pem=log_public_pem,
        expected_sequence=1,
        expected_previous_statement_sha256=TRANSPARENCY_ZERO_SHA256,
        expected_previous_rekor_uuid=None,
    )
    verified = verify_transparently_logged_acceptance(
        receipt,
        transparency_bundle=bundle,
        trusted_log_public_key_pem=log_public_pem,
        expected_sequence=1,
        expected_previous_statement_sha256=TRANSPARENCY_ZERO_SHA256,
        expected_previous_rekor_uuid=None,
    )

    assert missing.accepted is False
    assert missing.blockers == ("acceptance_transparency_bundle_missing",)
    assert verified.accepted is True
    assert verified.blockers == ()
    assert verified.rekor_log_index == 0
    assert verified.rekor_integrated_time == 1_785_082_401
    assert verified.transparency_bundle_sha256 == bundle["bundle_sha256"]


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        (
            lambda value: value["rekor_entry"]["verification"].__setitem__(
                "signedEntryTimestamp",
                base64.b64encode(b"forged").decode("ascii"),
            ),
            "acceptance_transparency_set_signature_invalid",
        ),
        (
            lambda value: value["rekor_entry"]["verification"][
                "inclusionProof"
            ].__setitem__("rootHash", "3" * 64),
            "acceptance_transparency_inclusion_proof_root_mismatch",
        ),
        (
            lambda value: value["statement"].__setitem__(
                "campaign_id",
                "campaign.substituted",
            ),
            "acceptance_transparency_statement_digest_invalid",
        ),
    ],
)
def test_transparency_log_tamper_fails_closed(mutation, blocker) -> None:
    receipt, bundle, log_public_pem = _transparency_fixture()
    attacked = copy.deepcopy(bundle)
    mutation(attacked)
    body = dict(attacked)
    body.pop("bundle_sha256")
    attacked["bundle_sha256"] = _transparency_digest(body)

    result = verify_transparently_logged_acceptance(
        receipt,
        transparency_bundle=attacked,
        trusted_log_public_key_pem=log_public_pem,
        expected_sequence=1,
        expected_previous_statement_sha256=TRANSPARENCY_ZERO_SHA256,
        expected_previous_rekor_uuid=None,
    )

    assert result.accepted is False
    assert result.blockers == (blocker,)


def test_transparency_rollback_pins_fail_closed() -> None:
    receipt, bundle, log_public_pem = _transparency_fixture()

    result = verify_transparently_logged_acceptance(
        receipt,
        transparency_bundle=bundle,
        trusted_log_public_key_pem=log_public_pem,
        expected_sequence=1,
        expected_previous_statement_sha256=TRANSPARENCY_ZERO_SHA256,
        expected_previous_rekor_uuid=None,
        minimum_log_index=0,
        minimum_integrated_time=1_785_082_402,
    )

    assert result.accepted is False
    assert result.blockers == ("acceptance_transparency_log_index_rollback",)


def test_physical_acceptance_cannot_promote_from_raw_producer_digests() -> None:
    mandate, certificate, evidence = _campaign()

    receipt = verify_acceptance_with_external_witnesses(
        certificate,
        evidence,
        mandate,
        now_ns=VERIFIED_AT_NS,
    )

    assert receipt.accepted is False
    assert set(receipt.blockers) == {
        "acceptance_preregistration_missing",
        "external_governance_witness_missing",
        "external_metrology_witness_missing",
    }
    assert "trusted_metrology_missing" in receipt.mandate_verification.verification.blockers
    assert "trusted_governance_missing" in receipt.mandate_verification.verification.blockers


def test_mandate_replay_rejects_readback_channel_substitution() -> None:
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
    substituted = copy.deepcopy(evidence)
    substituted["metrology_receipt"]["measurements"][0]["channel_id"] = (
        "fixture.substituted"
    )

    receipt = verify_acceptance_with_external_witnesses(
        certificate,
        substituted,
        mandate,
        preregistration_receipt=_preregistration(mandate, certificate),
        metrology_witness_bundle=metrology,
        governance_witness_bundle=governance,
        metrology_witness_key_sha256=metrology.public_key_sha256,
        governance_witness_key_sha256=governance.public_key_sha256,
        now_ns=VERIFIED_AT_NS,
    )

    assert receipt.accepted is False
    assert "mandate_live_channel_set_mismatch" in (
        receipt.mandate_verification.blockers
    )


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
                now_ns=VERIFIED_AT_NS,
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
                now_ns=VERIFIED_AT_NS,
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
        preregistration_receipt=_preregistration(mandate, certificate),
        metrology_witness_bundle=metrology,
        governance_witness_bundle=governance,
        metrology_witness_key_sha256=metrology.public_key_sha256,
        governance_witness_key_sha256=governance.public_key_sha256,
        now_ns=VERIFIED_AT_NS,
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
        preregistration_receipt=_preregistration(mandate, certificate),
        metrology_witness_bundle=metrology,
        governance_witness_bundle=governance,
        metrology_witness_key_sha256=metrology.public_key_sha256,
        governance_witness_key_sha256=governance.public_key_sha256,
        now_ns=VERIFIED_AT_NS,
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


def test_scalar_physical_witness_statement_requires_preregistration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mandate, certificate, _ = _campaign()
    store = AcceptanceCertificateStore(tmp_path / "acceptance")
    assert store.persist(certificate) is True

    class _MandateHandle:
        def get(self, campaign_id: str) -> AcceptanceVerificationMandate:
            assert campaign_id == mandate.campaign_id
            return mandate

        def close(self) -> None:
            return None

    class _MandateStoreFacade:
        @staticmethod
        def from_system(_path: Path) -> _MandateHandle:
            return _MandateHandle()

    monkeypatch.setattr(
        witness_tool,
        "AcceptanceMandateStore",
        _MandateStoreFacade,
    )
    common = [
        "statement",
        "--root",
        str(store.root),
        "--mandate-state",
        str(tmp_path / "mandates.encrypted.json"),
        "--campaign-id",
        mandate.campaign_id,
        "--role",
        AcceptanceWitnessRole.METROLOGY.value,
        "--witness-id",
        "metrology.external.fixture",
        "--sequence",
        "1",
        "--witnessed-at-ns",
        str(WITNESSED_AT_NS),
        "--statement-output",
        str(tmp_path / "statement.json"),
        "--payload-output",
        str(tmp_path / "statement.payload"),
    ]
    with pytest.raises(
        ValueError,
        match="acceptance_preregistration_arguments_missing",
    ):
        witness_tool.main(common)

    observed: list[int] = []

    def _verified(
        _args,
        actual_mandate: AcceptanceVerificationMandate,
        *,
        campaign_started_at_ns: int,
    ) -> PreregisteredAcceptanceReceipt:
        assert actual_mandate == mandate
        observed.append(campaign_started_at_ns)
        return _preregistration(mandate, certificate)

    monkeypatch.setattr(witness_tool, "_verified_preregistration", _verified)
    assert witness_tool.main(common) == 0
    statement = json.loads((tmp_path / "statement.json").read_text())
    assert statement["campaign_id"] == mandate.campaign_id
    assert statement["certificate_sha256"] == certificate.sha256
    assert observed == [certificate.started_at_ns]


def test_transparency_operator_assembles_only_verified_rekor_evidence(
    tmp_path: Path,
) -> None:
    _receipt, bundle, log_public_pem = _transparency_fixture()
    statement_path = tmp_path / "transparency-statement.json"
    signature_path = tmp_path / "producer-signature.raw"
    certificate_path = tmp_path / "producer-certificate.pem"
    entry_path = tmp_path / "rekor-entry.json"
    log_key_path = tmp_path / "rekor-log-key.pem"
    output_path = tmp_path / "transparency-bundle.json"
    statement_path.write_text(json.dumps(bundle["statement"]), encoding="utf-8")
    signature_path.write_bytes(
        base64.b64decode(bundle["producer_signature_b64"], validate=True)
    )
    certificate_path.write_bytes(
        base64.b64decode(bundle["producer_certificate_pem_b64"], validate=True)
    )
    entry_path.write_text(json.dumps(bundle["rekor_entry"]), encoding="utf-8")
    log_key_path.write_bytes(log_public_pem)
    command = [
        sys.executable,
        "tools/reality_reach/manage_acceptance_transparency.py",
        "assemble",
        "--statement",
        str(statement_path),
        "--producer-signature",
        str(signature_path),
        "--producer-certificate-pem",
        str(certificate_path),
        "--rekor-uuid",
        bundle["rekor_uuid"],
        "--rekor-entry",
        str(entry_path),
        "--trusted-log-public-key-pem",
        str(log_key_path),
        "--output",
        str(output_path),
    ]

    completed = subprocess.run(command, capture_output=True, text=True, timeout=20)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output_path.read_text(encoding="utf-8")) == bundle

    attacked_entry = copy.deepcopy(bundle["rekor_entry"])
    attacked_entry["verification"]["signedEntryTimestamp"] = base64.b64encode(
        b"forged"
    ).decode("ascii")
    entry_path.write_text(json.dumps(attacked_entry), encoding="utf-8")
    failed = subprocess.run(
        [*command[:-1], str(tmp_path / "attacked-bundle.json")],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert failed.returncode != 0
    assert "acceptance_transparency_set_signature_invalid" in failed.stderr
