from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.x509.oid import NameOID

from core.reality_reach.acceptance import AcceptanceEvidenceClass
from core.reality_reach.acceptance_mandate import (
    AcceptanceMandateError,
    AcceptanceMandateProvisionReceipt,
    AcceptanceVerificationMandate,
)
from core.reality_reach.acceptance_preregistration import (
    AcceptanceTrustPolicy,
    build_acceptance_preregistration_bundle,
    build_acceptance_preregistration_statement,
    persist_preregistered_acceptance_receipt,
    verify_acceptance_preregistration,
)
from core.reality_reach.acceptance_transparency import (
    ZERO_SHA256,
    AcceptanceTransparencyError,
)
from tools.reality_reach import manage_acceptance_preregistration as prereg_tool


def _mandate() -> tuple[
    AcceptanceVerificationMandate,
    AcceptanceMandateProvisionReceipt,
]:
    mandate = AcceptanceVerificationMandate(
        campaign_id="cp810.preregistered.fixture",
        connector_id="macos.acoustic.a1",
        adapter_id="macos_acoustic.fixture.adapter",
        expected_source_commit_sha256="sha256:" + "1" * 64,
        expected_physical_identity_sha256="sha256:" + "2" * 64,
        expected_evidence_class=AcceptanceEvidenceClass.LIVE,
        target=0.5,
        target_tolerance=0.0,
        scenario_id="",
        expected_live_channel_ids=("macos_acoustic.fixture.readback",),
        expected_simulated_channel_ids=(),
        required_cases=("calibration.monotone_transfer",),
        provisioned_at_ns=1_785_082_300_000_000_000,
        custody_sequence=7,
    )
    receipt = AcceptanceMandateProvisionReceipt(
        campaign_id=mandate.campaign_id,
        mandate_sha256=mandate.sha256,
        contract_sha256=mandate.contract_sha256,
        custody_identity_sha256="sha256:" + "3" * 64,
        provisioned_at_ns=mandate.provisioned_at_ns,
        created=True,
        custody_sequence=mandate.custody_sequence,
    )
    return mandate, receipt


def _rekor_bundle(
    statement: dict[str, object],
    *,
    log_key: ec.EllipticCurvePrivateKey | None = None,
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
    name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "Preregistration fixture")]
    )
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
    log_key = log_key or ec.generate_private_key(ec.SECP256R1())
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
        + "\n\u2014 rekor.sigstore.dev "
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
    bundle = build_acceptance_preregistration_bundle(
        statement=statement,
        producer_signature=signature,
        producer_certificate_pem=certificate,
        rekor_uuid=rekor_uuid,
        rekor_entry=entry,
        trusted_log_public_key_pem=log_public_pem,
    )
    return bundle, log_public_pem


def test_preregistration_proves_mandate_predates_campaign(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    mandate, provision_receipt = _mandate()
    assert (
        AcceptanceMandateProvisionReceipt.from_dict(provision_receipt.to_dict())
        == provision_receipt
    )
    issued_at = 1_785_082_400
    metrology_witness_key = Ed25519PrivateKey.generate()
    governance_witness_key = Ed25519PrivateKey.generate()
    preregistration_log_key = ec.generate_private_key(ec.SECP256R1())
    acceptance_log_key = ec.generate_private_key(ec.SECP256R1())
    metrology_witness_public_pem = metrology_witness_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    governance_witness_public_pem = governance_witness_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    preregistration_log_public_pem = preregistration_log_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    acceptance_log_public_pem = acceptance_log_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    trust_policy = AcceptanceTrustPolicy.from_public_key_material(
        metrology_witness_public_key=metrology_witness_public_pem,
        governance_witness_public_key=governance_witness_public_pem,
        preregistration_log_public_key_pem=preregistration_log_public_pem,
        acceptance_log_public_key_pem=acceptance_log_public_pem,
    )
    statement = build_acceptance_preregistration_statement(
        mandate,
        provision_receipt,
        trust_policy,
        sequence=1,
        previous_statement_sha256=ZERO_SHA256,
        previous_rekor_uuid=None,
        issued_at_unix=issued_at,
    )
    bundle, log_public_pem = _rekor_bundle(
        statement,
        log_key=preregistration_log_key,
    )
    campaign_started_at_ns = (issued_at + 2) * 1_000_000_000

    class _MandateHandle:
        def get(self, campaign_id: str) -> AcceptanceVerificationMandate:
            assert campaign_id == mandate.campaign_id
            return mandate

        def close(self) -> None:
            return None

    class _MandateStoreFacade:
        @staticmethod
        def from_system(_path):
            return _MandateHandle()

    monkeypatch.setattr(
        prereg_tool,
        "AcceptanceMandateStore",
        _MandateStoreFacade,
    )
    mandate_path = tmp_path / "mandate.json"
    provision_path = tmp_path / "provision-receipt.json"
    mandate_path.write_text(json.dumps(mandate.to_dict()), encoding="utf-8")
    provision_path.write_text(
        json.dumps(provision_receipt.to_dict()),
        encoding="utf-8",
    )
    operator_statement_path = tmp_path / "operator-preregistration.json"
    operator_payload_path = tmp_path / "operator-preregistration.payload"
    metrology_key_path = tmp_path / "metrology-witness-public.pem"
    governance_key_path = tmp_path / "governance-witness-public.pem"
    preregistration_log_key_path = tmp_path / "preregistration-log-public.pem"
    acceptance_log_key_path = tmp_path / "acceptance-log-public.pem"
    metrology_key_path.write_bytes(metrology_witness_public_pem)
    governance_key_path.write_bytes(governance_witness_public_pem)
    preregistration_log_key_path.write_bytes(preregistration_log_public_pem)
    acceptance_log_key_path.write_bytes(acceptance_log_public_pem)
    assert prereg_tool.main(
        [
            "statement",
            "--mandate-state",
            str(tmp_path / "mandates.encrypted.json"),
            "--campaign-id",
            mandate.campaign_id,
            "--provision-receipt",
            str(provision_path),
            "--metrology-witness-public-key",
            str(metrology_key_path),
            "--governance-witness-public-key",
            str(governance_key_path),
            "--preregistration-log-public-key-pem",
            str(preregistration_log_key_path),
            "--acceptance-log-public-key-pem",
            str(acceptance_log_key_path),
            "--sequence",
            "1",
            "--issued-at-unix",
            str(issued_at),
            "--statement-output",
            str(operator_statement_path),
            "--payload-output",
            str(operator_payload_path),
        ]
    ) == 0
    assert json.loads(operator_statement_path.read_text()) == statement
    capsys.readouterr()
    with pytest.raises(
        ValueError,
        match="acceptance_preregistration_trust_source_invalid",
    ):
        prereg_tool.main(
            [
                "statement",
                "--mandate-state",
                str(tmp_path / "mandates.encrypted.json"),
                "--campaign-id",
                mandate.campaign_id,
                "--provision-receipt",
                str(provision_path),
                "--metrology-witness-public-key",
                str(metrology_key_path),
                "--governance-witness-public-key",
                str(governance_key_path),
                "--preregistration-log-public-key-pem",
                str(preregistration_log_key_path),
                "--acceptance-log-key-sha256",
                trust_policy.acceptance_log_key_sha256,
                "--sequence",
                "1",
                "--issued-at-unix",
                str(issued_at),
                "--statement-output",
                str(tmp_path / "mixed-source-statement.json"),
                "--payload-output",
                str(tmp_path / "mixed-source-payload.bin"),
            ]
        )

    signature_path = tmp_path / "producer-signature.raw"
    certificate_path = tmp_path / "producer-certificate.pem"
    entry_path = tmp_path / "rekor-entry.json"
    log_key_path = tmp_path / "rekor-log-key.pem"
    bundle_path = tmp_path / "preregistration-bundle.json"
    signature_path.write_bytes(
        base64.b64decode(bundle["producer_signature_b64"], validate=True)
    )
    certificate_path.write_bytes(
        base64.b64decode(bundle["producer_certificate_pem_b64"], validate=True)
    )
    entry_path.write_text(json.dumps(bundle["rekor_entry"]), encoding="utf-8")
    log_key_path.write_bytes(log_public_pem)
    assembled = subprocess.run(
        [
            sys.executable,
            "tools/reality_reach/manage_acceptance_preregistration.py",
            "assemble",
            "--statement",
            str(operator_statement_path),
            "--producer-signature",
            str(signature_path),
            "--producer-certificate-pem",
            str(certificate_path),
            "--rekor-uuid",
            str(bundle["rekor_uuid"]),
            "--rekor-entry",
            str(entry_path),
            "--trusted-log-public-key-pem",
            str(log_key_path),
            "--output",
            str(bundle_path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert assembled.returncode == 0, assembled.stderr
    assert json.loads(bundle_path.read_text()) == bundle

    independent_path = tmp_path / "portable-preregistration-verdict.json"
    independent = subprocess.run(
        [
            sys.executable,
            "tools/reality_reach/manage_acceptance_preregistration.py",
            "verify",
            "--mandate-document",
            str(mandate_path),
            "--provision-receipt",
            str(provision_path),
            "--transparency-bundle",
            str(bundle_path),
            "--trusted-log-public-key-pem",
            str(log_key_path),
            "--campaign-started-at-ns",
            str(campaign_started_at_ns),
            "--sequence",
            "1",
            "--receipt-output",
            str(independent_path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert independent.returncode == 0, independent.stderr
    assert json.loads(independent_path.read_text())["accepted"] is True

    verified = verify_acceptance_preregistration(
        mandate,
        provision_receipt,
        transparency_bundle=bundle,
        trusted_log_public_key_pem=log_public_pem,
        campaign_started_at_ns=campaign_started_at_ns,
        expected_sequence=1,
        expected_previous_statement_sha256=ZERO_SHA256,
        expected_previous_rekor_uuid=None,
    )
    assert verified.accepted is True
    assert verified.rekor_integrated_time == issued_at + 1

    late = verify_acceptance_preregistration(
        mandate,
        provision_receipt,
        transparency_bundle=bundle,
        trusted_log_public_key_pem=log_public_pem,
        campaign_started_at_ns=(issued_at + 1) * 1_000_000_000,
        expected_sequence=1,
        expected_previous_statement_sha256=ZERO_SHA256,
        expected_previous_rekor_uuid=None,
    )
    missing = verify_acceptance_preregistration(
        mandate,
        provision_receipt,
        transparency_bundle=None,
        trusted_log_public_key_pem=log_public_pem,
        campaign_started_at_ns=campaign_started_at_ns,
        expected_sequence=1,
        expected_previous_statement_sha256=ZERO_SHA256,
        expected_previous_rekor_uuid=None,
    )
    assert late.accepted is False
    assert late.blockers == (
        "acceptance_preregistration_not_strictly_before_campaign",
    )
    reconstructed_late = replace(
        verified,
        rekor_integrated_time=campaign_started_at_ns // 1_000_000_000,
    )
    assert reconstructed_late.strictly_predates_campaign is False
    assert reconstructed_late.accepted is False
    assert missing.accepted is False
    assert missing.blockers == (
        "acceptance_transparency_bundle_missing",
        "acceptance_trust_policy_missing_or_invalid",
    )

    path = tmp_path / "preregistration.json"
    assert persist_preregistered_acceptance_receipt(verified, path) is True
    assert persist_preregistered_acceptance_receipt(verified, path) is False
    with pytest.raises(AcceptanceTransparencyError, match="receipt_collision"):
        persist_preregistered_acceptance_receipt(
            replace(verified, blockers=("tampered",)),
            path,
        )


def test_preregistration_rejects_posthoc_or_rebound_questions() -> None:
    mandate, provision_receipt = _mandate()
    trust_policy = AcceptanceTrustPolicy(
        metrology_witness_key_sha256="sha256:" + "4" * 64,
        governance_witness_key_sha256="sha256:" + "5" * 64,
        preregistration_log_key_sha256="sha256:" + "6" * 64,
        acceptance_log_key_sha256="sha256:" + "7" * 64,
    )
    with pytest.raises(
        AcceptanceTransparencyError,
        match="mandate_binding_invalid",
    ):
        build_acceptance_preregistration_statement(
            replace(mandate, target=0.75),
            provision_receipt,
            trust_policy,
            sequence=1,
            previous_statement_sha256=ZERO_SHA256,
            previous_rekor_uuid=None,
            issued_at_unix=1_785_082_400,
        )
    with pytest.raises(AcceptanceTransparencyError, match="genesis_invalid"):
        build_acceptance_preregistration_statement(
            mandate,
            provision_receipt,
            trust_policy,
            sequence=1,
            previous_statement_sha256="sha256:" + "9" * 64,
            previous_rekor_uuid=None,
            issued_at_unix=1_785_082_400,
        )
    attacked = provision_receipt.to_dict()
    attacked["custody_sequence"] = 8
    with pytest.raises(AcceptanceMandateError, match="provision_receipt_digest_invalid"):
        AcceptanceMandateProvisionReceipt.from_dict(attacked)


def test_trust_policy_admits_only_canonical_public_key_material() -> None:
    metrology = Ed25519PrivateKey.generate()
    governance = Ed25519PrivateKey.generate()
    preregistration_log = ec.generate_private_key(ec.SECP256R1())
    acceptance_log = ec.generate_private_key(ec.SECP256R1())
    metrology_raw = metrology.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    governance_pem = governance.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    preregistration_log_pem = preregistration_log.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    acceptance_log_pem = acceptance_log.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    policy = AcceptanceTrustPolicy.from_public_key_material(
        metrology_witness_public_key=metrology_raw,
        governance_witness_public_key=governance_pem,
        preregistration_log_public_key_pem=preregistration_log_pem,
        acceptance_log_public_key_pem=acceptance_log_pem,
    )

    assert policy.metrology_witness_key_sha256 == (
        "sha256:" + hashlib.sha256(metrology_raw).hexdigest()
    )
    assert AcceptanceTrustPolicy.from_dict(policy.to_dict()) == policy
    with pytest.raises(
        AcceptanceTransparencyError,
        match="acceptance_metrology_witness_key_invalid",
    ):
        AcceptanceTrustPolicy.from_public_key_material(
            metrology_witness_public_key=metrology.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
            governance_witness_public_key=governance_pem,
            preregistration_log_public_key_pem=preregistration_log_pem,
            acceptance_log_public_key_pem=acceptance_log_pem,
        )
    with pytest.raises(
        AcceptanceTransparencyError,
        match="acceptance_metrology_witness_key_type_invalid",
    ):
        AcceptanceTrustPolicy.from_public_key_material(
            metrology_witness_public_key=preregistration_log_pem,
            governance_witness_public_key=governance_pem,
            preregistration_log_public_key_pem=preregistration_log_pem,
            acceptance_log_public_key_pem=acceptance_log_pem,
        )
    with pytest.raises(
        AcceptanceTransparencyError,
        match="acceptance_preregistration_log_key_type_invalid",
    ):
        AcceptanceTrustPolicy.from_public_key_material(
            metrology_witness_public_key=metrology_raw,
            governance_witness_public_key=governance_pem,
            preregistration_log_public_key_pem=governance_pem,
            acceptance_log_public_key_pem=acceptance_log_pem,
        )
    with pytest.raises(
        AcceptanceTransparencyError,
        match="acceptance_acceptance_log_key_invalid",
    ):
        AcceptanceTrustPolicy.from_public_key_material(
            metrology_witness_public_key=metrology_raw,
            governance_witness_public_key=governance_pem,
            preregistration_log_public_key_pem=preregistration_log_pem,
            acceptance_log_public_key_pem=acceptance_log.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
        )
    with pytest.raises(ValueError, match="witness trust roots must be distinct"):
        AcceptanceTrustPolicy.from_public_key_material(
            metrology_witness_public_key=metrology_raw,
            governance_witness_public_key=metrology_raw,
            preregistration_log_public_key_pem=preregistration_log_pem,
            acceptance_log_public_key_pem=acceptance_log_pem,
        )


def test_preregistration_rejects_valid_unregistered_log_root() -> None:
    mandate, provision_receipt = _mandate()
    registered_log_key = ec.generate_private_key(ec.SECP256R1())
    substituted_log_key = ec.generate_private_key(ec.SECP256R1())
    registered_log_sha256 = "sha256:" + hashlib.sha256(
        registered_log_key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    ).hexdigest()
    trust_policy = AcceptanceTrustPolicy(
        metrology_witness_key_sha256="sha256:" + "4" * 64,
        governance_witness_key_sha256="sha256:" + "5" * 64,
        preregistration_log_key_sha256=registered_log_sha256,
        acceptance_log_key_sha256="sha256:" + "7" * 64,
    )
    issued_at = 1_785_082_400
    statement = build_acceptance_preregistration_statement(
        mandate,
        provision_receipt,
        trust_policy,
        sequence=1,
        previous_statement_sha256=ZERO_SHA256,
        previous_rekor_uuid=None,
        issued_at_unix=issued_at,
    )
    bundle, substituted_log_public_pem = _rekor_bundle(
        statement,
        log_key=substituted_log_key,
    )

    result = verify_acceptance_preregistration(
        mandate,
        provision_receipt,
        transparency_bundle=bundle,
        trusted_log_public_key_pem=substituted_log_public_pem,
        campaign_started_at_ns=(issued_at + 2) * 1_000_000_000,
        expected_sequence=1,
        expected_previous_statement_sha256=ZERO_SHA256,
        expected_previous_rekor_uuid=None,
    )

    assert result.accepted is False
    assert result.blockers == (
        "acceptance_preregistration_log_root_not_preregistered",
    )
