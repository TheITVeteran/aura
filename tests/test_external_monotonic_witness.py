from __future__ import annotations

import base64
import copy
import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.x509.oid import NameOID

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.learning.external_monotonic_witness import (
    EXTERNAL_WITNESS_STATEMENT_SCHEMA,
    REKOR_PUBLIC_GOOD_SERVER,
    REKOR_WITNESS_BUNDLE_SCHEMA,
    ZERO_SHA256,
    ExternalMonotonicWitnessError,
    build_external_witness_statement,
    build_rekor_witness_bundle,
    build_spark_059_production_audit_packet,
    validate_rekor_witness_bundle,
    validate_spark_059_production_audit_packet,
)

ISSUED_AT = 1_785_082_400
_ARTIFACTS = {
    "combined_lineage_publication": canonical_json_bytes(
        {
            "schema": "aura.rlc.cp400_combined_sft_lineage_publication_evidence.v1",
            "checkpoint": "CP400",
        }
    ),
    "external_audit_contract": canonical_json_bytes(
        {
            "schema": "aura.rlc.cp401_combined_sft_external_audit_evidence.v1",
            "checkpoint": "CP401",
        }
    ),
    "resident_tokenizer_admission": canonical_json_bytes(
        {
            "schema": "aura.rlc.cp402_verified_replay_sft_tokenizer_evidence.v1",
            "checkpoint": "CP402",
        }
    ),
}


def _packet():
    return build_spark_059_production_audit_packet(
        source_git_commit="1" * 40,
        artifact_payloads=_ARTIFACTS,
        production_replay_candidate_sha256=None,
        external_audit_bundle_sha256=None,
        resident_tokenizer_bundle_sha256=None,
    )


def _certificate(private_key: ed25519.Ed25519PrivateKey) -> bytes:
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Aura test witness")])
    start = datetime.fromtimestamp(ISSUED_AT - 60, tz=UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(private_key.public_key())
        .serial_number(1)
        .not_valid_before(start)
        .not_valid_after(start + timedelta(days=1))
        .sign(private_key, algorithm=None)
    )
    return certificate.public_bytes(serialization.Encoding.PEM)


def _log_key_id(public_key: ec.EllipticCurvePublicKey) -> str:
    der = public_key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(der).hexdigest()


def _fixture():
    packet = _packet()
    statement = build_external_witness_statement(
        audit_packet=packet,
        sequence=1,
        previous_statement_sha256=ZERO_SHA256,
        previous_rekor_uuid=None,
        issued_at_unix=ISSUED_AT,
    )
    statement_bytes = canonical_json_bytes(statement)
    producer_key = ed25519.Ed25519PrivateKey.generate()
    certificate = _certificate(producer_key)
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
    body_b64 = base64.b64encode(canonical_json_bytes(body)).decode("ascii")
    root = hashlib.sha256(b"\x00" + base64.b64decode(body_b64)).digest()
    log_key = ec.generate_private_key(ec.SECP256R1())
    log_public_pem = log_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    log_id = _log_key_id(log_key.public_key())
    checkpoint_text = (
        "rekor.sigstore.dev - 123\n"
        "1\n"
        f"{base64.b64encode(root).decode('ascii')}\n"
    )
    checkpoint_signature = log_key.sign(
        checkpoint_text.encode("utf-8"), ec.ECDSA(hashes.SHA256())
    )
    note_signature = bytes.fromhex(log_id[:8]) + checkpoint_signature
    checkpoint = (
        checkpoint_text
        + "\n— rekor.sigstore.dev "
        + base64.b64encode(note_signature).decode("ascii")
        + "\n"
    )
    entry = {
        "body": body_b64,
        "integratedTime": ISSUED_AT + 1,
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
            "signedEntryTimestamp": "pending",
        },
    }
    set_payload = canonical_json_bytes(
        {
            "body": entry["body"],
            "integratedTime": entry["integratedTime"],
            "logID": entry["logID"],
            "logIndex": entry["logIndex"],
        }
    )
    entry["verification"]["signedEntryTimestamp"] = base64.b64encode(
        log_key.sign(set_payload, ec.ECDSA(hashes.SHA256()))
    ).decode("ascii")
    rekor_uuid = f"{123:016x}{root.hex()}"
    bundle = build_rekor_witness_bundle(
        statement=statement,
        producer_signature=signature,
        producer_certificate_pem=certificate,
        rekor_uuid=rekor_uuid,
        rekor_entry=entry,
        trusted_log_public_key_pem=log_public_pem,
    )
    return packet, statement, bundle, log_public_pem


def _recommit(bundle: dict) -> dict:
    body = dict(bundle)
    body.pop("bundle_sha256", None)
    body["bundle_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return body


def _validate(packet, bundle, log_public_pem, **kwargs):
    return validate_rekor_witness_bundle(
        bundle,
        audit_packet=packet,
        trusted_log_public_key_pem=log_public_pem,
        expected_sequence=1,
        expected_previous_statement_sha256=ZERO_SHA256,
        expected_previous_rekor_uuid=None,
        **kwargs,
    )


def test_packet_is_explicitly_blocked_and_research_scope_excludes_replay():
    packet = validate_spark_059_production_audit_packet(_packet())
    assert packet["trainer_ready"] is False
    assert packet["training_authority"] == "none"
    assert packet["research_scope"] == {
        "small_checkpoint_falsification_may_use": ["structured_synthetic"],
        "forbidden_from_research_trainer": [
            "verified_replay_user_content",
            "evaluation_holdout",
        ],
        "production_promotion_allowed": False,
    }
    assert "production_verified_replay_candidate_absent" in packet["remaining_blockers"]


def test_realistic_offline_rekor_bundle_verifies_all_layers():
    packet, statement, bundle, log_public_pem = _fixture()
    result = _validate(packet, bundle, log_public_pem)
    assert statement["schema"] == EXTERNAL_WITNESS_STATEMENT_SCHEMA
    assert bundle["schema"] == REKOR_WITNESS_BUNDLE_SCHEMA
    assert bundle["rekor_server"] == REKOR_PUBLIC_GOOD_SERVER
    assert result["status"] == "externally_witnessed_audit_head_verified_offline"
    assert result["trainer_ready"] is False


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda bundle: bundle["rekor_entry"]["verification"].__setitem__(
                "signedEntryTimestamp", base64.b64encode(b"forged").decode("ascii")
            ),
            "external_witness_set_signature_invalid",
        ),
        (
            lambda bundle: bundle["rekor_entry"]["verification"][
                "inclusionProof"
            ].__setitem__("rootHash", "3" * 64),
            "external_witness_inclusion_proof_root_mismatch",
        ),
        (
            lambda bundle: bundle["rekor_entry"]["verification"][
                "inclusionProof"
            ].__setitem__(
                "checkpoint",
                bundle["rekor_entry"]["verification"]["inclusionProof"][
                    "checkpoint"
                ].replace("rekor.sigstore.dev - 123", "rekor.sigstore.dev - 124"),
            ),
            "external_witness_checkpoint_signature_invalid",
        ),
        (
            lambda bundle: bundle["rekor_entry"].__setitem__("logID", "4" * 64),
            "external_witness_rekor_log_identity_mismatch",
        ),
    ],
)
def test_log_evidence_forgery_fails_closed(mutation, error):
    packet, _statement, bundle, log_public_pem = _fixture()
    forged = copy.deepcopy(bundle)
    mutation(forged)
    forged = _recommit(forged)
    with pytest.raises(ExternalMonotonicWitnessError, match=error):
        _validate(packet, forged, log_public_pem)


def test_packet_or_statement_rebinding_fails_closed():
    packet, _statement, bundle, log_public_pem = _fixture()
    altered_packet = copy.deepcopy(packet)
    altered_packet["source_git_commit"] = "5" * 40
    packet_body = dict(altered_packet)
    packet_body.pop("packet_sha256")
    altered_packet["packet_sha256"] = hashlib.sha256(
        canonical_json_bytes(packet_body)
    ).hexdigest()
    with pytest.raises(
        ExternalMonotonicWitnessError, match="external_witness_statement_binding_invalid"
    ):
        _validate(altered_packet, bundle, log_public_pem)


def test_wrong_log_key_and_rollback_pins_fail_closed():
    packet, _statement, bundle, log_public_pem = _fixture()
    wrong_key = ec.generate_private_key(ec.SECP256R1()).public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    with pytest.raises(
        ExternalMonotonicWitnessError,
        match="external_witness_rekor_log_identity_mismatch",
    ):
        _validate(packet, bundle, wrong_key)
    with pytest.raises(
        ExternalMonotonicWitnessError, match="external_witness_log_index_rollback"
    ):
        _validate(packet, bundle, log_public_pem, minimum_log_index=0)
    with pytest.raises(
        ExternalMonotonicWitnessError, match="external_witness_integrated_time_rollback"
    ):
        _validate(
            packet,
            bundle,
            log_public_pem,
            minimum_integrated_time=ISSUED_AT + 2,
        )


def test_rekor_uuid_is_derived_from_tree_id_and_leaf_hash():
    packet, _statement, bundle, log_public_pem = _fixture()
    forged = copy.deepcopy(bundle)
    forged["rekor_uuid"] = "000000000000007b" + "8" * 64
    forged = _recommit(forged)
    with pytest.raises(
        ExternalMonotonicWitnessError,
        match="external_witness_rekor_uuid_binding_mismatch",
    ):
        _validate(packet, forged, log_public_pem)


def test_genesis_and_non_genesis_chain_contracts_are_distinct():
    packet = _packet()
    with pytest.raises(ExternalMonotonicWitnessError, match="external_witness_genesis_invalid"):
        build_external_witness_statement(
            audit_packet=packet,
            sequence=1,
            previous_statement_sha256="6" * 64,
            previous_rekor_uuid=None,
            issued_at_unix=ISSUED_AT,
        )
    statement = build_external_witness_statement(
        audit_packet=packet,
        sequence=2,
        previous_statement_sha256="6" * 64,
        previous_rekor_uuid="7" * 80,
        issued_at_unix=ISSUED_AT,
    )
    assert statement["sequence"] == 2


def test_unknown_bundle_fields_and_false_authority_are_rejected():
    packet, _statement, bundle, log_public_pem = _fixture()
    forged = copy.deepcopy(bundle)
    forged["trainer_ready"] = True
    forged = _recommit(forged)
    with pytest.raises(ExternalMonotonicWitnessError, match="external_witness_bundle_invalid"):
        _validate(packet, forged, log_public_pem)

    forged_packet = copy.deepcopy(packet)
    forged_packet["trainer_ready"] = True
    body = dict(forged_packet)
    body.pop("packet_sha256")
    forged_packet["packet_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    with pytest.raises(ExternalMonotonicWitnessError, match="spark_audit_packet_contract_invalid"):
        validate_spark_059_production_audit_packet(forged_packet)
