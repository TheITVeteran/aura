"""Offline-verifiable external witness for SPARK training audit heads.

Local hash chains detect accidental damage but their producer can rewrite the
entire history.  This module binds a canonical SPARK-059 production-audit
packet to a signed Rekor entry, verifies the transparency-log SET, verifies
the RFC 6962 inclusion proof and signed checkpoint, and makes the caller pin
the previous externally witnessed head.  It contains no network or private-key
operations and grants no trainer authority.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Final, Never

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes

SPARK_PRODUCTION_AUDIT_PACKET_SCHEMA: Final = (
    "aura.rlc.spark_059_production_audit_packet.v1"
)
EXTERNAL_WITNESS_STATEMENT_SCHEMA: Final = (
    "aura.rlc.external_monotonic_witness_statement.v1"
)
REKOR_WITNESS_BUNDLE_SCHEMA: Final = "aura.rlc.rekor_witness_bundle.v1"
EXTERNAL_WITNESS_VERSION: Final = "2026.07.26.1"
REKOR_PUBLIC_GOOD_SERVER: Final = "https://rekor.sigstore.dev"
ZERO_SHA256: Final = "0" * 64

_MAX_JSON_NODES = 300_000
_MAX_JSON_DEPTH = 128
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_WITNESS_DELAY_S = 60 * 60
_MAX_CERTIFICATE_BYTES = 64 * 1024
_MAX_SIGNATURE_BYTES = 64 * 1024
_REQUIRED_ARTIFACT_SCHEMAS = {
    "combined_lineage_publication": (
        "artifacts/current/cp400_combined_sft_lineage_publication_evidence.json",
        "aura.rlc.cp400_combined_sft_lineage_publication_evidence.v1",
    ),
    "external_audit_contract": (
        "artifacts/current/cp401_combined_sft_external_audit_evidence.json",
        "aura.rlc.cp401_combined_sft_external_audit_evidence.v1",
    ),
    "resident_tokenizer_admission": (
        "artifacts/current/cp402_verified_replay_sft_tokenizer_evidence.json",
        "aura.rlc.cp402_verified_replay_sft_tokenizer_evidence.v1",
    ),
}
_PACKET_KEYS = {
    "schema",
    "version",
    "campaign",
    "source_git_commit",
    "artifacts",
    "production_evidence",
    "gate_status",
    "research_scope",
    "remaining_blockers",
    "status",
    "trainer_ready",
    "training_authority",
    "packet_sha256",
}
_STATEMENT_KEYS = {
    "schema",
    "domain",
    "sequence",
    "previous_statement_sha256",
    "previous_rekor_uuid",
    "audit_packet_sha256",
    "source_git_commit",
    "issued_at_unix",
    "statement_sha256",
}
_BUNDLE_KEYS = {
    "schema",
    "statement",
    "producer_signature_b64",
    "producer_certificate_pem_b64",
    "producer_certificate_sha256",
    "rekor_server",
    "rekor_uuid",
    "rekor_entry",
    "trusted_log_key_sha256",
    "bundle_sha256",
}
_ENTRY_KEYS = {"body", "integratedTime", "logID", "logIndex", "verification"}
_VERIFICATION_KEYS = {"inclusionProof", "signedEntryTimestamp"}
_PROOF_KEYS = {"checkpoint", "hashes", "logIndex", "rootHash", "treeSize"}


class ExternalMonotonicWitnessError(ValueError):
    """Stable fail-closed witness or audit-packet error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    normalized = str(code or "").strip()
    if not normalized:
        normalized = "external_monotonic_witness_invalid"
    raise ExternalMonotonicWitnessError(normalized)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_oid(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_rekor_v1_uuid(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 80
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(canonical_json_bytes(value))


def _normalize(value: Any, *, code: str) -> Any:
    nodes = 0
    stack = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail(code)
        if isinstance(current, Mapping):
            if any(not isinstance(key, str) for key in current):
                _fail(code)
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)
        elif current is not None:
            if not isinstance(current, (str, int, float, bool)):
                _fail(code)
            if isinstance(current, float) and not math.isfinite(current):
                _fail(code)
    try:
        return json.loads(canonical_json_bytes(value))
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ExternalMonotonicWitnessError(code) from exc


def _strict_b64(value: Any, *, code: str, maximum: int) -> bytes:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(code)
    try:
        decoded = base64.b64decode(value, validate=True)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise ExternalMonotonicWitnessError(code) from exc
    if (
        not decoded
        or len(decoded) > maximum
        or base64.b64encode(decoded).decode("ascii") != value
    ):
        _fail(code)
    return decoded


def _strict_json_bytes(payload: bytes, *, code: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                _fail(code)
            value[key] = child
        return value

    try:
        value = json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=lambda _constant: _fail(code),
        )
    except (UnicodeDecodeError, ValueError, RecursionError, OverflowError) as exc:
        raise ExternalMonotonicWitnessError(code) from exc
    normalized = _normalize(value, code=code)
    if not isinstance(normalized, dict):
        _fail(code)
    return normalized


def _committed_document(
    raw: Any,
    *,
    keys: set[str],
    schema: str,
    digest_field: str,
    code: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != keys:
        _fail(code)
    document = _normalize(raw, code=code)
    if not isinstance(document, dict) or document.get("schema") != schema:
        _fail(code)
    body = dict(document)
    digest = body.pop(digest_field, None)
    if not _is_sha256(digest) or _sha_json(body) != digest:
        _fail(f"{code}_commitment_invalid")
    return document


def _production_gate_state(
    production_evidence: Mapping[str, str | None],
) -> tuple[dict[str, str], list[str]]:
    replay_present = production_evidence["production_replay_candidate_sha256"] is not None
    audit_present = production_evidence["external_audit_bundle_sha256"] is not None
    tokenizer_present = production_evidence["resident_tokenizer_bundle_sha256"] is not None
    blockers = []
    if not replay_present:
        blockers.append("production_verified_replay_candidate_absent")
    if not audit_present:
        blockers.append("independent_privacy_contamination_execution_audit_absent")
    if not tokenizer_present:
        blockers.append("production_replay_resident_tokenizer_receipt_absent")
    blockers.extend(
        [
            "bounded_resumable_trainer_authority_absent",
            "small_checkpoint_transfer_falsification_absent",
            "resident_32b_equal_compute_promotion_absent",
        ]
    )
    return (
        {
            "production_replay_custody": (
                "present_unverified" if replay_present else "blocked_absent"
            ),
            "external_privacy_contamination_execution": (
                "present_unverified" if audit_present else "blocked_absent"
            ),
            "resident_replay_tokenization": (
                "present_unverified" if tokenizer_present else "fixture_only"
            ),
            "trainer_authority": "blocked_absent",
            "small_checkpoint_transfer": "not_run",
            "resident_32b_promotion": "not_run",
        },
        blockers,
    )


def build_spark_059_production_audit_packet(
    *,
    source_git_commit: str,
    artifact_payloads: Mapping[str, bytes],
    production_replay_candidate_sha256: str | None,
    external_audit_bundle_sha256: str | None,
    resident_tokenizer_bundle_sha256: str | None,
) -> dict[str, Any]:
    """Commit the exact SPARK-059 audit state without inventing clearances."""

    if not _is_git_oid(source_git_commit):
        _fail("spark_audit_source_commit_invalid")
    if not isinstance(artifact_payloads, Mapping) or set(artifact_payloads) != set(
        _REQUIRED_ARTIFACT_SCHEMAS
    ):
        _fail("spark_audit_artifact_inventory_invalid")
    records = []
    for role, (path, expected_schema) in _REQUIRED_ARTIFACT_SCHEMAS.items():
        payload = artifact_payloads[role]
        if not isinstance(payload, bytes) or not payload or len(payload) > _MAX_ARTIFACT_BYTES:
            _fail(f"spark_audit_{role}_payload_invalid")
        document = _strict_json_bytes(payload, code=f"spark_audit_{role}_json_invalid")
        if document.get("schema") != expected_schema:
            _fail(f"spark_audit_{role}_schema_invalid")
        records.append(
            {
                "role": role,
                "path": path,
                "schema": expected_schema,
                "size_bytes": len(payload),
                "sha256": _sha_bytes(payload),
            }
        )

    evidence_values = {
        "production_replay_candidate_sha256": production_replay_candidate_sha256,
        "external_audit_bundle_sha256": external_audit_bundle_sha256,
        "resident_tokenizer_bundle_sha256": resident_tokenizer_bundle_sha256,
    }
    for name, value in evidence_values.items():
        if value is not None and not _is_sha256(value):
            _fail(f"spark_audit_{name}_invalid")

    gate_status, blockers = _production_gate_state(evidence_values)
    body = {
        "schema": SPARK_PRODUCTION_AUDIT_PACKET_SCHEMA,
        "version": EXTERNAL_WITNESS_VERSION,
        "campaign": "SPARK-059",
        "source_git_commit": source_git_commit,
        "artifacts": records,
        "production_evidence": evidence_values,
        "gate_status": gate_status,
        "research_scope": {
            "small_checkpoint_falsification_may_use": ["structured_synthetic"],
            "forbidden_from_research_trainer": [
                "verified_replay_user_content",
                "evaluation_holdout",
            ],
            "production_promotion_allowed": False,
        },
        "remaining_blockers": blockers,
        "status": "audit_state_committed_production_training_blocked",
        "trainer_ready": False,
        "training_authority": "none",
    }
    return {**body, "packet_sha256": _sha_json(body)}


def validate_spark_059_production_audit_packet(raw: Any) -> dict[str, Any]:
    """Validate packet commitment and the fail-closed production posture."""

    packet = _committed_document(
        raw,
        keys=_PACKET_KEYS,
        schema=SPARK_PRODUCTION_AUDIT_PACKET_SCHEMA,
        digest_field="packet_sha256",
        code="spark_audit_packet_invalid",
    )
    artifacts = packet.get("artifacts")
    if (
        packet.get("version") != EXTERNAL_WITNESS_VERSION
        or packet.get("campaign") != "SPARK-059"
        or not _is_git_oid(packet.get("source_git_commit"))
        or not isinstance(artifacts, list)
        or len(artifacts) != len(_REQUIRED_ARTIFACT_SCHEMAS)
        or packet.get("trainer_ready") is not False
        or packet.get("training_authority") != "none"
        or packet.get("status") != "audit_state_committed_production_training_blocked"
    ):
        _fail("spark_audit_packet_contract_invalid")
    by_role = {item.get("role"): item for item in artifacts if isinstance(item, dict)}
    if set(by_role) != set(_REQUIRED_ARTIFACT_SCHEMAS) or len(by_role) != len(artifacts):
        _fail("spark_audit_packet_artifacts_invalid")
    for role, (path, schema) in _REQUIRED_ARTIFACT_SCHEMAS.items():
        item = by_role[role]
        if (
            set(item) != {"role", "path", "schema", "size_bytes", "sha256"}
            or item.get("path") != path
            or item.get("schema") != schema
            or type(item.get("size_bytes")) is not int
            or item["size_bytes"] <= 0
            or not _is_sha256(item.get("sha256"))
        ):
            _fail("spark_audit_packet_artifacts_invalid")
    production = packet.get("production_evidence")
    gates = packet.get("gate_status")
    scope = packet.get("research_scope")
    blockers = packet.get("remaining_blockers")
    if (
        not isinstance(production, dict)
        or set(production)
        != {
            "production_replay_candidate_sha256",
            "external_audit_bundle_sha256",
            "resident_tokenizer_bundle_sha256",
        }
        or any(value is not None and not _is_sha256(value) for value in production.values())
        or not isinstance(gates, dict)
        or set(gates)
        != {
            "production_replay_custody",
            "external_privacy_contamination_execution",
            "resident_replay_tokenization",
            "trainer_authority",
            "small_checkpoint_transfer",
            "resident_32b_promotion",
        }
        or not isinstance(scope, dict)
        or scope
        != {
            "small_checkpoint_falsification_may_use": ["structured_synthetic"],
            "forbidden_from_research_trainer": [
                "verified_replay_user_content",
                "evaluation_holdout",
            ],
            "production_promotion_allowed": False,
        }
        or not isinstance(blockers, list)
        or not blockers
        or len(blockers) != len(set(blockers))
        or any(not isinstance(item, str) or not item for item in blockers)
    ):
        _fail("spark_audit_packet_state_invalid")
    expected_gates, expected_blockers = _production_gate_state(production)
    if (
        packet["gate_status"] != expected_gates
        or packet["remaining_blockers"] != expected_blockers
    ):
        _fail("spark_audit_packet_state_inconsistent")
    return packet


def build_external_witness_statement(
    *,
    audit_packet: Mapping[str, Any],
    sequence: int,
    previous_statement_sha256: str,
    previous_rekor_uuid: str | None,
    issued_at_unix: int,
) -> dict[str, Any]:
    """Build one caller-chain-pinned statement for external witnessing."""

    packet = validate_spark_059_production_audit_packet(audit_packet)
    if type(sequence) is not int or sequence <= 0:
        _fail("external_witness_sequence_invalid")
    if not _is_sha256(previous_statement_sha256):
        _fail("external_witness_previous_statement_invalid")
    if type(issued_at_unix) is not int or issued_at_unix <= 0:
        _fail("external_witness_issued_at_invalid")
    if sequence == 1:
        if previous_statement_sha256 != ZERO_SHA256 or previous_rekor_uuid is not None:
            _fail("external_witness_genesis_invalid")
    elif previous_statement_sha256 == ZERO_SHA256 or not _is_rekor_v1_uuid(
        previous_rekor_uuid
    ):
        _fail("external_witness_chain_invalid")
    body = {
        "schema": EXTERNAL_WITNESS_STATEMENT_SCHEMA,
        "domain": "aura.spark-059.production-audit-head",
        "sequence": sequence,
        "previous_statement_sha256": previous_statement_sha256,
        "previous_rekor_uuid": previous_rekor_uuid,
        "audit_packet_sha256": packet["packet_sha256"],
        "source_git_commit": packet["source_git_commit"],
        "issued_at_unix": issued_at_unix,
    }
    return {**body, "statement_sha256": _sha_json(body)}


def validate_external_witness_statement(
    raw: Any,
    *,
    audit_packet: Mapping[str, Any],
    expected_sequence: int,
    expected_previous_statement_sha256: str,
    expected_previous_rekor_uuid: str | None,
) -> dict[str, Any]:
    packet = validate_spark_059_production_audit_packet(audit_packet)
    statement = _committed_document(
        raw,
        keys=_STATEMENT_KEYS,
        schema=EXTERNAL_WITNESS_STATEMENT_SCHEMA,
        digest_field="statement_sha256",
        code="external_witness_statement_invalid",
    )
    if (
        statement.get("domain") != "aura.spark-059.production-audit-head"
        or statement.get("audit_packet_sha256") != packet["packet_sha256"]
        or statement.get("source_git_commit") != packet["source_git_commit"]
        or statement.get("sequence") != expected_sequence
        or statement.get("previous_statement_sha256")
        != expected_previous_statement_sha256
        or statement.get("previous_rekor_uuid") != expected_previous_rekor_uuid
        or type(statement.get("issued_at_unix")) is not int
        or statement["issued_at_unix"] <= 0
    ):
        _fail("external_witness_statement_binding_invalid")
    build_external_witness_statement(
        audit_packet=packet,
        sequence=expected_sequence,
        previous_statement_sha256=expected_previous_statement_sha256,
        previous_rekor_uuid=expected_previous_rekor_uuid,
        issued_at_unix=statement["issued_at_unix"],
    )
    return statement


def _verify_signature(public_key: Any, signature: bytes, payload: bytes, *, code: str) -> None:
    try:
        if isinstance(public_key, ed25519.Ed25519PublicKey):
            public_key.verify(signature, payload)
        elif isinstance(public_key, ec.EllipticCurvePublicKey):
            public_key.verify(signature, payload, ec.ECDSA(hashes.SHA256()))
        elif isinstance(public_key, rsa.RSAPublicKey):
            public_key.verify(signature, payload, padding.PKCS1v15(), hashes.SHA256())
        else:
            _fail(f"{code}_key_type_invalid")
    except (InvalidSignature, ValueError) as exc:
        raise ExternalMonotonicWitnessError(f"{code}_invalid") from exc


def _load_certificate(pem: bytes) -> x509.Certificate:
    try:
        certificate = x509.load_pem_x509_certificate(pem)
    except ValueError as exc:
        raise ExternalMonotonicWitnessError(
            "external_witness_producer_certificate_invalid"
        ) from exc
    return certificate


def _load_log_public_key(pem: bytes) -> tuple[Any, str]:
    try:
        key = serialization.load_pem_public_key(pem)
        der = key.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    except (TypeError, ValueError) as exc:
        raise ExternalMonotonicWitnessError("external_witness_log_key_invalid") from exc
    if not isinstance(key, ec.EllipticCurvePublicKey):
        _fail("external_witness_log_key_type_invalid")
    return key, _sha_bytes(der)


def _verify_inclusion_proof(
    *,
    body_b64: str,
    proof: Mapping[str, Any],
) -> bytes:
    if set(proof) != _PROOF_KEYS:
        _fail("external_witness_inclusion_proof_schema_invalid")
    tree_size = proof.get("treeSize")
    proof_index = proof.get("logIndex")
    root_hash = proof.get("rootHash")
    siblings = proof.get("hashes")
    if (
        type(tree_size) is not int
        or tree_size <= 0
        or type(proof_index) is not int
        or proof_index < 0
        or proof_index >= tree_size
        or not _is_sha256(root_hash)
        or not isinstance(siblings, list)
        or len(siblings) > 64
        or any(not _is_sha256(value) for value in siblings)
    ):
        _fail("external_witness_inclusion_proof_invalid")
    body = _strict_b64(
        body_b64,
        code="external_witness_rekor_body_invalid",
        maximum=_MAX_ARTIFACT_BYTES,
    )
    leaf_hash = hashlib.sha256(b"\x00" + body).digest()
    running = leaf_hash
    node = proof_index
    last = tree_size - 1
    for sibling_hex in siblings:
        sibling = bytes.fromhex(sibling_hex)
        if node & 1 or node == last:
            running = hashlib.sha256(b"\x01" + sibling + running).digest()
            while node and not (node & 1):
                node //= 2
                last //= 2
        else:
            running = hashlib.sha256(b"\x01" + running + sibling).digest()
        node //= 2
        last //= 2
    if node != 0 or running.hex() != root_hash:
        _fail("external_witness_inclusion_proof_root_mismatch")
    return leaf_hash


def _verify_checkpoint(
    checkpoint: Any,
    *,
    proof: Mapping[str, Any],
    log_id: str,
    log_public_key: Any,
) -> int:
    if not isinstance(checkpoint, str) or len(checkpoint) > 64 * 1024:
        _fail("external_witness_checkpoint_invalid")
    try:
        text, signature_block = checkpoint.rsplit("\n\n", 1)
    except ValueError as exc:
        raise ExternalMonotonicWitnessError("external_witness_checkpoint_invalid") from exc
    text_lines = text.splitlines()
    signature_lines = signature_block.splitlines()
    if len(text_lines) != 3 or len(signature_lines) != 1:
        _fail("external_witness_checkpoint_invalid")
    origin, tree_size_text, root_b64 = text_lines
    tree_id_text = origin.removeprefix("rekor.sigstore.dev - ")
    signature_parts = signature_lines[0].split(" ")
    if (
        not origin.startswith("rekor.sigstore.dev - ")
        or not tree_id_text.isascii()
        or not tree_id_text.isdigit()
        or int(tree_id_text) <= 0
        or int(tree_id_text) >= 2**64
        or not tree_size_text.isascii()
        or not tree_size_text.isdigit()
        or int(tree_size_text) != proof["treeSize"]
        or signature_parts[:2] != ["—", "rekor.sigstore.dev"]
        or len(signature_parts) != 3
    ):
        _fail("external_witness_checkpoint_binding_invalid")
    try:
        checkpoint_root = base64.b64decode(root_b64, validate=True)
        raw_signature = base64.b64decode(signature_parts[2], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ExternalMonotonicWitnessError(
            "external_witness_checkpoint_encoding_invalid"
        ) from exc
    if (
        len(checkpoint_root) != 32
        or checkpoint_root.hex() != proof["rootHash"]
        or len(raw_signature) < 5
        or raw_signature[:4].hex() != log_id[:8]
    ):
        _fail("external_witness_checkpoint_binding_invalid")
    _verify_signature(
        log_public_key,
        raw_signature[4:],
        (text + "\n").encode("utf-8"),
        code="external_witness_checkpoint_signature",
    )
    return int(tree_id_text)


def _verify_rekor_entry(
    *,
    entry: Mapping[str, Any],
    statement_bytes: bytes,
    producer_signature: bytes,
    producer_certificate_pem: bytes,
    trusted_log_public_key_pem: bytes,
    issued_at_unix: int,
    rekor_uuid: str,
) -> tuple[str, int, int]:
    if set(entry) != _ENTRY_KEYS:
        _fail("external_witness_rekor_entry_schema_invalid")
    body_b64 = entry.get("body")
    integrated_time = entry.get("integratedTime")
    log_id = entry.get("logID")
    log_index = entry.get("logIndex")
    verification = entry.get("verification")
    if (
        type(integrated_time) is not int
        or integrated_time <= 0
        or integrated_time < issued_at_unix
        or integrated_time - issued_at_unix > _MAX_WITNESS_DELAY_S
        or type(log_index) is not int
        or log_index < 0
        or not _is_sha256(log_id)
        or not isinstance(verification, Mapping)
        or set(verification) != _VERIFICATION_KEYS
    ):
        _fail("external_witness_rekor_entry_invalid")
    body_bytes = _strict_b64(
        body_b64,
        code="external_witness_rekor_body_invalid",
        maximum=_MAX_ARTIFACT_BYTES,
    )
    body = _strict_json_bytes(body_bytes, code="external_witness_rekor_body_json_invalid")
    if set(body) != {"apiVersion", "kind", "spec"}:
        _fail("external_witness_rekor_body_schema_invalid")
    spec = body.get("spec")
    if (
        body.get("apiVersion") != "0.0.1"
        or body.get("kind") != "rekord"
        or not isinstance(spec, dict)
        or set(spec) != {"data", "signature"}
    ):
        _fail("external_witness_rekor_body_schema_invalid")
    data = spec.get("data")
    signature_record = spec.get("signature")
    if (
        not isinstance(data, dict)
        or set(data) != {"hash"}
        or not isinstance(data.get("hash"), dict)
        or data["hash"]
        != {"algorithm": "sha256", "value": _sha_bytes(statement_bytes)}
        or not isinstance(signature_record, dict)
        or set(signature_record) != {"content", "format", "publicKey"}
        or signature_record.get("format") != "x509"
        or not isinstance(signature_record.get("publicKey"), dict)
        or set(signature_record["publicKey"]) != {"content"}
    ):
        _fail("external_witness_rekor_artifact_binding_invalid")
    logged_signature = _strict_b64(
        signature_record.get("content"),
        code="external_witness_rekor_signature_invalid",
        maximum=_MAX_SIGNATURE_BYTES,
    )
    logged_certificate = _strict_b64(
        signature_record["publicKey"].get("content"),
        code="external_witness_rekor_certificate_invalid",
        maximum=_MAX_CERTIFICATE_BYTES,
    )
    if logged_signature != producer_signature or logged_certificate != producer_certificate_pem:
        _fail("external_witness_rekor_signature_material_mismatch")

    log_public_key, trusted_log_key_sha256 = _load_log_public_key(
        trusted_log_public_key_pem
    )
    if log_id != trusted_log_key_sha256:
        _fail("external_witness_rekor_log_identity_mismatch")
    set_signature = _strict_b64(
        verification.get("signedEntryTimestamp"),
        code="external_witness_set_invalid",
        maximum=_MAX_SIGNATURE_BYTES,
    )
    set_payload = canonical_json_bytes(
        {
            "body": body_b64,
            "integratedTime": integrated_time,
            "logID": log_id,
            "logIndex": log_index,
        }
    )
    _verify_signature(
        log_public_key,
        set_signature,
        set_payload,
        code="external_witness_set_signature",
    )
    proof = verification.get("inclusionProof")
    if not isinstance(proof, Mapping):
        _fail("external_witness_inclusion_proof_invalid")
    leaf_hash = _verify_inclusion_proof(body_b64=body_b64, proof=proof)
    tree_id = _verify_checkpoint(
        proof.get("checkpoint"),
        proof=proof,
        log_id=log_id,
        log_public_key=log_public_key,
    )
    expected_uuid = f"{tree_id:016x}{leaf_hash.hex()}"
    if rekor_uuid != expected_uuid:
        _fail("external_witness_rekor_uuid_binding_mismatch")
    return trusted_log_key_sha256, log_index, integrated_time


def build_rekor_witness_bundle(
    *,
    statement: Mapping[str, Any],
    producer_signature: bytes,
    producer_certificate_pem: bytes,
    rekor_uuid: str,
    rekor_entry: Mapping[str, Any],
    trusted_log_public_key_pem: bytes,
) -> dict[str, Any]:
    """Verify all evidence first, then commit a portable offline bundle."""

    if not isinstance(statement, Mapping):
        _fail("external_witness_statement_invalid")
    normalized_statement = _normalize(statement, code="external_witness_statement_invalid")
    if not isinstance(normalized_statement, dict):
        _fail("external_witness_statement_invalid")
    statement_bytes = canonical_json_bytes(normalized_statement)
    if not _is_rekor_v1_uuid(rekor_uuid):
        _fail("external_witness_rekor_uuid_invalid")
    if (
        not isinstance(producer_signature, bytes)
        or not producer_signature
        or len(producer_signature) > _MAX_SIGNATURE_BYTES
        or not isinstance(producer_certificate_pem, bytes)
        or not producer_certificate_pem
        or len(producer_certificate_pem) > _MAX_CERTIFICATE_BYTES
    ):
        _fail("external_witness_producer_material_invalid")
    certificate = _load_certificate(producer_certificate_pem)
    _verify_signature(
        certificate.public_key(),
        producer_signature,
        statement_bytes,
        code="external_witness_producer_signature",
    )
    trusted_log_key_sha256, _log_index, integrated_time = _verify_rekor_entry(
        entry=rekor_entry,
        statement_bytes=statement_bytes,
        producer_signature=producer_signature,
        producer_certificate_pem=producer_certificate_pem,
        trusted_log_public_key_pem=trusted_log_public_key_pem,
        issued_at_unix=normalized_statement.get("issued_at_unix", 0),
        rekor_uuid=rekor_uuid,
    )
    observed = datetime.fromtimestamp(integrated_time, tz=UTC)
    if observed < certificate.not_valid_before_utc or observed > certificate.not_valid_after_utc:
        _fail("external_witness_producer_certificate_time_invalid")
    body = {
        "schema": REKOR_WITNESS_BUNDLE_SCHEMA,
        "statement": normalized_statement,
        "producer_signature_b64": base64.b64encode(producer_signature).decode("ascii"),
        "producer_certificate_pem_b64": base64.b64encode(
            producer_certificate_pem
        ).decode("ascii"),
        "producer_certificate_sha256": _sha_bytes(producer_certificate_pem),
        "rekor_server": REKOR_PUBLIC_GOOD_SERVER,
        "rekor_uuid": rekor_uuid,
        "rekor_entry": _normalize(rekor_entry, code="external_witness_rekor_entry_invalid"),
        "trusted_log_key_sha256": trusted_log_key_sha256,
    }
    return {**body, "bundle_sha256": _sha_json(body)}


def validate_rekor_witness_bundle(
    raw: Any,
    *,
    audit_packet: Mapping[str, Any],
    trusted_log_public_key_pem: bytes,
    expected_sequence: int,
    expected_previous_statement_sha256: str,
    expected_previous_rekor_uuid: str | None,
    minimum_log_index: int | None = None,
    minimum_integrated_time: int | None = None,
) -> dict[str, Any]:
    """Offline-verify an externally logged head against caller-pinned state."""

    bundle = _committed_document(
        raw,
        keys=_BUNDLE_KEYS,
        schema=REKOR_WITNESS_BUNDLE_SCHEMA,
        digest_field="bundle_sha256",
        code="external_witness_bundle_invalid",
    )
    if bundle.get("rekor_server") != REKOR_PUBLIC_GOOD_SERVER:
        _fail("external_witness_rekor_server_invalid")
    statement = validate_external_witness_statement(
        bundle.get("statement"),
        audit_packet=audit_packet,
        expected_sequence=expected_sequence,
        expected_previous_statement_sha256=expected_previous_statement_sha256,
        expected_previous_rekor_uuid=expected_previous_rekor_uuid,
    )
    signature = _strict_b64(
        bundle.get("producer_signature_b64"),
        code="external_witness_producer_signature_invalid",
        maximum=_MAX_SIGNATURE_BYTES,
    )
    certificate_pem = _strict_b64(
        bundle.get("producer_certificate_pem_b64"),
        code="external_witness_producer_certificate_invalid",
        maximum=_MAX_CERTIFICATE_BYTES,
    )
    if bundle.get("producer_certificate_sha256") != _sha_bytes(certificate_pem):
        _fail("external_witness_producer_certificate_commitment_mismatch")
    certificate = _load_certificate(certificate_pem)
    statement_bytes = canonical_json_bytes(statement)
    _verify_signature(
        certificate.public_key(),
        signature,
        statement_bytes,
        code="external_witness_producer_signature",
    )
    entry = bundle.get("rekor_entry")
    if not isinstance(entry, Mapping):
        _fail("external_witness_rekor_entry_invalid")
    trusted_log_key_sha256, log_index, integrated_time = _verify_rekor_entry(
        entry=entry,
        statement_bytes=statement_bytes,
        producer_signature=signature,
        producer_certificate_pem=certificate_pem,
        trusted_log_public_key_pem=trusted_log_public_key_pem,
        issued_at_unix=statement["issued_at_unix"],
        rekor_uuid=bundle["rekor_uuid"],
    )
    if bundle.get("trusted_log_key_sha256") != trusted_log_key_sha256:
        _fail("external_witness_log_key_commitment_mismatch")
    if not _is_rekor_v1_uuid(bundle.get("rekor_uuid")):
        _fail("external_witness_rekor_uuid_invalid")
    if minimum_log_index is not None and (
        type(minimum_log_index) is not int or log_index <= minimum_log_index
    ):
        _fail("external_witness_log_index_rollback")
    if minimum_integrated_time is not None and (
        type(minimum_integrated_time) is not int
        or integrated_time < minimum_integrated_time
    ):
        _fail("external_witness_integrated_time_rollback")
    observed = datetime.fromtimestamp(integrated_time, tz=UTC)
    if observed < certificate.not_valid_before_utc or observed > certificate.not_valid_after_utc:
        _fail("external_witness_producer_certificate_time_invalid")
    return {
        "schema": "aura.rlc.rekor_witness_validation.v1",
        "bundle_sha256": bundle["bundle_sha256"],
        "statement_sha256": statement["statement_sha256"],
        "audit_packet_sha256": statement["audit_packet_sha256"],
        "sequence": statement["sequence"],
        "rekor_uuid": bundle["rekor_uuid"],
        "rekor_log_index": log_index,
        "rekor_integrated_time": integrated_time,
        "trusted_log_key_sha256": trusted_log_key_sha256,
        "trainer_ready": False,
        "training_authority": "none",
        "status": "externally_witnessed_audit_head_verified_offline",
    }
