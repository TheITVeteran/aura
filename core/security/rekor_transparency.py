"""Strict offline verification for Rekor ``rekord`` transparency entries.

The verifier accepts only a caller-pinned log key and a complete portable
entry.  It verifies the producer signature material recorded by Rekor, the
signed entry timestamp, RFC 6962 inclusion proof, signed checkpoint, log UUID,
certificate validity, and bounded publication delay.  It performs no network
or private-key operations.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Never

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa

REKOR_PUBLIC_GOOD_SERVER = "https://rekor.sigstore.dev"

_ENTRY_KEYS = {"body", "integratedTime", "logID", "logIndex", "verification"}
_VERIFICATION_KEYS = {"inclusionProof", "signedEntryTimestamp"}
_PROOF_KEYS = {"checkpoint", "hashes", "logIndex", "rootHash", "treeSize"}
_MAX_JSON_NODES = 300_000
_MAX_JSON_DEPTH = 128
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_CERTIFICATE_BYTES = 64 * 1024
_MAX_SIGNATURE_BYTES = 64 * 1024


class RekorTransparencyError(ValueError):
    """Stable fail-closed Rekor verification error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class VerifiedRekorEntry:
    trusted_log_key_sha256: str
    log_index: int
    integrated_time: int
    rekor_uuid: str


def _fail(code: str) -> Never:
    raise RekorTransparencyError(code)


def _code(prefix: str, suffix: str) -> str:
    normalized = str(prefix or "").strip().strip("_")
    if not normalized:
        _fail("rekor_error_prefix_invalid")
    return f"{normalized}_{suffix}"


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise RekorTransparencyError("rekor_value_not_canonical_json") from exc


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
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
        return json.loads(_canonical_json_bytes(value))
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise RekorTransparencyError(code) from exc


def _strict_b64(value: Any, *, code: str, maximum: int) -> bytes:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail(code)
    try:
        decoded = base64.b64decode(value, validate=True)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise RekorTransparencyError(code) from exc
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
        raise RekorTransparencyError(code) from exc
    normalized = _normalize(value, code=code)
    if not isinstance(normalized, dict):
        _fail(code)
    return normalized


def verify_signature(
    public_key: Any,
    signature: bytes,
    payload: bytes,
    *,
    code: str,
) -> None:
    """Verify a supported detached signature using a caller-owned error code."""

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
        raise RekorTransparencyError(f"{code}_invalid") from exc


def load_x509_certificate(pem: bytes, *, code: str) -> x509.Certificate:
    try:
        return x509.load_pem_x509_certificate(pem)
    except ValueError as exc:
        raise RekorTransparencyError(code) from exc


def _load_log_public_key(pem: bytes, *, prefix: str) -> tuple[Any, str]:
    try:
        key = serialization.load_pem_public_key(pem)
        der = key.public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    except (TypeError, ValueError) as exc:
        raise RekorTransparencyError(_code(prefix, "log_key_invalid")) from exc
    if not isinstance(key, ec.EllipticCurvePublicKey):
        _fail(_code(prefix, "log_key_type_invalid"))
    return key, _sha_bytes(der)


def _verify_inclusion_proof(
    *,
    body_b64: str,
    proof: Mapping[str, Any],
    prefix: str,
) -> bytes:
    if set(proof) != _PROOF_KEYS:
        _fail(_code(prefix, "inclusion_proof_schema_invalid"))
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
        _fail(_code(prefix, "inclusion_proof_invalid"))
    body = _strict_b64(
        body_b64,
        code=_code(prefix, "rekor_body_invalid"),
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
        _fail(_code(prefix, "inclusion_proof_root_mismatch"))
    return leaf_hash


def _verify_checkpoint(
    checkpoint: Any,
    *,
    proof: Mapping[str, Any],
    log_id: str,
    log_public_key: Any,
    prefix: str,
) -> int:
    if not isinstance(checkpoint, str) or len(checkpoint) > 64 * 1024:
        _fail(_code(prefix, "checkpoint_invalid"))
    try:
        text, signature_block = checkpoint.rsplit("\n\n", 1)
    except ValueError as exc:
        raise RekorTransparencyError(_code(prefix, "checkpoint_invalid")) from exc
    text_lines = text.splitlines()
    signature_lines = signature_block.splitlines()
    if len(text_lines) != 3 or len(signature_lines) != 1:
        _fail(_code(prefix, "checkpoint_invalid"))
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
        _fail(_code(prefix, "checkpoint_binding_invalid"))
    try:
        checkpoint_root = base64.b64decode(root_b64, validate=True)
        raw_signature = base64.b64decode(signature_parts[2], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RekorTransparencyError(
            _code(prefix, "checkpoint_encoding_invalid")
        ) from exc
    if (
        len(checkpoint_root) != 32
        or checkpoint_root.hex() != proof["rootHash"]
        or len(raw_signature) < 5
        or raw_signature[:4].hex() != log_id[:8]
    ):
        _fail(_code(prefix, "checkpoint_binding_invalid"))
    verify_signature(
        log_public_key,
        raw_signature[4:],
        (text + "\n").encode("utf-8"),
        code=_code(prefix, "checkpoint_signature"),
    )
    return int(tree_id_text)


def verify_rekord_entry(
    *,
    entry: Mapping[str, Any],
    artifact_bytes: bytes,
    producer_signature: bytes,
    producer_certificate_pem: bytes,
    trusted_log_public_key_pem: bytes,
    issued_at_unix: int,
    rekor_uuid: str,
    code_prefix: str,
    maximum_witness_delay_s: int = 60 * 60,
) -> VerifiedRekorEntry:
    """Verify one complete Rekor ``rekord`` entry against pinned inputs."""

    if set(entry) != _ENTRY_KEYS:
        _fail(_code(code_prefix, "rekor_entry_schema_invalid"))
    if not _is_rekor_v1_uuid(rekor_uuid):
        _fail(_code(code_prefix, "rekor_uuid_invalid"))
    if (
        not isinstance(artifact_bytes, bytes)
        or not artifact_bytes
        or len(artifact_bytes) > _MAX_ARTIFACT_BYTES
        or not isinstance(producer_signature, bytes)
        or not producer_signature
        or len(producer_signature) > _MAX_SIGNATURE_BYTES
        or not isinstance(producer_certificate_pem, bytes)
        or not producer_certificate_pem
        or len(producer_certificate_pem) > _MAX_CERTIFICATE_BYTES
        or type(issued_at_unix) is not int
        or issued_at_unix <= 0
        or type(maximum_witness_delay_s) is not int
        or maximum_witness_delay_s <= 0
    ):
        _fail(_code(code_prefix, "rekor_input_invalid"))
    body_b64 = entry.get("body")
    integrated_time = entry.get("integratedTime")
    log_id = entry.get("logID")
    log_index = entry.get("logIndex")
    verification = entry.get("verification")
    if (
        not isinstance(body_b64, str)
        or not body_b64
        or type(integrated_time) is not int
        or integrated_time <= 0
        or integrated_time < issued_at_unix
        or integrated_time - issued_at_unix > maximum_witness_delay_s
        or type(log_index) is not int
        or log_index < 0
        or not _is_sha256(log_id)
        or not isinstance(verification, Mapping)
        or set(verification) != _VERIFICATION_KEYS
    ):
        _fail(_code(code_prefix, "rekor_entry_invalid"))
    body_bytes = _strict_b64(
        body_b64,
        code=_code(code_prefix, "rekor_body_invalid"),
        maximum=_MAX_ARTIFACT_BYTES,
    )
    body = _strict_json_bytes(
        body_bytes,
        code=_code(code_prefix, "rekor_body_json_invalid"),
    )
    if set(body) != {"apiVersion", "kind", "spec"}:
        _fail(_code(code_prefix, "rekor_body_schema_invalid"))
    spec = body.get("spec")
    if (
        body.get("apiVersion") != "0.0.1"
        or body.get("kind") != "rekord"
        or not isinstance(spec, dict)
        or set(spec) != {"data", "signature"}
    ):
        _fail(_code(code_prefix, "rekor_body_schema_invalid"))
    data = spec.get("data")
    signature_record = spec.get("signature")
    if (
        not isinstance(data, dict)
        or set(data) != {"hash"}
        or not isinstance(data.get("hash"), dict)
        or data["hash"] != {"algorithm": "sha256", "value": _sha_bytes(artifact_bytes)}
        or not isinstance(signature_record, dict)
        or set(signature_record) != {"content", "format", "publicKey"}
        or signature_record.get("format") != "x509"
        or not isinstance(signature_record.get("publicKey"), dict)
        or set(signature_record["publicKey"]) != {"content"}
    ):
        _fail(_code(code_prefix, "rekor_artifact_binding_invalid"))
    logged_signature = _strict_b64(
        signature_record.get("content"),
        code=_code(code_prefix, "rekor_signature_invalid"),
        maximum=_MAX_SIGNATURE_BYTES,
    )
    logged_certificate = _strict_b64(
        signature_record["publicKey"].get("content"),
        code=_code(code_prefix, "rekor_certificate_invalid"),
        maximum=_MAX_CERTIFICATE_BYTES,
    )
    if logged_signature != producer_signature or logged_certificate != producer_certificate_pem:
        _fail(_code(code_prefix, "rekor_signature_material_mismatch"))

    certificate = load_x509_certificate(
        producer_certificate_pem,
        code=_code(code_prefix, "producer_certificate_invalid"),
    )
    verify_signature(
        certificate.public_key(),
        producer_signature,
        artifact_bytes,
        code=_code(code_prefix, "producer_signature"),
    )
    log_public_key, trusted_log_key_sha256 = _load_log_public_key(
        trusted_log_public_key_pem,
        prefix=code_prefix,
    )
    if log_id != trusted_log_key_sha256:
        _fail(_code(code_prefix, "rekor_log_identity_mismatch"))
    set_signature = _strict_b64(
        verification.get("signedEntryTimestamp"),
        code=_code(code_prefix, "set_invalid"),
        maximum=_MAX_SIGNATURE_BYTES,
    )
    set_payload = _canonical_json_bytes(
        {
            "body": body_b64,
            "integratedTime": integrated_time,
            "logID": log_id,
            "logIndex": log_index,
        }
    )
    verify_signature(
        log_public_key,
        set_signature,
        set_payload,
        code=_code(code_prefix, "set_signature"),
    )
    proof = verification.get("inclusionProof")
    if not isinstance(proof, Mapping):
        _fail(_code(code_prefix, "inclusion_proof_invalid"))
    leaf_hash = _verify_inclusion_proof(
        body_b64=body_b64,
        proof=proof,
        prefix=code_prefix,
    )
    tree_id = _verify_checkpoint(
        proof.get("checkpoint"),
        proof=proof,
        log_id=log_id,
        log_public_key=log_public_key,
        prefix=code_prefix,
    )
    expected_uuid = f"{tree_id:016x}{leaf_hash.hex()}"
    if rekor_uuid != expected_uuid:
        _fail(_code(code_prefix, "rekor_uuid_binding_mismatch"))
    observed = datetime.fromtimestamp(integrated_time, tz=UTC)
    if observed < certificate.not_valid_before_utc or observed > certificate.not_valid_after_utc:
        _fail(_code(code_prefix, "producer_certificate_time_invalid"))
    return VerifiedRekorEntry(
        trusted_log_key_sha256=trusted_log_key_sha256,
        log_index=log_index,
        integrated_time=integrated_time,
        rekor_uuid=rekor_uuid,
    )


__all__ = [
    "REKOR_PUBLIC_GOOD_SERVER",
    "RekorTransparencyError",
    "VerifiedRekorEntry",
    "load_x509_certificate",
    "verify_rekord_entry",
    "verify_signature",
]
