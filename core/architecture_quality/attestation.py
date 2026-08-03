"""Pinned Ed25519 attestations for architecture evidence artifacts."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def attest_payload(
    payload: dict[str, Any],
    *,
    digest_field: str,
    signing_key_path: Path,
) -> dict[str, Any]:
    """Return a copied payload bound to its digest and external signer."""

    if digest_field in payload or "signature" in payload:
        raise ValueError("payload already contains attestation fields")
    attested = dict(payload)
    attested[digest_field] = payload_sha256(payload)
    attested["signature"] = sign_payload(attested, signing_key_path=signing_key_path)
    return attested


def sign_payload(payload: dict[str, Any], *, signing_key_path: Path) -> dict[str, str]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if not signing_key_path.is_file():
        raise ValueError(f"architecture evidence signing key not found: {signing_key_path}")
    private_key = serialization.load_pem_private_key(
        signing_key_path.read_bytes(),
        password=None,
    )
    if not isinstance(private_key, Ed25519PrivateKey):
        raise ValueError("architecture evidence signing key is not Ed25519")
    public_der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return {
        "algorithm": "ed25519",
        "public_key_sha256": hashlib.sha256(public_der).hexdigest(),
        "signature_b64": base64.b64encode(private_key.sign(canonical_bytes(payload))).decode(
            "ascii"
        ),
    }


def verify_attested_payload(
    data: dict[str, Any],
    *,
    digest_field: str,
    trusted_public_key_pem: bytes,
) -> None:
    """Verify content digest, signer identity, and Ed25519 signature."""

    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    claimed_digest = data.get(digest_field)
    signature_data = data.get("signature")
    if not isinstance(claimed_digest, str) or len(claimed_digest) != 64:
        raise ValueError(f"architecture evidence is missing {digest_field}")
    if not isinstance(signature_data, dict):
        raise ValueError("architecture evidence is missing its detached signature")
    unsigned = {
        key: value
        for key, value in data.items()
        if key not in {digest_field, "signature"}
    }
    if claimed_digest != payload_sha256(unsigned):
        raise ValueError("architecture evidence attestation mismatch")
    if signature_data.get("algorithm") != "ed25519":
        raise ValueError("architecture evidence signature algorithm is not Ed25519")
    try:
        public_key = serialization.load_pem_public_key(trusted_public_key_pem)
    except (TypeError, ValueError) as exc:
        raise ValueError("architecture evidence trust root is invalid") from exc
    if not isinstance(public_key, Ed25519PublicKey):
        raise ValueError("architecture evidence trust root is not Ed25519")
    public_der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    if signature_data.get("public_key_sha256") != hashlib.sha256(public_der).hexdigest():
        raise ValueError("architecture evidence signer does not match the pinned trust root")
    try:
        signature = base64.b64decode(signature_data.get("signature_b64", ""), validate=True)
        public_key.verify(
            signature,
            canonical_bytes({**unsigned, digest_field: claimed_digest}),
        )
    except (binascii.Error, InvalidSignature, TypeError, ValueError) as exc:
        raise ValueError("architecture evidence signature verification failed") from exc
