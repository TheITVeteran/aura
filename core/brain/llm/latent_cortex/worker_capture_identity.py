"""Boot-scoped signing identity for resident cognitive-state captures."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes

WORKER_CAPTURE_IDENTITY_SCHEMA: Final = "aura.rlc.worker_capture_identity.v1"
WORKER_CAPTURE_IDENTITY_FIELDS: Final = {
    "schema",
    "algorithm",
    "worker_boot_id",
    "worker_pid",
    "public_key_b64",
    "key_id",
    "signed_payload_sha256",
    "signature_b64",
    "identity_sha256",
}


class WorkerCaptureIdentityError(ValueError):
    """A stable worker capture-identity validation failure."""


@dataclass(frozen=True, slots=True)
class WorkerCaptureSigningIdentity:
    """Worker-private signer plus its public, self-authenticating identity."""

    private_key: Ed25519PrivateKey = field(repr=False)
    public_identity: dict[str, Any]


def _fail(code: str) -> None:
    if not isinstance(code, str) or not code or code != code.strip():
        raise WorkerCaptureIdentityError("worker_capture_identity_error_code_invalid")
    raise WorkerCaptureIdentityError(code)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _boot_id(value: Any) -> str:
    boot_id = str(value or "").strip().lower()
    if len(boot_id) != 32 or any(character not in "0123456789abcdef" for character in boot_id):
        _fail("worker_capture_identity_boot_id_invalid")
    return boot_id


def _public_raw(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def build_worker_capture_identity(
    *,
    worker_boot_id: str,
    worker_pid: int | None = None,
    private_key: Ed25519PrivateKey | None = None,
) -> WorkerCaptureSigningIdentity:
    """Generate one non-exported signer for one resident worker boot."""

    boot_id = _boot_id(worker_boot_id)
    pid = os.getpid() if worker_pid is None else worker_pid
    if type(pid) is not int or pid <= 0:
        _fail("worker_capture_identity_pid_invalid")
    signer = private_key or Ed25519PrivateKey.generate()
    if not isinstance(signer, Ed25519PrivateKey):
        _fail("worker_capture_identity_private_key_invalid")
    raw = _public_raw(signer.public_key())
    body = {
        "schema": WORKER_CAPTURE_IDENTITY_SCHEMA,
        "algorithm": "Ed25519",
        "worker_boot_id": boot_id,
        "worker_pid": pid,
        "public_key_b64": base64.b64encode(raw).decode("ascii"),
        "key_id": _sha256_bytes(raw),
    }
    signed = canonical_json_bytes(body)
    attested = {
        **body,
        "signed_payload_sha256": _sha256_bytes(signed),
        "signature_b64": base64.b64encode(signer.sign(signed)).decode("ascii"),
    }
    public_identity = {
        **attested,
        "identity_sha256": _sha256_bytes(canonical_json_bytes(attested)),
    }
    return WorkerCaptureSigningIdentity(
        private_key=signer,
        public_identity=public_identity,
    )


def validate_worker_capture_identity(value: Any) -> dict[str, Any]:
    """Validate the public half of a boot-scoped worker capture identity."""

    if not isinstance(value, dict) or set(value) != WORKER_CAPTURE_IDENTITY_FIELDS:
        _fail("worker_capture_identity_fields")
    identity = dict(value)
    boot_id = _boot_id(identity.get("worker_boot_id"))
    pid = identity.get("worker_pid")
    if (
        identity.get("schema") != WORKER_CAPTURE_IDENTITY_SCHEMA
        or identity.get("algorithm") != "Ed25519"
        or type(pid) is not int
        or pid <= 0
    ):
        _fail("worker_capture_identity_invalid")
    try:
        raw = base64.b64decode(identity["public_key_b64"], validate=True)
        signature = base64.b64decode(identity["signature_b64"], validate=True)
    except (binascii.Error, KeyError, TypeError, ValueError):
        _fail("worker_capture_identity_encoding_invalid")
    if (
        len(raw) != 32
        or identity.get("key_id") != _sha256_bytes(raw)
        or identity.get("identity_sha256")
        != _sha256_bytes(
            canonical_json_bytes(
                {
                    name: identity[name]
                    for name in WORKER_CAPTURE_IDENTITY_FIELDS - {"identity_sha256"}
                }
            )
        )
    ):
        _fail("worker_capture_identity_hash_mismatch")
    body = {
        "schema": WORKER_CAPTURE_IDENTITY_SCHEMA,
        "algorithm": "Ed25519",
        "worker_boot_id": boot_id,
        "worker_pid": pid,
        "public_key_b64": identity["public_key_b64"],
        "key_id": identity["key_id"],
    }
    signed = canonical_json_bytes(body)
    if identity.get("signed_payload_sha256") != _sha256_bytes(signed):
        _fail("worker_capture_identity_payload_mismatch")
    try:
        Ed25519PublicKey.from_public_bytes(raw).verify(signature, signed)
    except (InvalidSignature, ValueError) as exc:
        raise WorkerCaptureIdentityError("worker_capture_identity_signature_invalid") from exc
    return identity


__all__ = [
    "WORKER_CAPTURE_IDENTITY_FIELDS",
    "WORKER_CAPTURE_IDENTITY_SCHEMA",
    "WorkerCaptureIdentityError",
    "WorkerCaptureSigningIdentity",
    "build_worker_capture_identity",
    "validate_worker_capture_identity",
]
