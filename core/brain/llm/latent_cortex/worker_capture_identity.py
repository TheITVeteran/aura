"""Boot-scoped signing identity for resident cognitive-state captures."""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes

WORKER_CAPTURE_IDENTITY_SCHEMA_V1: Final = "aura.rlc.worker_capture_identity.v1"
WORKER_CAPTURE_IDENTITY_SCHEMA: Final = "aura.rlc.worker_capture_identity.v2"
WORKER_CAPTURE_LAUNCH_CHALLENGE_SCHEMA: Final = (
    "aura.rlc.worker_capture_launch_challenge.v1"
)
WORKER_CAPTURE_LAUNCH_ATTESTATION_SCHEMA: Final = (
    "aura.rlc.worker_capture_launch_attestation.v1"
)
WORKER_CAPTURE_ORIGIN_BINDING_SCHEMA: Final = "aura.rlc.worker_capture_origin_binding.v1"
_MAX_CHALLENGE_LIFETIME_S: Final = 600
_WORKER_CAPTURE_IDENTITY_FIELDS_V1: Final = {
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
WORKER_CAPTURE_IDENTITY_FIELDS: Final = _WORKER_CAPTURE_IDENTITY_FIELDS_V1 | {
    "launch_challenge_sha256",
}
WORKER_CAPTURE_LAUNCH_CHALLENGE_FIELDS: Final = {
    "schema",
    "algorithm",
    "challenge_id",
    "challenge_nonce_b64",
    "challenge_nonce_sha256",
    "issued_at_unix",
    "not_after_unix",
    "supervisor_public_key_b64",
    "supervisor_key_id",
    "challenge_sha256",
}
WORKER_CAPTURE_LAUNCH_ATTESTATION_FIELDS: Final = {
    "schema",
    "algorithm",
    "challenge_sha256",
    "worker_identity_sha256",
    "worker_boot_id",
    "worker_pid",
    "attested_at_unix",
    "supervisor_key_id",
    "signed_payload_sha256",
    "signature_b64",
    "attestation_sha256",
}
WORKER_CAPTURE_ORIGIN_BINDING_FIELDS: Final = {
    "schema",
    "worker_identity",
    "launch_challenge",
    "launch_attestation",
    "binding_sha256",
}


class WorkerCaptureIdentityError(ValueError):
    """A stable worker capture-identity validation failure."""


@dataclass(frozen=True, slots=True)
class WorkerCaptureSigningIdentity:
    """Worker-private signer plus its public, self-authenticating identity."""

    private_key: Ed25519PrivateKey = field(repr=False)
    public_identity: dict[str, Any]


@dataclass(frozen=True, slots=True)
class WorkerCaptureLaunchAuthority:
    """Parent-private launch signer and the one bounded public challenge."""

    private_key: Ed25519PrivateKey = field(repr=False)
    challenge: dict[str, Any]


def _fail(code: str) -> None:
    if not isinstance(code, str) or not code or code != code.strip():
        raise WorkerCaptureIdentityError("worker_capture_identity_error_code_invalid")
    raise WorkerCaptureIdentityError(code)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: Any, *, code: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        _fail(code)
    return digest


def _positive_int(value: Any, *, code: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(code)
    return value


def _hex_id(value: Any, *, code: str) -> str:
    identifier = str(value or "").strip().lower()
    if len(identifier) != 32 or any(
        character not in "0123456789abcdef" for character in identifier
    ):
        _fail(code)
    return identifier


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


def _decode_public_raw(value: Any, *, code: str) -> bytes:
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, TypeError, ValueError):
        _fail(code)
    if len(raw) != 32:
        _fail(code)
    return raw


def _expected_public_raw(value: Any) -> bytes:
    if isinstance(value, Ed25519PublicKey):
        return _public_raw(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) == 32:
            return raw
        try:
            public = serialization.load_pem_public_key(raw)
        except (TypeError, ValueError) as exc:
            raise WorkerCaptureIdentityError(
                "worker_capture_supervisor_public_key_invalid"
            ) from exc
        if isinstance(public, Ed25519PublicKey):
            return _public_raw(public)
    _fail("worker_capture_supervisor_public_key_invalid")


def build_worker_capture_launch_authority(
    *,
    issued_at_unix: int | None = None,
    lifetime_s: int = 300,
    private_key: Ed25519PrivateKey | None = None,
    challenge_nonce: bytes | None = None,
    challenge_id: str | None = None,
) -> WorkerCaptureLaunchAuthority:
    """Mint one parent-held signer and one challenge for a single worker spawn."""

    issued_at = int(time.time()) if issued_at_unix is None else _positive_int(
        issued_at_unix,
        code="worker_capture_launch_issued_at_invalid",
    )
    if type(lifetime_s) is not int or not 1 <= lifetime_s <= _MAX_CHALLENGE_LIFETIME_S:
        _fail("worker_capture_launch_lifetime_invalid")
    signer = private_key or Ed25519PrivateKey.generate()
    if not isinstance(signer, Ed25519PrivateKey):
        _fail("worker_capture_launch_private_key_invalid")
    nonce = challenge_nonce or secrets.token_bytes(32)
    if not isinstance(nonce, bytes) or len(nonce) != 32:
        _fail("worker_capture_launch_nonce_invalid")
    identifier = _hex_id(
        challenge_id or secrets.token_hex(16),
        code="worker_capture_launch_challenge_id_invalid",
    )
    supervisor_raw = _public_raw(signer.public_key())
    body = {
        "schema": WORKER_CAPTURE_LAUNCH_CHALLENGE_SCHEMA,
        "algorithm": "Ed25519",
        "challenge_id": identifier,
        "challenge_nonce_b64": base64.b64encode(nonce).decode("ascii"),
        "challenge_nonce_sha256": _sha256_bytes(nonce),
        "issued_at_unix": issued_at,
        "not_after_unix": issued_at + lifetime_s,
        "supervisor_public_key_b64": base64.b64encode(supervisor_raw).decode("ascii"),
        "supervisor_key_id": _sha256_bytes(supervisor_raw),
    }
    challenge = {**body, "challenge_sha256": _sha256_bytes(canonical_json_bytes(body))}
    return WorkerCaptureLaunchAuthority(private_key=signer, challenge=challenge)


def validate_worker_capture_launch_challenge(
    value: Any,
    *,
    now_unix: int | None = None,
    expected_supervisor_public_key: Any = None,
) -> dict[str, Any]:
    """Validate a public launch challenge, optionally for current admission."""

    if not isinstance(value, Mapping) or set(value) != WORKER_CAPTURE_LAUNCH_CHALLENGE_FIELDS:
        _fail("worker_capture_launch_challenge_fields")
    challenge = dict(value)
    _hex_id(
        challenge.get("challenge_id"),
        code="worker_capture_launch_challenge_id_invalid",
    )
    issued_at = _positive_int(
        challenge.get("issued_at_unix"),
        code="worker_capture_launch_issued_at_invalid",
    )
    not_after = _positive_int(
        challenge.get("not_after_unix"),
        code="worker_capture_launch_not_after_invalid",
    )
    if (
        challenge.get("schema") != WORKER_CAPTURE_LAUNCH_CHALLENGE_SCHEMA
        or challenge.get("algorithm") != "Ed25519"
        or not_after <= issued_at
        or not_after - issued_at > _MAX_CHALLENGE_LIFETIME_S
    ):
        _fail("worker_capture_launch_challenge_invalid")
    try:
        nonce = base64.b64decode(challenge.get("challenge_nonce_b64"), validate=True)
    except (binascii.Error, TypeError, ValueError):
        _fail("worker_capture_launch_nonce_invalid")
    supervisor_raw = _decode_public_raw(
        challenge.get("supervisor_public_key_b64"),
        code="worker_capture_supervisor_public_key_invalid",
    )
    if (
        len(nonce) != 32
        or challenge.get("challenge_nonce_sha256") != _sha256_bytes(nonce)
        or challenge.get("supervisor_key_id") != _sha256_bytes(supervisor_raw)
    ):
        _fail("worker_capture_launch_challenge_invalid")
    body = {
        name: challenge[name]
        for name in WORKER_CAPTURE_LAUNCH_CHALLENGE_FIELDS - {"challenge_sha256"}
    }
    if challenge.get("challenge_sha256") != _sha256_bytes(canonical_json_bytes(body)):
        _fail("worker_capture_launch_challenge_hash_mismatch")
    if expected_supervisor_public_key is not None and supervisor_raw != _expected_public_raw(
        expected_supervisor_public_key
    ):
        _fail("worker_capture_supervisor_public_key_mismatch")
    if now_unix is not None:
        current = _positive_int(now_unix, code="worker_capture_launch_now_invalid")
        if current < issued_at or current > not_after:
            _fail("worker_capture_launch_challenge_not_current")
    return challenge


def build_worker_capture_identity(
    *,
    worker_boot_id: str,
    worker_pid: int | None = None,
    private_key: Ed25519PrivateKey | None = None,
    launch_challenge: Mapping[str, Any] | None = None,
    now_unix: int | None = None,
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
    challenge = None
    if launch_challenge is not None:
        challenge = validate_worker_capture_launch_challenge(
            launch_challenge,
            now_unix=int(time.time()) if now_unix is None else now_unix,
        )
    schema = (
        WORKER_CAPTURE_IDENTITY_SCHEMA
        if challenge is not None
        else WORKER_CAPTURE_IDENTITY_SCHEMA_V1
    )
    body = {
        "schema": schema,
        "algorithm": "Ed25519",
        "worker_boot_id": boot_id,
        "worker_pid": pid,
        "public_key_b64": base64.b64encode(raw).decode("ascii"),
        "key_id": _sha256_bytes(raw),
    }
    if challenge is not None:
        body["launch_challenge_sha256"] = challenge["challenge_sha256"]
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

    if not isinstance(value, dict):
        _fail("worker_capture_identity_fields")
    identity = dict(value)
    schema = identity.get("schema")
    expected_fields = (
        WORKER_CAPTURE_IDENTITY_FIELDS
        if schema == WORKER_CAPTURE_IDENTITY_SCHEMA
        else _WORKER_CAPTURE_IDENTITY_FIELDS_V1
    )
    if schema not in {WORKER_CAPTURE_IDENTITY_SCHEMA_V1, WORKER_CAPTURE_IDENTITY_SCHEMA} or set(
        identity
    ) != expected_fields:
        _fail("worker_capture_identity_fields")
    boot_id = _boot_id(identity.get("worker_boot_id"))
    pid = identity.get("worker_pid")
    if (
        identity.get("algorithm") != "Ed25519"
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
                    for name in expected_fields - {"identity_sha256"}
                }
            )
        )
    ):
        _fail("worker_capture_identity_hash_mismatch")
    body = {
        "schema": schema,
        "algorithm": "Ed25519",
        "worker_boot_id": boot_id,
        "worker_pid": pid,
        "public_key_b64": identity["public_key_b64"],
        "key_id": identity["key_id"],
    }
    if schema == WORKER_CAPTURE_IDENTITY_SCHEMA:
        body["launch_challenge_sha256"] = _sha256(
            identity.get("launch_challenge_sha256"),
            code="worker_capture_launch_challenge_hash_invalid",
        )
    signed = canonical_json_bytes(body)
    if identity.get("signed_payload_sha256") != _sha256_bytes(signed):
        _fail("worker_capture_identity_payload_mismatch")
    try:
        Ed25519PublicKey.from_public_bytes(raw).verify(signature, signed)
    except (InvalidSignature, ValueError) as exc:
        raise WorkerCaptureIdentityError("worker_capture_identity_signature_invalid") from exc
    return identity


def build_worker_capture_launch_attestation(
    authority: WorkerCaptureLaunchAuthority,
    worker_identity: Mapping[str, Any],
    *,
    attested_at_unix: int,
    expected_worker_pid: int,
) -> dict[str, Any]:
    """Parent-sign one challenge-bound worker identity after process start."""

    if not isinstance(authority, WorkerCaptureLaunchAuthority):
        _fail("worker_capture_launch_authority_invalid")
    challenge = validate_worker_capture_launch_challenge(
        authority.challenge,
        now_unix=attested_at_unix,
        expected_supervisor_public_key=authority.private_key.public_key(),
    )
    identity = validate_worker_capture_identity(worker_identity)
    if identity.get("schema") != WORKER_CAPTURE_IDENTITY_SCHEMA:
        _fail("worker_capture_launch_identity_unbound")
    pid = _positive_int(
        expected_worker_pid,
        code="worker_capture_launch_expected_pid_invalid",
    )
    if (
        identity.get("worker_pid") != pid
        or identity.get("launch_challenge_sha256") != challenge["challenge_sha256"]
    ):
        _fail("worker_capture_launch_identity_mismatch")
    attested_at = _positive_int(
        attested_at_unix,
        code="worker_capture_launch_attested_at_invalid",
    )
    body = {
        "schema": WORKER_CAPTURE_LAUNCH_ATTESTATION_SCHEMA,
        "algorithm": "Ed25519",
        "challenge_sha256": challenge["challenge_sha256"],
        "worker_identity_sha256": identity["identity_sha256"],
        "worker_boot_id": identity["worker_boot_id"],
        "worker_pid": identity["worker_pid"],
        "attested_at_unix": attested_at,
        "supervisor_key_id": challenge["supervisor_key_id"],
    }
    signed = canonical_json_bytes(body)
    attested = {
        **body,
        "signed_payload_sha256": _sha256_bytes(signed),
        "signature_b64": base64.b64encode(authority.private_key.sign(signed)).decode("ascii"),
    }
    return {
        **attested,
        "attestation_sha256": _sha256_bytes(canonical_json_bytes(attested)),
    }


def validate_worker_capture_launch_attestation(
    value: Any,
    *,
    worker_identity: Mapping[str, Any],
    launch_challenge: Mapping[str, Any],
    expected_supervisor_public_key: Any,
    now_unix: int | None = None,
) -> dict[str, Any]:
    """Verify the parent signature from an independently expected public key."""

    if not isinstance(value, Mapping) or set(value) != WORKER_CAPTURE_LAUNCH_ATTESTATION_FIELDS:
        _fail("worker_capture_launch_attestation_fields")
    attestation = dict(value)
    identity = validate_worker_capture_identity(worker_identity)
    challenge = validate_worker_capture_launch_challenge(
        launch_challenge,
        now_unix=now_unix,
        expected_supervisor_public_key=expected_supervisor_public_key,
    )
    pid = _positive_int(
        attestation.get("worker_pid"),
        code="worker_capture_launch_attestation_pid_invalid",
    )
    attested_at = _positive_int(
        attestation.get("attested_at_unix"),
        code="worker_capture_launch_attested_at_invalid",
    )
    if (
        attestation.get("schema") != WORKER_CAPTURE_LAUNCH_ATTESTATION_SCHEMA
        or attestation.get("algorithm") != "Ed25519"
        or attestation.get("challenge_sha256") != challenge["challenge_sha256"]
        or attestation.get("worker_identity_sha256") != identity["identity_sha256"]
        or attestation.get("worker_boot_id") != identity["worker_boot_id"]
        or pid != identity["worker_pid"]
        or attestation.get("supervisor_key_id") != challenge["supervisor_key_id"]
        or identity.get("launch_challenge_sha256") != challenge["challenge_sha256"]
        or attested_at < challenge["issued_at_unix"]
        or attested_at > challenge["not_after_unix"]
    ):
        _fail("worker_capture_launch_attestation_invalid")
    body = {
        name: attestation[name]
        for name in WORKER_CAPTURE_LAUNCH_ATTESTATION_FIELDS
        - {"signed_payload_sha256", "signature_b64", "attestation_sha256"}
    }
    signed = canonical_json_bytes(body)
    if (
        attestation.get("signed_payload_sha256") != _sha256_bytes(signed)
        or attestation.get("attestation_sha256")
        != _sha256_bytes(
            canonical_json_bytes(
                {
                    name: attestation[name]
                    for name in WORKER_CAPTURE_LAUNCH_ATTESTATION_FIELDS
                    - {"attestation_sha256"}
                }
            )
        )
    ):
        _fail("worker_capture_launch_attestation_hash_mismatch")
    signature_b64 = attestation.get("signature_b64")
    try:
        signature = base64.b64decode(signature_b64, validate=True)
    except (binascii.Error, TypeError, ValueError):
        _fail("worker_capture_launch_attestation_encoding_invalid")
    try:
        Ed25519PublicKey.from_public_bytes(
            _expected_public_raw(expected_supervisor_public_key)
        ).verify(signature, signed)
    except (InvalidSignature, ValueError) as exc:
        raise WorkerCaptureIdentityError(
            "worker_capture_launch_attestation_signature_invalid"
        ) from exc
    return attestation


def build_worker_capture_origin_binding(
    authority: WorkerCaptureLaunchAuthority,
    worker_identity: Mapping[str, Any],
    *,
    attested_at_unix: int,
    expected_worker_pid: int,
) -> dict[str, Any]:
    """Build the public, independently verifiable worker launch chain."""

    identity = validate_worker_capture_identity(worker_identity)
    attestation = build_worker_capture_launch_attestation(
        authority,
        identity,
        attested_at_unix=attested_at_unix,
        expected_worker_pid=expected_worker_pid,
    )
    body = {
        "schema": WORKER_CAPTURE_ORIGIN_BINDING_SCHEMA,
        "worker_identity": identity,
        "launch_challenge": dict(authority.challenge),
        "launch_attestation": attestation,
    }
    return {**body, "binding_sha256": _sha256_bytes(canonical_json_bytes(body))}


def validate_worker_capture_origin_binding(
    value: Any,
    *,
    expected_supervisor_public_key: Any,
    now_unix: int | None = None,
) -> dict[str, Any]:
    """Validate a worker key as a child of the expected live supervisor."""

    if not isinstance(value, Mapping) or set(value) != WORKER_CAPTURE_ORIGIN_BINDING_FIELDS:
        _fail("worker_capture_origin_binding_fields")
    binding = dict(value)
    if binding.get("schema") != WORKER_CAPTURE_ORIGIN_BINDING_SCHEMA:
        _fail("worker_capture_origin_binding_invalid")
    body = {
        name: binding[name]
        for name in WORKER_CAPTURE_ORIGIN_BINDING_FIELDS - {"binding_sha256"}
    }
    if binding.get("binding_sha256") != _sha256_bytes(canonical_json_bytes(body)):
        _fail("worker_capture_origin_binding_hash_mismatch")
    identity = validate_worker_capture_identity(binding.get("worker_identity"))
    challenge = validate_worker_capture_launch_challenge(
        binding.get("launch_challenge"),
        now_unix=now_unix,
        expected_supervisor_public_key=expected_supervisor_public_key,
    )
    validate_worker_capture_launch_attestation(
        binding.get("launch_attestation"),
        worker_identity=identity,
        launch_challenge=challenge,
        expected_supervisor_public_key=expected_supervisor_public_key,
        now_unix=now_unix,
    )
    return binding


__all__ = [
    "WORKER_CAPTURE_IDENTITY_FIELDS",
    "WORKER_CAPTURE_IDENTITY_SCHEMA",
    "WORKER_CAPTURE_IDENTITY_SCHEMA_V1",
    "WORKER_CAPTURE_LAUNCH_ATTESTATION_FIELDS",
    "WORKER_CAPTURE_LAUNCH_ATTESTATION_SCHEMA",
    "WORKER_CAPTURE_LAUNCH_CHALLENGE_FIELDS",
    "WORKER_CAPTURE_LAUNCH_CHALLENGE_SCHEMA",
    "WORKER_CAPTURE_ORIGIN_BINDING_FIELDS",
    "WORKER_CAPTURE_ORIGIN_BINDING_SCHEMA",
    "WorkerCaptureIdentityError",
    "WorkerCaptureLaunchAuthority",
    "WorkerCaptureSigningIdentity",
    "build_worker_capture_identity",
    "build_worker_capture_launch_attestation",
    "build_worker_capture_launch_authority",
    "build_worker_capture_origin_binding",
    "validate_worker_capture_identity",
    "validate_worker_capture_launch_attestation",
    "validate_worker_capture_launch_challenge",
    "validate_worker_capture_origin_binding",
]
