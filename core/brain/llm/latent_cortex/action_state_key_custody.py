"""External key custody for SPARK private action-state snapshots."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from collections.abc import Mapping
from typing import Any, Never, Protocol, runtime_checkable

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.security.zenith_secrets import KeychainBackend, require_keychain_backend

SNAPSHOT_KEY_CUSTODY_IDENTITY_SCHEMA = "aura.rlc.snapshot_key_custody.identity.v1"
SNAPSHOT_WRAPPED_KEY_SCHEMA = "aura.rlc.snapshot_key_custody.wrapped_dek.v1"
_DEFAULT_SERVICE = "AuraRLCActionStateCapture"
_DEFAULT_ACCOUNT = "snapshot-wrapping-key-v1"


class SnapshotKeyCustodyError(RuntimeError):
    """Stable fail-closed key-custody error."""


def _fail(code: str) -> Never:
    error = SnapshotKeyCustodyError(code)
    raise error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _decode_b64(value: Any, *, role: str) -> bytes:
    if not isinstance(value, str):
        _fail(f"{role}_invalid")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        _fail(f"{role}_invalid")


@runtime_checkable
class SnapshotKeyCustodian(Protocol):
    """Opaque wrapping-key boundary; raw custody keys never enter the store."""

    @property
    def identity(self) -> Mapping[str, Any]: ...

    def wrap_data_key(self, data_key: bytes, *, context_sha256: str) -> Mapping[str, Any]: ...

    def unwrap_data_key(
        self,
        envelope: Mapping[str, Any],
        *,
        context_sha256: str,
    ) -> bytearray: ...


class KeychainSnapshotKeyCustodian:
    """Wrap snapshot DEKs under a key held by a strict Keychain backend."""

    def __init__(
        self,
        backend: KeychainBackend,
        *,
        service: str = _DEFAULT_SERVICE,
        account: str = _DEFAULT_ACCOUNT,
        create_if_missing: bool = True,
    ) -> None:
        if not service or not account:
            _fail("snapshot_key_custodian_name_invalid")
        self._backend = backend
        self._service = service
        self._account = account
        self._wrapping_key = self._load_or_create_key(create_if_missing=create_if_missing)
        key_id = _sha256_bytes(bytes(self._wrapping_key))
        identity_body = {
            "schema": SNAPSHOT_KEY_CUSTODY_IDENTITY_SCHEMA,
            "custody_class": "macos_keychain",
            "service": service,
            "account": account,
            "algorithm": "AES-256-GCM",
            "wrapping_key_id": key_id,
        }
        self._identity = {**identity_body, "identity_sha256": _digest(identity_body)}

    @classmethod
    def from_system(cls) -> KeychainSnapshotKeyCustodian:
        """Open a pre-provisioned native Keychain key without mutating custody."""

        return cls(require_keychain_backend(), create_if_missing=False)

    @classmethod
    def provision_system(cls) -> KeychainSnapshotKeyCustodian:
        """Provision once in the host coordinator before resident workers spawn."""

        return cls(require_keychain_backend(), create_if_missing=True)

    def _load_or_create_key(self, *, create_if_missing: bool) -> bytearray:
        try:
            encoded = self._backend.get_password(self._service, self._account)
        except Exception as exc:  # noqa: BLE001 - external backend boundary
            raise SnapshotKeyCustodyError("snapshot_key_custodian_read_failed") from exc
        if encoded is None:
            if not create_if_missing:
                _fail("snapshot_key_custodian_not_provisioned")
            candidate = secrets.token_bytes(32)
            encoded = base64.b64encode(candidate).decode("ascii")
            try:
                stored = self._backend.set_password(self._service, self._account, encoded)
                confirmed = self._backend.get_password(self._service, self._account)
            except Exception as exc:  # noqa: BLE001 - external backend boundary
                raise SnapshotKeyCustodyError("snapshot_key_custodian_write_failed") from exc
            if stored is not True or confirmed != encoded:
                _fail("snapshot_key_custodian_write_unconfirmed")
        key = _decode_b64(encoded, role="snapshot_key_custodian_key")
        if len(key) != 32:
            _fail("snapshot_key_custodian_key_invalid")
        return bytearray(key)

    @property
    def identity(self) -> Mapping[str, Any]:
        if not self._wrapping_key:
            _fail("snapshot_key_custodian_closed")
        return dict(self._identity)

    def _aad(self, context_sha256: str) -> bytes:
        if not _is_sha256(context_sha256):
            _fail("snapshot_key_context_invalid")
        return canonical_json_bytes(
            {
                "context_sha256": context_sha256,
                "custody_identity_sha256": self._identity["identity_sha256"],
                "schema": SNAPSHOT_WRAPPED_KEY_SCHEMA,
            }
        )

    def wrap_data_key(self, data_key: bytes, *, context_sha256: str) -> Mapping[str, Any]:
        if not self._wrapping_key:
            _fail("snapshot_key_custodian_closed")
        if not isinstance(data_key, bytes) or len(data_key) != 32:
            _fail("snapshot_data_key_invalid")
        nonce = secrets.token_bytes(12)
        wrapped = AESGCM(bytes(self._wrapping_key)).encrypt(
            nonce,
            data_key,
            self._aad(context_sha256),
        )
        body = {
            "schema": SNAPSHOT_WRAPPED_KEY_SCHEMA,
            "algorithm": "AES-256-GCM",
            "custody_identity": dict(self._identity),
            "context_sha256": context_sha256,
            "plaintext_key_sha256": _sha256_bytes(data_key),
            "nonce_b64": base64.b64encode(nonce).decode("ascii"),
            "wrapped_key_b64": base64.b64encode(wrapped).decode("ascii"),
        }
        return {**body, "envelope_sha256": _digest(body)}

    def unwrap_data_key(
        self,
        envelope: Mapping[str, Any],
        *,
        context_sha256: str,
    ) -> bytearray:
        if not self._wrapping_key:
            _fail("snapshot_key_custodian_closed")
        if not isinstance(envelope, Mapping):
            _fail("snapshot_wrapped_key_invalid")
        value = dict(envelope)
        expected_fields = {
            "schema",
            "algorithm",
            "custody_identity",
            "context_sha256",
            "plaintext_key_sha256",
            "nonce_b64",
            "wrapped_key_b64",
            "envelope_sha256",
        }
        body = {name: item for name, item in value.items() if name != "envelope_sha256"}
        if (
            set(value) != expected_fields
            or value.get("schema") != SNAPSHOT_WRAPPED_KEY_SCHEMA
            or value.get("algorithm") != "AES-256-GCM"
            or value.get("custody_identity") != self._identity
            or value.get("context_sha256") != context_sha256
            or not _is_sha256(value.get("plaintext_key_sha256"))
            or not _is_sha256(value.get("envelope_sha256"))
            or not hmac.compare_digest(value["envelope_sha256"], _digest(body))
        ):
            _fail("snapshot_wrapped_key_invalid")
        nonce = _decode_b64(value.get("nonce_b64"), role="snapshot_wrapped_key_nonce")
        wrapped = _decode_b64(value.get("wrapped_key_b64"), role="snapshot_wrapped_key_ciphertext")
        if len(nonce) != 12 or len(wrapped) != 48:
            _fail("snapshot_wrapped_key_invalid")
        try:
            data_key = AESGCM(bytes(self._wrapping_key)).decrypt(
                nonce,
                wrapped,
                self._aad(context_sha256),
            )
        except InvalidTag as exc:
            raise SnapshotKeyCustodyError("snapshot_wrapped_key_authentication_failed") from exc
        if len(data_key) != 32 or not hmac.compare_digest(
            _sha256_bytes(data_key),
            value["plaintext_key_sha256"],
        ):
            _fail("snapshot_unwrapped_key_invalid")
        return bytearray(data_key)

    def close(self) -> None:
        for index in range(len(self._wrapping_key)):
            self._wrapping_key[index] = 0
        self._wrapping_key.clear()


__all__ = [
    "SNAPSHOT_KEY_CUSTODY_IDENTITY_SCHEMA",
    "SNAPSHOT_WRAPPED_KEY_SCHEMA",
    "KeychainSnapshotKeyCustodian",
    "SnapshotKeyCustodian",
    "SnapshotKeyCustodyError",
]
