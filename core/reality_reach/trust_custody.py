"""Key-custodied, rollback-resistant storage for physical attachment trust.

The filesystem envelope contains ciphertext and non-secret commit metadata
only.  Encryption keys and the monotonic commit anchor live in macOS Keychain.
That separation makes a copied or edited trust file insufficient to mint,
extend, or replay a physical relationship.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import stat
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Never, Protocol, runtime_checkable

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core.runtime.atomic_writer import (
    atomic_write_text,
    durable_unlink,
    interprocess_file_lock,
)
from core.security.zenith_secrets import KeychainBackend, require_keychain_backend

TRUST_ENVELOPE_SCHEMA = "aura.reality-attachment-trust.envelope.v2"
TRUST_BODY_SCHEMA = "aura.reality-attachment-trust.body.v2"
TRUST_KEYRING_SCHEMA = "aura.reality-attachment-trust.keyring.v1"
TRUST_ANCHOR_SCHEMA = "aura.reality-attachment-trust.anchor.v1"
TRUST_CUSTODY_IDENTITY_SCHEMA = "aura.reality-attachment-trust.custody.v1"

_DEFAULT_SERVICE = "AuraRealityReachTrust"
_DEFAULT_KEYRING_ACCOUNT = "attachment-state-keyring-v1"
_DEFAULT_ANCHOR_ACCOUNT = "attachment-state-anchor-v1"
_ZERO_DIGEST = "sha256:" + "0" * 64
_MAX_KEY_VERSIONS = 4
_MAX_STATE_BYTES = 2 * 1024 * 1024
_MAX_ENVELOPE_BYTES = 3 * 1024 * 1024
_MAX_KEYRING_BYTES = 32 * 1024
_MAX_ANCHOR_BYTES = 8 * 1024


class AttachmentTrustStoreError(RuntimeError):
    """Stable fail-closed error raised at the physical trust boundary."""


def _fail(code: str) -> Never:
    raise AttachmentTrustStoreError(code)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AttachmentTrustStoreError("attachment_trust_json_invalid") from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _decode_b64(value: Any, *, role: str) -> bytes:
    if not isinstance(value, str):
        _fail(f"{role}_invalid")
    try:
        return base64.b64decode(value, validate=True)
    except (TypeError, ValueError) as exc:
        raise AttachmentTrustStoreError(f"{role}_invalid") from exc


def _strict_document(encoded: str, *, role: str) -> dict[str, Any]:
    def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(encoded, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AttachmentTrustStoreError(f"{role}_invalid") from exc
    if not isinstance(value, dict):
        _fail(f"{role}_invalid")
    return value


@runtime_checkable
class AttachmentTrustStore(Protocol):
    """Authenticated persistence boundary used by the attachment broker."""

    @property
    def identity(self) -> Mapping[str, Any]: ...

    def load(self) -> Mapping[str, Any] | None: ...

    def save(self, body: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def rotate_and_save(self, body: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def status(self) -> Mapping[str, Any]: ...


class KeychainAttachmentTrustStore:
    """Encrypt trust state and anchor its commit head outside the filesystem.

    Save ordering is deliberately file first, Keychain anchor second.  A crash
    between those operations leaves exactly one recoverable state: sequence
    ``anchor + 1`` whose predecessor is the current anchor.  Every older,
    divergent, or multi-step-ahead envelope is refused as rollback or replay.
    """

    def __init__(
        self,
        backend: KeychainBackend,
        state_path: Path,
        *,
        service: str = _DEFAULT_SERVICE,
        keyring_account: str = _DEFAULT_KEYRING_ACCOUNT,
        anchor_account: str = _DEFAULT_ANCHOR_ACCOUNT,
        create_if_missing: bool = True,
    ) -> None:
        if not isinstance(state_path, Path):
            raise TypeError("state_path must be a Path")
        if not service or not keyring_account or not anchor_account:
            raise ValueError("Keychain service and account names must be present")
        if keyring_account == anchor_account:
            raise ValueError("keyring and anchor accounts must be distinct")
        self._backend = backend
        self._state_path = state_path
        self._lock_path = state_path.with_name(state_path.name + ".lock")
        self._service = service
        self._keyring_account = keyring_account
        self._anchor_account = anchor_account
        self._closed = False
        self._last_error = ""
        self._recovered_commits = 0
        self._rotations = 0
        self._committed_sequence = 0
        self._keyring = self._load_or_create_keyring(create_if_missing=create_if_missing)
        self._identity = self._build_identity(self._keyring)

    @classmethod
    def provision_system(
        cls,
        state_path: Path,
    ) -> KeychainAttachmentTrustStore:
        """Provision the Keychain root on first boot and confirm its write."""

        return cls(require_keychain_backend(), state_path, create_if_missing=True)

    @classmethod
    def from_system(
        cls,
        state_path: Path,
    ) -> KeychainAttachmentTrustStore:
        """Open an already provisioned trust root without mutating Keychain."""

        return cls(require_keychain_backend(), state_path, create_if_missing=False)

    def _load_or_create_keyring(self, *, create_if_missing: bool) -> dict[str, Any]:
        try:
            encoded = self._backend.get_password(self._service, self._keyring_account)
        except Exception as exc:  # noqa: BLE001 - external credential boundary
            raise AttachmentTrustStoreError("attachment_trust_keyring_read_failed") from exc
        if encoded is None:
            if self._state_path.exists():
                _fail("attachment_trust_keyring_missing_for_existing_state")
            if not create_if_missing:
                _fail("attachment_trust_keyring_not_provisioned")
            now_ns = max(1, time.time_ns())
            key = secrets.token_bytes(32)
            body = {
                "schema": TRUST_KEYRING_SCHEMA,
                "custody_seed_b64": base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
                "active_version": 1,
                "keys": [
                    {
                        "version": 1,
                        "created_at_ns": now_ns,
                        "key_b64": base64.b64encode(key).decode("ascii"),
                        "key_sha256": _bytes_digest(key),
                    }
                ],
                "updated_at_ns": now_ns,
            }
            document = {**body, "keyring_sha256": _digest(body)}
            self._write_confirmed(self._keyring_account, document, role="keyring")
            return self._validate_keyring(document)
        if len(encoded.encode("utf-8")) > _MAX_KEYRING_BYTES:
            _fail("attachment_trust_keyring_too_large")
        return self._validate_keyring(_strict_document(encoded, role="attachment_trust_keyring"))

    def _validate_keyring(self, value: Mapping[str, Any]) -> dict[str, Any]:
        document = dict(value)
        expected = {
            "schema",
            "custody_seed_b64",
            "active_version",
            "keys",
            "updated_at_ns",
            "keyring_sha256",
        }
        body = {key: item for key, item in document.items() if key != "keyring_sha256"}
        if (
            set(document) != expected
            or document.get("schema") != TRUST_KEYRING_SCHEMA
            or not isinstance(document.get("active_version"), int)
            or isinstance(document.get("active_version"), bool)
            or int(document["active_version"]) <= 0
            or not isinstance(document.get("updated_at_ns"), int)
            or int(document["updated_at_ns"]) <= 0
            or not _is_digest(document.get("keyring_sha256"))
            or not hmac.compare_digest(str(document["keyring_sha256"]), _digest(body))
        ):
            _fail("attachment_trust_keyring_invalid")
        custody_seed = _decode_b64(
            document.get("custody_seed_b64"),
            role="attachment_trust_custody_seed",
        )
        if len(custody_seed) != 32:
            _fail("attachment_trust_custody_seed_invalid")
        raw_keys = document.get("keys")
        if not isinstance(raw_keys, list) or not 1 <= len(raw_keys) <= _MAX_KEY_VERSIONS:
            _fail("attachment_trust_keyring_invalid")
        seen: set[int] = set()
        for raw in raw_keys:
            if not isinstance(raw, dict) or set(raw) != {
                "version",
                "created_at_ns",
                "key_b64",
                "key_sha256",
            }:
                _fail("attachment_trust_key_invalid")
            version = raw.get("version")
            created_at_ns = raw.get("created_at_ns")
            if (
                not isinstance(version, int)
                or isinstance(version, bool)
                or version <= 0
                or version in seen
                or not isinstance(created_at_ns, int)
                or created_at_ns <= 0
            ):
                _fail("attachment_trust_key_invalid")
            key = _decode_b64(raw.get("key_b64"), role="attachment_trust_key")
            if (
                len(key) != 32
                or not _is_digest(raw.get("key_sha256"))
                or not hmac.compare_digest(str(raw["key_sha256"]), _bytes_digest(key))
            ):
                _fail("attachment_trust_key_invalid")
            seen.add(version)
        if int(document["active_version"]) not in seen:
            _fail("attachment_trust_active_key_missing")
        return document

    def _build_identity(self, keyring: Mapping[str, Any]) -> dict[str, Any]:
        seed = _decode_b64(
            keyring.get("custody_seed_b64"),
            role="attachment_trust_custody_seed",
        )
        body = {
            "schema": TRUST_CUSTODY_IDENTITY_SCHEMA,
            "custody_class": "macos_keychain",
            "service": self._service,
            "keyring_account": self._keyring_account,
            "anchor_account": self._anchor_account,
            "algorithm": "AES-256-GCM",
            "custody_id": _bytes_digest(seed),
        }
        return {**body, "identity_sha256": _digest(body)}

    @property
    def identity(self) -> Mapping[str, Any]:
        self._ensure_open()
        return dict(self._identity)

    def _ensure_open(self) -> None:
        if self._closed:
            _fail("attachment_trust_store_closed")

    def _write_confirmed(
        self,
        account: str,
        document: Mapping[str, Any],
        *,
        role: str,
    ) -> None:
        encoded = _canonical_bytes(document).decode("ascii")
        try:
            stored = self._backend.set_password(self._service, account, encoded)
            confirmed = self._backend.get_password(self._service, account)
        except Exception as exc:  # noqa: BLE001 - external credential boundary
            raise AttachmentTrustStoreError(f"attachment_trust_{role}_write_failed") from exc
        if stored is not True or confirmed != encoded:
            _fail(f"attachment_trust_{role}_write_unconfirmed")

    def _key_for_version(self, version: int) -> bytes:
        for item in self._keyring["keys"]:
            if int(item["version"]) == version:
                return _decode_b64(item["key_b64"], role="attachment_trust_key")
        _fail("attachment_trust_envelope_key_unavailable")

    def _read_anchor(self) -> dict[str, Any] | None:
        try:
            encoded = self._backend.get_password(self._service, self._anchor_account)
        except Exception as exc:  # noqa: BLE001 - external credential boundary
            raise AttachmentTrustStoreError("attachment_trust_anchor_read_failed") from exc
        if encoded is None:
            self._committed_sequence = 0
            return None
        if len(encoded.encode("utf-8")) > _MAX_ANCHOR_BYTES:
            _fail("attachment_trust_anchor_too_large")
        document = _strict_document(encoded, role="attachment_trust_anchor")
        expected = {
            "schema",
            "custody_identity_sha256",
            "sequence",
            "envelope_sha256",
            "key_version",
            "committed_at_ns",
            "anchor_sha256",
        }
        body = {key: item for key, item in document.items() if key != "anchor_sha256"}
        if (
            set(document) != expected
            or document.get("schema") != TRUST_ANCHOR_SCHEMA
            or document.get("custody_identity_sha256") != self._identity["identity_sha256"]
            or not isinstance(document.get("sequence"), int)
            or isinstance(document.get("sequence"), bool)
            or int(document["sequence"]) <= 0
            or not _is_digest(document.get("envelope_sha256"))
            or not isinstance(document.get("key_version"), int)
            or int(document["key_version"]) <= 0
            or not isinstance(document.get("committed_at_ns"), int)
            or int(document["committed_at_ns"]) <= 0
            or not _is_digest(document.get("anchor_sha256"))
            or not hmac.compare_digest(str(document["anchor_sha256"]), _digest(body))
        ):
            _fail("attachment_trust_anchor_invalid")
        self._committed_sequence = int(document["sequence"])
        return document

    def _commit_anchor(self, envelope: Mapping[str, Any]) -> None:
        body = {
            "schema": TRUST_ANCHOR_SCHEMA,
            "custody_identity_sha256": self._identity["identity_sha256"],
            "sequence": int(envelope["sequence"]),
            "envelope_sha256": str(envelope["envelope_sha256"]),
            "key_version": int(envelope["key_version"]),
            "committed_at_ns": max(1, time.time_ns()),
        }
        self._write_confirmed(
            self._anchor_account,
            {**body, "anchor_sha256": _digest(body)},
            role="anchor",
        )
        self._committed_sequence = int(envelope["sequence"])

    def _validate_envelope_shape(self, value: Mapping[str, Any]) -> dict[str, Any]:
        envelope = dict(value)
        expected = {
            "schema",
            "algorithm",
            "custody_identity_sha256",
            "key_version",
            "key_sha256",
            "sequence",
            "previous_envelope_sha256",
            "written_at_ns",
            "nonce_b64",
            "ciphertext_b64",
            "envelope_sha256",
        }
        body = {key: item for key, item in envelope.items() if key != "envelope_sha256"}
        if (
            set(envelope) != expected
            or envelope.get("schema") != TRUST_ENVELOPE_SCHEMA
            or envelope.get("algorithm") != "AES-256-GCM"
            or envelope.get("custody_identity_sha256") != self._identity["identity_sha256"]
            or not isinstance(envelope.get("key_version"), int)
            or isinstance(envelope.get("key_version"), bool)
            or int(envelope["key_version"]) <= 0
            or not _is_digest(envelope.get("key_sha256"))
            or not isinstance(envelope.get("sequence"), int)
            or isinstance(envelope.get("sequence"), bool)
            or int(envelope["sequence"]) <= 0
            or not _is_digest(envelope.get("previous_envelope_sha256"))
            or not isinstance(envelope.get("written_at_ns"), int)
            or int(envelope["written_at_ns"]) <= 0
            or not _is_digest(envelope.get("envelope_sha256"))
            or not hmac.compare_digest(str(envelope["envelope_sha256"]), _digest(body))
        ):
            _fail("attachment_trust_envelope_invalid")
        nonce = _decode_b64(envelope.get("nonce_b64"), role="attachment_trust_nonce")
        ciphertext = _decode_b64(
            envelope.get("ciphertext_b64"),
            role="attachment_trust_ciphertext",
        )
        if (
            len(nonce) != 12
            or len(ciphertext) < 17
            or len(ciphertext) > _MAX_STATE_BYTES + 16
        ):
            _fail("attachment_trust_envelope_invalid")
        return envelope

    @staticmethod
    def _aad(envelope: Mapping[str, Any]) -> bytes:
        return _canonical_bytes(
            {
                "schema": envelope["schema"],
                "algorithm": envelope["algorithm"],
                "custody_identity_sha256": envelope["custody_identity_sha256"],
                "key_version": envelope["key_version"],
                "key_sha256": envelope["key_sha256"],
                "sequence": envelope["sequence"],
                "previous_envelope_sha256": envelope["previous_envelope_sha256"],
                "written_at_ns": envelope["written_at_ns"],
            }
        )

    def _open_envelope(
        self,
        envelope: Mapping[str, Any],
        *,
        anchor: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], bool]:
        value = self._validate_envelope_shape(envelope)
        sequence = int(value["sequence"])
        envelope_sha256 = str(value["envelope_sha256"])
        crash_recovery = False
        if anchor is None:
            if sequence != 1 or value["previous_envelope_sha256"] != _ZERO_DIGEST:
                _fail("attachment_trust_unanchored_state_refused")
            crash_recovery = True
        else:
            anchor_sequence = int(anchor["sequence"])
            anchor_digest = str(anchor["envelope_sha256"])
            if sequence == anchor_sequence and hmac.compare_digest(
                envelope_sha256,
                anchor_digest,
            ):
                pass
            elif (
                sequence == anchor_sequence + 1
                and hmac.compare_digest(
                    str(value["previous_envelope_sha256"]),
                    anchor_digest,
                )
            ):
                crash_recovery = True
            elif sequence <= anchor_sequence:
                _fail("attachment_trust_rollback_or_replay_refused")
            else:
                _fail("attachment_trust_commit_fork_refused")
        key = self._key_for_version(int(value["key_version"]))
        if not hmac.compare_digest(_bytes_digest(key), str(value["key_sha256"])):
            _fail("attachment_trust_envelope_key_mismatch")
        nonce = _decode_b64(value["nonce_b64"], role="attachment_trust_nonce")
        ciphertext = _decode_b64(
            value["ciphertext_b64"],
            role="attachment_trust_ciphertext",
        )
        try:
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, self._aad(value))
        except InvalidTag as exc:
            raise AttachmentTrustStoreError(
                "attachment_trust_authentication_failed"
            ) from exc
        if len(plaintext) > _MAX_STATE_BYTES:
            _fail("attachment_trust_plaintext_too_large")
        try:
            decoded_plaintext = plaintext.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AttachmentTrustStoreError("attachment_trust_plaintext_invalid") from exc
        document = _strict_document(decoded_plaintext, role="attachment_trust_plaintext")
        if set(document) != {"schema", "body", "body_sha256"}:
            _fail("attachment_trust_plaintext_invalid")
        body = document.get("body")
        if (
            document.get("schema") != TRUST_BODY_SCHEMA
            or not isinstance(body, dict)
            or not _is_digest(document.get("body_sha256"))
            or not hmac.compare_digest(str(document["body_sha256"]), _digest(body))
        ):
            _fail("attachment_trust_plaintext_invalid")
        return body, crash_recovery

    def _read_state_document(self) -> dict[str, Any]:
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self._state_path, flags)
        except OSError as exc:
            raise AttachmentTrustStoreError("attachment_trust_state_read_failed") from exc
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                _fail("attachment_trust_state_not_regular")
            if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
                _fail("attachment_trust_state_owner_invalid")
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                _fail("attachment_trust_state_permissions_invalid")
            if metadata.st_size > _MAX_ENVELOPE_BYTES:
                _fail("attachment_trust_state_too_large")
            with os.fdopen(fd, "rb", closefd=False) as handle:
                payload = handle.read(_MAX_ENVELOPE_BYTES + 1)
            if len(payload) > _MAX_ENVELOPE_BYTES:
                _fail("attachment_trust_state_too_large")
        finally:
            os.close(fd)
        try:
            encoded = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise AttachmentTrustStoreError("attachment_trust_state_invalid") from exc
        return _strict_document(encoded, role="attachment_trust_state")

    def load(self) -> Mapping[str, Any] | None:
        self._ensure_open()
        with interprocess_file_lock(self._lock_path):
            try:
                anchor = self._read_anchor()
                if not self._state_path.exists():
                    if anchor is not None:
                        _fail("attachment_trust_state_missing_after_commit")
                    self._last_error = ""
                    return None
                body, recover = self._open_envelope(
                    self._read_state_document(),
                    anchor=anchor,
                )
                if recover:
                    envelope = self._validate_envelope_shape(self._read_state_document())
                    self._commit_anchor(envelope)
                    self._recovered_commits += 1
                self._last_error = ""
                return body
            except AttachmentTrustStoreError as exc:
                self._last_error = str(exc)
                raise

    def _seal(
        self,
        body: Mapping[str, Any],
        *,
        sequence: int,
        previous_envelope_sha256: str,
    ) -> dict[str, Any]:
        canonical_body = json.loads(_canonical_bytes(dict(body)).decode("ascii"))
        plaintext = _canonical_bytes(
            {
                "schema": TRUST_BODY_SCHEMA,
                "body": canonical_body,
                "body_sha256": _digest(canonical_body),
            }
        )
        if len(plaintext) > _MAX_STATE_BYTES:
            _fail("attachment_trust_plaintext_too_large")
        version = int(self._keyring["active_version"])
        key = self._key_for_version(version)
        header = {
            "schema": TRUST_ENVELOPE_SCHEMA,
            "algorithm": "AES-256-GCM",
            "custody_identity_sha256": self._identity["identity_sha256"],
            "key_version": version,
            "key_sha256": _bytes_digest(key),
            "sequence": sequence,
            "previous_envelope_sha256": previous_envelope_sha256,
            "written_at_ns": max(1, time.time_ns()),
        }
        nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(key).encrypt(nonce, plaintext, self._aad(header))
        envelope_body = {
            **header,
            "nonce_b64": base64.b64encode(nonce).decode("ascii"),
            "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
        }
        return {**envelope_body, "envelope_sha256": _digest(envelope_body)}

    def _save_locked(self, body: Mapping[str, Any]) -> dict[str, Any]:
        anchor = self._read_anchor()
        previous_document: dict[str, Any] | None = None
        if self._state_path.exists():
            previous_document = self._read_state_document()
            _existing, recover = self._open_envelope(
                previous_document,
                anchor=anchor,
            )
            if recover:
                current = self._validate_envelope_shape(previous_document)
                self._commit_anchor(current)
                self._recovered_commits += 1
                anchor = self._read_anchor()
        elif anchor is not None:
            _fail("attachment_trust_state_missing_after_commit")
        sequence = int(anchor["sequence"]) + 1 if anchor is not None else 1
        previous = str(anchor["envelope_sha256"]) if anchor is not None else _ZERO_DIGEST
        envelope = self._seal(
            body,
            sequence=sequence,
            previous_envelope_sha256=previous,
        )
        atomic_write_text(
            self._state_path,
            _canonical_bytes(envelope).decode("ascii"),
            mode=0o600,
        )
        try:
            self._commit_anchor(envelope)
        except AttachmentTrustStoreError as commit_error:
            try:
                if previous_document is None:
                    durable_unlink(self._state_path, missing_ok=True)
                else:
                    atomic_write_text(
                        self._state_path,
                        _canonical_bytes(previous_document).decode("ascii"),
                        mode=0o600,
                    )
            except OSError as rollback_error:
                raise AttachmentTrustStoreError(
                    "attachment_trust_anchor_and_file_rollback_failed"
                ) from rollback_error
            raise commit_error
        self._last_error = ""
        return {
            "sequence": sequence,
            "envelope_sha256": envelope["envelope_sha256"],
            "key_version": envelope["key_version"],
        }

    def save(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        self._ensure_open()
        if not isinstance(body, Mapping):
            raise TypeError("trust body must be a mapping")
        with interprocess_file_lock(self._lock_path):
            try:
                return self._save_locked(body)
            except AttachmentTrustStoreError as exc:
                self._last_error = str(exc)
                raise

    def _rotate_keyring_locked(self) -> None:
        anchor = self._read_anchor()
        active_version = int(self._keyring["active_version"])
        next_version = max(int(item["version"]) for item in self._keyring["keys"]) + 1
        key = secrets.token_bytes(32)
        keys = [dict(item) for item in self._keyring["keys"]]
        keys.append(
            {
                "version": next_version,
                "created_at_ns": max(1, time.time_ns()),
                "key_b64": base64.b64encode(key).decode("ascii"),
                "key_sha256": _bytes_digest(key),
            }
        )
        must_keep = {
            active_version,
            next_version,
            int(anchor["key_version"]) if anchor is not None else active_version,
        }
        retained = [item for item in keys if int(item["version"]) in must_keep]
        for item in sorted(keys, key=lambda entry: int(entry["version"]), reverse=True):
            if item in retained:
                continue
            if len(retained) >= _MAX_KEY_VERSIONS:
                break
            retained.append(item)
        retained.sort(key=lambda entry: int(entry["version"]))
        body = {
            "schema": TRUST_KEYRING_SCHEMA,
            "custody_seed_b64": self._keyring["custody_seed_b64"],
            "active_version": next_version,
            "keys": retained,
            "updated_at_ns": max(1, time.time_ns()),
        }
        document = {**body, "keyring_sha256": _digest(body)}
        self._write_confirmed(self._keyring_account, document, role="keyring")
        self._keyring = self._validate_keyring(document)
        self._rotations += 1

    def rotate_and_save(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        """Rotate custody and immediately re-encrypt the current trust head."""

        self._ensure_open()
        if not isinstance(body, Mapping):
            raise TypeError("trust body must be a mapping")
        with interprocess_file_lock(self._lock_path):
            try:
                if self._state_path.exists():
                    self.load()
                elif self._read_anchor() is not None:
                    _fail("attachment_trust_state_missing_after_commit")
                self._rotate_keyring_locked()
                return self._save_locked(body)
            except AttachmentTrustStoreError as exc:
                self._last_error = str(exc)
                raise

    def status(self) -> Mapping[str, Any]:
        try:
            self._ensure_open()
        except AttachmentTrustStoreError as exc:
            self._last_error = str(exc)
        return {
            "healthy": not self._closed and not self._last_error,
            "error": self._last_error,
            "custody_class": "macos_keychain",
            "identity_sha256": self._identity["identity_sha256"],
            "active_key_version": int(self._keyring["active_version"]),
            "retained_key_versions": len(self._keyring["keys"]),
            "committed_sequence": self._committed_sequence,
            "recovered_commits": self._recovered_commits,
            "rotations": self._rotations,
            "state_present": self._state_path.exists(),
        }

    def close(self) -> None:
        with interprocess_file_lock(self._lock_path):
            for item in self._keyring.get("keys", []):
                item["key_b64"] = ""
                item["key_sha256"] = _ZERO_DIGEST
            self._closed = True


__all__ = [
    "TRUST_ANCHOR_SCHEMA",
    "TRUST_BODY_SCHEMA",
    "TRUST_CUSTODY_IDENTITY_SCHEMA",
    "TRUST_ENVELOPE_SCHEMA",
    "TRUST_KEYRING_SCHEMA",
    "AttachmentTrustStore",
    "AttachmentTrustStoreError",
    "KeychainAttachmentTrustStore",
]
