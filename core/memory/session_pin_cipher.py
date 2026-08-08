"""Authenticated encryption for explicit user memory pins.

The filesystem and memory index receive ciphertext-only envelopes. The sole
data-encryption key lives in macOS Keychain, and the encrypted body binds each
pin to the principal, surface, and session that created it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core.security.zenith_secrets import KeychainBackend, require_keychain_backend

SESSION_PIN_ENVELOPE_SCHEMA = "aura.session_memory_pin.envelope.v3"
SESSION_PIN_BODY_SCHEMA = "aura.session_memory_pin.body.v1"
SESSION_PIN_INDEX_CONTENT = "Encrypted explicit user memory pin"

_KEYCHAIN_SERVICE = "AuraSessionMemoryPins"
_KEYCHAIN_ACCOUNT = "session-pin-aes256-v1"
_AAD_DOMAIN = b"aura.session-memory-pin.v3\0"
_MAX_CONTENT_CHARS = 240
_MAX_SOURCE_CHARS = 512
_MAX_SESSION_CHARS = 64
_MAX_PRINCIPAL_CHARS = 160
_MAX_SURFACE_CHARS = 32


class SessionPinCipherError(RuntimeError):
    """Stable fail-closed error at the explicit-memory persistence boundary."""


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
        raise SessionPinCipherError("session_pin_json_invalid") from exc


def _decode_key(encoded: str) -> bytes:
    try:
        key = base64.urlsafe_b64decode(encoded.encode("ascii"))
    except (ValueError, UnicodeError) as exc:
        raise SessionPinCipherError("session_pin_key_malformed") from exc
    if len(key) != 32:
        raise SessionPinCipherError("session_pin_key_length_invalid")
    return key


class SessionPinCipher:
    """Seal and open principal-bound memory-pin records with AES-256-GCM."""

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError("Session pin encryption requires a 32-byte key")
        self._cipher = AESGCM(key)
        self._key_id = "sha256:" + hashlib.sha256(key).hexdigest()

    @classmethod
    def from_backend(
        cls,
        backend: KeychainBackend,
        *,
        create_if_missing: bool = True,
    ) -> SessionPinCipher:
        encoded = backend.get_password(_KEYCHAIN_SERVICE, _KEYCHAIN_ACCOUNT)
        if not encoded:
            if not create_if_missing:
                raise SessionPinCipherError("session_pin_key_missing")
            encoded = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")
            if not backend.set_password(_KEYCHAIN_SERVICE, _KEYCHAIN_ACCOUNT, encoded):
                raise SessionPinCipherError("session_pin_key_write_rejected")
            confirmed = backend.get_password(_KEYCHAIN_SERVICE, _KEYCHAIN_ACCOUNT)
            if not confirmed or not hmac.compare_digest(confirmed, encoded):
                raise SessionPinCipherError("session_pin_key_write_unconfirmed")
        return cls(_decode_key(encoded))

    @classmethod
    def from_system(cls) -> SessionPinCipher:
        return cls.from_backend(require_keychain_backend())

    @property
    def key_id(self) -> str:
        return self._key_id

    def seal(
        self,
        *,
        content: str,
        source: str,
        timestamp: str,
        session_id: str,
        principal_id: str,
        principal_surface: str,
    ) -> dict[str, str]:
        body = {
            "content": str(content or "").strip()[:_MAX_CONTENT_CHARS],
            "principal_id": " ".join(str(principal_id or "").strip().split())[
                :_MAX_PRINCIPAL_CHARS
            ],
            "principal_surface": str(principal_surface or "").strip().casefold()[
                :_MAX_SURFACE_CHARS
            ],
            "schema": SESSION_PIN_BODY_SCHEMA,
            "session_id": str(session_id or "")[:_MAX_SESSION_CHARS],
            "source": str(source or "").strip()[:_MAX_SOURCE_CHARS],
            "timestamp": str(timestamp or ""),
        }
        if not body["content"]:
            raise SessionPinCipherError("session_pin_content_missing")
        if not body["principal_id"] or not body["principal_surface"]:
            raise SessionPinCipherError("session_pin_principal_binding_missing")
        header = {
            "key_id": self._key_id,
            "record_id": secrets.token_hex(16),
            "schema": SESSION_PIN_ENVELOPE_SCHEMA,
        }
        nonce = secrets.token_bytes(12)
        ciphertext = self._cipher.encrypt(
            nonce,
            _canonical_bytes(body),
            _AAD_DOMAIN + _canonical_bytes(header),
        )
        return {
            **header,
            "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
            "nonce_b64": base64.b64encode(nonce).decode("ascii"),
        }

    def open(self, envelope: Mapping[str, Any]) -> dict[str, str]:
        if str(envelope.get("schema") or "") != SESSION_PIN_ENVELOPE_SCHEMA:
            raise SessionPinCipherError("session_pin_envelope_schema_invalid")
        header = {
            "key_id": str(envelope.get("key_id") or ""),
            "record_id": str(envelope.get("record_id") or ""),
            "schema": SESSION_PIN_ENVELOPE_SCHEMA,
        }
        if header["key_id"] != self._key_id or len(header["record_id"]) != 32:
            raise SessionPinCipherError("session_pin_envelope_identity_invalid")
        try:
            nonce = base64.b64decode(str(envelope.get("nonce_b64") or ""), validate=True)
            ciphertext = base64.b64decode(
                str(envelope.get("ciphertext_b64") or ""), validate=True
            )
        except (TypeError, ValueError) as exc:
            raise SessionPinCipherError("session_pin_envelope_encoding_invalid") from exc
        if len(nonce) != 12 or len(ciphertext) < 17:
            raise SessionPinCipherError("session_pin_envelope_payload_invalid")
        try:
            raw = self._cipher.decrypt(
                nonce,
                ciphertext,
                _AAD_DOMAIN + _canonical_bytes(header),
            )
        except InvalidTag as exc:
            raise SessionPinCipherError("session_pin_authentication_failed") from exc
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SessionPinCipherError("session_pin_body_invalid") from exc
        if not isinstance(body, dict) or body.get("schema") != SESSION_PIN_BODY_SCHEMA:
            raise SessionPinCipherError("session_pin_body_schema_invalid")
        normalized = {
            "content": str(body.get("content") or "").strip()[:_MAX_CONTENT_CHARS],
            "principal_id": " ".join(
                str(body.get("principal_id") or "").strip().split()
            )[:_MAX_PRINCIPAL_CHARS],
            "principal_surface": str(body.get("principal_surface") or "")
            .strip()
            .casefold()[:_MAX_SURFACE_CHARS],
            "session_id": str(body.get("session_id") or "")[:_MAX_SESSION_CHARS],
            "source": str(body.get("source") or "").strip()[:_MAX_SOURCE_CHARS],
            "timestamp": str(body.get("timestamp") or ""),
        }
        if not normalized["content"]:
            raise SessionPinCipherError("session_pin_content_missing")
        if not normalized["principal_id"] or not normalized["principal_surface"]:
            raise SessionPinCipherError("session_pin_principal_binding_missing")
        return normalized
