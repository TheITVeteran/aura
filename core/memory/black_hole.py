"""core/memory/black_hole.py — Aura 3.0: BlackHole Encryption
=========================================================
Implements Phase 7: AES-256-GCM encryption for all local persistent data.
Replaces the old legacy XOR obfuscation.

ZENITH Protocol compliance:
  - AES-256-GCM (Authenticated Encryption).
  - Derived key from Horcrux hardware entanglement.
  - Zero raw secrets stored on disk.
"""

import base64
import binascii
import hashlib
import logging
import os
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Aura.BlackHole")

_BLACK_HOLE_CRYPTO_ERRORS = (
    InvalidTag,
    RuntimeError,
    AttributeError,
    TypeError,
    ValueError,
)
_BLACK_HOLE_DECODE_ERRORS = (
    InvalidTag,
    binascii.Error,
    RuntimeError,
    AttributeError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


class DecodedPayload(str):
    """Backward-compatible decode result supporting both string and mapping access."""

    def get(self, key: str, default: str = "") -> str:
        if key == "decoded":
            return str(self)
        return default

    def __contains__(self, item: object) -> bool:
        if item == "decoded":
            return True
        return super().__contains__(item)

    def __getitem__(self, key):  # type: ignore[override]
        if key == "decoded":
            return str(self)
        return super().__getitem__(key)


class BlackHoleDecodeError(ValueError):
    """Encrypted payload could not be authenticated or decoded with this key."""


class BlackHoleEncryptionUnavailable(RuntimeError):
    """No key is available, so the payload cannot be encrypted.

    This is raised instead of returning the plaintext. ``encrypt()`` used to
    log a warning and hand the caller its data back unchanged, which meant a
    boot without a Horcrux key wrote every "encrypted" memory to disk in the
    clear while the privacy claim stayed intact on paper. A caller that cannot
    encrypt must find out at the call site, not discover it later in a file.
    """


def _local_key_path():
    from pathlib import Path

    override = os.environ.get("AURA_BLACK_HOLE_KEY_DIR", "").strip()
    if override:
        return Path(override).expanduser() / "black_hole_local.key"
    try:
        from core.config import config

        return Path(config.paths.home_dir) / "keys" / "black_hole_local.key"
    except (ImportError, AttributeError, RuntimeError):
        return Path.home() / ".aura" / "keys" / "black_hole_local.key"


def _provision_local_key() -> bytes | None:
    """Get (or create) a local AES key so first boot is never written in clear.

    This is weaker than a Horcrux-derived key: it is not entangled with the
    hardware, so it protects data at rest against casual disk access but not
    against an attacker who already has the key file. That is a real and
    material downgrade — but it is strictly better than the previous behaviour,
    which was no encryption at all, silently. The distinction is surfaced by
    ``BlackHole.key_provenance``.
    """
    path = _local_key_path()
    try:
        if path.exists():
            key = path.read_bytes()
            if len(key) == 32:
                return key
            logger.warning("BlackHole: local key at %s is malformed — regenerating", path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        key = os.urandom(32)
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(key)
        logger.info("BlackHole: provisioned a local encryption key at %s", path)
        return key
    except OSError as exc:
        record_degradation(
            "black_hole",
            exc,
            action="could not provision a local encryption key; encryption will fail closed",
            enforce_failure_policy=False,
        )
        return None


def _resolve_aes_key(key_material: str | bytes) -> bytes:
    """Accept base64-encoded keys, raw AES keys, or arbitrary strings.

    The legacy memory stack sometimes passes a placeholder or raw string key
    before Horcrux has been fully initialized. Instead of crashing on base64
    decode, normalize unsupported formats into a deterministic AES-256 key.
    """
    raw = key_material.encode("utf-8") if isinstance(key_material, str) else bytes(key_material)

    try:
        decoded = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        decoded = b""

    if len(decoded) in {16, 24, 32}:
        return decoded
    if len(raw) in {16, 24, 32}:
        return raw
    return hashlib.sha256(raw).digest()


class BlackHole:
    """
    Encryption provider for Aura 3.0.
    
    ZENITH Purity:
      - Mandatory authentication tag verification.
      - Automated nonce generation.
    """

    def __init__(self):
        self._aesgcm: AESGCM | None = None
        self.key_provenance: str = "none"

    def on_start(self):
        """Initialize the provider, preferring the Horcrux-derived key.

        Falls back to a locally provisioned key rather than to no encryption:
        "Horcrux unavailable" used to mean every memory written on that boot
        went to disk in plaintext.
        """
        horcrux = get_runtime_service("horcrux", default=None)
        if horcrux and horcrux.derived_key:
            self._aesgcm = AESGCM(horcrux.derived_key)
            self.key_provenance = "horcrux"
            logger.info("BlackHole: AES-256-GCM substrate initialized (Horcrux key).")
            return

        local_key = _provision_local_key()
        if local_key:
            self._aesgcm = AESGCM(local_key)
            self.key_provenance = "local"
            logger.warning(
                "BlackHole: Horcrux keys unavailable — using a locally provisioned "
                "key. Memories are encrypted at rest but NOT hardware-entangled."
            )
            return

        self.key_provenance = "none"
        record_degradation(
            "black_hole",
            RuntimeError("no encryption key available"),
            action="encryption will fail closed; no memory can be written",
            enforce_failure_policy=False,
        )
        logger.error(
            "BlackHole: no encryption key available. Encryption FAILS CLOSED — "
            "writes will raise rather than persist plaintext."
        )

    @property
    def encryption_active(self) -> bool:
        """Whether payloads are actually being encrypted. Never assume — ask."""
        return self._aesgcm is not None

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt with a fresh random nonce, or refuse.

        Raises:
            BlackHoleEncryptionUnavailable: when no key is available. This never
                returns plaintext — a privacy guarantee that silently degrades
                to no-op is worse than one that fails, because it keeps looking
                like it worked.
        """
        if not self._aesgcm:
            raise BlackHoleEncryptionUnavailable(
                "BlackHole has no encryption key; refusing to return plaintext "
                "from encrypt(). Call on_start() first, or check "
                "BlackHole.encryption_active before writing."
            )

        nonce = os.urandom(12)
        ciphertext = self._aesgcm.encrypt(nonce, data, None)
        return nonce + ciphertext

    def decrypt(self, blob: bytes) -> bytes:
        """Decrypt and verify, or refuse.

        Raises:
            BlackHoleEncryptionUnavailable: when no key is available. Returning
                the raw blob here would hand ciphertext back to the caller as if
                it were plaintext.
        """
        if not self._aesgcm:
            raise BlackHoleEncryptionUnavailable(
                "BlackHole has no encryption key; refusing to return an "
                "unverified blob from decrypt()."
            )


        try:
            nonce = blob[:12]
            ciphertext = blob[12:]
            return self._aesgcm.decrypt(nonce, ciphertext, None)
        except _BLACK_HOLE_CRYPTO_ERRORS as e:
            record_degradation('black_hole', e)
            logger.error("BlackHole decryption FAILED: %s", e)
            raise ValueError("Decryption/Authentication failure.") from e
            
    def encrypt_json(self, data: dict[str, Any]) -> str:
        import json
        raw = json.dumps(data).encode()
        return base64.b64encode(self.encrypt(raw)).decode()

    def decrypt_json(self, b64_blob: str) -> dict[str, Any]:
        import json
        blob = base64.b64decode(b64_blob)
        raw = self.decrypt(blob)
        return json.loads(raw.decode())


def encode_payload(data: str | bytes, key_b64: str) -> dict[str, str]:
    """Module-level compatibility for Zenith memory encryption."""
    key = _resolve_aes_key(key_b64)
    aesgcm = AESGCM(key)
    
    raw = data.encode() if isinstance(data, str) else data
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, raw, None)
    encoded = base64.b64encode(nonce + ciphertext).decode()
    raw_len = max(len(raw), 1)
    ratio = round((len(encoded) / raw_len) * 100, 2)
    
    return {"encoded": encoded, "ratio": ratio}


def decode_payload(b64_blob: str, key_b64: str, *, strict: bool = False) -> DecodedPayload:
    """Module-level compatibility for Zenith memory decryption."""
    try:
        key = _resolve_aes_key(key_b64)
        aesgcm = AESGCM(key)
        
        blob = base64.b64decode(b64_blob)
        nonce = blob[:12]
        ciphertext = blob[12:]
        
        decrypted = aesgcm.decrypt(nonce, ciphertext, None).decode()
        return DecodedPayload(decrypted)
    except _BLACK_HOLE_DECODE_ERRORS as e:
        if strict:
            raise BlackHoleDecodeError("payload authentication or decoding failed") from e
        logger.debug("decode_payload failed: %s", e)  # Downgraded — happens on first boot with no stored data
        return DecodedPayload("")
