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
import stat
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from core.runtime.atomic_writer import atomic_write_bytes_if_absent, ensure_private_directory
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
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


_LOCAL_KEY_DIR_FLAG = declare(
    "AURA_BLACK_HOLE_KEY_DIR", kind=FlagKind.STRING, default="",
    description="Override directory for the BlackHole local encryption key",
    owner="core.memory.black_hole",
)


def _local_key_path():
    override = str(_LOCAL_KEY_DIR_FLAG.value() or "").strip()
    if override:
        return Path(override).expanduser() / "black_hole_local.key"
    try:
        from core.config import config

        return Path(config.paths.home_dir) / "keys" / "black_hole_local.key"
    except (ImportError, AttributeError, RuntimeError):
        return Path.home() / ".aura" / "keys" / "black_hole_local.key"


def _read_local_key(path: Path) -> bytes:
    """Read one stable owner-private AES key without following symlinks."""

    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"local key is not a regular file: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(str(path), flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"local key is not a regular file: {path}")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError("local key changed while opening")
        if hasattr(os, "getuid") and opened.st_uid != os.getuid():
            raise PermissionError("local key is not owned by the current user")
        if opened.st_mode & 0o077:
            os.fchmod(fd, 0o600)
            logger.warning("BlackHole: restricted local key permissions to 0600 at %s", path)
            opened = os.fstat(fd)
        key = os.read(fd, 33)
        if len(key) != 32 or os.read(fd, 1):
            raise ValueError(f"local key at {path} is malformed; expected exactly 32 bytes")
        after = os.fstat(fd)
        stable_attributes = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            stat.S_IMODE(after.st_mode),
            after.st_uid,
            after.st_gid,
        )
        opened_attributes = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            stat.S_IMODE(opened.st_mode),
            opened.st_uid,
            opened.st_gid,
        )
        if stable_attributes != opened_attributes:
            raise RuntimeError("local key changed while reading")
        if after.st_ctime_ns != opened.st_ctime_ns and not (
            opened.st_nlink == 2 and after.st_nlink == 1
        ):
            raise RuntimeError("local key metadata changed while reading")
        current = path.lstat()
        current_attributes = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            stat.S_IMODE(current.st_mode),
            current.st_uid,
            current.st_gid,
        )
        if not stat.S_ISREG(current.st_mode) or current_attributes != stable_attributes:
            raise RuntimeError("local key path changed while reading")
        return key
    finally:
        os.close(fd)


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
        ensure_private_directory(path.parent)
        if os.path.lexists(path):
            return _read_local_key(path)
        key = os.urandom(32)
        if atomic_write_bytes_if_absent(path, key, durable=True, mode=0o600):
            logger.info("BlackHole: provisioned a local encryption key at %s", path)
            return key
        return _read_local_key(path)
    except (OSError, RuntimeError, ValueError) as exc:
        record_degradation(
            "black_hole",
            exc,
            action=(
                "could not establish one stable local encryption identity; existing key "
                "material was preserved and encryption will fail closed"
            ),
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
        self.key_identity_sha256: str = ""

    def on_start(self):
        """Initialize the provider, preferring the Horcrux-derived key.

        Falls back to a locally provisioned key rather than to no encryption:
        "Horcrux unavailable" used to mean every memory written on that boot
        went to disk in plaintext.
        """
        horcrux = get_runtime_service("horcrux", default=None)
        derived_key = getattr(horcrux, "derived_key", None)
        if derived_key:
            self._aesgcm = AESGCM(derived_key)
            self.key_provenance = "horcrux"
            identity = getattr(horcrux, "key_identity_sha256", "")
            self.key_identity_sha256 = str(identity or "")
            if (
                len(self.key_identity_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.key_identity_sha256
                )
            ):
                self._aesgcm = None
                self.key_provenance = "none"
                self.key_identity_sha256 = ""
                raise RuntimeError("Horcrux key identity is unavailable")
            logger.info("BlackHole: AES-256-GCM substrate initialized (Horcrux key).")
            return

        local_key = _provision_local_key()
        if local_key:
            self._aesgcm = AESGCM(local_key)
            self.key_provenance = "local"
            self.key_identity_sha256 = hashlib.sha256(
                b"AURA-BLACK-HOLE-LOCAL-KEY-IDENTITY-v1\x00" + local_key
            ).hexdigest()
            logger.warning(
                "BlackHole: Horcrux keys unavailable — using a locally provisioned "
                "key. Memories are encrypted at rest but NOT hardware-entangled."
            )
            return

        self.key_provenance = "none"
        self.key_identity_sha256 = ""
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
