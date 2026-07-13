"""core/security/device_pairing.py
────────────────────────────────
Paired-device credentials for Aura's LAN embodiment surface.

This is the trust root that lets Bryan chat with Aura from his phone
without hand-copying the master ``AURA_API_TOKEN`` onto every device.
The design is deliberately narrow and fail-closed:

- Pairing begins ONLY from an owner-present surface (localhost desktop
  UI or master token). It mints a short-lived, single-use numeric code.
- A device on the LAN exchanges that code for its own bearer token.
  The token is shown exactly once; only its SHA-256 digest is persisted.
- Device tokens are *scoped*: the HTTP layer (interface/auth.py) only
  honors them on the conversation surface allowlist — never on the
  sovereign control surface (skill execution, reboot, hot-reload, …).
- Everything is revocable, audited, and disabled outright in
  ``internal_only_mode``.

Threat model notes (honest limits):
- Transport is plain HTTP on the home LAN. Tokens are revocable and
  scoped precisely because the wire is not TLS. Do not reuse device
  tokens for anything beyond the conversation surface.
- Pairing-code brute force: 8 random digits, ``_MAX_ATTEMPTS`` guesses,
  ``_CODE_TTL_SECONDS`` lifetime, one active code at a time.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import get_config
from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Security.DevicePairing")

TOKEN_PREFIX = "adt1"
_CODE_TTL_SECONDS = 180.0
_MAX_ATTEMPTS = 5
_CODE_DIGITS = 8
_SCHEMA_VERSION = 2
_LAST_SEEN_PERSIST_INTERVAL = 300.0

_REGISTRY_ERRORS = (OSError, RuntimeError, TypeError, ValueError, KeyError)

# The only scope minted today. The HTTP layer maps it to a path
# allowlist; new scopes must be added there deliberately, not here.
SCOPE_CONVERSATION = "conversation"


class PairingError(Exception):
    """Pairing failed for a reason safe to show the remote device."""


class PairingDisabledError(PairingError):
    """Pairing is administratively unavailable (internal-only mode)."""


@dataclass
class PairedDevice:
    device_id: str
    name: str
    token_sha256: str
    scopes: tuple[str, ...]
    created_at: float
    last_seen: float
    principal_id: str = ""
    revoked: bool = False

    def public_view(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "scopes": list(self.scopes),
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "principal_bound": bool(self.principal_id),
            "revoked": self.revoked,
        }


@dataclass
class _PairingChallenge:
    pairing_id: str
    code: str
    expires_at: float
    principal_id: str
    attempts_left: int = _MAX_ATTEMPTS
    consumed: bool = False


@dataclass
class DevicePairingRegistry:
    """In-memory registry with governed, atomic JSON persistence."""

    path: Path
    devices: dict[str, PairedDevice] = field(default_factory=dict)
    _challenge: _PairingChallenge | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_seen_dirty_since: float = 0.0

    # ── construction ────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path) -> DevicePairingRegistry:
        registry = cls(path=path)
        try:
            if path.exists():
                import json

                document = json.loads(path.read_text(encoding="utf-8"))
                # file_write_gateway wraps documents in a schema envelope.
                payload = document.get("payload", document)
                for row in payload.get("devices", []):
                    device = PairedDevice(
                        device_id=str(row["device_id"]),
                        name=str(row.get("name", "device")),
                        token_sha256=str(row["token_sha256"]),
                        scopes=tuple(row.get("scopes", [SCOPE_CONVERSATION])),
                        created_at=float(row.get("created_at", 0.0)),
                        last_seen=float(row.get("last_seen", 0.0)),
                        principal_id=_sanitize_principal_id(row.get("principal_id")),
                        revoked=bool(row.get("revoked", False)),
                    )
                    registry.devices[device.device_id] = device
        except _REGISTRY_ERRORS as exc:
            # A corrupt registry must not brick auth — but it must also
            # never authorize anyone. Empty registry = nobody paired.
            record_degradation("security.device_pairing.load", exc)
            logger.error("Paired-device registry unreadable; starting empty: %s", exc)
            registry.devices = {}
        return registry

    # ── pairing lifecycle ───────────────────────────────────────

    def _pairing_enabled(self) -> bool:
        return not bool(getattr(get_config().security, "internal_only_mode", False))

    def begin_pairing(self, principal_id: str) -> dict[str, Any]:
        """Mint a short-lived single-use pairing code. Owner surface only —
        the caller (route layer) is responsible for owner authentication."""
        if not self._pairing_enabled():
            raise PairingDisabledError("Pairing is disabled in internal-only mode")
        normalized_principal = _sanitize_principal_id(principal_id)
        if not normalized_principal:
            raise PairingError("Pairing requires a verified relational principal")
        with self._lock:
            code = "".join(secrets.choice("0123456789") for _ in range(_CODE_DIGITS))
            self._challenge = _PairingChallenge(
                pairing_id=secrets.token_urlsafe(8),
                code=code,
                expires_at=time.time() + _CODE_TTL_SECONDS,
                principal_id=normalized_principal,
            )
            return {
                "pairing_id": self._challenge.pairing_id,
                "code": code,
                "expires_at": self._challenge.expires_at,
                "ttl_seconds": _CODE_TTL_SECONDS,
            }

    def cancel_pairing(self) -> None:
        with self._lock:
            self._challenge = None

    async def complete_pairing(self, code: str, device_name: str) -> dict[str, Any]:
        """Exchange a pairing code for a device token. The token is returned
        exactly once and never persisted in the clear."""
        if not self._pairing_enabled():
            raise PairingDisabledError("Pairing is disabled in internal-only mode")
        with self._lock:
            challenge = self._challenge
            now = time.time()
            if challenge is None or challenge.consumed or now > challenge.expires_at:
                self._challenge = None
                raise PairingError("No active pairing code — start pairing from the desktop first")
            if challenge.attempts_left <= 0:
                self._challenge = None
                raise PairingError("Too many attempts — start pairing again from the desktop")
            challenge.attempts_left -= 1
            supplied = str(code or "").strip().replace(" ", "").replace("-", "")
            if not hmac.compare_digest(supplied, challenge.code):
                if challenge.attempts_left <= 0:
                    self._challenge = None
                raise PairingError("Incorrect pairing code")
            challenge.consumed = True
            self._challenge = None

            device_id = secrets.token_hex(8)
            secret = secrets.token_urlsafe(32)
            token = f"{TOKEN_PREFIX}.{device_id}.{secret}"
            device = PairedDevice(
                device_id=device_id,
                name=_sanitize_device_name(device_name),
                token_sha256=hashlib.sha256(secret.encode("utf-8")).hexdigest(),
                scopes=(SCOPE_CONVERSATION,),
                created_at=now,
                last_seen=now,
                principal_id=challenge.principal_id,
            )
            self.devices[device_id] = device
            snapshot = self._snapshot_locked()
        await self._persist(snapshot)
        await self._audit("device_paired", {"device_id": device_id, "name": device.name})
        return {"device_id": device_id, "token": token, "name": device.name}

    # ── verification ────────────────────────────────────────────

    def verify_token(self, token: str | None) -> PairedDevice | None:
        """Constant-time-verified lookup. Pure in-memory; safe on the hot path."""
        if not self._pairing_enabled():
            return None
        if not token:
            return None
        parts = str(token).split(".")
        if len(parts) != 3 or parts[0] != TOKEN_PREFIX:
            return None
        _, device_id, secret = parts
        with self._lock:
            device = self.devices.get(device_id)
            if device is None or device.revoked:
                return None
            supplied_hash = hashlib.sha256(secret.encode("utf-8")).hexdigest()
            if not hmac.compare_digest(supplied_hash, device.token_sha256):
                return None
            now = time.time()
            device.last_seen = now
            if not self._last_seen_dirty_since:
                self._last_seen_dirty_since = now
            return device

    # ── administration ──────────────────────────────────────────

    def list_devices(self) -> list[dict[str, Any]]:
        with self._lock:
            return [d.public_view() for d in self.devices.values()]

    async def revoke_device(self, device_id: str) -> bool:
        with self._lock:
            device = self.devices.get(str(device_id))
            if device is None:
                return False
            device.revoked = True
            snapshot = self._snapshot_locked()
        await self._persist(snapshot)
        await self._audit("device_revoked", {"device_id": device_id, "name": device.name})
        return True

    async def flush_last_seen(self) -> None:
        """Opportunistic persistence of last_seen, throttled so the
        conversation hot path never owns a disk write."""
        with self._lock:
            if not self._last_seen_dirty_since:
                return
            if time.time() - self._last_seen_dirty_since < _LAST_SEEN_PERSIST_INTERVAL:
                return
            self._last_seen_dirty_since = 0.0
            snapshot = self._snapshot_locked()
        await self._persist(snapshot)

    # ── persistence ─────────────────────────────────────────────

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "devices": [
                {
                    "device_id": d.device_id,
                    "name": d.name,
                    "token_sha256": d.token_sha256,
                    "scopes": list(d.scopes),
                    "created_at": d.created_at,
                    "last_seen": d.last_seen,
                    "principal_id": d.principal_id,
                    "revoked": d.revoked,
                }
                for d in self.devices.values()
            ],
        }

    async def _persist(self, snapshot: dict[str, Any]) -> None:
        try:
            gateway = get_file_write_gateway()
            with local_internal_governed_scope(
                "security.device_pairing",
                domain="file_write",
                receipt_prefix="device-pairing",
            ):
                await gateway.ensure_directory_async(
                    self.path.parent, source="security.device_pairing"
                )
                await gateway.write_json_async(
                    self.path,
                    snapshot,
                    schema_version=_SCHEMA_VERSION,
                    schema_name="paired_devices",
                    source="security.device_pairing",
                )
        except _REGISTRY_ERRORS as exc:
            record_degradation("security.device_pairing.persist", exc)
            logger.error("Failed to persist paired-device registry: %s", exc)

    async def _audit(self, action: str, details: dict[str, Any]) -> None:
        try:
            from core.security.audit_log import SecurityAuditLogger

            await asyncio.to_thread(SecurityAuditLogger().log_event, action, details)
        except _REGISTRY_ERRORS as exc:
            record_degradation("security.device_pairing.audit", exc)


def _sanitize_device_name(raw: str) -> str:
    cleaned = "".join(ch for ch in str(raw or "") if ch.isprintable()).strip()
    return (cleaned or "device")[:64]


def _sanitize_principal_id(raw: Any) -> str:
    return " ".join(str(raw or "").strip().split()).casefold()[:160]


# ── module singleton ─────────────────────────────────────────────

_registry: DevicePairingRegistry | None = None
_registry_lock = threading.Lock()


def registry_path() -> Path:
    return Path(get_config().paths.data_dir) / "security" / "paired_devices.json"


def get_device_registry() -> DevicePairingRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = DevicePairingRegistry.load(registry_path())
    return _registry


def reset_device_registry_for_tests(path: Path | None = None) -> DevicePairingRegistry:
    """Test seam: swap the singleton for one rooted at a temp path."""
    global _registry
    with _registry_lock:
        _registry = DevicePairingRegistry.load(path or registry_path())
        return _registry
