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
_SCHEMA_VERSION = 3
_LAST_SEEN_PERSIST_INTERVAL = 300.0
# A connect nonce is single-use; the window only has to cover one round
# trip on a LAN, not a human typing a code.
_CONNECT_NONCE_TTL_SECONDS = 60.0

_REGISTRY_ERRORS = (OSError, RuntimeError, TypeError, ValueError, KeyError)

# The only scope minted today. The HTTP layer maps it to a path
# allowlist; new scopes must be added there deliberately, not here.
SCOPE_CONVERSATION = "conversation"
# Voice is never minted at pairing time: it must be granted explicitly
# by the owner, per device, after pairing (deny-by-default posture).
SCOPE_VOICE = "voice"
GRANTABLE_SCOPES = frozenset({SCOPE_VOICE})


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
    # What the device said it can serve, pinned at pairing. A declaration,
    # never a grant: scopes still decide what Aura may *use*. This only
    # narrows — a device cannot be asked for something it never claimed.
    capabilities: tuple[str, ...] = ()
    # Pinned identity. A token replayed from a different kind of device
    # fails the connect signature rather than being honoured.
    platform: str = ""
    device_family: str = ""

    @property
    def manifest_sha256(self) -> str:
        return _manifest_digest(self.capabilities)

    @property
    def metadata_pinned(self) -> bool:
        return bool(self.platform or self.device_family or self.capabilities)

    def public_view(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "scopes": list(self.scopes),
            "created_at": self.created_at,
            "last_seen": self.last_seen,
            "principal_bound": bool(self.principal_id),
            "revoked": self.revoked,
            "capabilities": list(self.capabilities),
            "platform": self.platform,
            "device_family": self.device_family,
            "manifest_sha256": self.manifest_sha256,
            "metadata_pinned": self.metadata_pinned,
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
    _presence_noted_at: dict[str, float] = field(default_factory=dict)
    # device_id -> (nonce, expires_at). One outstanding challenge per
    # device; a new begin_connect replaces the old one, so a nonce cannot
    # be banked for later.
    _connect_nonces: dict[str, tuple[str, float]] = field(default_factory=dict)

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
                        capabilities=_sanitize_capabilities(row.get("capabilities", ())),
                        platform=_sanitize_identity_field(row.get("platform")),
                        device_family=_sanitize_identity_field(row.get("device_family")),
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

    async def complete_pairing(
        self,
        code: str,
        device_name: str,
        *,
        platform: str = "",
        device_family: str = "",
        capabilities: Any = (),
    ) -> dict[str, Any]:
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
                capabilities=_sanitize_capabilities(capabilities),
                platform=_sanitize_identity_field(platform),
                device_family=_sanitize_identity_field(device_family),
            )
            self.devices[device_id] = device
            snapshot = self._snapshot_locked()
        await self._persist(snapshot)
        await self._audit(
            "device_paired",
            {
                "device_id": device_id,
                "name": device.name,
                "platform": device.platform,
                "device_family": device.device_family,
                "capabilities": list(device.capabilities),
                "manifest_sha256": device.manifest_sha256,
            },
        )
        return {
            "device_id": device_id,
            "token": token,
            "name": device.name,
            "capabilities": list(device.capabilities),
            "manifest_sha256": device.manifest_sha256,
        }

    # ── connect handshake ───────────────────────────────────────
    #
    # The bearer token alone is possession-is-authentication, and this
    # module's own threat model admits the wire is plain HTTP on the LAN.
    # A connect that replays a captured token from anywhere would be
    # honoured. OpenClaw signs a gateway-chosen nonce and binds platform
    # and deviceFamily into the signed payload, pinning the metadata so a
    # change forces re-pairing; same idea here.
    #
    # The HMAC key is the stored token digest, not the token secret. The
    # device can derive it (it holds the secret), the server already has
    # it, and the secret itself never crosses the wire on connect. The
    # honest limit: the registry file therefore holds material equivalent
    # to a connect credential, exactly as it already held a verifier for
    # the bearer token.

    def begin_connect(self, device_id: str) -> dict[str, Any]:
        """Issue a single-use nonce for this device to sign."""
        with self._lock:
            device = self.devices.get(str(device_id or ""))
            if device is None or device.revoked:
                raise PairingError("Unknown device")
            nonce = secrets.token_urlsafe(24)
            expires_at = time.time() + _CONNECT_NONCE_TTL_SECONDS
            self._connect_nonces[device.device_id] = (nonce, expires_at)
            return {"nonce": nonce, "expires_at": expires_at, "ttl_seconds": _CONNECT_NONCE_TTL_SECONDS}

    @staticmethod
    def connect_signature(
        *,
        token_sha256: str,
        nonce: str,
        device_id: str,
        platform: str,
        device_family: str,
        manifest_sha256: str,
    ) -> str:
        """The canonical signature both sides compute independently."""
        payload = _FIELD_SEPARATOR.join(
            (
                "aura-connect-v1",
                str(device_id or ""),
                str(nonce or ""),
                _sanitize_identity_field(platform),
                _sanitize_identity_field(device_family),
                str(manifest_sha256 or ""),
            )
        )
        return hmac.new(
            str(token_sha256 or "").encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def verify_connect(
        self,
        device_id: str,
        *,
        nonce: str,
        signature: str,
        platform: str = "",
        device_family: str = "",
        capabilities: Any = (),
    ) -> PairedDevice:
        """Verify a signed connect, enforcing pinned identity.

        Raises PairingError on anything unverified. Returns the device on
        success.
        """
        declared_platform = _sanitize_identity_field(platform)
        declared_family = _sanitize_identity_field(device_family)
        declared_caps = _sanitize_capabilities(capabilities)
        persist_snapshot: dict[str, Any] | None = None
        pinned_now = False

        with self._lock:
            device = self.devices.get(str(device_id or ""))
            if device is None or device.revoked:
                raise PairingError("Unknown device")

            issued = self._connect_nonces.pop(device.device_id, None)
            if issued is None:
                raise PairingError("No connect challenge outstanding for this device")
            issued_nonce, expires_at = issued
            if time.time() > expires_at:
                raise PairingError("Connect challenge expired")
            if not hmac.compare_digest(str(nonce or ""), issued_nonce):
                raise PairingError("Connect challenge mismatch")

            # A device paired before metadata existed has nothing pinned.
            # Pin what it declares on this connect rather than locking the
            # owner out of a device that was legitimately paired earlier;
            # every connect after this one is enforced.
            if not device.metadata_pinned:
                device.platform = declared_platform
                device.device_family = declared_family
                device.capabilities = declared_caps
                pinned_now = True
            else:
                if (
                    declared_platform != device.platform
                    or declared_family != device.device_family
                ):
                    raise PairingError(
                        "Device identity changed since pairing — pair this device again"
                    )
                if declared_caps and declared_caps != device.capabilities:
                    raise PairingError(
                        "Device capability manifest changed since pairing — pair this device again"
                    )

            expected = self.connect_signature(
                token_sha256=device.token_sha256,
                nonce=issued_nonce,
                device_id=device.device_id,
                platform=device.platform,
                device_family=device.device_family,
                manifest_sha256=device.manifest_sha256,
            )
            if not hmac.compare_digest(str(signature or ""), expected):
                if pinned_now:
                    # Do not keep metadata a failed signature "declared".
                    device.platform = ""
                    device.device_family = ""
                    device.capabilities = ()
                raise PairingError("Connect signature invalid")

            device.last_seen = time.time()
            if pinned_now:
                persist_snapshot = self._snapshot_locked()

        if persist_snapshot is not None:
            await self._persist(persist_snapshot)
            await self._audit(
                "device_metadata_pinned",
                {
                    "device_id": device.device_id,
                    "platform": device.platform,
                    "device_family": device.device_family,
                    "manifest_sha256": device.manifest_sha256,
                },
            )
        return device

    def device_can_serve(self, device_id: str, capability: str) -> bool:
        """Whether this device declared it can serve this capability.

        A declaration is not permission — scopes still decide what Aura
        may use. This only narrows: a device cannot be asked for something
        it never claimed to have. A device with no manifest declares
        nothing, so it can serve nothing through this path.
        """
        wanted = _sanitize_capability(capability)
        if not wanted:
            return False
        with self._lock:
            device = self.devices.get(str(device_id or ""))
            if device is None or device.revoked:
                return False
            return wanted in device.capabilities

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
        self._note_presence(device, now)
        return device

    def _note_presence(self, device: PairedDevice, now: float) -> None:
        """Surface device reachability as a world-state belief (throttled):
        'Bryan is reachable on his phone' becomes a perceptual fact with a
        TTL, so Aura's cognition can act on presence and its expiry."""
        if now - self._presence_noted_at.get(device.device_id, 0.0) < 60.0:
            return
        self._presence_noted_at[device.device_id] = now
        try:
            from core.world_state import get_world_state

            get_world_state().set_belief(
                f"device_presence.{device.device_id}",
                {"device_id": device.device_id, "name": device.name, "last_seen": now},
                confidence=0.95,
                source="device_pairing",
                ttl=600.0,
            )
        except _REGISTRY_ERRORS as exc:
            record_degradation("security.device_pairing.presence", exc)

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

    async def grant_scope(self, device_id: str, scope: str) -> bool:
        """Owner-granted scope widening (e.g. voice). Only scopes in
        GRANTABLE_SCOPES may ever be added post-pairing."""
        scope = str(scope or "").strip().lower()
        if scope not in GRANTABLE_SCOPES:
            raise PairingError(f"scope '{scope}' is not grantable")
        with self._lock:
            device = self.devices.get(str(device_id))
            if device is None or device.revoked:
                return False
            if scope not in device.scopes:
                device.scopes = tuple(device.scopes) + (scope,)
            snapshot = self._snapshot_locked()
        await self._persist(snapshot)
        await self._audit("device_scope_granted",
                          {"device_id": device_id, "scope": scope})
        return True

    async def revoke_scope(self, device_id: str, scope: str) -> bool:
        scope = str(scope or "").strip().lower()
        with self._lock:
            device = self.devices.get(str(device_id))
            if device is None:
                return False
            device.scopes = tuple(s for s in device.scopes if s != scope)
            snapshot = self._snapshot_locked()
        await self._persist(snapshot)
        await self._audit("device_scope_revoked",
                          {"device_id": device_id, "scope": scope})
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
                    "capabilities": list(d.capabilities),
                    "platform": d.platform,
                    "device_family": d.device_family,
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


# Signed fields are joined with the unit separator, so they must not be
# able to contain one — otherwise a device could shift the boundary
# between two fields and make one signature satisfy two different claims.
_FIELD_SEPARATOR = "\x1f"


def _sanitize_identity_field(raw: Any) -> str:
    cleaned = "".join(
        ch for ch in str(raw or "") if ch.isprintable() and ch != _FIELD_SEPARATOR
    ).strip()
    return cleaned.casefold()[:48]


def _sanitize_capability(raw: Any) -> str:
    """A capability is a dotted lowercase token: ``camera.capture``."""
    cleaned = "".join(
        ch
        for ch in str(raw or "").strip().casefold()
        if ch.isalnum() or ch in {".", "_", "-"}
    )
    return cleaned[:64]


def _sanitize_capabilities(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, (str, bytes)):
        return ()
    try:
        items = list(raw or [])
    except TypeError:
        return ()
    seen: list[str] = []
    for item in items[:64]:
        capability = _sanitize_capability(item)
        if capability and capability not in seen:
            seen.append(capability)
    # Sorted so the digest is a property of the SET, not of the order the
    # device happened to list them in.
    return tuple(sorted(seen))


def _manifest_digest(capabilities: tuple[str, ...]) -> str:
    payload = _FIELD_SEPARATOR.join(capabilities)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
