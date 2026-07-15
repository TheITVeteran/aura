"""core/governance/capability_chain.py
=====================================
The cryptographic join between the Will and every consequential sink.

Aura had all the pieces of a constitutional authority system — the Will could
make and sign a legitimate decision, sinks could require a token, receipts
could record an operation — but the pieces were never *joined*. The sink could
not establish that the token it accepted came from the Will. A token was an
8-character uuid in a process-global dict (``core/agency/capability_system.py``),
mintable by anyone who imported the module, and callers could skip even that by
setting ``ctx["_capability_token_verified"] = True``. That broke the invariant
underlying the strongest claims Aura makes about itself.

This module closes the chain. A :class:`SignedCapability` is an immutable,
signed grant that binds, in one signature:

    outcome        the Will's actual decision (only approving outcomes issue)
    domain         the ActionDomain the decision was made in
    action_digest  a digest of the concrete action + its parameters
    issuer         who minted it
    key_id         which key minted it
    nonce          single-use, replay-defeating
    expires_at     wall-clock expiry
    receipt_id     back-link to the Will's provenance record

Verification is fail-closed and rejects, at the moment of execution:

    fabricated       — no valid signature over the payload
    altered          — any field mutated after issue (signature no longer binds)
    refused          — an outcome the Will never approves
    expired          — past ``expires_at`` (or not yet valid)
    replayed         — a nonce already consumed
    domain-mismatch  — issued for one ActionDomain, presented at another
    action-mismatch  — issued for one action digest, presented for another
    revoked          — explicitly revoked before use

Threat model — stated honestly
------------------------------
Under Ed25519 (the default when ``cryptography`` is importable) the asymmetry is
real and load-bearing: :class:`CapabilityVerifier` loads **only the public key**,
so a sink is *structurally incapable* of minting a capability it would accept.
Only :class:`WillCapabilityIssuer`, which holds the private key, can mint.

What this defends against is the actual failure that was present: structural
bypass, self-asserted verification flags, fabricated or LLM-authored governance
contexts, replay across turns, and domain/action confusion. What it does **not**
defend against is malicious in-process code that reads the private key off disk
and mints deliberately. Aura is a single trust domain; a capability chain cannot
manufacture a boundary the process does not have. The honest claim is: forgery
is no longer reachable by accident, refactor, or a fabricated dict — it now
requires deliberate key theft, which is auditable. Do not claim more than that.

On the HMAC fallback that asymmetry is lost (verifiers can mint), so the fallback
is a degraded mode: it is recorded in ``key_id`` as ``hmac-*``, surfaced in
:func:`capability_chain_status`, and callers that require the strong property can
assert :func:`issuer_is_asymmetric`.

Dependency discipline
---------------------
This module deliberately does not import ``core.governance.will`` (the Will
issues capabilities, so that would be circular) and does not route its nonce
ledger through ``core/runtime/file_write_gateway.py`` (the gateway is itself a
consequential sink and will be capability-gated; routing the ledger through it
would make the authority system depend on the thing it authorizes). The ledger
uses a direct atomic tmp+rename with fsync kept off the caller's path.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger("Aura.Governance.CapabilityChain")

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    _ED25519_AVAILABLE = True
except (ImportError, AttributeError):  # pragma: no cover - declared runtime dep
    _ED25519_AVAILABLE = False

    class InvalidSignature(Exception):  # type: ignore[no-redef]
        pass


CAPABILITY_SCHEMA_VERSION = 1

# Domain-separation tag. Signatures are computed over this tag + payload so a
# signature minted for some other Aura structure can never be replayed as a
# capability, and vice versa.
_SIGNING_TAG = b"aura.governance.capability.v1\x00"

# Outcomes the Will considers authorizing. Anything else must never mint, and a
# forged capability carrying a non-approving outcome is rejected at verify time
# even if its signature were somehow valid.
APPROVING_OUTCOMES: frozenset[str] = frozenset({"proceed", "constrain", "critical"})

DEFAULT_TTL_S = 300.0
MAX_TTL_S = 3600.0
# Small tolerance for clock jitter between issue and verify on the same host.
_CLOCK_SKEW_TOLERANCE_S = 2.0

_NONCE_LEDGER_CAP = 20_000


class CapabilityDenial(StrEnum):
    """Why a capability was refused. Every value is a hard denial."""

    MISSING = "missing"
    MALFORMED = "malformed"
    SCHEMA_MISMATCH = "schema_mismatch"
    UNKNOWN_ISSUER = "unknown_issuer"
    BAD_SIGNATURE = "bad_signature"
    NOT_APPROVED = "not_approved"
    EXPIRED = "expired"
    NOT_YET_VALID = "not_yet_valid"
    REPLAYED = "replayed"
    REVOKED = "revoked"
    DOMAIN_MISMATCH = "domain_mismatch"
    ACTION_MISMATCH = "action_mismatch"


class CapabilityViolation(Exception):
    """Raised at a consequential sink when authority cannot be established.

    This is the fail-closed path. It carries the machine-readable denial so
    sinks can record a truthful receipt rather than a generic error.
    """

    def __init__(self, denial: CapabilityDenial, detail: str = "", *, sink: str = ""):
        self.denial = denial
        self.detail = detail
        self.sink = sink
        super().__init__(
            f"Capability denied at sink '{sink or 'unknown'}': {denial.value}"
            + (f" ({detail})" if detail else "")
        )


# ---------------------------------------------------------------------------
# Action digests
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    """Deterministic JSON. Both issuer and sink must produce identical bytes."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_fallback,
    )


def _json_fallback(obj: Any) -> str:
    """Stable rendering for values json cannot encode natively.

    Falls back to the repr of the type rather than the value for unknown
    objects: a digest must never depend on a memory address.
    """
    if isinstance(obj, (set, frozenset)):
        return _canonical_json(sorted(str(o) for o in obj))
    if isinstance(obj, bytes):
        return hashlib.sha256(obj).hexdigest()
    if isinstance(obj, Path):
        return str(obj)
    return f"<{type(obj).__name__}>"


def compute_action_digest(action: str, payload: Any = None) -> str:
    """Digest binding a capability to one concrete action and its parameters.

    The digest is what makes a capability non-transferable between actions: a
    grant minted for ``read_file(/etc/hosts)`` cannot be presented for
    ``shell_command(rm -rf /)`` even though both are tool executions in the same
    domain.
    """
    body = _canonical_json({"action": str(action or ""), "payload": payload})
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The capability
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignedCapability:
    """An immutable, signed grant of authority from the Will to one action.

    Frozen because a capability that can be mutated after issue is not a
    capability. Any mutation must go through :meth:`replace`-style
    reconstruction, which invalidates the signature — which is the point.
    """

    capability_id: str
    schema_version: int
    outcome: str
    domain: str
    action_digest: str
    issuer: str
    key_id: str
    nonce: str
    receipt_id: str
    scope: str
    constraints: tuple[str, ...]
    issued_at: float
    expires_at: float
    signature: str = ""

    # -- serialization ----------------------------------------------------
    def signing_payload(self) -> bytes:
        """Canonical bytes covered by the signature (everything but the sig)."""
        return _SIGNING_TAG + _canonical_json(
            {
                "capability_id": self.capability_id,
                "schema_version": self.schema_version,
                "outcome": self.outcome,
                "domain": self.domain,
                "action_digest": self.action_digest,
                "issuer": self.issuer,
                "key_id": self.key_id,
                "nonce": self.nonce,
                "receipt_id": self.receipt_id,
                "scope": self.scope,
                "constraints": list(self.constraints),
                "issued_at": round(float(self.issued_at), 6),
                "expires_at": round(float(self.expires_at), 6),
            }
        ).encode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "schema_version": self.schema_version,
            "outcome": self.outcome,
            "domain": self.domain,
            "action_digest": self.action_digest,
            "issuer": self.issuer,
            "key_id": self.key_id,
            "nonce": self.nonce,
            "receipt_id": self.receipt_id,
            "scope": self.scope,
            "constraints": list(self.constraints),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: Any) -> SignedCapability:
        """Rebuild from transport. Raises ValueError on anything malformed.

        Deliberately strict: a capability that arrives with a missing or
        wrong-typed field is a fabrication attempt, not a compatibility
        problem to paper over.
        """
        if not isinstance(data, dict):
            raise ValueError(f"capability must be a mapping, got {type(data).__name__}")
        try:
            return cls(
                capability_id=str(data["capability_id"]),
                schema_version=int(data["schema_version"]),
                outcome=str(data["outcome"]),
                domain=str(data["domain"]),
                action_digest=str(data["action_digest"]),
                issuer=str(data["issuer"]),
                key_id=str(data["key_id"]),
                nonce=str(data["nonce"]),
                receipt_id=str(data["receipt_id"]),
                scope=str(data["scope"]),
                constraints=tuple(str(c) for c in (data.get("constraints") or ())),
                issued_at=float(data["issued_at"]),
                expires_at=float(data["expires_at"]),
                signature=str(data.get("signature") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed capability: {exc}") from exc

    def redacted(self) -> dict[str, Any]:
        """Log-safe view. Never log the full nonce or signature."""
        return {
            "capability_id": self.capability_id,
            "outcome": self.outcome,
            "domain": self.domain,
            "action_digest": self.action_digest[:12],
            "issuer": self.issuer,
            "key_id": self.key_id,
            "receipt_id": self.receipt_id,
            "expires_in_s": round(self.expires_at - time.time(), 3),
        }


@dataclass(frozen=True, slots=True)
class VerificationResult:
    ok: bool
    denial: CapabilityDenial | None = None
    detail: str = ""
    capability: SignedCapability | None = None

    def raise_if_denied(self, sink: str = "") -> SignedCapability:
        if not self.ok or self.capability is None:
            raise CapabilityViolation(
                self.denial or CapabilityDenial.MALFORMED, self.detail, sink=sink
            )
        return self.capability


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def _key_dir() -> Path:
    override = os.environ.get("AURA_CAPABILITY_KEY_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    try:
        from core.config import config

        return Path(config.paths.home_dir) / "keys"
    except (ImportError, AttributeError, RuntimeError):
        return Path.home() / ".aura" / "keys"


def _priv_path() -> Path:
    return _key_dir() / "will_capability_ed25519_priv.pem"


def _pub_path() -> Path:
    return _key_dir() / "will_capability_ed25519_pub.pem"


def _hmac_path() -> Path:
    return _key_dir() / "will_capability_hmac.key"


def _key_id_for(material: bytes, kind: str) -> str:
    return f"{kind}-{hashlib.sha256(material).hexdigest()[:16]}"


class _KeyMaterial:
    """Loads/creates the Will's capability keys.

    The private key is created 0600 inside a 0700 directory. If key storage is
    unavailable we fall back to a process-ephemeral key rather than to no
    signing at all: an ephemeral key still defeats fabrication and replay
    within the process lifetime, and capabilities do not outlive the process
    because their TTL is minutes.
    """

    _lock = threading.RLock()
    _cache: dict[str, Any] = {}

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._cache.clear()

    @classmethod
    def _ensure_dir(cls) -> bool:
        try:
            d = _key_dir()
            d.mkdir(parents=True, exist_ok=True, mode=0o700)
            if os.name != "nt":
                os.chmod(d, 0o700)
            return True
        except OSError as exc:
            logger.warning(
                "Capability key storage unavailable at %s: %s — using an "
                "ephemeral in-process key for this run.",
                _key_dir(),
                exc,
            )
            return False

    @classmethod
    def load(cls) -> dict[str, Any]:
        with cls._lock:
            if cls._cache:
                return cls._cache
            cls._cache = cls._load_uncached()
            return cls._cache

    @classmethod
    def _load_uncached(cls) -> dict[str, Any]:
        storage = cls._ensure_dir()
        forced_hmac = os.environ.get("AURA_CAPABILITY_FORCE_HMAC", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }

        if _ED25519_AVAILABLE and not forced_hmac:
            try:
                priv, created = cls._load_or_create_ed25519(storage)
                pub = priv.public_key()
                pub_raw = pub.public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )
                if created and storage:
                    cls._write_public(pub)
                return {
                    "algorithm": "ed25519",
                    "private": priv,
                    "public": pub,
                    "key_id": _key_id_for(pub_raw, "ed25519"),
                    "asymmetric": True,
                    "persisted": storage,
                }
            except (OSError, ValueError, TypeError) as exc:
                logger.error(
                    "Ed25519 capability key unusable (%s) — falling back to HMAC. "
                    "Sinks can now mint; this is a DEGRADED authority mode.",
                    exc,
                )

        secret = cls._load_or_create_hmac(storage)
        return {
            "algorithm": "hmac-sha256",
            "private": secret,
            "public": secret,
            "key_id": _key_id_for(secret, "hmac"),
            "asymmetric": False,
            "persisted": storage,
        }

    @classmethod
    def _load_or_create_ed25519(cls, storage: bool) -> tuple[Any, bool]:
        path = _priv_path()
        if storage and path.exists():
            with open(path, "rb") as fh:
                loaded = serialization.load_pem_private_key(fh.read(), password=None)
            if not isinstance(loaded, Ed25519PrivateKey):
                raise ValueError(f"{path} is not an Ed25519 private key")
            return loaded, False

        priv = Ed25519PrivateKey.generate()
        if storage:
            data = priv.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            logger.info("Minted a new Will capability signing key at %s", path)
        return priv, True

    @classmethod
    def _write_public(cls, pub: Any) -> None:
        try:
            data = pub.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            fd = os.open(str(_pub_path()), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
        except OSError as exc:
            logger.warning("Could not publish capability public key: %s", exc)

    @classmethod
    def _load_or_create_hmac(cls, storage: bool) -> bytes:
        path = _hmac_path()
        if storage and path.exists():
            try:
                with open(path, "rb") as fh:
                    secret = fh.read()
                if len(secret) >= 32:
                    return secret
                logger.warning("Capability HMAC key at %s is too short — regenerating", path)
            except OSError as exc:
                logger.warning("Capability HMAC key unreadable (%s) — regenerating", exc)

        secret = os.urandom(32)
        if storage:
            try:
                fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "wb") as fh:
                    fh.write(secret)
            except OSError as exc:
                logger.warning("Could not persist capability HMAC key: %s", exc)
        return secret


def issuer_is_asymmetric() -> bool:
    """True when sinks verify with a public key they cannot mint with.

    Call this in any test or gate that needs the *strong* property rather than
    merely "a signature was checked".
    """
    return bool(_KeyMaterial.load().get("asymmetric"))


def capability_chain_status() -> dict[str, Any]:
    """Operator-facing truth about the authority chain's current strength."""
    keys = _KeyMaterial.load()
    return {
        "algorithm": keys["algorithm"],
        "key_id": keys["key_id"],
        "asymmetric": bool(keys["asymmetric"]),
        "keys_persisted": bool(keys["persisted"]),
        "degraded": not bool(keys["asymmetric"]),
        "nonce_ledger_size": get_nonce_ledger().size(),
        "note": (
            "Sinks verify with a public key and cannot mint."
            if keys["asymmetric"]
            else "DEGRADED: symmetric HMAC — any holder of the key can mint."
        ),
    }


_VALID_ENFORCEMENT_MODES = frozenset({"strict", "warn", "off"})


def capability_enforcement_mode(default: str = "strict") -> str:
    """How hard sinks enforce the chain. See ``_capability_chain_denial``.

    An unrecognized value resolves to ``strict`` rather than to the caller's
    default: a typo in an env var must not silently disable governance.
    """
    raw = os.environ.get("AURA_CAPABILITY_ENFORCEMENT", "").strip().lower()
    if not raw:
        return default if default in _VALID_ENFORCEMENT_MODES else "strict"
    if raw not in _VALID_ENFORCEMENT_MODES:
        logger.error(
            "AURA_CAPABILITY_ENFORCEMENT=%r is not one of %s — falling back to "
            "strict rather than disabling governance on a typo.",
            raw,
            sorted(_VALID_ENFORCEMENT_MODES),
        )
        return "strict"
    return raw


def _sign(payload: bytes) -> str:
    keys = _KeyMaterial.load()
    if keys["asymmetric"]:
        return keys["private"].sign(payload).hex()
    return hmac.new(keys["private"], payload, hashlib.sha256).hexdigest()


def _verify_signature(payload: bytes, signature: str, key_id: str) -> bool:
    keys = _KeyMaterial.load()
    if not signature:
        return False
    if key_id != keys["key_id"]:
        # A capability minted under a key we do not hold. Never accept it: an
        # unknown key_id is indistinguishable from an attacker naming their own.
        return False
    if keys["asymmetric"]:
        try:
            keys["public"].verify(bytes.fromhex(signature), payload)
            return True
        except (InvalidSignature, ValueError):
            return False
    expected = hmac.new(keys["private"], payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# Nonce ledger (replay defence)
# ---------------------------------------------------------------------------


class NonceLedger:
    """Single-use enforcement for capability nonces.

    Persisted so a capability captured before a restart cannot be replayed
    after one while still inside its TTL. Bounded, and pruned of entries whose
    capabilities have expired anyway — an expired capability is already
    rejected on the expiry check, so retaining its nonce buys nothing.
    """

    def __init__(self, path: Path | None = None):
        self._lock = threading.RLock()
        self._seen: dict[str, float] = {}
        self._path = path or (_key_dir().parent / "governance" / "capability_nonces.json")
        self._dirty = False
        self._load()

    def _load(self) -> None:
        try:
            if not self._path.exists():
                return
            with open(self._path, encoding="utf-8") as fh:
                data = json.load(fh)
            now = time.time()
            if isinstance(data, dict):
                self._seen = {
                    str(k): float(v)
                    for k, v in data.get("nonces", {}).items()
                    if float(v) > now
                }
        except (OSError, ValueError, TypeError) as exc:
            # A corrupt ledger must not open a replay window. Start empty and
            # say so loudly rather than silently accepting everything.
            logger.error(
                "Capability nonce ledger at %s unreadable (%s) — starting empty. "
                "Replay protection covers only this process until it repopulates.",
                self._path,
                exc,
            )
            self._seen = {}

    def _prune(self, now: float) -> None:
        if len(self._seen) <= _NONCE_LEDGER_CAP:
            expired = [k for k, exp in self._seen.items() if exp <= now]
            for k in expired:
                self._seen.pop(k, None)
            return
        # Over cap even after expiry pruning: drop the soonest-to-expire.
        for k, _exp in sorted(self._seen.items(), key=lambda kv: kv[1])[
            : len(self._seen) - _NONCE_LEDGER_CAP
        ]:
            self._seen.pop(k, None)

    def consume(self, nonce: str, expires_at: float) -> bool:
        """Claim a nonce. False if it was already claimed (a replay)."""
        now = time.time()
        with self._lock:
            self._prune(now)
            if nonce in self._seen:
                return False
            self._seen[nonce] = float(expires_at)
            self._dirty = True
        return True

    def seen(self, nonce: str) -> bool:
        with self._lock:
            return nonce in self._seen

    def size(self) -> int:
        with self._lock:
            return len(self._seen)

    def flush(self) -> None:
        """Persist. Direct atomic write — see the module docstring on why this
        does not go through the file write gateway."""
        with self._lock:
            if not self._dirty:
                return
            snapshot = dict(self._seen)
            self._dirty = False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(f".{os.getpid()}.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"nonces": snapshot, "saved_at": time.time()}, fh)
            os.replace(tmp, self._path)
        except (OSError, ValueError) as exc:
            logger.warning("Could not persist capability nonce ledger: %s", exc)

    def reset(self) -> None:
        with self._lock:
            self._seen.clear()
            self._dirty = False


_ledger: NonceLedger | None = None
_ledger_lock = threading.RLock()


def get_nonce_ledger() -> NonceLedger:
    global _ledger
    with _ledger_lock:
        if _ledger is None:
            _ledger = NonceLedger()
        return _ledger


def reset_capability_chain() -> None:
    """Test seam. Drops cached keys and the nonce ledger."""
    global _ledger
    with _ledger_lock:
        _ledger = None
    _KeyMaterial.reset()


# ---------------------------------------------------------------------------
# Issuer — only the Will
# ---------------------------------------------------------------------------


class WillCapabilityIssuer:
    """Mints signed capabilities from real Will decisions.

    There is deliberately no ``issue(domain, action)`` convenience overload that
    skips the decision. A capability exists only as the signed shadow of a
    decision the Will actually made; if you have no decision, you get no
    capability. That is the invariant this whole module exists to enforce.
    """

    def __init__(self, issuer_name: str = "UnifiedWill"):
        self._issuer = issuer_name
        self._revoked: set[str] = set()
        self._lock = threading.RLock()

    # -- issue ------------------------------------------------------------
    def issue_from_decision(
        self,
        decision: Any,
        *,
        action: str,
        payload: Any = None,
        action_digest: str | None = None,
        scope: str = "",
        ttl_s: float = DEFAULT_TTL_S,
    ) -> SignedCapability:
        """Mint a capability bound to ``decision`` and this exact action.

        Raises:
            CapabilityViolation: if the decision did not approve. A refusal can
                never become a capability — that is the whole point of asking.
        """
        outcome = _outcome_str(decision)
        if outcome not in APPROVING_OUTCOMES:
            raise CapabilityViolation(
                CapabilityDenial.NOT_APPROVED,
                f"Will returned '{outcome}' for action '{action}' — refusing to mint",
                sink=action,
            )

        domain = _domain_str(decision)
        if not domain:
            raise CapabilityViolation(
                CapabilityDenial.MALFORMED,
                f"decision for '{action}' carries no domain",
                sink=action,
            )

        digest = action_digest or compute_action_digest(action, payload)
        ttl = max(1.0, min(float(ttl_s), MAX_TTL_S))
        now = time.time()
        keys = _KeyMaterial.load()

        unsigned = SignedCapability(
            capability_id=f"cap-{uuid.uuid4()}",
            schema_version=CAPABILITY_SCHEMA_VERSION,
            outcome=outcome,
            domain=domain,
            action_digest=digest,
            issuer=self._issuer,
            key_id=str(keys["key_id"]),
            nonce=uuid.uuid4().hex,
            receipt_id=str(getattr(decision, "receipt_id", "") or ""),
            scope=str(scope or ""),
            constraints=tuple(str(c) for c in (getattr(decision, "constraints", ()) or ())),
            issued_at=now,
            expires_at=now + ttl,
        )
        signed = replace(unsigned, signature=_sign(unsigned.signing_payload()))
        logger.debug("Issued capability %s", signed.redacted())
        return signed

    # -- revoke -----------------------------------------------------------
    def revoke(self, capability_id: str) -> None:
        with self._lock:
            self._revoked.add(str(capability_id))

    def is_revoked(self, capability_id: str) -> bool:
        with self._lock:
            return str(capability_id) in self._revoked

    @property
    def issuer_name(self) -> str:
        return self._issuer


def _outcome_str(decision: Any) -> str:
    raw = getattr(decision, "outcome", None)
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw)).strip().lower()


def _domain_str(decision: Any) -> str:
    raw = getattr(decision, "domain", None)
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw)).strip().lower()


_issuer: WillCapabilityIssuer | None = None
_issuer_lock = threading.RLock()


def get_capability_issuer() -> WillCapabilityIssuer:
    global _issuer
    with _issuer_lock:
        if _issuer is None:
            _issuer = WillCapabilityIssuer()
        return _issuer


# ---------------------------------------------------------------------------
# Verifier — every consequential sink
# ---------------------------------------------------------------------------


class CapabilityVerifier:
    """Verifies capabilities at a sink. Holds no minting authority.

    Under Ed25519 this object literally cannot produce a capability it would
    accept: it touches only the public key. That is the structural property the
    old dict-lookup token could not offer.
    """

    def verify(
        self,
        capability: Any,
        *,
        expected_domain: Any = None,
        expected_action_digest: str | None = None,
        consume: bool = True,
        now: float | None = None,
    ) -> VerificationResult:
        now = time.time() if now is None else now

        # -- shape --------------------------------------------------------
        if capability is None:
            return VerificationResult(False, CapabilityDenial.MISSING, "no capability presented")
        if isinstance(capability, dict):
            try:
                cap = SignedCapability.from_dict(capability)
            except ValueError as exc:
                return VerificationResult(False, CapabilityDenial.MALFORMED, str(exc))
        elif isinstance(capability, SignedCapability):
            cap = capability
        else:
            return VerificationResult(
                False,
                CapabilityDenial.MALFORMED,
                f"unsupported capability type {type(capability).__name__}",
            )

        if cap.schema_version != CAPABILITY_SCHEMA_VERSION:
            return VerificationResult(
                False,
                CapabilityDenial.SCHEMA_MISMATCH,
                f"schema {cap.schema_version} != {CAPABILITY_SCHEMA_VERSION}",
            )

        # -- signature first ---------------------------------------------
        # Everything below trusts the field values, so nothing below may run
        # until the signature proves those values are the Will's and unaltered.
        if not _verify_signature(cap.signing_payload(), cap.signature, cap.key_id):
            return VerificationResult(
                False,
                CapabilityDenial.BAD_SIGNATURE,
                f"signature does not verify under key {cap.key_id}",
            )

        # -- authority ----------------------------------------------------
        if cap.outcome not in APPROVING_OUTCOMES:
            return VerificationResult(
                False, CapabilityDenial.NOT_APPROVED, f"outcome '{cap.outcome}'"
            )

        if get_capability_issuer().is_revoked(cap.capability_id):
            return VerificationResult(False, CapabilityDenial.REVOKED, cap.capability_id)

        # -- validity window ----------------------------------------------
        if now + _CLOCK_SKEW_TOLERANCE_S < cap.issued_at:
            return VerificationResult(
                False,
                CapabilityDenial.NOT_YET_VALID,
                f"issued {cap.issued_at - now:.3f}s in the future",
            )
        if now >= cap.expires_at:
            return VerificationResult(
                False,
                CapabilityDenial.EXPIRED,
                f"expired {now - cap.expires_at:.3f}s ago",
            )

        # -- binding ------------------------------------------------------
        if expected_domain is not None:
            want = str(getattr(expected_domain, "value", expected_domain)).strip().lower()
            if want != cap.domain:
                return VerificationResult(
                    False,
                    CapabilityDenial.DOMAIN_MISMATCH,
                    f"issued for '{cap.domain}', presented at '{want}'",
                )

        if expected_action_digest is not None:
            if not hmac.compare_digest(str(expected_action_digest), cap.action_digest):
                return VerificationResult(
                    False,
                    CapabilityDenial.ACTION_MISMATCH,
                    f"issued for {cap.action_digest[:12]}…, "
                    f"presented for {str(expected_action_digest)[:12]}…",
                )

        # -- single use ---------------------------------------------------
        # Last, so a capability is not burned by a request that fails an
        # earlier check — otherwise a domain typo would consume real authority.
        if consume:
            if not get_nonce_ledger().consume(cap.nonce, cap.expires_at):
                return VerificationResult(
                    False, CapabilityDenial.REPLAYED, f"nonce already used ({cap.capability_id})"
                )

        return VerificationResult(True, None, "", cap)


_verifier: CapabilityVerifier | None = None
_verifier_lock = threading.RLock()


def get_capability_verifier() -> CapabilityVerifier:
    global _verifier
    with _verifier_lock:
        if _verifier is None:
            _verifier = CapabilityVerifier()
        return _verifier


# ---------------------------------------------------------------------------
# The sink-side enforcement point
# ---------------------------------------------------------------------------

_CAPABILITY_CTX_KEY = "signed_capability"


def attach_capability(ctx: dict[str, Any], cap: SignedCapability) -> dict[str, Any]:
    """Put a capability into an execution context for a downstream sink."""
    ctx[_CAPABILITY_CTX_KEY] = cap.to_dict()
    ctx["capability_id"] = cap.capability_id
    ctx["will_receipt_id"] = cap.receipt_id
    return ctx


def capability_from_context(ctx: Any) -> Any:
    if not isinstance(ctx, dict):
        return None
    return ctx.get(_CAPABILITY_CTX_KEY)


def enforce_capability(
    ctx: Any,
    *,
    sink: str,
    domain: Any,
    action: str,
    payload: Any = None,
    action_digest: str | None = None,
    consume: bool = True,
) -> SignedCapability:
    """Fail-closed authority check for a consequential sink.

    This is the function that closes the chain. Call it at the moment of
    execution — not at plan time, not at admission time — so that what is
    authorized is exactly what runs.

    Raises:
        CapabilityViolation: always, when authority cannot be established.
    """
    digest = action_digest or compute_action_digest(action, payload)
    result = get_capability_verifier().verify(
        capability_from_context(ctx),
        expected_domain=domain,
        expected_action_digest=digest,
        consume=consume,
    )
    if not result.ok:
        logger.warning(
            "🔒 Capability DENIED at sink '%s' for action '%s': %s (%s)",
            sink,
            action,
            (result.denial.value if result.denial else "unknown"),
            result.detail,
        )
    return result.raise_if_denied(sink=sink)


__all__ = [
    "APPROVING_OUTCOMES",
    "CAPABILITY_SCHEMA_VERSION",
    "CapabilityDenial",
    "CapabilityVerifier",
    "CapabilityViolation",
    "NonceLedger",
    "SignedCapability",
    "VerificationResult",
    "WillCapabilityIssuer",
    "attach_capability",
    "capability_chain_status",
    "capability_enforcement_mode",
    "capability_from_context",
    "compute_action_digest",
    "enforce_capability",
    "get_capability_issuer",
    "get_capability_verifier",
    "get_nonce_ledger",
    "issuer_is_asymmetric",
    "reset_capability_chain",
]
"""
    core.governance.capability_chain — end of module
"""
