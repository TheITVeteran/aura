"""Canonical ownership and ingress accounting for every microphone source.

Opening a microphone is not a stateless read.  It holds an exclusive host
resource, survives across many callbacks, and must stop immediately when the
owner revokes ``voice.input_enabled``.  Historically Aura had several direct
owners (the resident voice engine, browser duplex voice, perception probes,
skills, and older voice pipelines) which could all truthfully say *their* mic
was enabled while contending for the same device.

This module is deliberately below those consumers.  A lease is required
before any capture API is touched, and every audio frame is admitted through
``AudioIngressBroker`` before it reaches ASR.  Focused conversation may
preempt passive/ambient listening; equal-priority or lower-priority consumers
are refused with a named reason.  Paired-device microphones use their own
resource group because they are physically distinct from the host mic.

This is resource arbitration and an owner privacy boundary, not an approval
gate on Aura's agency.  Aura may decide to listen or converse whenever her
runtime permits it, but two parts of her cannot own one physical microphone at
the same time and an off switch cannot be advisory.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.permission_gates import microphone_allowed

logger = logging.getLogger("Voice.MicrophoneAuthority")

STALE_MICROPHONE_LEASE_S = 30.0

_MODE_PRIORITY = {
    "passive": 10,
    "ambient": 20,
    "snapshot": 25,
    "focused": 30,
    "calibration": 40,
}


@dataclass(frozen=True)
class MicrophoneDenial:
    reason: str
    detail: str = ""
    remedy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "granted": False,
            "reason": self.reason,
            "detail": self.detail,
            "remedy": self.remedy,
        }


@dataclass
class MicrophoneLease:
    lease_id: str
    holder: str
    principal: str
    source: str
    group: str
    mode: str
    session_id: str
    generation: int
    acquired_at: float
    last_seen: float
    preemptible: bool
    revoke_callback: Callable[[str], Any] | None = field(default=None, repr=False)
    _released: bool = field(default=False, repr=False)
    _revoked_reason: str = field(default="", repr=False)

    @property
    def active(self) -> bool:
        return not self._released

    @property
    def revoked_reason(self) -> str:
        return self._revoked_reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "holder": self.holder,
            "principal": self.principal,
            "source": self.source,
            "group": self.group,
            "mode": self.mode,
            "session_id": self.session_id or None,
            "generation": self.generation,
            "held_for_s": round(max(0.0, time.monotonic() - self.acquired_at), 3),
            "idle_for_s": round(max(0.0, time.monotonic() - self.last_seen), 3),
            "preemptible": self.preemptible,
            "active": self.active,
            "revoked_reason": self.revoked_reason or None,
        }


class MicrophoneAuthority:
    """One lease table for host and paired-device microphones."""

    def __init__(self) -> None:
        self._lock = checked_lock(
            "microphone_authority",
            rank=LockRank.RESOURCE,
        )
        self._leases: dict[str, MicrophoneLease] = {}
        self._generation = 0
        self._denials: dict[str, int] = {}
        self._preemptions = 0
        self._revocations = 0

    @staticmethod
    def resource_group(*, source: str, principal: str) -> str:
        """Map capture transports to the physical resource they hold."""
        principal = str(principal or "owner:local")
        if principal.startswith("paired:"):
            return f"remote_microphone:{principal.removeprefix('paired:')}"
        del source
        return "host_microphone"

    def acquire(
        self,
        holder: str,
        *,
        principal: str,
        source: str,
        mode: str,
        session_id: str = "",
        preemptible: bool = True,
        revoke_callback: Callable[[str], Any] | None = None,
    ) -> MicrophoneLease | MicrophoneDenial:
        """Reserve a physical microphone before any capture API is touched."""
        if not microphone_allowed():
            return self._deny(
                "owner_disabled",
                "voice.input_enabled is off in Aura's runtime settings.",
                "Turn microphone input on in Runtime Settings.",
            )

        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode not in _MODE_PRIORITY:
            return self._deny(
                "invalid_mode",
                f"Unknown microphone mode {mode!r}.",
                "Use passive, ambient, snapshot, focused, or calibration.",
            )

        normalized_source = str(source or "unknown")
        normalized_principal = str(principal or "owner:local")
        group = self.resource_group(
            source=normalized_source,
            principal=normalized_principal,
        )
        displaced: MicrophoneLease | None = None
        now = time.monotonic()

        with self._lock:
            existing = self._leases.get(group)
            if existing is not None and existing.active:
                idle_s = now - existing.last_seen
                if idle_s >= STALE_MICROPHONE_LEASE_S:
                    displaced = self._revoke_locked(existing, "stale_holder_reclaimed")
                    self._revocations += 1
                elif (
                    _MODE_PRIORITY[normalized_mode] > _MODE_PRIORITY[existing.mode]
                    and existing.preemptible
                ):
                    displaced = self._revoke_locked(
                        existing,
                        f"preempted_by:{holder}:{normalized_mode}",
                    )
                    self._preemptions += 1
                else:
                    return self._deny_locked(
                        "device_busy",
                        f"{existing.holder} owns {group} in {existing.mode} mode.",
                        "Wait for the current capture to finish or use a higher-priority focused session.",
                    )

            self._generation += 1
            lease = MicrophoneLease(
                lease_id=uuid.uuid4().hex,
                holder=str(holder or "unknown"),
                principal=normalized_principal,
                source=normalized_source,
                group=group,
                mode=normalized_mode,
                session_id=str(session_id or ""),
                generation=self._generation,
                acquired_at=now,
                last_seen=now,
                preemptible=bool(preemptible),
                revoke_callback=revoke_callback,
            )
            self._leases[group] = lease

        if displaced is not None:
            self._notify_revocation(displaced)
        logger.info(
            "Microphone lease %s acquired by %s (%s/%s)",
            lease.lease_id[:10],
            lease.holder,
            lease.source,
            lease.mode,
        )
        return lease

    def heartbeat(self, lease: MicrophoneLease | None) -> bool:
        if lease is None or not lease.active:
            return False
        with self._lock:
            current = self._leases.get(lease.group)
            if current is not lease or not lease.active:
                return False
            lease.last_seen = time.monotonic()
            return True

    def validate(self, lease: MicrophoneLease | None) -> tuple[bool, str]:
        if lease is None:
            return False, "lease_missing"
        if not microphone_allowed():
            self.revoke(lease, reason="owner_disabled")
            return False, "owner_disabled"
        with self._lock:
            current = self._leases.get(lease.group)
            if current is lease and lease.active:
                return True, "active"
        return False, lease.revoked_reason or "lease_not_authoritative"

    def release(self, lease: MicrophoneLease | None, *, reason: str = "released") -> bool:
        if lease is None:
            return False
        with self._lock:
            current = self._leases.get(lease.group)
            was_current = current is lease and lease.active
            if was_current:
                self._leases.pop(lease.group, None)
            if lease.active:
                lease._released = True
                lease._revoked_reason = str(reason or "released")
            self._generation += 1
        if was_current:
            logger.info("Microphone lease %s released (%s)", lease.lease_id[:10], reason)
        return was_current

    def revoke(self, lease: MicrophoneLease | None, *, reason: str) -> bool:
        if lease is None:
            return False
        with self._lock:
            current = self._leases.get(lease.group)
            if current is not lease or not lease.active:
                return False
            revoked = self._revoke_locked(lease, reason)
            self._revocations += 1
            self._generation += 1
        self._notify_revocation(revoked)
        return True

    def revoke_all(self, *, reason: str) -> dict[str, Any]:
        """Invalidate all microphone transports and notify their owners."""
        with self._lock:
            revoked = [
                self._revoke_locked(lease, reason)
                for lease in tuple(self._leases.values())
                if lease.active
            ]
            if revoked:
                self._revocations += len(revoked)
                self._generation += 1
        for lease in revoked:
            self._notify_revocation(lease)
        return {
            "revoked": len(revoked),
            "holders": [lease.holder for lease in revoked],
            "reason": reason,
        }

    def _revoke_locked(self, lease: MicrophoneLease, reason: str) -> MicrophoneLease:
        if self._leases.get(lease.group) is lease:
            self._leases.pop(lease.group, None)
        lease._released = True
        lease._revoked_reason = str(reason or "revoked")
        return lease

    @staticmethod
    def _notify_revocation(lease: MicrophoneLease) -> None:
        callback = lease.revoke_callback
        if callback is None:
            return
        try:
            callback(lease.revoked_reason)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "microphone_authority",
                exc,
                severity="warning",
                action="microphone holder revocation callback failed",
                extra={"holder": lease.holder, "reason": lease.revoked_reason},
                enforce_failure_policy=False,
            )

    def _deny(self, reason: str, detail: str, remedy: str) -> MicrophoneDenial:
        with self._lock:
            return self._deny_locked(reason, detail, remedy)

    def _deny_locked(
        self,
        reason: str,
        detail: str,
        remedy: str,
    ) -> MicrophoneDenial:
        self._denials[reason] = self._denials.get(reason, 0) + 1
        return MicrophoneDenial(reason=reason, detail=detail, remedy=remedy)

    def state(self) -> dict[str, Any]:
        with self._lock:
            holders = {
                group: lease.to_dict()
                for group, lease in self._leases.items()
                if lease.active
            }
            generation = self._generation
            denials = dict(self._denials)
            preemptions = self._preemptions
            revocations = self._revocations
        return {
            "schema": "aura.voice.microphone_authority.v1",
            "input_permitted": microphone_allowed(),
            # A lease is admission to open/retain capture, not proof that a
            # hardware callback has produced a frame. Ingress activity is
            # measured separately by AudioIngressBroker.
            "lease_active": bool(holders),
            "holders": holders,
            "generation": generation,
            "preemptions": preemptions,
            "revocations": revocations,
            "denials": denials,
        }


class AudioIngressBroker:
    """Admit and meter audio only while its source owns a live lease."""

    def __init__(self, authority: MicrophoneAuthority) -> None:
        self._authority = authority
        self._lock = checked_lock("audio_ingress_broker", rank=LockRank.LEAF)
        self._frames: dict[str, int] = {}
        self._bytes: dict[str, int] = {}
        self._rejected = 0

    def admit(self, lease: MicrophoneLease | None, byte_count: int) -> bool:
        if byte_count <= 0 or not self._authority.heartbeat(lease):
            with self._lock:
                self._rejected += 1
            return False
        assert lease is not None
        with self._lock:
            self._frames[lease.lease_id] = self._frames.get(lease.lease_id, 0) + 1
            self._bytes[lease.lease_id] = self._bytes.get(lease.lease_id, 0) + int(byte_count)
        return True

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": "aura.voice.audio_ingress_broker.v1",
                "frames_by_lease": dict(self._frames),
                "bytes_by_lease": dict(self._bytes),
                "rejected_frames": self._rejected,
            }


_authority = MicrophoneAuthority()
_broker = AudioIngressBroker(_authority)


def get_microphone_authority() -> MicrophoneAuthority:
    return _authority


def get_audio_ingress_broker() -> AudioIngressBroker:
    return _broker


__all__ = [
    "STALE_MICROPHONE_LEASE_S",
    "AudioIngressBroker",
    "MicrophoneAuthority",
    "MicrophoneDenial",
    "MicrophoneLease",
    "get_audio_ingress_broker",
    "get_microphone_authority",
]
