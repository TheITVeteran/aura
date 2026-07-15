"""One-time, action-bound user confirmations for runtime policy overlays."""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from core.executive.standing_authority import canonical_arguments_digest

logger = logging.getLogger("Aura.ActionConfirmation")

_PENDING_TTL_SECONDS = 300.0
_AUTHORIZED_TTL_SECONDS = 60.0
_MAX_CHALLENGES = 256


def action_confirmation_fingerprint(
    *,
    tool_name: Any,
    arguments: Mapping[str, Any] | None,
    source: Any,
    risk_level: Any,
    effect_scope: Any,
) -> str:
    """Return a stable, content-hiding identity for one proposed action."""

    body = {
        "tool": str(tool_name or "").strip().lower(),
        "arguments_sha256": canonical_arguments_digest(arguments),
        "source": str(source or "unknown").strip().lower(),
        "risk_level": str(risk_level or "unknown").strip().lower(),
        "effect_scope": str(effect_scope or "unknown").strip().lower(),
    }
    encoded = json.dumps(
        body,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class _Challenge:
    challenge_id: str
    action_fingerprint: str
    tool_name: str
    created_at: float
    pending_deadline: float
    authorized_at: float = 0.0
    authorization_deadline: float = 0.0
    consumed_at: float = 0.0


class ActionConfirmationRegistry:
    """Process-local registry for expiring, one-use confirmation challenges."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        pending_ttl_seconds: float = _PENDING_TTL_SECONDS,
        authorized_ttl_seconds: float = _AUTHORIZED_TTL_SECONDS,
        max_challenges: int = _MAX_CHALLENGES,
    ) -> None:
        self._clock = clock
        self._pending_ttl_seconds = max(1.0, float(pending_ttl_seconds))
        self._authorized_ttl_seconds = max(1.0, float(authorized_ttl_seconds))
        self._max_challenges = max(8, int(max_challenges))
        self._lock = threading.RLock()
        self._challenges: OrderedDict[str, _Challenge] = OrderedDict()

    @property
    def authorized_ttl_seconds(self) -> float:
        return self._authorized_ttl_seconds

    def issue(self, *, action_fingerprint: str, tool_name: str) -> dict[str, Any]:
        fingerprint = self._validated_fingerprint(action_fingerprint)
        now = self._clock()
        challenge = _Challenge(
            challenge_id="action-confirm-" + secrets.token_urlsafe(32),
            action_fingerprint=fingerprint,
            tool_name=str(tool_name or "action")[:120],
            created_at=now,
            pending_deadline=now + self._pending_ttl_seconds,
        )
        with self._lock:
            self._prune_locked(now)
            self._challenges[challenge.challenge_id] = challenge
            while len(self._challenges) > self._max_challenges:
                self._challenges.popitem(last=False)
        logger.info(
            "Issued action-bound confirmation challenge %s for %s.",
            challenge.challenge_id,
            challenge.tool_name,
        )
        return {
            "challenge_id": challenge.challenge_id,
            "tool": challenge.tool_name,
            "pending_expires_in_seconds": self._pending_ttl_seconds,
            "one_time": True,
            "action_bound": True,
        }

    def authorize(self, challenge_id: str) -> dict[str, Any]:
        normalized = self._validated_challenge_id(challenge_id)
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            challenge = self._challenges.get(normalized)
            if challenge is None:
                raise KeyError("action_confirmation_challenge_not_found")
            if challenge.consumed_at > 0.0:
                raise RuntimeError("action_confirmation_already_consumed")
            if now > challenge.pending_deadline:
                self._challenges.pop(normalized, None)
                raise RuntimeError("action_confirmation_challenge_expired")
            challenge.authorized_at = now
            challenge.authorization_deadline = now + self._authorized_ttl_seconds
            self._challenges.move_to_end(normalized)
        logger.info("Authorized one-time action confirmation %s.", normalized)
        return {
            "ok": True,
            "challenge_id": normalized,
            "tool": challenge.tool_name,
            "authorization_expires_in_seconds": self._authorized_ttl_seconds,
            "one_time": True,
            "action_bound": True,
        }

    def consume_authorized(self, action_fingerprint: str) -> tuple[bool, str]:
        """Atomically consume one authorized challenge for the exact action."""

        fingerprint = self._validated_fingerprint(action_fingerprint)
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            for challenge in reversed(tuple(self._challenges.values())):
                if challenge.action_fingerprint != fingerprint:
                    continue
                if challenge.consumed_at > 0.0:
                    continue
                if challenge.authorized_at <= 0.0:
                    continue
                if now > challenge.authorization_deadline:
                    continue
                challenge.consumed_at = now
                logger.info(
                    "Consumed action-bound confirmation %s for %s.",
                    challenge.challenge_id,
                    challenge.tool_name,
                )
                return True, challenge.challenge_id
        return False, "action_confirmation_missing_or_expired"

    def revoke_authorization(self, challenge_id: str) -> None:
        """Undo an unconsumed authorization when a downstream acknowledgement fails."""

        normalized = self._validated_challenge_id(challenge_id)
        with self._lock:
            challenge = self._challenges.get(normalized)
            if challenge is None or challenge.consumed_at > 0.0:
                return
            challenge.authorized_at = 0.0
            challenge.authorization_deadline = 0.0

    def cancel(self, challenge_id: str) -> bool:
        """Remove an unconsumed challenge after the user abandons the prompt."""

        normalized = self._validated_challenge_id(challenge_id)
        with self._lock:
            challenge = self._challenges.get(normalized)
            if challenge is None or challenge.consumed_at > 0.0:
                return False
            self._challenges.pop(normalized, None)
            return True

    def clear_for_tests(self) -> None:
        with self._lock:
            self._challenges.clear()

    def _prune_locked(self, now: float) -> None:
        expired = [
            challenge_id
            for challenge_id, challenge in self._challenges.items()
            if (
                challenge.consumed_at > 0.0
                or (
                    challenge.authorized_at > 0.0
                    and now > challenge.authorization_deadline
                )
                or (
                    challenge.authorized_at <= 0.0
                    and now > challenge.pending_deadline
                )
            )
        ]
        for challenge_id in expired:
            self._challenges.pop(challenge_id, None)

    @staticmethod
    def _validated_fingerprint(value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized.startswith("sha256:") or len(normalized) != 71:
            raise ValueError("invalid_action_confirmation_fingerprint")
        return normalized

    @staticmethod
    def _validated_challenge_id(value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized.startswith("action-confirm-") or len(normalized) > 160:
            raise ValueError("invalid_action_confirmation_challenge_id")
        if any(ord(character) < 33 or ord(character) > 126 for character in normalized):
            raise ValueError("invalid_action_confirmation_challenge_id")
        return normalized


_REGISTRY: ActionConfirmationRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_action_confirmation_registry() -> ActionConfirmationRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            if _REGISTRY is None:
                _REGISTRY = ActionConfirmationRegistry()
    return _REGISTRY


__all__ = [
    "ActionConfirmationRegistry",
    "action_confirmation_fingerprint",
    "get_action_confirmation_registry",
]
