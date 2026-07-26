"""Actuator authority verification — a boolean in a dict is not authorization.

CP126 raised the same critical finding against every privileged actuator
(``8900fa05``, ``27651212``, ``9f94bf4d``, ``251ada47``, ``bdb4255d``,
``5ce6b589``, ``5acd8c38``, …): each one gated on ``params["_aura_authorized"]``,
which the ActuatorRegistry injects after the AuthorityGateway approves a call.
Any direct caller could set that key themselves and execute a privileged
actuator with no gateway, no principal, and no receipt.

The missing piece was never the token — ``core.runtime.capability_tokens``
already models capability/scope/issuer/expiry/one-shot-consume, and the
registry already issues one. Nothing *verified* it.

This module closes that gap with two halves:

* :func:`actuator_authorization` — the registry wraps execution in it, putting
  the authorization on a ContextVar (which propagates into ``asyncio.to_thread``,
  where actuator bodies actually run).
* :func:`verify_actuator_authority` — every privileged actuator calls it instead
  of reading the raw boolean. It requires BOTH a live authorization context and,
  when a capability token id is present, a token that actually validates.

Fail-closed direction: a fabricated flag with no context is refused; an expired,
revoked, unknown, or wrong-capability token is refused. Being unable to prove
authorization is not authorization.
"""
from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ActuatorAuthorization",
    "actuator_authorization",
    "current_authorization",
    "verify_actuator_authority",
]


@dataclass(frozen=True)
class ActuatorAuthorization:
    """Proof that the registry/AuthorityGateway approved this specific call."""

    actuator: str
    capability_token_id: str | None = None
    decision_reason: str = ""
    principal: str = ""


_ACTIVE: contextvars.ContextVar[ActuatorAuthorization | None] = contextvars.ContextVar(
    "aura_actuator_authorization", default=None
)
_lock = threading.RLock()


@contextmanager
def actuator_authorization(
    actuator: str,
    *,
    capability_token_id: str | None = None,
    decision_reason: str = "",
    principal: str = "",
):
    """Mark the dynamic extent in which ``actuator`` is genuinely authorized.

    Only the ActuatorRegistry (after a successful AuthorityGateway decision)
    should enter this scope.
    """
    token = _ACTIVE.set(
        ActuatorAuthorization(
            actuator=str(actuator),
            capability_token_id=capability_token_id,
            decision_reason=str(decision_reason or ""),
            principal=str(principal or ""),
        )
    )
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def current_authorization() -> ActuatorAuthorization | None:
    """The authorization in force for this execution, if any."""
    return _ACTIVE.get()


def _token_is_valid(token_id: str, actuator: str) -> tuple[bool, str]:
    """Validate a capability token id against the store (fail closed)."""
    try:
        from core.runtime.capability_tokens import TokenStatus, get_capability_token_store

        token = get_capability_token_store().get(token_id)
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
        # The store is unavailable — we cannot prove the token, so we cannot
        # claim it authorized anything.
        return False, "capability token store unavailable"
    if token is None:
        return False, "capability token is unknown"
    status = getattr(token, "status", None)
    if status not in (TokenStatus.ISSUED, TokenStatus.USED):
        # Use the enum VALUE ("revoked"/"expired"), not its repr.
        return False, f"capability token is {getattr(status, 'value', status) or 'invalid'}"
    import time as _time

    if _time.time() >= float(getattr(token, "expires_at", 0.0) or 0.0):
        return False, "capability token has expired"
    capability = str(getattr(token, "capability", "") or "")
    if capability and actuator and capability not in (actuator, "*"):
        return False, f"capability token is scoped to '{capability}', not '{actuator}'"
    return True, ""


def verify_actuator_authority(
    params: Any,
    *,
    actuator: str,
    require_context: bool = True,
) -> tuple[bool, str]:
    """Verify that this actuator call is genuinely authorized.

    Returns ``(ok, reason)``. Replaces ``if not params.get("_aura_authorized")``.

    Checks, in order:

    1. The legacy flag must still be present (callers/tests that never set it
       are refused exactly as before).
    2. A live :func:`actuator_authorization` context must exist — this is what a
       direct caller who fabricates the flag cannot produce.
    3. If a capability token id is supplied (by the caller or the context), it
       must resolve to a live, unexpired token scoped to this actuator.
    """
    if not isinstance(params, dict) or not params.get("_aura_authorized"):
        return False, (
            f"{actuator} requires ActuatorRegistry/AuthorityGateway authorization."
        )

    auth = current_authorization()
    if require_context and auth is None:
        # The flag was set by someone who never went through the registry.
        return False, (
            f"{actuator} refused: '_aura_authorized' was set without a registry "
            "authorization context (the flag alone is not authorization)."
        )
    if auth is not None and auth.actuator not in (actuator, "*"):
        return False, (
            f"{actuator} refused: the active authorization is for '{auth.actuator}'."
        )

    token_id = params.get("_capability_token_id") or (auth.capability_token_id if auth else None)
    if token_id:
        ok, reason = _token_is_valid(str(token_id), actuator)
        if not ok:
            return False, f"{actuator} refused: {reason}."
    return True, ""
