"""Expected-backpressure discipline for background cognition.

A bounded background generation (pruner consolidation, crucible debate
stage) timing out while the foreground conversation lane holds the model
is ROUTINE: the background yielded, nothing broke. The old behavior booked
these as warning+ degradations, and fail-closed subsystems escalate
warning+ to CRITICAL — so a chat turn landing mid-consolidation minted
critical incidents and spiked existential threat (observed live July 2026:
INC-1783068731-0001 sovereign_pruner, INC-1783068780-0002
dialectical_crucible, both plain TimeoutErrors under foreground load).

The discipline (CLAUDE.md convention, now enforced in one place):

- foreground busy + timeout   -> info log, count it, NO degradation record
- persistent (N consecutive)  -> ONE warning-level degradation (real signal)
- foreground idle + timeout   -> real degradation immediately (something IS wrong)
- success                     -> counter resets

Callers should also prefer *yielding before starting*: check
``foreground_inference_active()`` at stage entry and skip the cycle
entirely rather than compete with the user's turn and time out.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Runtime.Backpressure")

_counters: dict[str, int] = {}
_last_event_at: dict[str, float] = {}
_lock = threading.Lock()

# A consecutive-failure streak older than this is stale — the pressure
# window has passed; start counting fresh.
_STREAK_RESET_S = 900.0


def foreground_inference_active() -> bool:
    """Best-effort: is the live conversation lane using the model right now?"""
    try:
        from core.brain.inference_gate import InferenceGate

        return bool(
            InferenceGate._foreground_user_turn_active()
            or InferenceGate._foreground_owner_active()
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return False


def clear_backpressure(subsystem: str) -> None:
    """Call on success: the pressure window is over for this subsystem."""
    with _lock:
        _counters.pop(subsystem, None)
        _last_event_at.pop(subsystem, None)


def record_expected_backpressure(
    subsystem: str,
    error: BaseException,
    *,
    action: str,
    escalate_after: int = 3,
    extra: dict[str, Any] | None = None,
) -> str:
    """Handle a bounded-background timeout with backpressure discipline.

    Returns "yielded" when the event was absorbed as expected backpressure,
    "escalated" when it was recorded as a real degradation (persistent
    streak, or the foreground lane was idle so the timeout is unexplained).
    """
    now = time.monotonic()
    with _lock:
        if now - _last_event_at.get(subsystem, now) > _STREAK_RESET_S:
            _counters[subsystem] = 0
        _last_event_at[subsystem] = now
        _counters[subsystem] = _counters.get(subsystem, 0) + 1
        streak = _counters[subsystem]

    # CRITICAL: a background timeout must NEVER fail-close its subsystem. Many
    # of these (sovereign_pruner, dialectical_crucible) are on the fail-closed
    # list, and record_degradation escalates ANY warning+ on a fail-closed
    # service to a CRITICAL SERVICE FAILURE that RAISES and drives the runtime
    # into failure-lockdown 1.00 — which is exactly what pinned the live kernel
    # unhealthy and left the desktop stuck on "Connecting to runtime" (2026-07-04).
    # Record on a dedicated ``.backpressure`` channel (not a registered
    # fail-closed service) with the policy disabled, so the event stays visible
    # without ever locking the mind down over a slow housekeeping pass.
    backpressure_channel = f"{subsystem}.backpressure"

    if not foreground_inference_active():
        record_degradation(
            backpressure_channel,
            error,
            severity="warning",
            action=f"{action} (foreground idle — timeout is unexplained, not backpressure)",
            extra=extra,
            enforce_failure_policy=False,
        )
        return "escalated"

    if streak >= max(1, int(escalate_after)):
        record_degradation(
            backpressure_channel,
            error,
            severity="warning",
            action=(
                f"{action} (persistent: {streak} consecutive under foreground load — "
                "backpressure is no longer transient)"
            ),
            extra=extra,
            enforce_failure_policy=False,
        )
        with _lock:
            _counters[subsystem] = 0
        return "escalated"

    logger.info(
        "[BACKPRESSURE] %s yielded to foreground inference (%d/%d): %s: %s → %s",
        subsystem,
        streak,
        escalate_after,
        type(error).__name__,
        str(error)[:120],
        action,
    )
    return "yielded"


def reset_backpressure_state() -> None:
    """Testing hook."""
    with _lock:
        _counters.clear()
        _last_event_at.clear()


__all__ = [
    "foreground_inference_active",
    "record_expected_backpressure",
    "clear_backpressure",
    "reset_backpressure_state",
]
