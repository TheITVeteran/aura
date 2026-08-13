"""core/senses/vision_policy.py — one meaning for "can Aura see".

``AURA_ENABLE_PROACTIVE_VISION`` was read in four places with three different
defaults:

* ``core/senses/continuous_perception.py`` — ``os.getenv(..., "0") == "1"``
* ``core/senses/pulse_manager.py``         — ``os.getenv(..., "0") == "1"``
* ``core/senses/continuous_vision.py``     — default ``"1"``, opt-out
* ``core/orchestrator/main.py``            — ``_env_flag(..., True)``

So with the variable unset — the normal case — the continuous-vision engine and
the orchestrator believed ambient vision was ON while the perception engine and
the pulse manager believed it was OFF. Nothing was inconsistent enough to fail;
it just meant "does Aura watch the screen" had no single answer, and which half
of the system you asked decided what you were told.

That is the shape this codebase already refuses elsewhere. ``core/runtime/flags.py``
raises on a conflicting re-declaration precisely because "a knob must have
exactly one meaning", and this flag never went through that registry, so nothing
caught the divergence.

The resolved default is **ON**. Ambient perception is part of the live Aura
rather than an extra, and the two readers that defaulted it off were the
accident — the orchestrator, which owns whether the loop starts at all, already
defaulted it on. Turning it off remains one explicit assignment.

Headless boots never get ambient vision regardless: there is no screen to read,
and ``continuous_vision`` already refused on ``AURA_HEADLESS``. That refusal
lives here now so every caller inherits it instead of one caller remembering it.
"""

from __future__ import annotations

import os

__all__ = ["proactive_vision_enabled", "vision_policy_reason"]

_TRUE = frozenset({"1", "true", "yes", "on", "enabled"})
_FALSE = frozenset({"0", "false", "no", "off", "disabled"})


def _flag(name: str) -> bool | None:
    """Tri-state read: True, False, or None when the operator said nothing."""

    raw = str(os.environ.get(name, "") or "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return None


def vision_policy_reason() -> str:
    """Why ambient vision is off, or "" when it is on.

    A reason string rather than a bare bool so the log line at the call site
    can say which condition refused — "headless" and "operator turned it off"
    are different facts and used to be reported identically.
    """

    if _flag("AURA_HEADLESS") is True:
        return "headless"
    if _flag("AURA_ENABLE_PROACTIVE_VISION") is False:
        return "operator_disabled"
    return ""


def proactive_vision_enabled() -> bool:
    """Whether ambient/proactive screen perception may run.

    Default ON when the operator has said nothing.
    """

    return not vision_policy_reason()
