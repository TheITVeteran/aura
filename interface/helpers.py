"""interface/helpers.py
─────────────────────
Shared helpers used by multiple route files and the main server module.
"""
from __future__ import annotations

import logging
import time

from core.exceptions import ContainerError
from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger("Aura.Server.Helpers")
_USER_SPOKE_HOOK_ERRORS = (
    ImportError,
    AttributeError,
    ContainerError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _record_user_spoke_hook_failure(hook: str, exc: BaseException) -> None:
    record_degradation('helpers', exc)
    logger.debug("User-spoke hook '%s' skipped: %s", hook, exc)


def _notify_user_spoke(message: str = ""):
    """
    Central hook called whenever the user sends any message (WS or REST).
    Updates all proactive presence/communication systems so they respect the
    active-conversation window and do not monologue into a silent room.

    Pass the message text so ProactivePresence can detect away signals
    (e.g. "heading to the gym") and suppress autonomous chat accordingly.
    """
    try:
        from core.runtime.foreground_guard import notify_user_spoke

        notify_user_spoke(message)
    except _USER_SPOKE_HOOK_ERRORS as _e:
        _record_user_spoke_hook_failure("foreground_guard", _e)

    try:
        orch = get_runtime_service("orchestrator", default=None)
    except _USER_SPOKE_HOOK_ERRORS as _e:
        _record_user_spoke_hook_failure("orchestrator_lookup", _e)
        return

    if not orch:
        return

    # Phase-30 ProactivePresence tracks interaction time and away signals.
    pp = getattr(orch, "proactive_presence", None)
    if pp:
        try:
            if message and hasattr(pp, "mark_user_spoke_with_message"):
                pp.mark_user_spoke_with_message(message)
            elif hasattr(pp, "mark_user_spoke"):
                pp.mark_user_spoke()
        except _USER_SPOKE_HOOK_ERRORS as _e:
            _record_user_spoke_hook_failure("proactive_presence", _e)

    # Older ProactiveCommunicationManager resets unanswered backoff.
    pc = getattr(orch, "proactive_comm", None)
    if pc and hasattr(pc, "record_user_interaction"):
        try:
            pc.record_user_interaction()
        except _USER_SPOKE_HOOK_ERRORS as _e:
            _record_user_spoke_hook_failure("proactive_comm", _e)

    # ProactiveInitiativeEngine if attached.
    pie = getattr(orch, "proactive_initiative_engine", None)
    if pie and hasattr(pie, "register_user_interaction"):
        try:
            pie.register_user_interaction()
        except _USER_SPOKE_HOOK_ERRORS as _e:
            _record_user_spoke_hook_failure("proactive_initiative_engine", _e)

    try:
        orch._last_user_interaction_time = time.time()
    except _USER_SPOKE_HOOK_ERRORS as _e:
        _record_user_spoke_hook_failure("last_user_interaction_time", _e)
