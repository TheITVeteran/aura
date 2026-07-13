"""interface/routes/system.py
─────────────────────────────
Extracted from server.py — Health, telemetry, metrics, bootstrap,
and all collector/diagnostic helpers.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, cast

import fastapi.responses as fastapi_responses
from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse, StreamingResponse

from core.config import config
from core.container import ServiceContainer
from core.health.boot_status import build_boot_health_snapshot
from core.health.conversation_lane import conversation_lane_is_busy
from core.runtime import resource_psutil as psutil
from core.runtime.errors import record_degradation
from core.runtime.health_contract import (
    REQUIRED_HEALTH_PROBE_GROUPS,
    required_probe_blockers,
    required_probe_groups_pass,
)
from core.runtime.service_access import optional_service
from core.runtime.shutdown_coordinator import (
    get_shutdown_coordinator,
    is_shutdown_requested,
)
from core.runtime.task_ownership import create_tracked_task
from core.runtime.version import VERSION, version_string
from core.scheduler import scheduler
from core.tools.runtime_tools import get_runtime_state
from interface.auth import (
    _require_internal,
    _restore_owner_session_from_request,
    paired_device_session_id,
    request_access_profile,
)
from interface.websocket_manager import broadcast_bus, runtime_heartbeat_payload, ws_manager

_SYSTEM_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    asyncio.InvalidStateError,
    asyncio.QueueEmpty,
    asyncio.QueueFull,
    json.JSONDecodeError,
    psutil.Error,
    subprocess.SubprocessError,
)

_TOOL_CATALOG_BOOTSTRAP_MAX_ITEMS = 256
_TOOL_CATALOG_BOOTSTRAP_READ_BUDGET_S = 0.35


def _shutdown_health_status() -> dict[str, object]:
    try:
        return get_shutdown_coordinator().get_status()
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        return {
            "running": False,
            "request": {"requested": is_shutdown_requested()},
            "report": None,
            "error": repr(exc),
        }


def _stopping_boot_health_payload() -> tuple[dict[str, Any], int] | None:
    shutdown = _shutdown_health_status()
    request = shutdown.get("request")
    if not isinstance(request, dict) or request.get("requested") is not True:
        return None
    return (
        {
            "ready": False,
            "status": "stopping",
            "system_ready": False,
            "launcher_ready": False,
            "conversation_ready": False,
            "boot_phase": "runtime_shutdown",
            "required_probes": {"all_passed": False},
            "blockers": ["runtime_shutdown"],
            "shutdown": shutdown,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        },
        503,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _env_positive_float(name: str, default: float) -> float:
    value = _safe_float(os.getenv(name, ""), default)
    return value if value > 0.0 else default


async def _optional_threaded_status(
    label: str,
    fn: Any,
    *,
    timeout_s: float = 0.18,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read optional health-panel data without blocking the API loop."""

    fallback_payload = {"_stale": True, "reason": "status_unavailable"}
    if fallback:
        fallback_payload.update(fallback)
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(fn),
            timeout=max(0.05, float(timeout_s)),
        )
        return dict(result or {}) if isinstance(result, dict) else {"value": result, "_stale": False}
    except TimeoutError:
        logger.debug("Optional health status %s timed out after %.2fs", label, timeout_s)
        fallback_payload["reason"] = "status_timeout"
        return fallback_payload
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system.optional_status", exc)
        logger.debug("Optional health status %s failed: %s", label, exc)
        fallback_payload["reason"] = f"{type(exc).__name__}: {exc}"
        return fallback_payload


def _runtime_component_status(
    service_name: str,
    *status_methods: str,
) -> dict[str, Any]:
    """Read a registered background component without instantiating a new one."""

    service = ServiceContainer.get(service_name, default=None)
    if service is None:
        return {"registered": False, "running": False, "reason": "not_registered"}
    for method_name in status_methods:
        method = getattr(service, method_name, None)
        if not callable(method):
            continue
        try:
            status = method()
            if inspect.isawaitable(status):
                close = getattr(status, "close", None)
                if callable(close):
                    close()
                return {
                    "registered": True,
                    "running": False,
                    "reason": "async_status_not_supported_in_health_snapshot",
                }
            if isinstance(status, dict):
                return {"registered": True, **status}
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "system.background_runtime",
                exc,
                action=f"marked {service_name} status unavailable",
            )
            return {
                "registered": True,
                "running": False,
                "reason": f"status_error:{type(exc).__name__}",
            }
    task = getattr(service, "_task", None) or getattr(service, "_background_task", None)
    running = bool(
        getattr(service, "_running", False)
        or getattr(service, "_started", False)
        or (task is not None and not task.done())
    )
    return {"registered": True, "running": running}


def _collect_full_runtime_status(
    pneuma_data: dict[str, Any],
    mhaf_data: dict[str, Any],
) -> dict[str, Any]:
    """Report whether a normal desktop launch actually started Aura's organs."""

    from core.runtime.background_policy import (
        background_activity_reason,
        background_cognition_disabled_reason,
        background_loop_start_reason,
        foreground_only_runtime,
    )
    from core.runtime.desktop_boot_safety import (
        desktop_resource_guard_enabled,
        desktop_safe_boot_enabled,
    )

    conductor = _runtime_component_status("autonomy_conductor", "status")
    conductor["running"] = bool(conductor.get("active", False))
    overt = _runtime_component_status("overt_action_loop", "status")
    jobs = conductor.get("jobs", {}) if isinstance(conductor.get("jobs"), dict) else {}
    overt["scheduled"] = bool(
        conductor.get("active") and "overt_action_cycle" in jobs
    )
    overt["running"] = bool(overt.get("scheduled") and overt.get("enabled", False))
    agency = ServiceContainer.get("agency_core", default=None)
    swarm = getattr(agency, "swarm", None)
    deliberation = (
        swarm.get_status()
        if swarm is not None and callable(getattr(swarm, "get_status", None))
        else {"available": False, "active_shards": 0}
    )
    deliberation["registered"] = bool(swarm is not None)
    deliberation["scheduled"] = bool(
        conductor.get("active") and "internal_deliberation_cycle" in jobs
    )
    deliberation["running"] = bool(
        deliberation.get("registered") and deliberation.get("scheduled")
    )

    components = {
        "pneuma": {
            "registered": ServiceContainer.get("pneuma", default=None) is not None,
            "running": bool(pneuma_data.get("online")),
            **pneuma_data,
        },
        "mhaf": {
            "registered": ServiceContainer.get("mhaf", default=None) is not None,
            "running": bool(mhaf_data.get("online")),
            **mhaf_data,
        },
        "curiosity": _runtime_component_status("curiosity_engine", "get_status"),
        "proactive_communication": _runtime_component_status("proactive_comm", "get_status"),
        "autonomous_initiative": _runtime_component_status(
            "autonomous_initiative_loop",
            "get_status",
        ),
        "subjective_choice": _runtime_component_status(
            "subjective_choice_engine",
            "get_status",
            "status",
        ),
        "ambient_life_director": _runtime_component_status(
            "ambient_life_director",
            "get_status",
            "status",
        ),
        "research": _runtime_component_status("research_cycle", "get_status"),
        "self_healing": _runtime_component_status("self_healing", "get_status"),
        "self_modification": _runtime_component_status(
            "self_modification_engine", "runtime_status"
        ),
        "consciousness_stream": _runtime_component_status("consciousness"),
        "autonomy_conductor": conductor,
        "overt_action": overt,
        "deliberation": deliberation,
        "wake_word": _runtime_component_status("wake_word", "get_status"),
        "screen_perception": _runtime_component_status("screen_perception", "get_status"),
        "perceptual_pump": _runtime_component_status("perceptual_pump", "get_status"),
        "cognitive_situation": _runtime_component_status(
            "cognitive_situation",
            "get_status",
            "status",
        ),
        "imagination_engine": _runtime_component_status(
            "imagination_engine",
            "get_status",
            "status",
            "snapshot",
        ),
        "timescale_bridge": _runtime_component_status(
            "timescale_bridge",
            "get_status",
            "status",
        ),
        "ambient_developer_stream": _runtime_component_status(
            "ambient_developer_stream",
            "get_status",
            "status",
        ),
        "autonomic_reflection_loop": _runtime_component_status(
            "autonomic_reflection_loop",
            "get_status",
            "status",
        ),
    }
    resource_guard = desktop_resource_guard_enabled()
    expected = (
        resource_guard
        and not foreground_only_runtime()
        and not background_cognition_disabled_reason()
    )
    required = (
        "pneuma",
        "mhaf",
        "curiosity",
        "proactive_communication",
        "autonomous_initiative",
        "subjective_choice",
        "ambient_life_director",
        "research",
        "self_healing",
        "self_modification",
        "consciousness_stream",
        "autonomy_conductor",
        "overt_action",
        "deliberation",
        "wake_word",
        "screen_perception",
        "perceptual_pump",
        "cognitive_situation",
        "imagination_engine",
        "timescale_bridge",
        "ambient_developer_stream",
        "autonomic_reflection_loop",
    )
    blockers = [name for name in required if not components[name].get("running", False)]
    running_required = [
        name for name in required if components[name].get("running", False)
    ]
    disabled_reason = background_cognition_disabled_reason(
        allow_desktop_safe_boot=True,
    )
    loop_start_reason = background_loop_start_reason(
        allow_desktop_safe_boot=True,
    )
    orchestrator = ServiceContainer.get("orchestrator", default=None)
    activity_reason = background_activity_reason(
        orchestrator,
        min_idle_seconds=0.0,
        allow_no_user_anchor=True,
        allow_desktop_safe_boot=True,
    )
    background_enabled = bool(
        resource_guard and not foreground_only_runtime() and not disabled_reason
    )
    return {
        "profile": (
            "foreground_only"
            if foreground_only_runtime()
            else "protected_full_desktop"
            if desktop_safe_boot_enabled() and resource_guard
            else "full_desktop"
            if resource_guard
            else "recovery_safe_boot"
            if desktop_safe_boot_enabled()
            else "server_or_test"
        ),
        "full_runtime_expected": expected,
        "resource_guard_enabled": resource_guard,
        "ready": bool(expected and not blockers),
        "blockers": blockers,
        "background_cognition": {
            "enabled": background_enabled,
            "active": bool(background_enabled and not blockers),
            "loops_allowed": not bool(loop_start_reason),
            "loop_start_reason": loop_start_reason,
            "work_admission": "deferred" if activity_reason else "allowed",
            "work_defer_reason": activity_reason,
            "registered_required_count": len(required),
            "running_required_count": len(running_required),
            "offline_required": blockers,
        },
        "components": components,
    }


try:
    ORJSONResponse = fastapi_responses.ORJSONResponse
except _SYSTEM_RECOVERABLE_ERRORS:
    ORJSONResponse = JSONResponse

logger = logging.getLogger("Aura.Server.System")

router = APIRouter()

_DESKTOP_ACCESS_CACHE_TTL_S = _env_positive_float("AURA_DESKTOP_ACCESS_CACHE_TTL_S", 30.0)
_DESKTOP_ACCESS_DEGRADED_CACHE_TTL_S = _env_positive_float(
    "AURA_DESKTOP_ACCESS_DEGRADED_CACHE_TTL_S",
    15.0,
)
_DESKTOP_ACCESS_NATIVE_PROBE_TIMEOUT_S = _env_positive_float(
    "AURA_DESKTOP_ACCESS_NATIVE_PROBE_TIMEOUT_S",
    6.0,
)
_DESKTOP_ACCESS_DIRECT_PROBE_TIMEOUT_S = _env_positive_float(
    "AURA_DESKTOP_ACCESS_DIRECT_PROBE_TIMEOUT_S",
    2.0,
)
_DESKTOP_ACCESS_MENU_CLOCK_TIMEOUT_S = _env_positive_float(
    "AURA_DESKTOP_ACCESS_MENU_CLOCK_TIMEOUT_S",
    0.5,
)
_SSE_IDLE_HEARTBEAT_S = _env_positive_float("AURA_SSE_IDLE_HEARTBEAT_S", 15.0)
_SSE_QUEUE_BACKLOG_LIMIT = max(1, _safe_int(os.getenv("AURA_SSE_QUEUE_BACKLOG_LIMIT", ""), 100))
_HEALTH_PROBE_TIMEOUT_S = _env_positive_float("AURA_HEALTH_PROBE_TIMEOUT_S", 2.5)
_HEALTH_PROBE_DEGRADATION_THRESHOLD = max(
    2,
    _safe_int(os.getenv("AURA_HEALTH_PROBE_DEGRADATION_THRESHOLD", ""), 3),
)
_HEALTH_PROBE_STUCK_THRESHOLD_S = max(
    10.0,
    _env_positive_float(
        "AURA_HEALTH_PROBE_STUCK_THRESHOLD_S",
        max(30.0, _HEALTH_PROBE_TIMEOUT_S * 8.0),
    ),
)
_HEALTH_PROBE_LOCKS = {
    False: threading.Lock(),
    True: threading.Lock(),
}
# Backward-compatible canonical-runtime lock for direct contract tests.
_HEALTH_PROBE_LOCK = _HEALTH_PROBE_LOCKS[False]
_HEALTH_PROBE_STATE_LOCK = threading.Lock()
_HEALTH_PROBE_STATE: dict[str, Any] = {
    "generation": 0,
    "consecutive_failures": 0,
    "total_timeouts": 0,
    "total_contentions": 0,
    "total_terminal_failures": 0,
    "timeout_recorded_generation": 0,
    "stuck_recorded_generation": 0,
    "last_failure_reason": "",
    "last_failure_at_unix": 0.0,
    "escalated": False,
}
_HEALTH_PROBE_FUTURES: dict[bool, Future[tuple[dict[str, Any], int]]] = {}
_HEALTH_PROBE_GENERATIONS: dict[bool, int] = {}
_HEALTH_PROBE_STARTED_AT: dict[bool, float] = {}
_HEALTH_PROBE_EXECUTOR = ThreadPoolExecutor(
    max_workers=max(2, min(4, _safe_int(os.getenv("AURA_HEALTH_PROBE_WORKERS", ""), 2))),
    thread_name_prefix="AuraHealthProbe",
)
_HEALTH_CACHE_TTL_S = _env_positive_float("AURA_HEALTH_CACHE_TTL_S", 5.0)
_HEALTH_STALE_CACHE_TTL_S = max(
    _HEALTH_CACHE_TTL_S,
    _env_positive_float("AURA_HEALTH_STALE_CACHE_TTL_S", 30.0),
)
_HEALTH_MANIFEST_FALLBACK_TTL_S = _env_positive_float(
    "AURA_HEALTH_MANIFEST_FALLBACK_TTL_S",
    15.0,
)
_UI_SHELL_ERROR_BODY = Body(default=None)
_boot_health_cache_lock = threading.Lock()
_boot_health_cache: dict[bool, dict[str, Any]] = {
    False: {"captured_at": 0.0, "payload": None, "status_code": 503},
    True: {"captured_at": 0.0, "payload": None, "status_code": 503},
}
_desktop_access_cache: dict[str, Any] = {
    "captured_at": 0.0,
    "payload": None,
}
_DESKTOP_ACCESS_PROBE_TASKS: dict[
    asyncio.AbstractEventLoop,
    asyncio.Task[dict[str, Any]],
] = {}
_DESKTOP_ACCESS_PROBE_STATE_LOCK = threading.Lock()
_DESKTOP_ACCESS_PROBE_STATE: dict[str, Any] = {
    "total_timeouts": 0,
    "total_failures": 0,
    "active_streaks": {},
    "last_issue": "",
    "last_issue_at_unix": 0.0,
}
_desktop_access_request_state: dict[str, Any] = {}


def _health_probe_state_snapshot() -> dict[str, Any]:
    now = time.monotonic()
    with _HEALTH_PROBE_STATE_LOCK:
        active_entries = {
            bool(surface): {
                "generation": int(_HEALTH_PROBE_GENERATIONS.get(surface) or 0),
                "started_at": float(_HEALTH_PROBE_STARTED_AT.get(surface) or 0.0),
            }
            for surface, future in _HEALTH_PROBE_FUTURES.items()
            if not future.done()
        }
        active_since = min(
            (
                float(entry["started_at"])
                for entry in active_entries.values()
                if float(entry["started_at"]) > 0.0
            ),
            default=0.0,
        )
        active_generations = sorted(
            int(entry["generation"])
            for entry in active_entries.values()
            if int(entry["generation"]) > 0
        )
        return {
            "active": bool(active_entries),
            "active_count": len(active_entries),
            "active_age_s": round(max(0.0, now - active_since), 3)
            if active_since > 0.0
            else 0.0,
            "consecutive_failures": int(
                _HEALTH_PROBE_STATE.get("consecutive_failures") or 0
            ),
            "active_generation": active_generations[-1] if active_generations else 0,
            "active_generations": active_generations,
            "active_surfaces": sorted(
                "gui_proxy" if surface else "runtime"
                for surface in active_entries
            ),
            "generation": int(_HEALTH_PROBE_STATE.get("generation") or 0),
            "total_timeouts": int(_HEALTH_PROBE_STATE.get("total_timeouts") or 0),
            "total_contentions": int(
                _HEALTH_PROBE_STATE.get("total_contentions") or 0
            ),
            "total_terminal_failures": int(
                _HEALTH_PROBE_STATE.get("total_terminal_failures") or 0
            ),
            "last_failure_reason": str(
                _HEALTH_PROBE_STATE.get("last_failure_reason") or ""
            ),
            "last_failure_at_unix": float(
                _HEALTH_PROBE_STATE.get("last_failure_at_unix") or 0.0
            ),
            "escalated": bool(_HEALTH_PROBE_STATE.get("escalated", False)),
            "degradation_threshold": _HEALTH_PROBE_DEGRADATION_THRESHOLD,
            "stuck_threshold_s": _HEALTH_PROBE_STUCK_THRESHOLD_S,
        }


def _attach_health_probe_state(
    result: tuple[dict[str, Any], int],
) -> tuple[dict[str, Any], int]:
    payload, status_code = result
    enriched = dict(payload)
    enriched["health_probe_runtime"] = _health_probe_state_snapshot()
    return enriched, status_code


def _reset_health_probe_state_for_test() -> None:
    with _HEALTH_PROBE_STATE_LOCK:
        for surface, future in list(_HEALTH_PROBE_FUTURES.items()):
            if future.done():
                _HEALTH_PROBE_FUTURES.pop(surface, None)
                _HEALTH_PROBE_GENERATIONS.pop(surface, None)
                _HEALTH_PROBE_STARTED_AT.pop(surface, None)
        _HEALTH_PROBE_STATE.update(
            generation=0,
            consecutive_failures=0,
            total_timeouts=0,
            total_contentions=0,
            total_terminal_failures=0,
            timeout_recorded_generation=0,
            stuck_recorded_generation=0,
            last_failure_reason="",
            last_failure_at_unix=0.0,
            escalated=False,
        )


def _reset_boot_health_cache_for_test() -> None:
    with _boot_health_cache_lock:
        for entry in _boot_health_cache.values():
            entry.update(
                captured_at=0.0,
                payload=None,
                status_code=503,
            )


def _desktop_access_empty_payload() -> dict[str, Any]:
    return {
        "screen_recording": {"granted": False, "status": "unknown", "guidance": ""},
        "accessibility": {"granted": False, "status": "unknown", "guidance": ""},
        "automation": {"granted": False, "status": "unknown", "guidance": ""},
        "direct_screen_recording": {"granted": False, "status": "unknown", "guidance": ""},
        "direct_accessibility": {"granted": False, "status": "unknown", "guidance": ""},
        "direct_automation": {"granted": False, "status": "unknown", "guidance": ""},
        "screen_capture_ready": False,
        "desktop_control_ready": False,
        "screen_text_ready": False,
        "direct_screen_capture_ready": False,
        "direct_desktop_control_ready": False,
        "direct_screen_text_ready": False,
        "menu_clock_ready": False,
        "menu_clock_text": "",
        "menu_clock_error": "",
        "frontmost_app": "",
        "pyautogui_ready": False,
        "pyautogui_error": "",
        "permission_confidence": "unknown",
        "permission_assumptions": [],
        "process_identity": {},
        "effective_app_identity": {},
        "desktop_access_diagnosis": [],
        "tcc_repair_plan": {},
        "tcc_request_state": dict(_desktop_access_request_state),
        "native_bridge_probe": {},
        "overall_status": "pending",
        "blocking_permissions": [],
        "reported_blocking_permissions": [],
        "direct_blocking_permissions": [],
        "unverified_permissions": [],
        "reported_probe_unavailable_permissions": [],
        "direct_probe_unavailable_permissions": [],
        "direct_probe_available": False,
        "cache_age_s": 0.0,
        "cache_stale": False,
        "probe_mode": "empty",
        "probe_runtime": _desktop_access_probe_state_snapshot(),
    }


def _desktop_access_cache_ttl(payload: Any) -> float:
    if not isinstance(payload, dict):
        return 0.0
    if (
        payload.get("overall_status") == "ready"
        and payload.get("permission_confidence") == "direct"
        and not payload.get("blocking_permissions")
    ):
        return max(1.0, _DESKTOP_ACCESS_CACHE_TTL_S)
    return max(0.25, min(_DESKTOP_ACCESS_CACHE_TTL_S, _DESKTOP_ACCESS_DEGRADED_CACHE_TTL_S))


def _desktop_access_cached_copy(
    payload: dict[str, Any],
    *,
    captured_at: float,
    stale: bool = False,
    probe_mode: str = "cached",
) -> dict[str, Any]:
    copied = dict(payload)
    age = max(0.0, time.monotonic() - float(captured_at or 0.0))
    copied["cache_age_s"] = round(age, 3)
    copied["cache_stale"] = bool(stale)
    copied["probe_mode"] = probe_mode
    copied["cache_ttl_s"] = _desktop_access_cache_ttl(payload)
    return copied


def _desktop_access_probe_state_snapshot() -> dict[str, Any]:
    with _DESKTOP_ACCESS_PROBE_STATE_LOCK:
        return {
            "total_timeouts": int(
                _DESKTOP_ACCESS_PROBE_STATE.get("total_timeouts", 0) or 0
            ),
            "total_failures": int(
                _DESKTOP_ACCESS_PROBE_STATE.get("total_failures", 0) or 0
            ),
            "active_streaks": dict(
                _DESKTOP_ACCESS_PROBE_STATE.get("active_streaks", {}) or {}
            ),
            "last_issue": str(
                _DESKTOP_ACCESS_PROBE_STATE.get("last_issue", "") or ""
            ),
            "last_issue_at_unix": float(
                _DESKTOP_ACCESS_PROBE_STATE.get("last_issue_at_unix", 0.0) or 0.0
            ),
        }


def _record_desktop_access_probe_issue(
    probe: str,
    target: str,
    exc: BaseException,
) -> tuple[str, int]:
    issue = "timeout" if isinstance(exc, TimeoutError) else "probe_error"
    key = f"{probe}:{target}"
    with _DESKTOP_ACCESS_PROBE_STATE_LOCK:
        streaks = _DESKTOP_ACCESS_PROBE_STATE.setdefault("active_streaks", {})
        streak = int(streaks.get(key, 0) or 0) + 1
        streaks[key] = streak
        counter = "total_timeouts" if issue == "timeout" else "total_failures"
        _DESKTOP_ACCESS_PROBE_STATE[counter] = int(
            _DESKTOP_ACCESS_PROBE_STATE.get(counter, 0) or 0
        ) + 1
        detail = str(exc)[:240] or type(exc).__name__
        _DESKTOP_ACCESS_PROBE_STATE["last_issue"] = f"{key}:{detail}"
        _DESKTOP_ACCESS_PROBE_STATE["last_issue_at_unix"] = time.time()
    if streak == 1 or streak % 15 == 0:
        logger.warning(
            "Desktop access diagnostic %s for %s (%s, streak=%d)",
            issue,
            target,
            detail,
            streak,
        )
    else:
        logger.debug(
            "Desktop access diagnostic %s for %s suppressed (streak=%d)",
            issue,
            target,
            streak,
        )
    return issue, streak


def _mark_desktop_access_probe_success(probe: str, target: str) -> None:
    key = f"{probe}:{target}"
    with _DESKTOP_ACCESS_PROBE_STATE_LOCK:
        streaks = _DESKTOP_ACCESS_PROBE_STATE.setdefault("active_streaks", {})
        recovered = int(streaks.pop(key, 0) or 0)
    if recovered:
        logger.info(
            "Desktop access diagnostic recovered for %s after %d failed samples",
            target,
            recovered,
        )


def _desktop_access_probe_unavailable(
    guard: Any,
    ptype: Any,
    *,
    probe: str,
    exc: BaseException,
) -> dict[str, Any]:
    target = str(getattr(ptype, "name", ptype) or "unknown").lower()
    issue, streak = _record_desktop_access_probe_issue(probe, target, exc)
    guidance_getter = getattr(guard, "get_guidance", None)
    guidance = guidance_getter(ptype) if callable(guidance_getter) else ""
    return {
        "granted": False,
        "status": issue,
        "guidance": guidance,
        "detail": str(exc)[:240] or type(exc).__name__,
        "direct_probe": probe == "direct",
        "probe_unavailable": True,
        "retryable": True,
        "failure_streak": streak,
    }


# ── Collector Helpers ─────────────────────────────────────────

def _mark_runtime_service_progress(source: str) -> None:
    """Best-effort proof that the live desktop/API lane is actively serving."""
    try:
        from core.resilience.stall_watchdog import mark_runtime_service_progress

        mark_runtime_service_progress(source)
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        logger.debug("Runtime service progress marker skipped for %s: %s", source, exc)

def _fallback_conversation_lane_status(reason: str) -> dict[str, Any]:
    desired_endpoint: str | None = None
    background_endpoint: str | None = None
    try:
        from core.brain.llm.model_registry import BRAINSTEM_ENDPOINT, PRIMARY_ENDPOINT

        desired_endpoint = PRIMARY_ENDPOINT
        background_endpoint = BRAINSTEM_ENDPOINT
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Conversation lane fallback endpoint lookup failed: %s", exc)

    return {
        "desired_model": "Cortex (32B)",
        "desired_endpoint": desired_endpoint,
        "foreground_endpoint": desired_endpoint,
        "background_endpoint": background_endpoint,
        "foreground_tier": "local",
        "background_tier": "local_fast",
        "state": "degraded",
        "last_failure_reason": str(reason or "conversation_lane_status_unavailable")[:240],
        "conversation_ready": False,
        "last_transition_at": time.time(),
        "warmup_attempted": False,
        "warmup_in_flight": False,
        "expected_model": "Cortex (32B)",
        "detected_models": [],
        "runtime_identity_ok": False,
        "kernel_tick_age_s": None,
    }


def _collect_recent_degraded_events(limit: int = 12) -> list[dict[str, Any]]:
    try:
        from core.health.degraded_events import get_recent_degraded_events

        return get_recent_degraded_events(limit=limit)
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Recent degraded event collection failed: %s", exc)
        return []


def _collect_conversation_lane_status() -> dict[str, Any]:
    return _collect_conversation_lane_status_resilient()


def _collect_conversation_lane_status_resilient() -> dict[str, Any]:
    """Import and delegate to the canonical implementation in chat routes."""
    overridden = globals().get("_collect_conversation_lane_status")
    if callable(overridden) and overridden is not _NATIVE_CONVERSATION_LANE_STATUS_WRAPPER:
        try:
            lane = overridden()
            if isinstance(lane, dict):
                return lane
            raise TypeError(f"conversation lane collector returned {type(lane).__name__}")
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation("system", exc)
            logger.debug("Overridden conversation lane status unavailable: %s", exc)
            return _fallback_conversation_lane_status(str(exc))

    try:
        from interface.routes.chat import _collect_conversation_lane_status as _impl

        lane = _impl()
        if isinstance(lane, dict):
            return lane
        raise TypeError(f"conversation lane collector returned {type(lane).__name__}")
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Conversation lane status unavailable: %s", exc)
        return _fallback_conversation_lane_status(str(exc))


def _conversation_lane_is_standby(lane: dict[str, Any] | None) -> bool:
    return _conversation_lane_is_standby_resilient(lane)


def _conversation_lane_is_standby_resilient(lane: dict[str, Any] | None) -> bool:
    overridden = globals().get("_conversation_lane_is_standby")
    if callable(overridden) and overridden is not _NATIVE_CONVERSATION_LANE_STANDBY_WRAPPER:
        try:
            return overridden(lane)
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation("system", exc)
            logger.debug("Overridden conversation lane standby helper unavailable: %s", exc)

    try:
        from interface.routes.chat import _conversation_lane_is_standby as _impl

        return _impl(lane)
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Conversation lane standby helper unavailable: %s", exc)
        lane = dict(lane or {})
        state = str(lane.get("state", "") or "").strip().lower()
        return (
            not bool(lane.get("conversation_ready", False))
            and state in {"cold", "closed", ""}
            and not bool(lane.get("warmup_attempted", False))
            and not bool(lane.get("warmup_in_flight", False))
        )


def _conversation_lane_user_message(lane: dict[str, Any], **kwargs) -> str:
    return _conversation_lane_user_message_resilient(lane, **kwargs)


def _conversation_lane_user_message_resilient(lane: dict[str, Any], **kwargs) -> str:
    overridden = globals().get("_conversation_lane_user_message")
    if callable(overridden) and overridden is not _NATIVE_CONVERSATION_LANE_MESSAGE_WRAPPER:
        try:
            return overridden(lane, **kwargs)
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation("system", exc)
            logger.debug("Overridden conversation lane message helper unavailable: %s", exc)

    try:
        from interface.routes.chat import _conversation_lane_user_message as _impl

        return _impl(lane, **kwargs)
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Conversation lane message helper unavailable: %s", exc)
        reason = str((lane or {}).get("last_failure_reason") or exc or "status unavailable")
        return f"The conversation lane is degraded right now: {reason[:180]}"


_NATIVE_CONVERSATION_LANE_STATUS_WRAPPER = _collect_conversation_lane_status
_NATIVE_CONVERSATION_LANE_STANDBY_WRAPPER = _conversation_lane_is_standby
_NATIVE_CONVERSATION_LANE_MESSAGE_WRAPPER = _conversation_lane_user_message


def _attach_launch_provenance_contract(
    payload: dict[str, Any],
    status_code: int,
    *,
    provenance: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Prevent an orphaned, stale, or incorrectly signed app runtime from looking ready."""

    if provenance is None:
        try:
            from core.runtime.launch_provenance import collect_runtime_launch_provenance

            provenance = collect_runtime_launch_provenance(config.paths.project_root)
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            launched_from_app = str(os.environ.get("AURA_LAUNCHED_FROM_APP", "")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            provenance = {
                "schema": "aura.launch_provenance.v1",
                "required": launched_from_app,
                "verified": False,
                "source_verified": False,
                "issues": [f"provenance_collection_failed:{type(exc).__name__}"],
            }
            logger.warning("Launch provenance collection failed: %s", exc)

    result = dict(payload)
    result["launch_provenance"] = provenance
    checks = dict(result.get("checks") or {})
    required = bool(provenance.get("required"))
    verified = bool(provenance.get("verified"))
    checks["launch_provenance"] = verified if required else True
    result["checks"] = checks
    if not required or verified:
        return result, status_code

    blockers = [str(item) for item in result.get("blockers", []) if str(item)]
    if "launch_provenance" not in blockers:
        blockers.append("launch_provenance")
    result.update(
        {
            "ready": False,
            "launcher_ready": False,
            "system_ready": False,
            "status": "degraded",
            "status_message": (
                "Aura's runtime is alive, but its signed app/source provenance is not verified."
            ),
            "boot_phase": "launch_provenance_failed",
            "blockers": blockers,
        }
    )
    return result, 503


def _fallback_launch_provenance(manifest_snapshot: Any = None) -> dict[str, Any]:
    """Return non-blocking conservative evidence for event-loop fallback paths."""

    snapshot = dict(manifest_snapshot) if isinstance(manifest_snapshot, dict) else {}
    launched_from_app = str(os.environ.get("AURA_LAUNCHED_FROM_APP", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    required = bool(snapshot.get("required", launched_from_app))
    if not required:
        return {
            **snapshot,
            "schema": str(snapshot.get("schema") or "aura.launch_provenance.v1"),
            "required": False,
            "verified": False,
            "launch_mode": str(snapshot.get("launch_mode") or "direct"),
            "issues": list(snapshot.get("issues") or []),
        }
    issues = [str(item) for item in snapshot.get("issues", []) if str(item)]
    if "launch_provenance_live_refresh_unavailable" not in issues:
        issues.append("launch_provenance_live_refresh_unavailable")
    return {
        **snapshot,
        "schema": str(snapshot.get("schema") or "aura.launch_provenance.v1"),
        "required": True,
        "verified": False,
        "issues": issues,
    }


def _build_boot_health_payload_sync(*, is_gui_proxy: bool) -> tuple[dict[str, Any], int]:
    """Build boot health with a single-flight guard for HTTP readiness probes."""

    stopping = _stopping_boot_health_payload()
    if stopping is not None:
        return stopping

    probe_lock = _HEALTH_PROBE_LOCKS[bool(is_gui_proxy)]
    acquired = probe_lock.acquire(False)
    if not acquired:
        raise TimeoutError("health_probe_already_running")
    try:
        orch = ServiceContainer.get("orchestrator", default=None)
        rt = _get_runtime_state_safe()
        conversation_lane = _collect_conversation_lane_status_resilient()
        try:
            payload, status_code = build_boot_health_snapshot(
                orch,
                rt,
                is_gui_proxy=is_gui_proxy,
                conversation_lane=conversation_lane,
            )
            payload, status_code = _attach_launch_provenance_contract(payload, status_code)
            _store_boot_health_cache(
                payload,
                status_code,
                is_gui_proxy=is_gui_proxy,
            )
            return payload, status_code
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation("system", exc)
            logger.error("Boot health snapshot failed: %s", exc, exc_info=True)
            payload = {
                "ready": False,
                "status": "degraded",
                "issues": [str(exc)],
                "conversation_lane": conversation_lane,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }
            payload, status_code = _attach_launch_provenance_contract(payload, 503)
            _store_boot_health_cache(
                payload,
                status_code,
                is_gui_proxy=is_gui_proxy,
            )
            return payload, status_code
    finally:
        probe_lock.release()


def _store_boot_health_cache(
    payload: dict[str, Any],
    status_code: int,
    *,
    is_gui_proxy: bool = False,
) -> None:
    with _boot_health_cache_lock:
        entry = _boot_health_cache[bool(is_gui_proxy)]
        entry["captured_at"] = time.monotonic()
        entry["payload"] = dict(payload)
        entry["status_code"] = int(status_code)


def _fresh_boot_health_payload(
    *,
    is_gui_proxy: bool,
) -> tuple[dict[str, Any], int] | None:
    now = time.monotonic()
    with _boot_health_cache_lock:
        entry = _boot_health_cache[bool(is_gui_proxy)]
        captured_at = float(entry.get("captured_at") or 0.0)
        payload = entry.get("payload")
        status_code = int(entry.get("status_code") or 503)
    age_s = max(0.0, now - captured_at) if captured_at > 0.0 else float("inf")
    if (
        not isinstance(payload, dict)
        or "ready" not in payload
        or age_s > _HEALTH_CACHE_TTL_S
    ):
        return None
    cached = dict(payload)
    cached["cache_status"] = "fresh"
    cached["cache_reason"] = "health_cache_ttl"
    cached["cache_age_s"] = round(age_s, 3)
    return cached, status_code


def _complete_health_probe_future(
    future: Future[tuple[dict[str, Any], int]],
    generation: int,
    *,
    is_gui_proxy: bool,
) -> None:
    failure: Exception | None = None
    result: tuple[dict[str, Any], int] | None = None
    try:
        candidate = future.result()
        if (
            not isinstance(candidate, tuple)
            or len(candidate) != 2
            or not isinstance(candidate[0], dict)
        ):
            raise TypeError("health probe returned an invalid payload contract")
        result = candidate
    except Exception as exc:  # noqa: BLE001 - final Future completion boundary
        failure = exc

    should_escalate = False
    with _HEALTH_PROBE_STATE_LOCK:
        surface = bool(is_gui_proxy)
        if _HEALTH_PROBE_FUTURES.get(surface) is not future:
            return
        _HEALTH_PROBE_FUTURES.pop(surface, None)
        _HEALTH_PROBE_GENERATIONS.pop(surface, None)
        _HEALTH_PROBE_STARTED_AT.pop(surface, None)
        if result is not None:
            _store_boot_health_cache(
                result[0],
                result[1],
                is_gui_proxy=surface,
            )
        if failure is None:
            _HEALTH_PROBE_STATE["consecutive_failures"] = 0
            _HEALTH_PROBE_STATE["last_failure_reason"] = ""
            _HEALTH_PROBE_STATE["escalated"] = False
        else:
            reason = f"health_probe_exception:{type(failure).__name__}"
            _HEALTH_PROBE_STATE["total_terminal_failures"] = int(
                _HEALTH_PROBE_STATE.get("total_terminal_failures") or 0
            ) + 1
            _HEALTH_PROBE_STATE["consecutive_failures"] = int(
                _HEALTH_PROBE_STATE.get("consecutive_failures") or 0
            ) + 1
            _HEALTH_PROBE_STATE["last_failure_reason"] = reason
            _HEALTH_PROBE_STATE["last_failure_at_unix"] = time.time()
            should_escalate = bool(
                int(_HEALTH_PROBE_STATE.get("consecutive_failures") or 0)
                >= _HEALTH_PROBE_DEGRADATION_THRESHOLD
                and not bool(_HEALTH_PROBE_STATE.get("escalated", False))
            )
            if should_escalate:
                _HEALTH_PROBE_STATE["escalated"] = True

    if failure is None:
        return
    if should_escalate:
        record_degradation(
            "system",
            failure,
            severity="warning",
            action="escalated distinct terminal health-probe failures",
            extra=_health_probe_state_snapshot(),
            enforce_failure_policy=False,
        )
    else:
        logger.warning(
            "Boot-health probe generation %d failed: %s",
            generation,
            failure,
        )


def _start_or_join_health_probe(
    *,
    is_gui_proxy: bool,
) -> tuple[Future[tuple[dict[str, Any], int]], int, bool]:
    with _HEALTH_PROBE_STATE_LOCK:
        surface = bool(is_gui_proxy)
        existing = _HEALTH_PROBE_FUTURES.get(surface)
        if existing is not None:
            generation = int(
                _HEALTH_PROBE_GENERATIONS.get(surface)
                or _HEALTH_PROBE_STATE.get("generation")
                or 0
            )
            if not existing.done():
                _HEALTH_PROBE_STATE["total_contentions"] = int(
                    _HEALTH_PROBE_STATE.get("total_contentions") or 0
                ) + 1
                return existing, generation, False
            return existing, generation, True

        generation = int(_HEALTH_PROBE_STATE.get("generation") or 0) + 1
        _HEALTH_PROBE_STATE["generation"] = generation
        _HEALTH_PROBE_GENERATIONS[surface] = generation
        _HEALTH_PROBE_STARTED_AT[surface] = time.monotonic()
        future = _HEALTH_PROBE_EXECUTOR.submit(
            _build_boot_health_payload_sync,
            is_gui_proxy=surface,
        )
        _HEALTH_PROBE_FUTURES[surface] = future

    future.add_done_callback(
        lambda completed, probe_generation=generation, probe_surface=surface: _complete_health_probe_future(
            completed,
            probe_generation,
            is_gui_proxy=probe_surface,
        )
    )
    return future, generation, True


def _record_health_probe_wait_timeout(generation: int) -> tuple[dict[str, Any], bool]:
    recorded = False
    with _HEALTH_PROBE_STATE_LOCK:
        if int(_HEALTH_PROBE_STATE.get("timeout_recorded_generation") or 0) != generation:
            recorded = True
            _HEALTH_PROBE_STATE["timeout_recorded_generation"] = generation
            _HEALTH_PROBE_STATE["total_timeouts"] = int(
                _HEALTH_PROBE_STATE.get("total_timeouts") or 0
            ) + 1
    return _health_probe_state_snapshot(), recorded


def _record_stuck_health_probe_once(
    generation: int,
    *,
    is_gui_proxy: bool,
) -> tuple[dict[str, Any], bool, bool]:
    now_monotonic = time.monotonic()
    recorded = False
    should_escalate = False
    with _HEALTH_PROBE_STATE_LOCK:
        surface = bool(is_gui_proxy)
        active_since = float(_HEALTH_PROBE_STARTED_AT.get(surface) or 0.0)
        active_generation = int(_HEALTH_PROBE_GENERATIONS.get(surface) or 0)
        active_age_s = (
            max(0.0, now_monotonic - active_since) if active_since > 0.0 else 0.0
        )
        if (
            active_generation == generation
            and active_age_s >= _HEALTH_PROBE_STUCK_THRESHOLD_S
            and int(_HEALTH_PROBE_STATE.get("stuck_recorded_generation") or 0)
            != generation
        ):
            recorded = True
            _HEALTH_PROBE_STATE["stuck_recorded_generation"] = generation
            _HEALTH_PROBE_STATE["total_terminal_failures"] = int(
                _HEALTH_PROBE_STATE.get("total_terminal_failures") or 0
            ) + 1
            _HEALTH_PROBE_STATE["consecutive_failures"] = int(
                _HEALTH_PROBE_STATE.get("consecutive_failures") or 0
            ) + 1
            _HEALTH_PROBE_STATE["last_failure_reason"] = "health_probe_stuck"
            _HEALTH_PROBE_STATE["last_failure_at_unix"] = time.time()
            should_escalate = bool(
                int(_HEALTH_PROBE_STATE.get("consecutive_failures") or 0)
                >= _HEALTH_PROBE_DEGRADATION_THRESHOLD
                and not bool(_HEALTH_PROBE_STATE.get("escalated", False))
            )
            if should_escalate:
                _HEALTH_PROBE_STATE["escalated"] = True
    return _health_probe_state_snapshot(), recorded, should_escalate


def _cached_boot_health_payload(
    reason: str,
    *,
    is_gui_proxy: bool,
) -> tuple[dict[str, Any], int]:
    stopping = _stopping_boot_health_payload()
    if stopping is not None:
        return stopping
    now = time.monotonic()
    with _boot_health_cache_lock:
        entry = _boot_health_cache[bool(is_gui_proxy)]
        captured_at = float(entry.get("captured_at") or 0.0)
        payload = entry.get("payload")
        status_code = int(entry.get("status_code") or 503)

    cache_age_s = max(0.0, now - captured_at) if captured_at > 0.0 else float("inf")
    if (
        isinstance(payload, dict)
        and "ready" in payload
        and cache_age_s <= _HEALTH_CACHE_TTL_S
    ):
        cached = dict(payload)
        cached["cache_status"] = "fresh"
        cached["cache_reason"] = reason
        cached["cache_age_s"] = round(cache_age_s, 3)
        return cached, status_code
    if (
        reason in {"health_probe_in_flight", "health_probe_timeout"}
        and isinstance(payload, dict)
        and "ready" in payload
        and cache_age_s <= _HEALTH_STALE_CACHE_TTL_S
    ):
        cached = dict(payload)
        cached["cache_status"] = "stale_while_revalidate"
        cached["cache_reason"] = reason
        cached["cache_age_s"] = round(cache_age_s, 3)
        cached["cache_stale"] = True
        return cached, status_code

    manifest_payload = _runtime_manifest_boot_health_payload(reason)
    if manifest_payload is not None:
        manifest_body, manifest_status = manifest_payload
        fallback = _fallback_launch_provenance(manifest_body.get("launch_provenance"))
        return _attach_launch_provenance_contract(
            manifest_body,
            manifest_status,
            provenance=fallback,
        )

    return _attach_launch_provenance_contract(
        {
            "ready": False,
            "status": "unhealthy",
            "issues": [reason],
            "required_probes": {"all_passed": False},
            "blockers": [reason],
            "boot_phase": reason,
            "conversation_ready": False,
            "cache_status": "miss",
            "cache_reason": reason,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        },
        503,
        provenance=_fallback_launch_provenance(),
    )


def _runtime_manifest_boot_health_payload(reason: str) -> tuple[dict[str, Any], int] | None:
    try:
        manifest_path = config.paths.project_root / "artifacts" / "current" / "runtime_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        readiness = manifest.get("readiness_snapshot")
        if not isinstance(readiness, dict):
            return None
        generated_at = _safe_float(manifest.get("generated_at_unix"), 0.0)
        manifest_age_s = max(0.0, time.time() - generated_at) if generated_at > 0.0 else float("inf")
        if manifest_age_s > _HEALTH_MANIFEST_FALLBACK_TTL_S:
            return (
                {
                    "ready": False,
                    "status": "unhealthy",
                    "system_ready": False,
                    "launcher_ready": False,
                    "conversation_ready": False,
                    "boot_phase": "manifest_stale",
                    "required_probes": {"all_passed": False},
                    "blockers": ["health_manifest_stale", reason],
                    "cache_status": "manifest_stale",
                    "cache_reason": reason,
                    "manifest_age_s": round(manifest_age_s, 3),
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                },
                503,
            )
        ready = bool(readiness.get("ready") is True)
        blockers = [str(item) for item in readiness.get("required_probe_blockers", []) if str(item)]
        if not ready and not blockers:
            blockers = [reason]
        status_code = 200 if ready and not blockers else 503
        required_probes: dict[str, Any] = {"all_passed": ready}
        for group_name, components in REQUIRED_HEALTH_PROBE_GROUPS.items():
            required_probes[group_name] = {
                "ok": ready,
                "components": {component: ready for component in components},
            }
        return (
            {
                "ready": ready,
                "status": "ready" if status_code == 200 else "unhealthy",
                "system_ready": ready,
                "launcher_ready": ready,
                "conversation_ready": ready,
                "boot_phase": "manifest_ready" if ready else "manifest_unhealthy",
                "required_probes": required_probes,
                "blockers": blockers,
                "cache_status": "manifest",
                "cache_reason": reason,
                "manifest_generated_at_unix": manifest.get("generated_at_unix"),
                "manifest_age_s": round(manifest_age_s, 3),
                "launch_provenance": manifest.get("launch_provenance"),
                "timestamp": datetime.now(tz=UTC).isoformat(),
            },
            status_code,
        )
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Runtime manifest health fallback failed: %s", exc)
        return None


async def _build_boot_health_payload_bounded(*, is_gui_proxy: bool) -> tuple[dict[str, Any], int]:
    """Return a boot-health snapshot without allowing probes to hang the HTTP loop."""

    fresh = _fresh_boot_health_payload(is_gui_proxy=is_gui_proxy)
    if fresh is not None:
        return _attach_health_probe_state(fresh)

    future, generation, created = _start_or_join_health_probe(
        is_gui_proxy=is_gui_proxy,
    )
    if not created:
        probe_state, newly_stuck, should_escalate = _record_stuck_health_probe_once(
            generation,
            is_gui_proxy=is_gui_proxy,
        )
        if should_escalate:
            record_degradation(
                "system",
                TimeoutError(
                    "distinct health probes exceeded the explicit stuck threshold"
                ),
                severity="warning",
                action="escalated distinct stuck health-probe generations",
                extra=probe_state,
                enforce_failure_policy=False,
            )
        elif newly_stuck:
            logger.warning(
                "Boot-health probe generation %d exceeded the %.1fs stuck threshold; "
                "recorded once while callers continue using bounded fallback evidence.",
                generation,
                _HEALTH_PROBE_STUCK_THRESHOLD_S,
            )
        return _attach_health_probe_state(
            _cached_boot_health_payload(
                "health_probe_in_flight",
                is_gui_proxy=is_gui_proxy,
            )
        )

    try:
        result = await asyncio.wait_for(
            asyncio.shield(asyncio.wrap_future(future)),
            timeout=_HEALTH_PROBE_TIMEOUT_S,
        )
        return _attach_health_probe_state(result)
    except TimeoutError:
        probe_state, timeout_recorded = _record_health_probe_wait_timeout(generation)
        if timeout_recorded:
            logger.warning(
                "Boot-health probe generation %d exceeded the %.1fs HTTP wait budget; "
                "the singleflight remains active and later polls will reuse its result.",
                generation,
                _HEALTH_PROBE_TIMEOUT_S,
            )
        return _attach_health_probe_state(
            _cached_boot_health_payload(
                "health_probe_timeout",
                is_gui_proxy=is_gui_proxy,
            )
        )
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        logger.warning(
            "Boot-health probe generation %d ended before returning a payload: %s",
            generation,
            exc,
        )
        return _attach_health_probe_state(
            _cached_boot_health_payload(
                "health_probe_failed",
                is_gui_proxy=is_gui_proxy,
            )
        )


def _get_runtime_state_safe() -> dict[str, Any]:
    try:
        rt = get_runtime_state()
        if isinstance(rt, dict):
            return rt
        raise TypeError(f"runtime state returned {type(rt).__name__}")
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Runtime state snapshot failed: %s", exc)
        return {
            "state": {},
            "status": "degraded",
            "error": str(exc)[:240],
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }


def _collect_stability_details() -> dict[str, Any]:
    details: dict[str, Any] = {
        "status": "unknown",
        "healthy": False,
        "active_issues": [],
    }
    try:
        guardian = ServiceContainer.get("stability_guardian", default=None)
        if guardian is None:
            details["status"] = "unavailable"
            details["active_issues"].append(
                {
                    "name": "stability_guardian",
                    "message": "StabilityGuardian is not registered.",
                    "severity": "warning",
                    "action_taken": "withhold healthy status until guardian is online",
                }
            )
        elif hasattr(guardian, "get_latest_report"):
            report = guardian.get_latest_report() or {}
            checks = report.get("checks", []) if isinstance(report, dict) else []
            active_issues = []
            for check in checks:
                if not bool(check.get("healthy", False)):
                    active_issues.append(
                        {
                            "name": check.get("name", "unknown"),
                            "message": check.get("message", ""),
                            "severity": check.get("severity", "warning"),
                            "action_taken": check.get("action_taken"),
                        }
                    )
            if report:
                details["healthy"] = bool(report.get("overall_healthy", False))
                details["status"] = "healthy" if details["healthy"] else "degraded"
                details["active_issues"] = active_issues
                details["memory_pct"] = report.get("memory_pct")
                details["cpu_pct"] = report.get("cpu_pct")
            elif hasattr(guardian, "get_health_summary"):
                summary = guardian.get_health_summary()
                if isinstance(summary, dict):
                    details["healthy"] = bool(summary.get("healthy", False))
                    details["status"] = str(summary.get("status") or "unknown")
                    details["active_issues"] = list(summary.get("active_issues") or [])
                    if details["status"] == "initializing":
                        details["healthy"] = False
            else:
                details["status"] = "no_report"
                details["active_issues"] = [
                    {
                        "name": "stability_report",
                        "message": "StabilityGuardian has not produced a health report.",
                        "severity": "warning",
                        "action_taken": "withhold healthy status until probes run",
                    }
                ]
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Stability detail collection failed: %s", exc)

    try:
        lane = _collect_conversation_lane_status_resilient()
        if isinstance(lane, dict) and not bool(lane.get("conversation_ready", False)):
            details["healthy"] = False
            if details.get("status") == "unknown":
                details["status"] = "degraded"
            details.setdefault("active_issues", []).append(
                {
                    "name": "conversation_lane",
                    "message": _conversation_lane_user_message_resilient(lane),
                    "severity": "warning" if str(lane.get("state", "") or "").lower() != "failed" else "error",
                    "action_taken": None,
                }
            )
        if isinstance(lane, dict) and not bool(lane.get("runtime_identity_ok", True)):
            details["healthy"] = False
            if details.get("status") == "unknown":
                details["status"] = "degraded"
            details.setdefault("active_issues", []).append(
                {
                    "name": "conversation_lane_model_mismatch",
                    "message": (
                        f"Expected {lane.get('expected_model') or 'the configured Cortex model'}, "
                        f"but detected {', '.join(lane.get('detected_models') or []) or 'an unexpected runtime model'} "
                        "on the reserved conversation lane."
                    ),
                    "severity": "error",
                    "action_taken": None,
                }
            )
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Conversation lane stability detail merge failed: %s", exc)
    if details.get("status") == "unknown":
        details["status"] = "healthy" if bool(details.get("healthy", False)) else "degraded"
    return details


def _normalize_percentish(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if abs(number) <= 1.0:
        number *= 100.0
    return max(0.0, min(100.0, number))


def _json_safe(value: Any) -> Any:
    """Recursively coerce runtime payloads into JSON-safe primitives."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Unable to coerce scalar-like value with item(): %s", exc)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _json_safe(value.tolist())
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Unable to coerce array-like value with tolist(): %s", exc)
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(coerced) or math.isinf(coerced):
        return None
    return coerced


def _collect_liquid_state_payload(
    ls_data: dict[str, Any],
    *,
    runtime_state: dict[str, Any],
    homeostasis_data: dict[str, Any],
) -> dict[str, Any]:
    runtime_affect = runtime_state.get("affect", {}) if isinstance(runtime_state.get("affect"), dict) else {}
    payload: dict[str, Any] = {}

    def _pick_metric(key: str, *, runtime_fallback: Any = None) -> float | None:
        primary = _normalize_percentish(ls_data.get(key))
        fallback = _normalize_percentish(runtime_fallback if runtime_fallback is not None else runtime_affect.get(key))
        if primary is None:
            return fallback
        if primary == 0.0 and fallback not in (None, 0.0):
            return fallback
        return primary

    derived_frustration = runtime_affect.get("frustration")
    if derived_frustration is None:
        try:
            valence = float(runtime_affect.get("valence"))
            if valence < 0.0:
                derived_frustration = min(100.0, abs(valence) * 100.0)
        except (TypeError, ValueError):
            derived_frustration = None

    for key in ("energy", "curiosity", "frustration", "focus", "confidence"):
        runtime_fallback = None
        if key == "frustration":
            runtime_fallback = derived_frustration
        elif key == "curiosity":
            runtime_fallback = runtime_affect.get("curiosity", homeostasis_data.get("curiosity"))
        elif key == "confidence":
            runtime_fallback = runtime_affect.get(
                "confidence",
                _homeostasis_vitality_value(homeostasis_data),
            )
        normalized = _pick_metric(key, runtime_fallback=runtime_fallback)
        if normalized is not None:
            payload[key] = round(normalized, 1)

    if "confidence" not in payload:
        normalized = _normalize_percentish(_homeostasis_vitality_value(homeostasis_data))
        if normalized is not None:
            payload["confidence"] = round(normalized, 1)

    if ls_data.get("mood") is not None:
        payload["mood"] = ls_data.get("mood")
    elif runtime_affect.get("mood") is not None:
        payload["mood"] = runtime_affect.get("mood")

    if isinstance(ls_data.get("vad"), dict):
        payload["vad"] = ls_data["vad"]

    return payload


def _homeostasis_vitality_value(homeostasis_data: dict[str, Any]) -> Any:
    """Return the public vitality/confidence source from homeostasis data.

    ``will_to_live`` is retained as an internal legacy key in the homeostasis
    subsystem. Public health payloads should prefer operational labels so UI and
    API consumers do not treat a homeostatic scalar as proof of subjectivity.
    """
    for key in ("operational_confidence", "vitality", "will_to_live"):
        value = homeostasis_data.get(key)
        if value is not None:
            return value
    return None


def _collect_homeostasis_public_payload(homeostasis_data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(homeostasis_data or {})
    legacy_vitality = payload.pop("will_to_live", None)
    vitality_source = None
    for key in ("vitality", "operational_confidence"):
        value = payload.get(key)
        if value is not None:
            vitality_source = value
            break
    if vitality_source is None:
        vitality_source = legacy_vitality
    normalized = _normalize_percentish(vitality_source)
    if normalized is not None:
        value = round(normalized / 100.0, 4)
        payload.setdefault("vitality", value)
        payload.setdefault("operational_confidence", value)
    return payload


async def _collect_soma_payload() -> dict[str, Any]:
    def _system_fallback() -> dict[str, Any]:
        try:
            cpu_pct = float(psutil.cpu_percent(interval=None) or 0.0) / 100.0
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            ram_pct = float(getattr(ram, "percent", 0.0) or 0.0) / 100.0
            disk_pct = float(getattr(disk, "percent", 0.0) or 0.0) / 100.0
            vitality = max(0.0, 1.0 - (max(cpu_pct, ram_pct, disk_pct) * 0.2))
            return {
                "thermal_load": cpu_pct,
                "resource_anxiety": ram_pct,
                "vitality": vitality,
            }
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation('system', exc)
            logger.debug("Soma fallback telemetry failed: %s", exc)
            return {}

    soma = ServiceContainer.get("soma", default=None)
    if not soma:
        return _system_fallback()

    if hasattr(soma, "pulse"):
        try:
            await asyncio.wait_for(soma.pulse(), timeout=0.25)
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation('system', exc)
            logger.debug("Soma pulse refresh failed: %s", exc)

    try:
        if hasattr(soma, "get_status"):
            raw = soma.get_status() or {}
            if isinstance(raw.get("soma"), dict):
                payload = dict(raw["soma"])
                if payload:
                    return payload
            if isinstance(raw, dict) and {"thermal_load", "resource_anxiety", "vitality"} & set(raw.keys()):
                payload = {
                    "thermal_load": float(raw.get("thermal_load", 0.0) or 0.0),
                    "resource_anxiety": float(raw.get("resource_anxiety", 0.0) or 0.0),
                    "vitality": float(raw.get("vitality", 0.0) or 0.0),
                }
                if payload:
                    return payload
        if hasattr(soma, "get_health"):
            raw = soma.get_health() or {}
            if isinstance(raw, dict):
                payload = {
                    "thermal_load": float(raw.get("thermal_load", 0.0) or 0.0),
                    "resource_anxiety": float(raw.get("resource_anxiety", 0.0) or 0.0),
                    "vitality": float(raw.get("vitality", 0.0) or 0.0),
                }
                if any(value > 0.0 for value in payload.values()):
                    return payload
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Soma status collection failed: %s", exc)
    return _system_fallback()


def _collect_tool_catalog() -> list[dict[str, Any]]:
    engine = optional_service("capability_engine", default=None)
    if not engine:
        return []

    try:
        raw_catalog: Any = None
        if hasattr(engine, "iter_tool_catalog"):
            raw_catalog = engine.iter_tool_catalog(include_inactive=True)
        elif hasattr(engine, "get_tool_catalog"):
            get_tool_catalog = engine.get_tool_catalog
            if inspect.isgeneratorfunction(get_tool_catalog):
                raw_catalog = get_tool_catalog(include_inactive=True)
            else:
                logger.warning(
                    "Skipping materialized tool catalog during UI bootstrap; "
                    "capability_engine should expose iter_tool_catalog()."
                )
                return []

        if raw_catalog is None:
            return []

        catalog: list[dict[str, Any]] = []
        started_at = time.monotonic()
        for index, item in enumerate(raw_catalog):
            if index >= _TOOL_CATALOG_BOOTSTRAP_MAX_ITEMS:
                break
            if time.monotonic() - started_at > _TOOL_CATALOG_BOOTSTRAP_READ_BUDGET_S:
                break
            if isinstance(item, dict):
                catalog.append(item)
        catalog.sort(
            key=lambda item: (
                0 if bool(item.get("available")) else 1,
                0 if bool(item.get("active")) else 1,
                str(item.get("name") or ""),
            )
        )
        return catalog
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Tool catalog collection failed: %s", exc)
    return []


def _collect_commitment_summary() -> dict[str, Any]:
    try:
        from core.agency.commitment_engine import get_commitment_engine

        engine = get_commitment_engine()
        active = engine.get_active_commitments()
        return {
            "active_count": len(active),
            "reliability_score": round(float(engine.reliability_score), 4),
            "active": [
                {
                    "id": item.id,
                    "description": item.description,
                    "outcome": item.outcome,
                    "status": item.status.value if hasattr(item.status, "value") else str(item.status),
                    "hours_remaining": round(float(item.hours_remaining()), 2),
                }
                for item in active[:5]
            ],
        }
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Commitment summary collection failed: %s", exc)
        return {"active_count": 0, "reliability_score": 1.0, "active": []}


def _collect_voice_summary() -> dict[str, Any]:
    try:
        from interface.routes.privacy import get_voice_engine_fn

        _voice_engine_fn = get_voice_engine_fn()
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Voice engine resolver unavailable: %s", exc)
        _voice_engine_fn = None
    voice_available = bool(_voice_engine_fn)
    summary = {
        "available": voice_available,
        "microphone_enabled": voice_available,
        "speaking_enabled": voice_available,
        "listening": False,
        "auto_listen": False,
        "server_capture": False,
        "capture_available": False,
        "stt_available": False,
        "stt_initialized": False,
        "streaming_available": voice_available,
        "state": "ready" if voice_available else "unavailable",
    }
    try:
        voice = _voice_engine_fn() if _voice_engine_fn else None
        if voice is not None:
            microphone_enabled = bool(getattr(voice, "microphone_enabled", True))
            speaking_enabled = bool(getattr(voice, "speaking_enabled", True))
            listening = bool(
                getattr(voice, "_mic_listening", False)
                or getattr(voice, "is_listening", False)
            )
            summary["microphone_enabled"] = microphone_enabled
            summary["speaking_enabled"] = speaking_enabled
            summary["listening"] = listening
            if hasattr(voice, "get_status"):
                voice_status = voice.get_status() or {}
                if isinstance(voice_status, dict):
                    summary["auto_listen"] = bool(voice_status.get("auto_listen", False))
                    summary["server_capture"] = bool(voice_status.get("server_capture", False))
                    summary["capture_available"] = bool(voice_status.get("capture_available", False))
                    summary["stt_available"] = bool(voice_status.get("stt_available", False))
                    summary["stt_initialized"] = bool(voice_status.get("stt_initialized", False))
                    summary["capture_backend"] = voice_status.get("capture_backend")
                    summary["stt_backend"] = voice_status.get("stt_backend")
                    summary["stt"] = voice_status.get("stt")
                    summary["tts"] = voice_status.get("tts")
            if not microphone_enabled and not speaking_enabled:
                summary["state"] = "muted"
            else:
                voice_state = getattr(getattr(voice, "state", None), "name", "") or ""
                if voice_state:
                    summary["state"] = str(voice_state).lower()
                else:
                    summary["state"] = "listening" if getattr(voice, "is_listening", False) else "ready"
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Voice summary collection failed: %s", exc)
    return summary


async def _probe_desktop_access_summary(*, allow_probe: bool = True) -> dict[str, Any]:
    cached_payload = _desktop_access_cache.get("payload")
    cached_at = float(_desktop_access_cache.get("captured_at", 0.0) or 0.0)
    if (
        isinstance(cached_payload, dict)
        and (time.monotonic() - cached_at) < _desktop_access_cache_ttl(cached_payload)
    ):
        return _desktop_access_cached_copy(cached_payload, captured_at=cached_at)
    if not allow_probe:
        if isinstance(cached_payload, dict):
            return _desktop_access_cached_copy(
                cached_payload,
                captured_at=cached_at,
                stale=True,
                probe_mode="stale_cached",
            )
        payload = _desktop_access_empty_payload()
        payload["probe_mode"] = "fast_pending"
        payload["overall_status"] = "pending"
        payload["permission_confidence"] = "pending"
        payload["desktop_access_diagnosis"] = [
            "Desktop permission probing is handled by /api/system/desktop-access so health checks stay fast."
        ]
        return payload

    payload: dict[str, Any] = _desktop_access_empty_payload()
    payload["probe_mode"] = "full"
    try:
        from core.security.permission_guard import PermissionType, get_permission_guard
        from core.skills._pyautogui_runtime import get_pyautogui

        native_ready = False
        resident_native_ready = False
        if sys.platform == "darwin":
            try:
                from core.security.native_desktop_bridge import probe_native_desktop_bridge

                native_probe = await asyncio.wait_for(
                    asyncio.to_thread(probe_native_desktop_bridge, force=False),
                    timeout=max(0.2, _DESKTOP_ACCESS_NATIVE_PROBE_TIMEOUT_S),
                )
                _mark_desktop_access_probe_success("native_bridge", "resident")
                payload["native_bridge_probe"] = (
                    native_probe if isinstance(native_probe, dict)
                    else {"ok": False, "error": f"invalid:{type(native_probe).__name__}"}
                )
                resident_native_ready = bool(
                    isinstance(native_probe, dict)
                    and native_probe.get("ok")
                    and native_probe.get("bridge_transport") == "resident_ipc"
                    and all(
                        bool(native_probe.get(key))
                        for key in ("screen_recording", "accessibility", "automation")
                    )
                )
                native_ready = resident_native_ready
            except (TimeoutError, *_SYSTEM_RECOVERABLE_ERRORS) as exc:
                issue, streak = _record_desktop_access_probe_issue(
                    "native_bridge",
                    "resident",
                    exc,
                )
                payload["native_bridge_probe"] = {
                    "ok": False,
                    "error": str(exc)[:240] or type(exc).__name__,
                    "status": issue,
                    "probe_unavailable": True,
                    "retryable": True,
                    "failure_streak": streak,
                }

        guard = ServiceContainer.get("permission_guard", default=None) or get_permission_guard()
        if guard:
            identity_probe = getattr(guard, "current_process_identity", None)
            if callable(identity_probe):
                try:
                    payload["process_identity"] = identity_probe()
                except _SYSTEM_RECOVERABLE_ERRORS as exc:
                    _record_desktop_access_probe_issue(
                        "identity",
                        "current_process",
                        exc,
                    )
            if not native_ready:
                async def _bounded_reported_probe(ptype: Any) -> dict[str, Any]:
                    target = ptype.name.lower()
                    try:
                        result = await asyncio.wait_for(
                            guard.check_permission(ptype, force=False),
                            timeout=max(0.2, _DESKTOP_ACCESS_DIRECT_PROBE_TIMEOUT_S),
                        )
                        _mark_desktop_access_probe_success("reported", target)
                        return result if isinstance(result, dict) else {
                            "granted": False,
                            "status": "invalid_probe_result",
                            "guidance": "",
                            "detail": f"got {type(result).__name__}",
                        }
                    except (TimeoutError, *_SYSTEM_RECOVERABLE_ERRORS) as exc:
                        return _desktop_access_probe_unavailable(
                            guard,
                            ptype,
                            probe="reported",
                            exc=exc,
                        )

                screen, accessibility, automation = await asyncio.gather(
                    _bounded_reported_probe(PermissionType.SCREEN),
                    _bounded_reported_probe(PermissionType.ACCESSIBILITY),
                    _bounded_reported_probe(PermissionType.AUTOMATION),
                )
                payload["screen_recording"] = screen
                payload["accessibility"] = accessibility
                payload["automation"] = automation
                payload["frontmost_app"] = str(automation.get("detail", "") or "")
                direct_probe = getattr(guard, "check_permission_direct_local", None)
                if not callable(direct_probe):
                    direct_probe = getattr(guard, "check_permission_direct", None)
                if callable(direct_probe):
                    reported_by_type = {
                        PermissionType.SCREEN: screen,
                        PermissionType.ACCESSIBILITY: accessibility,
                        PermissionType.AUTOMATION: automation,
                    }

                    async def _bounded_direct_probe(ptype: Any) -> dict[str, Any]:
                        target = ptype.name.lower()
                        reported = reported_by_type.get(ptype, {})
                        if isinstance(reported, dict) and reported.get("probe_unavailable"):
                            inherited = dict(reported)
                            inherited["direct_probe"] = True
                            inherited["probe_source"] = "reported_probe"
                            return inherited
                        try:
                            result = await asyncio.wait_for(
                                direct_probe(ptype),
                                timeout=max(0.2, _DESKTOP_ACCESS_DIRECT_PROBE_TIMEOUT_S),
                            )
                            _mark_desktop_access_probe_success("direct", target)
                            return result if isinstance(result, dict) else {
                                "granted": False,
                                "status": "invalid_probe_result",
                                "guidance": "",
                                "detail": f"got {type(result).__name__}",
                                "direct_probe": True,
                            }
                        except (TimeoutError, *_SYSTEM_RECOVERABLE_ERRORS) as exc:
                            return _desktop_access_probe_unavailable(
                                guard,
                                ptype,
                                probe="direct",
                                exc=exc,
                            )

                    try:
                        direct_screen, direct_accessibility, direct_automation = await asyncio.gather(
                            _bounded_direct_probe(PermissionType.SCREEN),
                            _bounded_direct_probe(PermissionType.ACCESSIBILITY),
                            _bounded_direct_probe(PermissionType.AUTOMATION),
                        )
                        payload["direct_screen_recording"] = direct_screen
                        payload["direct_accessibility"] = direct_accessibility
                        payload["direct_automation"] = direct_automation
                    except (TimeoutError, *_SYSTEM_RECOVERABLE_ERRORS) as exc:
                        _record_desktop_access_probe_issue(
                            "direct_group",
                            "permissions",
                            exc,
                        )

        native_bridge = payload.get("native_bridge_probe")
        native_bridge_is_resident = (
            isinstance(native_bridge, dict)
            and native_bridge.get("ok")
            and native_bridge.get("bridge_transport") == "resident_ipc"
        )
        if native_bridge_is_resident:
            payload["effective_app_identity"] = {
                "bundle_identifier": str(native_bridge.get("bundle_identifier", "") or ""),
                "bridge_executable": str(native_bridge.get("bridge_executable", "") or ""),
                "bridge_transport": str(native_bridge.get("bridge_transport", "") or ""),
                "code_signature": native_bridge.get("code_signature")
                if isinstance(native_bridge.get("code_signature"), dict)
                else {},
            }
            native_common = {
                "status": "active_native_bridge",
                "guidance": "",
                "native_bridge": True,
                "bridge_executable": str(native_bridge.get("bridge_executable", "") or ""),
                "bundle_identifier": str(native_bridge.get("bundle_identifier", "") or ""),
                "direct_probe": True,
            }
            if native_bridge.get("screen_recording"):
                screen_result = {"granted": True, **native_common}
                payload["screen_recording"] = screen_result
                payload["direct_screen_recording"] = screen_result
            if native_bridge.get("accessibility"):
                accessibility_result = {"granted": True, **native_common}
                payload["accessibility"] = accessibility_result
                payload["direct_accessibility"] = accessibility_result
            if native_bridge.get("automation"):
                automation_result = {
                    "granted": True,
                    **native_common,
                    "frontmost_app": str(native_bridge.get("frontmost_app", "") or ""),
                }
                payload["automation"] = automation_result
                payload["direct_automation"] = automation_result

        pyautogui, pyautogui_error = get_pyautogui()
        payload["pyautogui_ready"] = pyautogui is not None
        if pyautogui_error:
            payload["pyautogui_error"] = str(pyautogui_error)[:240]

        screen_granted = bool((payload["screen_recording"] or {}).get("granted"))
        accessibility_granted = bool((payload["accessibility"] or {}).get("granted"))
        automation_granted = bool((payload["automation"] or {}).get("granted"))
        direct_screen_granted = bool((payload["direct_screen_recording"] or {}).get("granted"))
        direct_accessibility_granted = bool((payload["direct_accessibility"] or {}).get("granted"))
        direct_automation_granted = bool((payload["direct_automation"] or {}).get("granted"))
        unavailable_statuses = {
            "",
            "unknown",
            "deferred",
            "timeout",
            "probe_error",
            "probe_failed",
            "invalid_probe_result",
            "dependency_missing",
            "resident_bridge_required",
            "unverified_assertion",
            "asserted_env",
        }

        def _probe_has_evidence(result: Any) -> bool:
            return bool(
                isinstance(result, dict)
                and not result.get("probe_unavailable")
                and str(result.get("status") or "").lower()
                not in unavailable_statuses
            )

        reported_results = {
            "screen_recording": payload["screen_recording"],
            "accessibility": payload["accessibility"],
            "automation": payload["automation"],
        }
        direct_results = {
            "screen_recording": payload["direct_screen_recording"],
            "accessibility": payload["direct_accessibility"],
            "automation": payload["direct_automation"],
        }
        reported_probe_unavailable_permissions = [
            name for name, result in reported_results.items()
            if not _probe_has_evidence(result)
        ]
        direct_probe_unavailable_permissions = [
            name for name, result in direct_results.items()
            if not _probe_has_evidence(result)
        ]
        unverified_permissions = [
            name for name in reported_results
            if not _probe_has_evidence(direct_results[name])
            and not _probe_has_evidence(reported_results[name])
        ]
        payload["reported_probe_unavailable_permissions"] = (
            reported_probe_unavailable_permissions
        )
        payload["direct_probe_unavailable_permissions"] = (
            direct_probe_unavailable_permissions
        )
        payload["unverified_permissions"] = unverified_permissions
        direct_probe_available = any(
            _probe_has_evidence(result) for result in direct_results.values()
        )
        payload["direct_probe_available"] = direct_probe_available
        payload["reported_screen_capture_ready"] = screen_granted
        payload["reported_desktop_control_ready"] = accessibility_granted and bool(payload["pyautogui_ready"])
        payload["reported_screen_text_ready"] = automation_granted and accessibility_granted
        payload["direct_screen_capture_ready"] = direct_screen_granted
        payload["direct_desktop_control_ready"] = direct_accessibility_granted and bool(payload["pyautogui_ready"])
        payload["direct_screen_text_ready"] = direct_automation_granted and direct_accessibility_granted
        effective_screen_granted = (
            direct_screen_granted
            if _probe_has_evidence(payload["direct_screen_recording"])
            else screen_granted
        )
        effective_accessibility_granted = (
            direct_accessibility_granted
            if _probe_has_evidence(payload["direct_accessibility"])
            else accessibility_granted
        )
        effective_automation_granted = (
            direct_automation_granted
            if _probe_has_evidence(payload["direct_automation"])
            else automation_granted
        )
        payload["screen_capture_ready"] = effective_screen_granted
        payload["desktop_control_ready"] = (
            effective_accessibility_granted
        ) and bool(payload["pyautogui_ready"])
        payload["screen_text_ready"] = (
            effective_automation_granted and effective_accessibility_granted
        )
        payload["menu_clock_ready"] = (
            effective_automation_granted and effective_accessibility_granted
        )
        if payload["menu_clock_ready"]:
            from core.skills.computer_use import ComputerUseSkill

            def _probe_menu_clock() -> dict[str, Any]:
                from core.governance_context import local_internal_governed_scope
                skill = ComputerUseSkill()
                try:
                    with local_internal_governed_scope("system.probe_menu_clock", domain="tool_execution"):
                        text = skill._read_menu_clock_macos()
                    return {"ready": True, "text": text[:240]}
                except _SYSTEM_RECOVERABLE_ERRORS as exc:
                    return {"ready": False, "error": str(exc)[:240]}

            try:
                menu_clock_probe = await asyncio.wait_for(
                    asyncio.to_thread(_probe_menu_clock),
                    timeout=max(0.25, _DESKTOP_ACCESS_MENU_CLOCK_TIMEOUT_S),
                )
            except (TimeoutError, *_SYSTEM_RECOVERABLE_ERRORS) as exc:
                _record_desktop_access_probe_issue(
                    "menu_clock",
                    "system_events",
                    exc,
                )
                menu_clock_probe = {
                    "ready": False,
                    "error": str(exc)[:240] or type(exc).__name__,
                }
            payload["menu_clock_ready"] = bool(menu_clock_probe.get("ready"))
            payload["menu_clock_text"] = str(menu_clock_probe.get("text", "") or "")
            payload["menu_clock_error"] = str(menu_clock_probe.get("error", "") or "")
        primary_ready = [
            payload["screen_capture_ready"],
            payload["desktop_control_ready"],
            payload["screen_text_ready"],
        ]
        reported_primary_ready = [
            payload["reported_screen_capture_ready"],
            payload["reported_desktop_control_ready"],
            payload["reported_screen_text_ready"],
        ]
        direct_primary_ready = [
            payload["direct_screen_capture_ready"],
            payload["direct_desktop_control_ready"],
            payload["direct_screen_text_ready"],
        ]
        payload["permission_assumptions"] = [
            name for name, result in (
                ("screen_recording", payload["screen_recording"]),
                ("accessibility", payload["accessibility"]),
                ("automation", payload["automation"]),
            )
            if str((result or {}).get("status") or "") == "asserted_env"
        ]
        reported_blocking_permissions = [
            name for name, granted in (
                ("screen_recording", screen_granted),
                ("accessibility", accessibility_granted),
                ("automation", automation_granted),
            ) if not granted
        ]
        direct_blocking_permissions = [
            name for name, granted in (
                ("screen_recording", direct_screen_granted),
                ("accessibility", direct_accessibility_granted),
                ("automation", direct_automation_granted),
            ) if not granted
        ]
        payload["reported_blocking_permissions"] = reported_blocking_permissions
        payload["direct_blocking_permissions"] = direct_blocking_permissions
        payload["blocking_permissions"] = [
            name for name, granted in (
                ("screen_recording", effective_screen_granted),
                ("accessibility", effective_accessibility_granted),
                ("automation", effective_automation_granted),
            ) if not granted
        ]
        payload["permission_confidence"] = (
            "direct"
            if all(direct_primary_ready) else
            "partial_direct"
            if any(direct_primary_ready) else
            "claims_only"
            if direct_probe_available and all(reported_primary_ready) and payload["permission_assumptions"] else
            "asserted_env"
            if all(reported_primary_ready) and payload["permission_assumptions"] else
            "unavailable"
            if unverified_permissions and not any(primary_ready) else
            "unverified"
            if payload["permission_assumptions"] else
            "blocked"
        )
        payload["overall_status"] = (
            "ready"
            if all(direct_primary_ready) else
            "claims_only"
            if direct_probe_available and all(reported_primary_ready) and payload["permission_assumptions"] else
            "assumed_ready"
            if all(reported_primary_ready) and payload["permission_assumptions"] else
            "partial"
            if any(direct_primary_ready) or (not direct_probe_available and any(primary_ready)) else
            "probe_unavailable"
            if unverified_permissions and not any(primary_ready) else
            "partial"
            if any(
                bool((payload[key] or {}).get("granted"))
                for key in ("screen_recording", "accessibility", "automation")
            ) else
            "blocked"
        )
        diagnosis: list[str] = []
        if payload.get("unverified_permissions"):
            diagnosis.append(
                "One or more passive permission probes were unavailable; Aura is preserving the distinction between unknown and macOS-denied access."
            )
        signature = {}
        if isinstance(payload.get("effective_app_identity"), dict):
            signature = payload["effective_app_identity"].get("code_signature") or {}
        if isinstance(signature, dict) and signature.get("stable_tcc_identity") is False:
            if signature.get("adhoc") or str(signature.get("signature") or "").strip().lower() == "adhoc":
                diagnosis.append(
                    "Aura.app is ad-hoc signed, so macOS permissions can attach to a stale rebuild instead of the currently running app."
                )
            else:
                diagnosis.append(
                    "Aura.app does not expose a stable signing authority, so macOS may not retain permissions reliably across rebuilds."
                )
            hint = str(signature.get("tcc_repair_hint") or "").strip()
            if hint:
                diagnosis.append(hint)
        if native_bridge_is_resident and payload["blocking_permissions"]:
            diagnosis.append(
                "The resident Aura.app bridge is reachable, but macOS denies the requested TCC grants for this exact app identity."
            )
            bundle_identifier = str(native_bridge.get("bundle_identifier") or "com.aura.desktop")
            bridge_executable = str(native_bridge.get("bridge_executable") or "/Applications/Aura.app/Contents/MacOS/aura-launcher")
            payload["tcc_repair_plan"] = {
                "reason": "resident_bridge_denied_current_tcc_grants",
                "bundle_identifier": bundle_identifier,
                "bridge_executable": bridge_executable,
                "blocking_permissions": list(payload["blocking_permissions"]),
                "commands": [
                    f"tccutil reset ScreenCapture {bundle_identifier}",
                    f"tccutil reset Accessibility {bundle_identifier}",
                ],
                "manual_steps": [
                    "Quit Aura completely.",
                    "Run the reset commands for the current Aura.app bundle identifier.",
                    "Open /Applications/Aura.app.",
                    "Approve Screen Recording and Accessibility when macOS prompts.",
                    "If System Settings still shows Aura as enabled but the bridge is denied, remove the Aura row with the minus button and add /Applications/Aura.app again.",
                ],
                "request_state": dict(_desktop_access_request_state),
                "verification_endpoint": "/api/system/desktop-access",
            }
        if isinstance(native_bridge, dict) and native_bridge.get("bridge_transport") == "one_shot_subprocess":
            diagnosis.append(
                "A diagnostic one-shot Aura.app bridge responded, but the resident Aura.app bridge is not alive; durable desktop control is blocked until the signed app stays resident."
            )
        if (
            payload.get("process_identity", {}).get("bundle_identifier") == "org.python.python"
            and payload.get("overall_status") != "ready"
        ):
            diagnosis.append(
                "The cognitive runtime is a Python child; durable desktop control should route through the resident Aura.app bridge, not Python's own TCC row."
            )
        payload["desktop_access_diagnosis"] = diagnosis
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Desktop access summary collection failed: %s", exc)
    payload["captured_at_unix"] = time.time()
    payload["probe_runtime"] = _desktop_access_probe_state_snapshot()
    payload["cache_ttl_s"] = _desktop_access_cache_ttl(payload)
    _desktop_access_cache["captured_at"] = time.monotonic()
    _desktop_access_cache["payload"] = payload
    return payload


async def _collect_desktop_access_summary(*, allow_probe: bool = True) -> dict[str, Any]:
    """Share one full desktop probe per event loop and preserve it on caller cancel."""
    cached_payload = _desktop_access_cache.get("payload")
    cached_at = float(_desktop_access_cache.get("captured_at", 0.0) or 0.0)
    if (
        isinstance(cached_payload, dict)
        and (time.monotonic() - cached_at) < _desktop_access_cache_ttl(cached_payload)
    ):
        return _desktop_access_cached_copy(cached_payload, captured_at=cached_at)
    if not allow_probe:
        return await _probe_desktop_access_summary(allow_probe=False)

    loop = asyncio.get_running_loop()
    task = _DESKTOP_ACCESS_PROBE_TASKS.get(loop)
    shared = task is not None and not task.done()
    if task is None or task.done():
        task = create_tracked_task(
            _probe_desktop_access_summary(allow_probe=True),
            name="system.desktop_access.shared_probe",
            owner="system.desktop_access",
        )
        _DESKTOP_ACCESS_PROBE_TASKS[loop] = task

        def _clear(completed: asyncio.Task[dict[str, Any]]) -> None:
            if _DESKTOP_ACCESS_PROBE_TASKS.get(loop) is completed:
                _DESKTOP_ACCESS_PROBE_TASKS.pop(loop, None)
            if completed.cancelled():
                return
            try:
                completed.exception()
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                return

        task.add_done_callback(_clear)

    result = await asyncio.shield(task)
    if not shared:
        return result
    copied = dict(result)
    copied["probe_mode"] = "shared_probe"
    copied["singleflight_shared"] = True
    return copied


@router.get("/system/desktop-access")
async def desktop_access_summary() -> dict[str, Any]:
    return await _collect_desktop_access_summary()


@router.post("/system/desktop-access/request-screen")
async def request_screen_access() -> dict[str, Any]:
    try:
        native_result: dict[str, Any] = {}
        try:
            from core.security.native_desktop_bridge import invoke_native_desktop_bridge

            native_result = invoke_native_desktop_bridge(
                "request_screen",
                read_only=True,
                timeout=45.0,
                prefer_one_shot=False,
            )
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "system.desktop_access.native_request_screen",
                exc,
                action="falling back to Python Screen Recording request",
                severity="warning",
            )
        if native_result:
            granted = bool(native_result.get("screen_recording"))
            status = "granted" if granted else "approval_required"
            _desktop_access_request_state["screen_recording"] = {
                "requested": True,
                "granted": granted,
                "status": status,
                "target": "Aura.app",
                "bundle_identifier": str(native_result.get("bundle_identifier") or ""),
                "requested_at": time.time(),
                "detail": (
                    "macOS still requires user approval in Screen Recording for /Applications/Aura.app"
                    if not granted else
                    "Screen Recording is granted for the signed Aura.app bridge"
                ),
            }
            _desktop_access_cache["captured_at"] = 0.0
            return {
                "requested": True,
                "granted": granted,
                "status": status,
                "approval_required": not granted,
                "native_bridge": native_result,
                "target": "Aura.app",
            }

        from core.security.permission_guard import get_permission_guard

        guard = get_permission_guard()
        request = getattr(guard, "request_screen_capture_access", None)
        granted = bool(request()) if callable(request) else False
        _desktop_access_request_state["screen_recording"] = {
            "requested": callable(request),
            "granted": granted,
            "status": "granted" if granted else "approval_required",
            "target": "Python runtime",
            "requested_at": time.time(),
        }
        _desktop_access_cache["captured_at"] = 0.0
        return {
            "requested": callable(request),
            "granted": granted,
            "status": "granted" if granted else "approval_required",
            "approval_required": not granted,
        }
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "system.desktop_access.request_screen",
            exc,
            action="reported Screen Recording request failure",
            severity="warning",
        )
        return {"requested": False, "granted": False, "error": str(exc)[:240]}


@router.post("/system/desktop-access/request-accessibility")
async def request_accessibility_access() -> dict[str, Any]:
    try:
        native_result: dict[str, Any] = {}
        try:
            from core.security.native_desktop_bridge import invoke_native_desktop_bridge

            native_result = invoke_native_desktop_bridge(
                "request_accessibility",
                read_only=True,
                timeout=45.0,
                prefer_one_shot=False,
            )
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "system.desktop_access.native_request_accessibility",
                exc,
                action="falling back to Python Accessibility request",
                severity="warning",
            )
        if native_result:
            granted = bool(native_result.get("accessibility"))
            status = "granted" if granted else "approval_required"
            _desktop_access_request_state["accessibility"] = {
                "requested": True,
                "granted": granted,
                "status": status,
                "target": "Aura.app",
                "bundle_identifier": str(native_result.get("bundle_identifier") or ""),
                "requested_at": time.time(),
                "detail": (
                    "macOS still requires user approval in Accessibility for /Applications/Aura.app"
                    if not granted else
                    "Accessibility is granted for the signed Aura.app bridge"
                ),
            }
            _desktop_access_cache["captured_at"] = 0.0
            return {
                "requested": True,
                "granted": granted,
                "status": status,
                "approval_required": not granted,
                "native_bridge": native_result,
                "target": "Aura.app",
            }

        from core.security.permission_guard import get_permission_guard

        guard = get_permission_guard()
        request = getattr(guard, "request_accessibility_trust", None)
        granted = bool(request()) if callable(request) else False
        _desktop_access_request_state["accessibility"] = {
            "requested": callable(request),
            "granted": granted,
            "status": "granted" if granted else "approval_required",
            "target": "Python runtime",
            "requested_at": time.time(),
        }
        _desktop_access_cache["captured_at"] = 0.0
        return {
            "requested": callable(request),
            "granted": granted,
            "status": "granted" if granted else "approval_required",
            "approval_required": not granted,
        }
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "system.desktop_access.request_accessibility",
            exc,
            action="reported Accessibility request failure",
            severity="warning",
        )
        return {"requested": False, "granted": False, "error": str(exc)[:240]}


@router.post("/system/desktop-access/open-settings/{permission}")
async def open_desktop_access_settings(permission: str) -> dict[str, Any]:
    aliases = {
        "screen": "SCREEN",
        "screen_recording": "SCREEN",
        "screencapture": "SCREEN",
        "accessibility": "ACCESSIBILITY",
        "automation": "AUTOMATION",
    }
    normalized = aliases.get(str(permission or "").strip().lower())
    if not normalized:
        return {
            "opened": False,
            "permission": permission,
            "error": "unknown_permission",
        }
    try:
        from core.security.permission_setup import open_settings_pane

        opened = bool(open_settings_pane(normalized))
        return {
            "opened": opened,
            "permission": normalized,
            "target": "System Settings",
        }
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "system.desktop_access.open_settings",
            exc,
            action="reported desktop permission settings launch failure",
            severity="warning",
        )
        return {
            "opened": False,
            "permission": normalized,
            "error": str(exc)[:240],
        }


def _collect_neurodynamic_status() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "idle",
        "action": "",
        "uncertainty": 0.0,
        "confidence": 0.0,
        "advisory_only": True,
        "authority_gateway_required_for_effects": True,
    }
    try:
        advisor = ServiceContainer.get("spiking_active_inference", default=None)
        if advisor is None or not hasattr(advisor, "snapshot"):
            return payload
        snapshot = advisor.snapshot() or {}
        if not isinstance(snapshot, dict):
            return payload
        governance = snapshot.get("governance") or {}
        if not isinstance(governance, dict):
            governance = {}
        payload.update(
            {
                "status": str(snapshot.get("status") or "active"),
                "action": str(snapshot.get("action") or ""),
                "uncertainty": _safe_float(snapshot.get("uncertainty"), 0.0),
                "confidence": _safe_float(snapshot.get("confidence"), 0.0),
                "advisory_only": bool(governance.get("advisory_only", True)),
                "authority_gateway_required_for_effects": bool(
                    governance.get("authority_gateway_required_for_effects", True)
                ),
            }
        )
        features = snapshot.get("features")
        if isinstance(features, dict):
            payload["features"] = {
                "tool_pressure": _safe_float(features.get("tool_pressure"), 0.0),
                "error_pressure": _safe_float(features.get("error_pressure"), 0.0),
                "memory_pressure": _safe_float(features.get("memory_pressure"), 0.0),
            }
        stability = snapshot.get("stability")
        if isinstance(stability, dict):
            payload["stability"] = {
                "spectral_radius": _safe_float(stability.get("spectral_radius"), 0.0),
                "entropy": _safe_float(stability.get("entropy"), 0.0),
                "winner_margin": _safe_float(stability.get("winner_margin"), 0.0),
                "decision_instability": _safe_float(
                    stability.get("decision_instability"), 0.0
                ),
                "ode_spectral_abscissa": _safe_float(
                    stability.get("ode_spectral_abscissa"), 0.0
                ),
                "fixed_point_residual": _safe_float(
                    stability.get("fixed_point_residual"), 0.0
                ),
                "bifurcation_pressure": _safe_float(
                    stability.get("bifurcation_pressure"), 0.0
                ),
            }
        working_memory = snapshot.get("working_memory")
        if isinstance(working_memory, dict):
            payload["working_memory"] = {
                "admission": str(working_memory.get("admission") or "unknown"),
                "admitted": bool(working_memory.get("admitted", True)),
                "queue_load": _safe_float(working_memory.get("queue_load"), 0.0),
                "overload_pressure": _safe_float(working_memory.get("overload_pressure"), 0.0),
                "utilization": _safe_float(working_memory.get("utilization"), 0.0),
                "expected_wait_s": _safe_float(working_memory.get("expected_wait_s"), 0.0),
            }
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Neurodynamic status collection failed: %s", exc)
    return payload


def _collect_imagination_status() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "idle",
        "frames": 0,
        "latest": None,
        "working_memory": {},
        "attractor_bias": {},
        "eligibility_trace": {},
        "recent_outcomes": [],
        "advisory_only": True,
        "no_external_effects": True,
        "authority_gateway_required_for_effects": True,
    }
    try:
        engine = ServiceContainer.get("imagination_engine", default=None)
        if engine is None or not hasattr(engine, "snapshot"):
            return payload
        snapshot = engine.snapshot() or {}
        if not isinstance(snapshot, dict):
            return payload
        governance = snapshot.get("governance") or {}
        if not isinstance(governance, dict):
            governance = {}
        payload.update(
            {
                "status": str(snapshot.get("status") or "active"),
                "frames": int(_safe_float(snapshot.get("frames"), 0.0)),
                "latest": snapshot.get("latest") if isinstance(snapshot.get("latest"), dict) else None,
                "working_memory": snapshot.get("working_memory") if isinstance(snapshot.get("working_memory"), dict) else {},
                "attractor_bias": snapshot.get("attractor_bias") if isinstance(snapshot.get("attractor_bias"), dict) else {},
                "eligibility_trace": snapshot.get("eligibility_trace") if isinstance(snapshot.get("eligibility_trace"), dict) else {},
                "recent_outcomes": snapshot.get("recent_outcomes") if isinstance(snapshot.get("recent_outcomes"), list) else [],
                "advisory_only": bool(governance.get("advisory_only", True)),
                "no_external_effects": bool(governance.get("no_external_effects", True)),
                "authority_gateway_required_for_effects": bool(
                    governance.get("authority_gateway_required_for_effects", True)
                ),
            }
        )
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Imagination status collection failed: %s", exc)
    return payload


def _collect_runtime_capabilities(conversation_lane: dict[str, Any] | None = None) -> dict[str, Any]:
    lane = conversation_lane if isinstance(conversation_lane, dict) else _collect_conversation_lane_status_resilient()
    payload: dict[str, Any] = {
        "local_backend": "unknown",
        "local_runtime": "offline",
        "conversation_model": str(lane.get("desired_model", "") or ""),
        "conversation_endpoint": str(lane.get("desired_endpoint", "") or ""),
        "conversation_state": str(lane.get("state", "") or ""),
        "conversation_ready": bool(lane.get("conversation_ready", False)),
        "neurodynamic_advisor": _collect_neurodynamic_status(),
        "imagination_engine": _collect_imagination_status(),
    }
    try:
        from core.brain.llm.model_registry import (
            ACTIVE_MODEL,
            BRAINSTEM_MODEL,
            DEEP_MODEL,
            FALLBACK_MODEL,
            get_local_backend,
        )

        payload.update(
            {
                "local_backend": get_local_backend(),
                "cortex_model": ACTIVE_MODEL,
                "solver_model": DEEP_MODEL,
                "brainstem_model": BRAINSTEM_MODEL,
                "fallback_model": FALLBACK_MODEL,
            }
        )
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Runtime capability backend lookup failed: %s", exc)

    state = str(payload.get("conversation_state", "") or "").lower()
    if bool(payload.get("conversation_ready")):
        payload["local_runtime"] = "online"
    elif _conversation_lane_is_standby_resilient(lane):
        payload["local_runtime"] = "standby"
    elif state in {"cold", "warming", "spawning", "handshaking", "recovering", "ready"}:
        payload["local_runtime"] = "warming"
    elif state == "failed":
        payload["local_runtime"] = "degraded"
    return payload


def _derive_ui_status_flags(
    *,
    state_summary: dict[str, Any],
    executive_status: dict[str, Any],
    boot_snapshot: dict[str, Any],
    tool_catalog: list[dict[str, Any]],
) -> list[str]:
    flags: list[str] = []
    if not bool(boot_snapshot.get("ready", False)):
        flags.append("booting")
    if bool(state_summary.get("thermal_guard")):
        flags.append("thermal_guard")
    if _safe_float(state_summary.get("coherence_score"), 1.0) < 0.72:
        flags.append("coherence_low")
    if _safe_float(state_summary.get("fragmentation_score"), 0.0) > 0.4:
        flags.append("fragmentation_high")
    if _safe_int(state_summary.get("contradiction_count"), 0) > 3:
        flags.append("contradictions_present")
    epistemics = state_summary.get("epistemics", {}) or {}
    if _safe_int(epistemics.get("contested"), 0) > 0:
        flags.append("beliefs_contested")
    unavailable_count = sum(1 for tool in tool_catalog if not bool(tool.get("available")))
    if unavailable_count >= 3:
        flags.append("tool_unavailable")
    if str(executive_status.get("last_target") or "").strip().lower() == "secondary":
        flags.append("executive_hold")
    return flags


# ── Routes ────────────────────────────────────────────────────

@router.get("/telemetry/stream")
async def telemetry_stream(request: Request):
    """Server-Sent Events stream for HUD telemetry."""
    _require_internal(request)

    async def event_generator():
        try:
            init_payload = {
                "type": "telemetry",
                "cpu_usage": psutil.cpu_percent(interval=None),
                "memory_usage": psutil.virtual_memory().percent,
                "timestamp": time.time(),
            }
        except _SYSTEM_RECOVERABLE_ERRORS as e:
            record_degradation("system", e)
            logger.debug("SSE initial telemetry snapshot failed: %s", e)
            init_payload = {"type": "telemetry", "cpu_usage": 0.0, "memory_usage": 0.0, "timestamp": time.time()}
        init_data = json.dumps(init_payload)
        yield f"event: telemetry\ndata: {init_data}\n\n"

        q = None
        try:
            q = await broadcast_bus.subscribe()
            while not await request.is_disconnected():
                while q.qsize() > _SSE_QUEUE_BACKLOG_LIMIT:
                    try:
                        q.get_nowait()
                        q.task_done()
                    except asyncio.QueueEmpty:
                        break

                try:
                    item = await asyncio.wait_for(q.get(), timeout=_SSE_IDLE_HEARTBEAT_S)
                except TimeoutError:
                    heartbeat = json.dumps(runtime_heartbeat_payload("heartbeat"))
                    yield f"event: heartbeat\ndata: {heartbeat}\n\n"
                    continue

                try:
                    _priority, _ts, msg = item
                    safe_msg = _json_safe(msg) if isinstance(msg, dict) else {"type": "message", "payload": _json_safe(msg)}
                    msg_type = str(safe_msg.get("type", "message") or "message")
                    data = json.dumps(safe_msg)
                    yield f"event: {msg_type}\ndata: {data}\n\n"
                except asyncio.CancelledError:
                    break
                except _SYSTEM_RECOVERABLE_ERRORS as e:
                    record_degradation('system', e)
                    logger.debug("SSE generate error: %s", e)
                    await asyncio.sleep(0.1)
                    continue
                finally:
                    q.task_done()
        finally:
            if q is not None:
                await broadcast_bus.unsubscribe(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/metrics", tags=["metrics"])
async def metrics(request: Request):
    """System metrics for monitoring (JSON format, backwards compatible)."""
    _require_internal(request)
    try:
        from core.runtime.health_contract import runtime_health_report

        orch = ServiceContainer.get("orchestrator", default=None)
        orch_status = orch.get_status() if orch else {}
        contract = runtime_health_report()

        return {
            "status": contract.get("status", "unknown"),
            "healthy": bool(contract.get("healthy", False)),
            "operational": bool(contract.get("operational", False)),
            "required_probes": contract.get("required_probes", {}),
            "uptime": time.time() - (orch_status.get("start_time", time.time()) if orch_status else time.time()),
            "active_connections": ws_manager.count(),
            "cycle_count": orch_status.get("cycle_count", 0),
            "cpu_usage": float(int(psutil.cpu_percent() * 10)) / 10.0 if 'psutil' in sys.modules else 0,
            "memory_usage": float(int(psutil.virtual_memory().percent * 10)) / 10.0 if 'psutil' in sys.modules else 0,
        }
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.error("Metrics collection failed: %s", e, exc_info=True)
        return ORJSONResponse({"status": "error", "message": "Metrics collection failed"}, status_code=500)


@router.get("/metrics/prometheus", tags=["metrics"])
async def metrics_prometheus(request: Request):
    """Prometheus-compatible metrics in text exposition format.

    Scrape this endpoint with Prometheus or any compatible collector.
    """
    _require_internal(request)
    try:
        from fastapi.responses import Response

        from core.observability.metrics import get_metrics

        text = get_metrics().render_prometheus()
        return Response(
            content=text,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.error("Prometheus metrics render failed: %s", e, exc_info=True)
        return ORJSONResponse(
            {"status": "error", "message": "Prometheus metrics unavailable"},
            status_code=500,
        )


@router.get("/system/incidents", tags=["health"])
async def api_system_incidents(request: Request, minutes: float = 60.0):
    """Receipt-backed incident narrative over Aura's own forensics.

    Deterministic synthesis of stall dumps, degraded events, the memory
    sentinel, and boot timings into causal episodes — 'what happened and
    why', with a receipt for every claim. This is the operator's answer to
    'why was she slow?' without an hour of grep.
    """
    try:
        from core.observability.incident_narrator import get_incident_narrator

        minutes = max(1.0, min(float(minutes), 24 * 60.0))
        report = await asyncio.to_thread(get_incident_narrator().narrate, minutes)
        return JSONResponse(_json_safe(report))
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "system",
            exc,
            action="returned empty incident narrative after narrator failure",
        )
        logger.warning("Incident narrative unavailable: %s", exc)
        return JSONResponse(
            {
                "schema": "aura.incident_narrative.v1",
                "episodes": [],
                "error": "incident narrative unavailable",
            },
            status_code=200,
        )


@router.get("/system/memory/growth", tags=["health"])
async def api_system_memory_growth(request: Request, top: int = 25):
    """Allocation-growth attribution for the idle-leak investigation.

    Requires a launch with AURA_RUNTIME_HYGIENE_TRACEMALLOC=1 (opt-in;
    ~2x allocation overhead). First call arms the baseline snapshot;
    later calls return the top-N call sites by size growth since the
    baseline — the direct answer to 'WHAT is growing', not just how much.
    """
    try:
        from core.runtime.runtime_hygiene import get_runtime_hygiene

        hygiene = get_runtime_hygiene()
        if not hasattr(hygiene, "allocation_growth"):
            return JSONResponse(
                {"available": False, "reason": "runtime_hygiene_unavailable"},
                status_code=200,
            )
        report = await asyncio.to_thread(hygiene.allocation_growth, top)
        return JSONResponse(_json_safe(report))
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "system",
            exc,
            action="returned unavailable memory-growth report after hygiene failure",
        )
        return JSONResponse(
            {"available": False, "reason": "memory_growth_failed"},
            status_code=200,
        )


@router.get("/system/learning", tags=["health"])
async def api_system_learning(request: Request):
    """The weight-learning stack's live state, receipts included.

    One view over the whole loop: the compounding scheduler (when it last
    trained, what happened), the self-play flywheel (practice bursts,
    correct-rate trace, pairs produced), the lineage ledger's verdict (the
    only place a compounding claim may come from), and the expert-adapter
    library. This is the operator's answer to 'what has she learned lately?'
    """
    payload: dict = {"schema": "aura.learning_status.v1"}
    try:
        from core.container import ServiceContainer

        def _collect() -> dict:
            out: dict = {}
            scheduler = ServiceContainer.get("weight_compounding", default=None)
            if scheduler is not None and hasattr(scheduler, "get_status"):
                out["compounding"] = scheduler.get_status()
            flywheel = ServiceContainer.get("selfplay_flywheel", default=None)
            if flywheel is not None and hasattr(flywheel, "get_status"):
                out["selfplay"] = flywheel.get_status()
            try:
                from core.learning.verifiable_preference_harness import (
                    get_verifiable_preference_harness,
                )

                out["preference_store"] = get_verifiable_preference_harness().stats()
            except _SYSTEM_RECOVERABLE_ERRORS:
                out["preference_store"] = {"error": "unavailable"}
            try:
                from core.brain.expert_lora_library import get_expert_lora_library

                out["expert_library"] = get_expert_lora_library().stats()
            except _SYSTEM_RECOVERABLE_ERRORS:
                out["expert_library"] = {"error": "unavailable"}
            try:
                from core.runtime.service_access import resolve_practice_director

                director = resolve_practice_director(default=None)
                if director is not None and hasattr(director, "get_status"):
                    out["practice_director"] = director.get_status()
            except _SYSTEM_RECOVERABLE_ERRORS:
                out["practice_director"] = {"error": "unavailable"}
            return out

        payload.update(await asyncio.to_thread(_collect))
        return JSONResponse(_json_safe(payload))
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation(
            "system",
            exc,
            action="returned degraded learning status after collection failure",
        )
        logger.warning("Learning status unavailable: %s", exc)
        payload["error"] = "learning status unavailable"
        return JSONResponse(_json_safe(payload), status_code=200)


@router.get("/healthz", tags=["health"])
async def healthz(request: Request):
    """Liveness probe: is the process alive and responsive?

    Returns 200 if the server can respond to HTTP at all.
    Used by orchestrators (systemd, launchd, docker) to detect crashes.
    """
    try:
        from core.observability.metrics import check_liveness

        result = check_liveness()
        if is_shutdown_requested():
            result = dict(result)
            result["status"] = "stopping"
            result["shutdown"] = _shutdown_health_status()
        return JSONResponse(result, status_code=200)
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.warning("Liveness check degraded; returning process-level alive response: %s", exc)
        payload: dict[str, Any] = {"status": "alive", "pid": os.getpid()}
        if is_shutdown_requested():
            payload["status"] = "stopping"
            payload["shutdown"] = _shutdown_health_status()
        return JSONResponse(payload, status_code=200)


@router.get("/readyz", tags=["health"])
async def readyz(request: Request):
    """Readiness probe: can Aura accept and process requests?

    Returns 200 if ready, 503 if not. Checks:
    - Last tick completed recently
    - Substrate state is finite
    - Database is accessible
    """
    if is_shutdown_requested():
        return JSONResponse(
            {
                "status": "stopping",
                "ready": False,
                "issues": ["runtime_shutdown"],
                "shutdown": _shutdown_health_status(),
            },
            status_code=503,
        )
    try:
        from core.observability.metrics import check_readiness

        result = check_readiness()
        status_code = 200 if result.get("ready", False) else 503
        return JSONResponse(result, status_code=status_code)
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        return JSONResponse(
            {"status": "not_ready", "ready": False, "issues": [str(e)]},
            status_code=503,
        )


@router.get("/incidents", tags=["observability"])
async def incidents(request: Request):
    """Active incidents and incident manager summary."""
    _require_internal(request)
    try:
        from core.resilience.incident_manager import get_incident_manager

        manager = get_incident_manager()
        return JSONResponse({
            "summary": manager.get_summary(),
            "active": manager.get_active(),
        })
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        return JSONResponse(
            {"summary": {}, "active": [], "error": str(e)},
            status_code=200,
        )


@router.get("/db-maintenance", tags=["observability"])
async def db_maintenance_status(request: Request):
    """Database maintenance status and last run results."""
    _require_internal(request)
    try:
        from core.persistence.db_maintenance import get_db_maintenance
        return JSONResponse(get_db_maintenance().get_status())
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        return JSONResponse({"error": str(e)}, status_code=200)


@router.get("/resources", tags=["observability"])
async def resource_status(request: Request):
    """Resource governor status: thermal, memory, inference."""
    _require_internal(request)
    try:
        from core.resource.resource_governor import get_resource_governor
        return JSONResponse(get_resource_governor().get_status())
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        return JSONResponse({"error": str(e)}, status_code=200)


@router.get("/initiative-overflow", tags=["observability"])
async def initiative_overflow_status(request: Request):
    """Initiative overflow and skill gap status."""
    _require_internal(request)
    try:
        from core.autonomy.initiative_overflow import get_initiative_overflow
        return JSONResponse(get_initiative_overflow().get_status())
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        return JSONResponse({"error": str(e)}, status_code=200)


@router.get("/user-engagement", tags=["observability"])
async def user_engagement_status(request: Request):
    """User response tracking and engagement metrics."""
    _require_internal(request)
    try:
        from core.autonomy.user_response_tracker import get_user_response_tracker
        return JSONResponse(get_user_response_tracker().get_status())
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        return JSONResponse({"error": str(e)}, status_code=200)


@router.get("/gemini-usage")
async def gemini_usage(request: Request):
    """Return daily Gemini API usage stats."""
    _require_internal(request)
    try:
        from core.brain.llm.gemini_adapter import DailyRateLimiter
        orch = ServiceContainer.get("orchestrator", default=None)
        if orch and hasattr(orch, 'cognitive_engine'):
            brain = getattr(orch.cognitive_engine, 'brain', None) or getattr(orch.cognitive_engine, '_brain', None)
            if brain and hasattr(brain, 'llm_router'):
                for _name, adapter in brain.llm_router.adapters.items():
                    if hasattr(adapter, 'rate_limiter'):
                        return JSONResponse(adapter.rate_limiter.get_usage())
        from core.config import config
        state_path = str(config.paths.data_dir / "gemini_rate_state.json")
        limiter = DailyRateLimiter(state_path=state_path)
        return JSONResponse(limiter.get_usage())
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/health")
async def api_health(request: Request):
    _mark_runtime_service_progress("api.health")
    try:
        from interface.routes.privacy import get_voice_engine_fn

        _voice_engine_fn = get_voice_engine_fn()
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation("system", e)
        logger.debug("Voice engine resolver unavailable for health payload: %s", e)
        _voice_engine_fn = None

    _restore_owner_session_from_request(request)
    orch       = ServiceContainer.get("orchestrator", default=None)
    rt         = _get_runtime_state_safe()
    runtime_payload = rt.get("state", {}) if isinstance(rt.get("state"), dict) else {}
    status_obj = getattr(orch, "status", None)

    initialized = getattr(status_obj, "initialized", False)
    connected   = orch is not None and getattr(status_obj, "running", False)

    try:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        p_core = per_cpu[0] if len(per_cpu) > 1 else cpu
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Hardware stats collection failed: %s", e)
        cpu, ram, p_core = 0, 0, 0

    orch_status = {}
    if orch and hasattr(orch, "get_status"):
        try:
            orch_status = orch.get_status()
        except _SYSTEM_RECOVERABLE_ERRORS as e:
            record_degradation('system', e)
            logger.debug("get_status failed: %s", e)
    conversation_lane = _collect_conversation_lane_status_resilient()
    boot_snapshot, _ = build_boot_health_snapshot(
        orch,
        rt,
        is_gui_proxy=os.environ.get("AURA_GUI_PROXY") == "1",
        conversation_lane=conversation_lane,
    )
    connected = bool(
        boot_snapshot.get("system_ready", False)
        or (
            boot_snapshot.get("ready", False)
            and boot_snapshot.get("conversation_ready", False)
        )
    )

    ls_data = {}
    try:
        ls = ServiceContainer.get("liquid_substrate", default=None) or ServiceContainer.get("liquid_state", default=None)
        if ls and hasattr(ls, "get_status"):
            ls_data = ls.get_status()

        vad_data = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0, "_stale": True}
        engine = ServiceContainer.get("cognitive_engine", default=None)
        if engine and hasattr(engine, "consciousness"):
            v_state = await asyncio.wait_for(
                engine.consciousness.substrate.get_state_summary(),
                timeout=0.25,
            )
            vad_data = {
                "valence": v_state.get("valence", 0.0),
                "arousal": v_state.get("arousal", 0.0),
                "dominance": v_state.get("dominance", 0.0),
                "volatility": v_state.get("volatility", 0.0),
                "_stale": False,
            }
            ls_dict = cast(dict, ls_data)
            ls_dict["vad"] = vad_data
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Liquid state/VAD lookup failed: %s", e)
    curiosity_status = orch_status.get("curiosity_status", {})

    transcendence_data = {"meta_evolution": {"active": False, "acceleration_factor": 1.0}}
    try:
        meta = ServiceContainer.get("meta_cognition", default=None)
        if meta:
            transcendence_data["meta_evolution"] = meta.get_health()
            transcendence_data["meta_evolution"]["active"] = True
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Transcendence status collection failed: %s", e)

    # Agency: derive from energy + curiosity + active autonomous thought.
    _energy_raw = _normalize_percentish(ls_data.get("energy")) or 0.0
    _curiosity_raw = _normalize_percentish(ls_data.get("curiosity")) or 0.0
    thought_task = getattr(orch, "_current_thought_task", None) if orch else None
    try:
        _thinking = bool(thought_task and hasattr(thought_task, "done") and not thought_task.done())
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation("system", e)
        logger.debug("Current thought task status failed: %s", e)
        _thinking = False
    _agency_score = (_energy_raw * 0.4 + _curiosity_raw * 0.4 + (30.0 if _thinking else 0.0))
    _agency_score = min(100.0, max(0.0, _agency_score))

    scratchpad_engine = ServiceContainer.get("scratchpad_engine", default=None)
    subconscious_loop = ServiceContainer.get("subconscious_loop", default=None)
    subconscious_active = bool(
        subconscious_loop is not None
        and getattr(subconscious_loop, "_running", False)
    )

    cortex = {
        "agency":    float(int(_agency_score * 10)) / 10.0,
        "curiosity": float(int(_curiosity_raw * 10)) / 10.0,
        "fixes":     orch_status.get("stats", {}).get("modifications_made", 0),
        "beliefs":   0,
        "episodes":  0,
        "active_topic": curiosity_status.get("active_topic", "None"),
        "goals":     orch_status.get("stats", {}).get("goals_processed", 0),
        "autonomy":  config.security.aura_full_autonomy,
        "stealth":   config.security.enable_stealth_mode,
        "scratchpad": scratchpad_engine is not None,
        "forge":      ServiceContainer.get("hephaestus_engine", default=None) is not None,
        "subconscious": "dreaming" if subconscious_active and _safe_float(getattr(orch, "boredom", 0), 0.0) > 45 else ("awake" if subconscious_active else "idle"),
        "unity":      ServiceContainer.get("soma", default=None) is not None,
        "p_core_usage": float(int(_safe_float(p_core) * 10)) / 10.0,
        "singularity_factor": float(int(_safe_float(transcendence_data.get("meta_evolution", {}).get("acceleration_factor"), 1.0) * 100)) / 100.0,
        "meta_loop_active": transcendence_data.get("meta_evolution", {}).get("active", False)
    }

    if config.security.force_unity_on:
        cortex["unity"] = True
    try:
        if orch and hasattr(orch, "self_model") and orch.self_model:
            cortex["beliefs"] = len(getattr(orch.self_model, "beliefs", []))

        ep_mem = ServiceContainer.get("episodic_memory", default=None)
        if ep_mem and hasattr(ep_mem, "get_summary_cached"):
            # Off-loop + TTL-cached: the fresh get_summary() runs eight
            # aggregate queries and stalled the event loop for 5.1s live.
            ep_summary = await asyncio.to_thread(ep_mem.get_summary_cached)
            cortex["episodes"] = ep_summary.get("total_episodes", 0)
        elif ep_mem and hasattr(ep_mem, "get_summary"):
            ep_summary = await asyncio.to_thread(ep_mem.get_summary)
            cortex["episodes"] = ep_summary.get("total_episodes", 0)
        else:
            mem_mgr = ServiceContainer.get("memory_manager", default=None)
            if mem_mgr and hasattr(mem_mgr, "get_stats"):
                mem_stats = mem_mgr.get_stats()
                cortex["episodes"] = mem_stats.get("episodic_count", 0)
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Cortex supplementary metrics failed: %s", e)

    moral_data = {}
    try:
        moral = ServiceContainer.get("moral", default=None)
        moral_data = moral.get_health() if moral and hasattr(moral, "get_health") else {}
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation("system", e)
        logger.debug("Moral health collection failed: %s", e)

    homeo_data = {}
    try:
        homeostasis = ServiceContainer.get("homeostasis", default=None)
        homeo_data = homeostasis.get_health() if homeostasis and hasattr(homeostasis, "get_health") else {}
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation("system", e)
        logger.debug("Homeostasis health collection failed: %s", e)
    homeostasis_payload = _collect_homeostasis_public_payload(
        homeo_data if isinstance(homeo_data, dict) else {}
    )
    liquid_state_payload = _collect_liquid_state_payload(
        cast(dict[str, Any], ls_data if isinstance(ls_data, dict) else {}),
        runtime_state=runtime_payload if isinstance(runtime_payload, dict) else {},
        homeostasis_data=homeostasis_payload,
    )
    soma_data = await _collect_soma_payload()

    social_data = {"depth": 0.0}
    try:
        social = ServiceContainer.get("social", default=None)
        social_data = social.get_health() if social and hasattr(social, "get_health") else social_data
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation("system", e)
        logger.debug("Social health collection failed: %s", e)

    swarm_data = {"active_count": 0}
    try:
        swarm_data = orch.swarm_status if orch and hasattr(orch, 'swarm_status') else swarm_data
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation("system", e)
        logger.debug("Swarm status collection failed: %s", e)

    executive_closure_data = {}
    try:
        executive_closure_data = orch_status.get("executive_closure", {}) or {}
        if not executive_closure_data:
            executive_closure = ServiceContainer.get("executive_closure", default=None)
            if executive_closure and hasattr(executive_closure, "get_status"):
                executive_closure_data = executive_closure.get_status()
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Executive closure status collection failed: %s", e)

    consciousness_evidence = {}
    try:
        consciousness_evidence = orch_status.get("consciousness_evidence", {}) or {}
        if not consciousness_evidence:
            evidence = ServiceContainer.get("consciousness_evidence", default=None)
            if evidence and hasattr(evidence, "snapshot"):
                consciousness_evidence = evidence.snapshot()
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Consciousness evidence collection failed: %s", e)

    executive_authority_data = {}
    try:
        executive_authority = ServiceContainer.get("executive_authority", default=None)
        if executive_authority and hasattr(executive_authority, "get_status"):
            executive_authority_data = executive_authority.get_status()
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Executive authority status collection failed: %s", e)

    interaction_signals_data = {}
    try:
        interaction_signals = ServiceContainer.get("interaction_signals", default=None)
        if interaction_signals and hasattr(interaction_signals, "get_status"):
            interaction_signals_data = interaction_signals.get_status()
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Interaction signal status collection failed: %s", e)

    # ── Resilience Status ──
    resilience_data: dict[str, Any] = {"circuit_breakers": {}, "snapshot": "unknown", "llm_tier": "unknown"}
    try:
        voice = ServiceContainer.get("voice_engine", default=None)
        if voice:
            for attr_name in ("_stt_breaker", "_tts_breaker"):
                breaker = getattr(voice, attr_name, None)
                if breaker and hasattr(breaker, "state"):
                    cast(dict[str, Any], resilience_data["circuit_breakers"])[breaker.name] = breaker.state.value

        cog = ServiceContainer.get("cognitive_engine", default=None)
        if cog:
            for attr_name in dir(cog):
                obj = getattr(cog, attr_name, None)
                if obj and hasattr(obj, "state") and hasattr(obj, "name") and hasattr(obj.state, "value"):
                    if "breaker" in attr_name.lower():
                        cast(dict[str, Any], resilience_data["circuit_breakers"])[obj.name] = obj.state.value

        snap_mgr = ServiceContainer.get("snapshot_manager", default=None)
        if snap_mgr and hasattr(snap_mgr, "snapshot_file"):
            resilience_data["snapshot"] = "saved" if snap_mgr.snapshot_file.exists() else "none"

        llm_router = ServiceContainer.get("llm_router", default=None)
        tier_value = conversation_lane.get("foreground_tier")
        if llm_router and hasattr(llm_router, "get_health_report"):
            report = llm_router.get_health_report()
            tier_value = report.get("foreground_tier") or tier_value
        if not tier_value and cog:
            tier_value = (getattr(cog, "_current_tier", None)
                          or getattr(cog, "last_tier", None))
        if tier_value:
            resilience_data["llm_tier"] = str(tier_value)
        else:
            if llm_router and hasattr(llm_router, "_active_model"):
                model = str(getattr(llm_router, "_active_model", "") or "")
                resilience_data["llm_tier"] = "local" if "mlx" in model.lower() or "local" in model.lower() else "cloud"

        resilience_data["active_endpoint"] = conversation_lane.get("foreground_endpoint")
        resilience_data["background_endpoint"] = conversation_lane.get("background_endpoint")
        resilience_data["conversation_lane"] = conversation_lane
        if llm_router:
            if hasattr(llm_router, "endpoints"):
                ep_status = {}
                for name, ep in llm_router.endpoints.items():
                    ep_status[name] = {
                        "tier": getattr(ep, "tier", "unknown"),
                        "available": ep.is_available() if hasattr(ep, "is_available") else True,
                        "state": ep.state.value if hasattr(ep, "state") and hasattr(ep.state, "value") else "unknown",
                    }
                resilience_data["llm_endpoints"] = ep_status

        resilience_data["hardening_active"] = ServiceContainer.get("stability_guardian", default=None) is not None
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Resilience status collection failed: %s", e)

    # ── Qualia Status ──
    qualia_data: dict[str, Any] = {"pri": 0.0, "q_norm": 0.0, "dominant_dim": "none", "in_attractor": False, "_stale": True}
    try:
        qualia = ServiceContainer.get("qualia_synthesizer", default=None)
        if not qualia and orch:
            qualia = getattr(orch, "qualia", None)
        if qualia:
            qualia_data["_stale"] = False
            qualia_data["pri"] = round(float(getattr(qualia, "pri", 0.0)), 4)
            qualia_data["q_norm"] = round(float(getattr(qualia, "q_norm", 0.0)), 4)
            qualia_data["dominant_dim"] = getattr(qualia, "_history", None) and len(qualia._history) > 0 and qualia._history[-1].dominant_dimension or "none"
            qualia_data["in_attractor"] = getattr(qualia, "_in_attractor", False)
            qualia_data["identity_coherence"] = round(float(getattr(qualia, "identity_drift_score", 1.0)) * 100, 1)
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Qualia status collection failed: %s", e)

    # ── Mycelial Network Status ──
    mycelial_data: dict[str, Any] = {"nodes": 0, "edges": 0, "health": "offline"}
    try:
        mycelium = ServiceContainer.get("mycelial_network", default=None)
        if mycelium:
            if hasattr(mycelium, "pathways") and hasattr(mycelium, "hyphae"):
                mycelial_data["nodes"] = len(mycelium.pathways)
                mycelial_data["edges"] = len(mycelium.hyphae)
            mycelial_data["health"] = "online"
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Mycelial network status collection failed: %s", e)

    # ── PNEUMA Engine Status ──
    pneuma_data: dict[str, Any] = {"temperature": 0.7, "arousal": 0.0, "stability": 0.0,
                   "attractor_count": 0, "efe_score": 0.0, "online": False, "_stale": True}
    try:
        from core.pneuma.pneuma import get_pneuma
        pn = get_pneuma()
        if pn:
            runtime_state = pn.get_state_dict()
            pneuma_data["online"] = bool(runtime_state.get("online", False))
            pneuma_data["_stale"] = not bool(runtime_state.get("online", False))
            pneuma_data["temperature"] = round(pn.get_llm_temperature(), 3)
            pe = getattr(pn, "precision", None)
            if pe and hasattr(pe, "fhn"):
                s = pe.fhn.state
                pneuma_data["arousal"] = round(float(s.v), 3)
                pneuma_data["stability"] = round(float(s.w), 3)
            tm = getattr(pn, "topo_memory", None)
            if tm:
                pneuma_data["attractor_count"] = int(tm.attractor_count)
            pneuma_data["tick_count"] = runtime_state.get("tick_count", 0)
            pneuma_data["last_tick"] = runtime_state.get("last_tick", 0.0)
            pneuma_data["loop_errors"] = runtime_state.get("loop_errors", 0)
            pneuma_data["compute_budget"] = runtime_state.get("compute_budget", {})
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("PNEUMA status collection failed: %s", e)

    # ── MHAF Field Status ──
    mhaf_data: dict[str, Any] = {"phi": 0.0, "nodes": 0, "edges": 0, "free_energy": 0.0,
                 "lexicon_size": 0, "online": False, "_stale": True}
    try:
        from core.consciousness.mhaf_field import get_mhaf
        mhaf = get_mhaf()
        if mhaf:
            runtime_state = mhaf.get_state_dict()
            mhaf_data["online"] = bool(runtime_state.get("online", False))
            mhaf_data["_stale"] = not bool(runtime_state.get("online", False))
            mhaf_data["nodes"] = len(mhaf._nodes)
            mhaf_data["edges"] = len(mhaf._edges)
            mhaf_data["free_energy"] = round(float(mhaf._free_energy), 4)
            mhaf_data["tick_count"] = runtime_state.get("tick_count", 0)
            mhaf_data["last_tick"] = runtime_state.get("last_tick", 0.0)
            mhaf_data["loop_errors"] = runtime_state.get("loop_errors", 0)
            mhaf_data["compute_budget"] = runtime_state.get("compute_budget", {})
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("MHAF status collection failed: %s", e)
    # Wire real PhiCore IIT 4.0 phi into the MHAF data (replaces the surrogate)
    try:
        phi_core = ServiceContainer.get("phi_core", default=None)
        if phi_core is not None:
            result = phi_core._last_result
            live_phi = 0.0
            if hasattr(phi_core, "get_live_phi"):
                live_phi = float(phi_core.get_live_phi(include_surrogate=True))
            if live_phi > 0.0:
                mhaf_data["phi"] = round(live_phi, 4)
                mhaf_data["phi_source"] = "phi_s" if result is not None else "surrogate"
            if result is not None:
                mhaf_data["phi"] = round(float(result.phi_s), 4)
                mhaf_data["phi_complex"] = result.is_complex
                mhaf_data["phi_mip"] = result.mip_description
                mhaf_data["phi_samples"] = result.tpm_n_samples
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("PhiCore status collection failed: %s", e)
    if mhaf_data.get("phi", 0.0) <= 0.0:
        try:
            closed_loop = ServiceContainer.get("closed_causal_loop", default=None)
            if closed_loop is not None and hasattr(closed_loop, "get_status"):
                closed_loop_phi = float(
                    ((closed_loop.get_status() or {}).get("phi") or {}).get("estimate") or 0.0
                )
                if closed_loop_phi > 0.0:
                    mhaf_data["phi"] = round(closed_loop_phi, 4)
                    mhaf_data["phi_source"] = "closed_loop"
        except _SYSTEM_RECOVERABLE_ERRORS as e:
            record_degradation('system', e)
            logger.debug("Closed-loop phi fallback failed: %s", e)
    try:
        from core.consciousness.neologism_engine import get_neologism_engine
        neo = get_neologism_engine()
        if neo:
            mhaf_data["lexicon_size"] = len(neo._lexicon)
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Neologism lexicon count failed: %s", e)

    # ── Security Status ──
    security_data: dict[str, Any] = {
        "trust_level": "unknown", "threat_score": 0.0,
        "integrity_ok": True, "passphrase_set": False, "_stale": True,
    }
    try:
        from core.security.trust_engine import get_trust_engine
        te = get_trust_engine()
        ts = te.get_status()
        security_data["trust_level"] = ts.get("level", "guest")
        security_data["_stale"] = False
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Security status collection failed: %s", e)
    try:
        from core.security.emergency_protocol import get_emergency_protocol
        ep = get_emergency_protocol()
        eps = ep.get_status()
        security_data["threat_score"] = eps.get("threat_score", 0.0)
        security_data["threat_level"] = eps.get("threat_level", "none")
    except _SYSTEM_RECOVERABLE_ERRORS as _exc:
        record_degradation('system', _exc)
        logger.debug("Emergency protocol status collection failed: %s", _exc)
    try:
        from core.security.integrity_guardian import get_integrity_guardian
        igs = get_integrity_guardian().get_status()
        security_data["integrity_ok"] = bool(
            igs.get("integrity_ok", igs.get("alert_count", 0) == 0)
        )
        security_data["integrity_files"] = igs.get("manifest_files", 0)
    except _SYSTEM_RECOVERABLE_ERRORS as _exc:
        record_degradation('system', _exc)
        logger.debug("Integrity guardian status collection failed: %s", _exc)
    try:
        from core.security.user_recognizer import get_user_recognizer
        security_data["passphrase_set"] = get_user_recognizer().has_passphrase()
    except _SYSTEM_RECOVERABLE_ERRORS as _exc:
        record_degradation('system', _exc)
        logger.debug("User recognizer status collection failed: %s", _exc)

    # ── Circadian State ──
    circadian_data: dict[str, Any] = {}
    try:
        from core.senses.circadian import get_circadian
        ce = get_circadian()
        ce.update()
        s = ce.state
        circadian_data = {
            "phase": s.phase.value,
            "arousal_baseline": round(s.arousal_baseline, 3),
            "energy_modifier": round(s.energy_modifier, 3),
            "cognitive_mode": s.cognitive_mode,
        }
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Circadian status collection failed: %s", e)

    # ── Substrate Learning ──
    substrate_data: dict[str, Any] = {}
    try:
        from core.consciousness.crsm_lora_bridge import get_crsm_lora_bridge
        substrate_data["lora_bridge"] = await _optional_threaded_status(
            "crsm_lora_bridge",
            lambda: get_crsm_lora_bridge().get_status(),
            timeout_s=0.18,
            fallback={"loop": None},
        )
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("LoRA bridge status failed: %s", e)
    try:
        from core.consciousness.experience_consolidator import get_experience_consolidator
        substrate_data["consolidator"] = await _optional_threaded_status(
            "experience_consolidator",
            lambda: get_experience_consolidator().get_status(),
            timeout_s=0.18,
        )
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Consolidator status failed: %s", e)

    # ── Morphogenesis Status ──
    morphogenesis_data: dict[str, Any] = {"online": False, "cells": 0, "organs": 0, "_stale": True}
    try:
        morpho_rt = ServiceContainer.get("morphogenetic_runtime", default=None)
        if morpho_rt is not None and hasattr(morpho_rt, "status"):
            ms = morpho_rt.status()
            morphogenesis_data = {
                "online": ms.get("running", False),
                "enabled": ms.get("enabled", False),
                "tick": ms.get("tick", 0),
                "cells": ms.get("registry", {}).get("cells", 0),
                "organs": ms.get("registry", {}).get("organs", 0),
                "queued_signals": ms.get("queued_signals", 0),
                "last_tick_error": ms.get("last_tick_error", ""),
                "_stale": False,
            }
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Morphogenesis status collection failed: %s", e)

    # ── Terminal Fallback Status ──
    terminal_data: dict[str, Any] = {"active": False, "pending": 0, "watchdog": False}
    try:
        from core.conversation.terminal_chat import get_terminal_fallback, get_terminal_watchdog
        tf = get_terminal_fallback()
        terminal_data["active"] = tf.is_active
        terminal_data["pending"] = len(tf._pending)
        tw = get_terminal_watchdog()
        terminal_data["watchdog"] = tw._running if tw else False
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.debug("Terminal fallback status collection failed: %s", e)

    desktop_access_data = await _collect_desktop_access_summary(allow_probe=False)
    imagination_data = _collect_imagination_status()

    # ── Final Response Assembly ──
    try:
        voice_mod = _voice_engine_fn() if _voice_engine_fn else None
        smc_mod = ServiceContainer.get("sensory_motor_cortex", default=None)
        from interface.routes.privacy import get_browser_camera_privacy

        browser_camera_privacy = get_browser_camera_privacy()

        privacy_data = {
            "camera_enabled": bool(browser_camera_privacy.get("enabled", False)),
            "camera_mode": browser_camera_privacy.get("mode", "off"),
            "camera_reason": browser_camera_privacy.get("reason"),
            "continuous_camera_enabled": getattr(smc_mod, "camera_enabled", False),
            "microphone_enabled": getattr(voice_mod, "microphone_enabled", True),
            "microphone_listening": bool(
                getattr(voice_mod, "_mic_listening", False)
                or getattr(voice_mod, "is_listening", False)
            ),
            "speaking_enabled": getattr(voice_mod, "speaking_enabled", True),
        }

        full_runtime = _collect_full_runtime_status(pneuma_data, mhaf_data)

        conversation_ready = bool(conversation_lane.get("conversation_ready", False))
        conversation_busy = conversation_lane_is_busy(conversation_lane)
        lane_is_standby = _conversation_lane_is_standby_resilient(conversation_lane)
        service_ok = bool(boot_snapshot.get("system_ready", False))
        required_probes = boot_snapshot.get("required_probes", {})
        probe_blockers = required_probe_blockers(required_probes)
        required_probes_ok = required_probe_groups_pass(required_probes)
        health_blockers = list(dict.fromkeys(
            [str(item) for item in (boot_snapshot.get("blockers", []) or []) if str(item)]
            + probe_blockers
        ))
        health_blockers = _normalize_conversation_health_blockers(
            health_blockers,
            conversation_ready=conversation_ready,
            conversation_busy=conversation_busy,
        )
        if full_runtime.get("full_runtime_expected") and not full_runtime.get("ready"):
            health_blockers.extend(
                f"full_runtime:{name}"
                for name in full_runtime.get("blockers", [])
            )
            health_blockers = list(dict.fromkeys(health_blockers))
        healthy_ready = bool(
            service_ok
            and required_probes_ok
            and conversation_ready
            and not health_blockers
        )
        integrity_report = _collect_runtime_integrity_report()
        integrity_payload = _runtime_integrity_public_payload(integrity_report)
        proof_readiness_healthy = bool(integrity_payload.get("proof_readiness", False))
        certification_ready = bool(healthy_ready and proof_readiness_healthy)
        diagnostics_data = {
            "stability_guardian": _collect_stability_details(),
            "recent_degraded_events": _collect_recent_degraded_events(),
        }

        health_status = (
            "ok"
            if healthy_ready else
            "standby"
            if service_ok and lane_is_standby else
            "unavailable"
            if service_ok and str(conversation_lane.get("state", "") or "").lower() == "failed" else
            "recovering"
            if service_ok and str(conversation_lane.get("state", "") or "").lower() == "recovering" else
            "working"
            if service_ok and conversation_busy else
            "warming"
            if service_ok and not conversation_ready else
            "booting"
        )

        payload = {
            "status":      health_status,
            "healthy":     healthy_ready,
            "version":     version_string("full"),
            "connected":   connected,
            "initialized": initialized,
            "cycle_count": orch_status.get("cycle_count", getattr(status_obj, "cycle_count", 0)),
            "uptime":      round(float(time.time() - (getattr(status_obj, "start_time", None) or getattr(orch, "start_time", None) or time.time())), 1),
            "cpu_usage":   cpu,
            "ram_usage":   ram,
            "cortex":      cortex,
            "liquid_state": liquid_state_payload,
            "soma":        soma_data,
            "moral":       moral_data,
            "homeostasis": homeostasis_payload,
            "social":      social_data,
            "swarm":       swarm_data,
            "resilience":  resilience_data,
            "qualia":         qualia_data,
            "mycelial":       mycelial_data,
            "pneuma":         pneuma_data,
            "mhaf":           mhaf_data,
            "security":       security_data,
            "circadian":      circadian_data,
            "substrate":      substrate_data,
            "morphogenesis":  morphogenesis_data,
            "terminal":       terminal_data,
            "desktop_access": desktop_access_data,
            "imagination":    imagination_data,
            "transcendence": transcendence_data,
            "privacy":        privacy_data,
            "executive_closure": executive_closure_data,
            "consciousness_evidence": consciousness_evidence,
            "executive_authority": executive_authority_data,
            "interaction_signals": interaction_signals_data,
            "integrity": integrity_payload,
            "full_runtime": full_runtime,
            "full_runtime_ready": bool(full_runtime.get("ready")),
            "proof_readiness_healthy": proof_readiness_healthy,
            "certification_ready": certification_ready,
            "integrity_blockers": integrity_payload.get("proof_blockers", []),
            "conversation_lane": conversation_lane,
            "diagnostics": diagnostics_data,
            "readiness_contract": {
                "healthy": healthy_ready,
                "system_ready": service_ok,
                "conversation_ready": conversation_ready,
                "conversation_busy": conversation_busy,
                "runtime_probe_healthy": required_probes_ok,
                "full_runtime_ready": bool(full_runtime.get("ready")),
                "full_runtime": full_runtime,
                "proof_readiness_healthy": proof_readiness_healthy,
                "certification_ready": certification_ready,
                "integrity": integrity_payload,
                "integrity_blockers": integrity_payload.get("proof_blockers", []),
                "required_probes": required_probes,
                "blockers": health_blockers,
            },
            "runtime_probe_healthy": required_probes_ok,
            "conversation_ready": conversation_ready,
            "conversation_busy": conversation_busy,
            "required_probes": required_probes,
            "blockers": health_blockers,
            "runtime":        rt,
            "scheduler":      scheduler.get_health(),
            "boot":           boot_snapshot,
            "timestamp":      datetime.now(tz=UTC).isoformat(),
        }
    except _SYSTEM_RECOVERABLE_ERRORS as e:
        record_degradation('system', e)
        logger.error("Final health payload assembly failed: %s", e)
        payload = {
            "status": "degraded",
            "error": str(e),
            "version": version_string("full"),
            "uptime": 0.0,
            "cycle_count": 0,
            "cpu_usage": 0,
            "ram_usage": 0,
            "timestamp": datetime.now(tz=UTC).isoformat()
        }

    shutdown = _shutdown_health_status()
    shutdown_request = shutdown.get("request")
    if isinstance(shutdown_request, dict) and shutdown_request.get("requested") is True:
        payload["status"] = "stopping"
        payload["healthy"] = False
        payload["connected"] = False
        payload["conversation_ready"] = False
        payload["runtime_probe_healthy"] = False
        payload["certification_ready"] = False
        payload["shutdown"] = shutdown
        required_probe_payload = payload.get("required_probes")
        if isinstance(required_probe_payload, dict):
            required_probe_payload["all_passed"] = False
        blockers = [str(item) for item in payload.get("blockers", [])]
        if "runtime_shutdown" not in blockers:
            blockers.insert(0, "runtime_shutdown")
        payload["blockers"] = blockers
        readiness = payload.get("readiness_contract")
        if isinstance(readiness, dict):
            readiness["healthy"] = False
            readiness["system_ready"] = False
            readiness["conversation_ready"] = False
            readiness["runtime_probe_healthy"] = False
            readiness["certification_ready"] = False
            readiness["blockers"] = blockers
    else:
        payload["shutdown"] = shutdown

    return JSONResponse(_json_safe(payload))


@router.get("/tools/catalog")
async def api_tools_catalog():
    catalog = _collect_tool_catalog()
    engine = optional_service("capability_engine", default=None)
    health = (
        engine.get_catalog_health()
        if engine is not None and hasattr(engine, "get_catalog_health")
        else {"ready": False, "reason": "capability_engine_unavailable"}
    )
    return JSONResponse({"tools": catalog, "count": len(catalog), "health": health})


@router.get("/ui/bootstrap")
async def api_ui_bootstrap(request: Request = None):
    _mark_runtime_service_progress("api.ui.bootstrap")
    _restore_owner_session_from_request(request)
    access_profile = request_access_profile(request)
    conversation_only = bool(access_profile.get("conversation_only", True))
    orch = ServiceContainer.get("orchestrator", default=None)
    rt = _get_runtime_state_safe()
    constitutional_status = {}
    executive_status = {}
    state_summary = {
        "current_objective": "",
        "pending_initiatives": 0,
        "active_goals": 0,
        "policy_mode": "unknown",
        "health": {},
        "rolling_summary": "",
        "coherence_score": 1.0,
        "fragmentation_score": 0.0,
        "contradiction_count": 0,
        "phenomenal_state": "",
        "thermal_guard": False,
        "health_flags": [],
        "epistemics": {},
    }

    try:
        from core.constitution import get_constitutional_core

        constitutional_core = get_constitutional_core(orch)
        constitutional_status = constitutional_core.get_status()
        state_summary = constitutional_core.snapshot()
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Bootstrap constitutional snapshot failed: %s", exc)

    try:
        executive_authority = ServiceContainer.get("executive_authority", default=None)
        if executive_authority and hasattr(executive_authority, "get_status"):
            executive_status = executive_authority.get_status()
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Bootstrap executive snapshot failed: %s", exc)

    interaction_signals_data = {}
    try:
        interaction_signals = ServiceContainer.get("interaction_signals", default=None)
        if interaction_signals and hasattr(interaction_signals, "get_status"):
            interaction_signals_data = interaction_signals.get_status()
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.debug("Bootstrap interaction signal snapshot failed: %s", exc)

    tool_catalog = _collect_tool_catalog()
    conversation_lane = _collect_conversation_lane_status_resilient()
    boot_snapshot, _status_code = build_boot_health_snapshot(
        orch,
        rt,
        is_gui_proxy=os.environ.get("AURA_GUI_PROXY") == "1",
        conversation_lane=conversation_lane,
    )
    status_obj = getattr(orch, "status", None)
    recent_conversation: list[dict[str, Any]] = []
    try:
        from interface.routes.chat import _conversation_log, _conversation_log_lock

        async with _conversation_log_lock:
            recent_conversation = list(_conversation_log)[-40:]
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Bootstrap conversation log snapshot failed: %s", exc)
    if conversation_only:
        session_id = paired_device_session_id(request)
        recent_conversation = [
            entry
            for entry in recent_conversation
            if session_id
            and str(entry.get("session_id") or "") == session_id
        ]

    static_dir = config.paths.project_root / "interface" / "static"
    shell_dist_dir = static_dir / "shell" / "dist"
    legacy_ui_index = static_dir / "index.html"

    legacy_ui_status = {
        "shell": "legacy_shell" if legacy_ui_index.exists() else "react_shell",
        "legacy_fallback_available": legacy_ui_index.exists(),
        "experimental_shell_available": (shell_dist_dir / "index.html").exists(),
        "experimental_shell_enabled": os.environ.get("AURA_ENABLE_REACT_SHELL", "").strip().lower()
        in {"1", "true", "yes", "on"},
    }
    legacy_ui_status["canonical_shell"] = (
        "legacy_shell"
        if legacy_ui_index.exists() and not legacy_ui_status["experimental_shell_enabled"]
        else "react_shell"
    )
    shell_status_helper = globals().get("_collect_legacy_shell_status")
    if callable(shell_status_helper):
        try:
            helper_payload = shell_status_helper() or {}
            if isinstance(helper_payload, dict):
                legacy_ui_status.update(helper_payload)
        except _SYSTEM_RECOVERABLE_ERRORS as exc:
            record_degradation('system', exc)
            logger.debug("Bootstrap legacy shell status sync failed: %s", exc)

    try:
        bootstrap_cpu = psutil.cpu_percent(interval=None)
        bootstrap_ram = psutil.virtual_memory().percent
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Bootstrap telemetry resource sample failed: %s", exc)
        bootstrap_cpu = 0.0
        bootstrap_ram = 0.0

    payload = {
        "identity": {
            "name": "Aura Luna",
            "version": version_string("full"),
            "build": VERSION,
        },
        "session": {
            "connected": bool(
                boot_snapshot.get("system_ready", False)
                or (
                    boot_snapshot.get("ready", False)
                    and boot_snapshot.get("conversation_ready", False)
                )
            ),
            "initialized": bool(getattr(status_obj, "initialized", False)),
            "websocket_clients": ws_manager.count(),
            "is_gui_proxy": os.environ.get("AURA_GUI_PROXY") == "1",
        },
        "access": access_profile,
        "constitutional": constitutional_status,
        "executive": executive_status,
        "state": state_summary,
        "commitments": _collect_commitment_summary(),
        "tools": tool_catalog,
        "capabilities": _collect_runtime_capabilities(conversation_lane),
        "desktop_access": await _collect_desktop_access_summary(allow_probe=False),
        "conversation": {
            "recent": recent_conversation,
            "count": len(recent_conversation),
            "lane": conversation_lane,
        },
        "voice": _collect_voice_summary(),
        "interaction_signals": interaction_signals_data,
        "telemetry": {
            "cpu_usage": bootstrap_cpu,
            "ram_usage": bootstrap_ram,
            "runtime": rt,
            "boot": boot_snapshot,
        },
        "diagnostics": {
            "stability_guardian": _collect_stability_details(),
            "recent_degraded_events": _collect_recent_degraded_events(),
        },
        "ui": {
            "shell": legacy_ui_status.get("shell", "legacy_shell" if legacy_ui_index.exists() else "react_shell"),
            "legacy_fallback_available": bool(legacy_ui_status.get("legacy_fallback_available", legacy_ui_index.exists())),
            "experimental_shell_available": bool(legacy_ui_status.get("experimental_shell_available", (shell_dist_dir / "index.html").exists())),
            "experimental_shell_enabled": bool(legacy_ui_status.get("experimental_shell_enabled", False)),
            "canonical_shell": legacy_ui_status.get("canonical_shell", legacy_ui_status.get("shell", "legacy_shell")),
            "status_flags": _derive_ui_status_flags(
                state_summary=state_summary,
                executive_status=executive_status,
                boot_snapshot=boot_snapshot,
                tool_catalog=tool_catalog,
            ),
        },
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }
    if conversation_only:
        lane = payload["conversation"].get("lane") or {}
        public_lane = {
            key: lane.get(key)
            for key in (
                "state",
                "conversation_ready",
                "active_generation",
                "active_generations",
            )
            if key in lane
        }
        boot = payload["telemetry"].get("boot") or {}
        public_boot = {
            key: boot.get(key)
            for key in (
                "ready",
                "status",
                "system_ready",
                "conversation_ready",
                "progress",
            )
            if key in boot
        }
        public_flags = [
            flag
            for flag in payload["ui"].get("status_flags", [])
            if flag == "booting"
        ]
        payload.update(
            {
                "session": {
                    "connected": bool(payload["session"].get("connected", False)),
                    "surface": "paired_device",
                },
                "constitutional": {},
                "executive": {},
                "state": {},
                "commitments": {},
                "tools": [],
                "capabilities": {"conversation": True, "world_read": True},
                "desktop_access": {
                    "available": False,
                    "overall_status": "surface_not_authorized",
                },
                "voice": {"available": False, "state": "surface_not_authorized"},
                "interaction_signals": {},
                "telemetry": {"runtime": {}, "boot": public_boot},
                "diagnostics": {},
                "ui": {"status_flags": public_flags},
            }
        )
        payload["conversation"] = {
            "recent": recent_conversation,
            "count": len(recent_conversation),
            "lane": public_lane,
        }
    return JSONResponse(_json_safe(payload))


@router.post("/ui/shell-error")
async def api_ui_shell_error(payload: dict[str, Any] | None = _UI_SHELL_ERROR_BODY):
    """Record desktop shell render faults without blocking UI recovery."""
    safe_payload = _json_safe(payload if isinstance(payload, dict) else {})
    message = str(safe_payload.get("error") or "unknown shell render fault")[:500]
    logger.error("Aura desktop shell render fault: %s", message)
    try:
        await broadcast_bus.publish(
            {
                "kind": "log",
                "level": "error",
                "source": "Aura.Desktop.Shell",
                "message": f"Desktop shell render fault recovered: {message}",
                "payload": safe_payload,
                "event_ts": datetime.now(tz=UTC).isoformat(),
            },
            priority=0,
        )
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Shell error broadcast failed: %s", exc)
    return JSONResponse({"ok": True})


@router.get("/health/boot")
async def api_boot_health():
    _mark_runtime_service_progress("api.health.boot")
    payload, status_code = await _build_boot_health_payload_bounded(
        is_gui_proxy=os.environ.get("AURA_GUI_PROXY") == "1",
    )
    return JSONResponse(payload, status_code=status_code)


def _heartbeat_probe_blockers(required_probes: Any) -> list[str]:
    """Return blockers that make a readiness heartbeat unhealthy.

    A healthy heartbeat is a launch contract, not a process ping. It must have
    every required probe group and every group must report ok.
    """
    return required_probe_blockers(required_probes)


def _normalize_conversation_health_blockers(
    blockers: list[Any],
    *,
    conversation_ready: bool,
    conversation_busy: bool = False,
) -> list[str]:
    """Merge health blockers without preserving stale conversation failures."""
    normalized = [
        str(item)
        for item in (blockers or [])
        if str(item or "").strip()
    ]
    if conversation_ready or conversation_busy:
        normalized = [
            item
            for item in normalized
            if item != "conversation_ready"
            and not item.startswith("conversation_lane:")
            and not item.startswith("conversation_reason:")
        ]
    elif "conversation_ready" not in normalized:
        normalized.append("conversation_ready")
    return list(dict.fromkeys(normalized))


def _collect_runtime_integrity_report() -> dict[str, Any]:
    """Return the throttled proof/learning integrity audit.

    This is intentionally separated from launch readiness. CRSM/CAA learning
    debt should not masquerade as a clean proof state, but it should also not
    make the desktop shell refuse to open when kernel/inference/memory/tool
    probes are otherwise safe.
    """
    try:
        from core.runtime.integrity_audit import maybe_run

        report = maybe_run()
        if isinstance(report, dict):
            return report
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.debug("Runtime integrity audit unavailable: %s", exc)
    return {
        "healthy": False,
        "concerns": ["integrity_audit_unavailable"],
        "strict_mode": False,
        "degradations": {},
        "crsm_loop": {},
        "caa_readiness": {},
        "at": time.time(),
    }


def _runtime_integrity_blockers(report: dict[str, Any] | None) -> list[str]:
    if not isinstance(report, dict):
        return ["integrity_audit_unavailable"]
    concerns = [
        str(item).strip()
        for item in (report.get("concerns") or [])
        if str(item or "").strip()
    ]
    if bool(report.get("healthy", False)) and not concerns:
        return []
    return [f"integrity:{concern}" for concern in (concerns or ["integrity_unknown"])]


def _runtime_integrity_proof_blockers(report: dict[str, Any] | None) -> list[str]:
    if not isinstance(report, dict):
        return ["integrity:integrity_audit_unavailable"]
    blockers = list(_runtime_integrity_blockers(report))
    advisory = [
        str(item).strip()
        for item in (report.get("advisory") or [])
        if str(item or "").strip()
    ]
    blockers.extend(f"integrity:{item}" for item in advisory)
    return list(dict.fromkeys(blockers))


def _runtime_integrity_public_payload(report: dict[str, Any] | None) -> dict[str, Any]:
    report = report if isinstance(report, dict) else {}
    blockers = _runtime_integrity_blockers(report)
    proof_blockers = _runtime_integrity_proof_blockers(report)
    return {
        "healthy": not blockers,
        "status": "healthy" if not blockers else "degraded",
        "concerns": [
            str(item)
            for item in (report.get("concerns") or [])
            if str(item or "").strip()
        ],
        "advisory": [
            str(item)
            for item in (report.get("advisory") or [])
            if str(item or "").strip()
        ],
        "blockers": blockers,
        "proof_blockers": proof_blockers,
        "proof_readiness": not proof_blockers,
        "operational_blocking": bool(report.get("strict_mode", False)) and bool(blockers),
        "crsm_loop": report.get("crsm_loop") or {},
        "caa_readiness": report.get("caa_readiness") or {},
        "at": report.get("at"),
    }


@router.get("/health/mind_tick")
async def api_mind_tick_diagnostics():
    """Diagnostic: MindTick's internal liveness state — is the supervised loop
    running, tick_count, last successful/progress timestamps + their ages, the
    active tick stage, consecutive failures, liveness-repair count. Read-only.

    Built 2026-07-07 to pin the false-death → launcher-respawn loop: is_alive()
    flipping False at exactly 180s means the boot-grace branch fired, i.e. BOTH
    progress timestamps are still 0 — the loop body never marked progress. This
    surfaces whether the loop is running at all and where it is stuck.
    """
    import time as _t

    mt = ServiceContainer.get("mind_tick", default=None)
    if mt is None or not hasattr(mt, "get_health_status"):
        return JSONResponse({"error": "mind_tick unavailable"}, status_code=503)
    try:
        status = dict(mt.get_health_status())
    except _SYSTEM_RECOVERABLE_ERRORS as exc:  # diagnostic must never itself 500 the health lane
        return JSONResponse({"error": f"get_health_status failed: {exc}"}, status_code=200)
    now = _t.time()
    for key in ("last_successful_tick_at", "last_loop_progress_at", "active_tick_started_at"):
        value = float(status.get(key) or 0.0)
        status[key + "_age_s"] = round(now - value, 1) if value > 0 else None
    return JSONResponse(_json_safe(status) if "_json_safe" in globals() else status)


@router.get("/health/heartbeat")
async def api_heartbeat():
    """Readiness heartbeat for GUI/runtime watchdogs.

    This is intentionally not a process-only ping. It may report healthy only
    when the kernel, inference, memory, scheduler, and tool-governance probes
    pass through the canonical boot health contract.
    """
    _mark_runtime_service_progress("api.health.heartbeat")
    payload, status_code = await _build_boot_health_payload_bounded(
        is_gui_proxy=False,
    )
    conversation_lane = _collect_conversation_lane_status_resilient()
    conversation_ready = bool(conversation_lane.get("conversation_ready", False))
    conversation_busy = conversation_lane_is_busy(conversation_lane)
    required_probes = payload.get("required_probes", {})
    probe_blockers = _heartbeat_probe_blockers(required_probes)
    integrity_report = _collect_runtime_integrity_report()
    integrity_payload = _runtime_integrity_public_payload(integrity_report)
    proof_readiness_healthy = bool(integrity_payload.get("proof_readiness", False))
    blockers = _normalize_conversation_health_blockers(
        list(payload.get("blockers", []) or []) + probe_blockers,
        conversation_ready=conversation_ready,
        conversation_busy=conversation_busy,
    )
    runtime_probe_healthy = not probe_blockers
    healthy = (
        status_code in {200, 202}
        and bool(payload.get("system_ready", payload.get("ready", False)))
        and runtime_probe_healthy
        and conversation_ready
        and not blockers
    )
    if not healthy and not (runtime_probe_healthy and conversation_busy and not blockers):
        status_code = 503
    status = "healthy" if healthy else "working" if runtime_probe_healthy and conversation_busy else "unhealthy"
    heartbeat_payload = {
        "status": status,
        "healthy": healthy,
        "runtime_probe_healthy": runtime_probe_healthy,
        "time": time.time(),
        "required_probes": required_probes,
        "blockers": blockers,
        "boot_phase": payload.get("boot_phase"),
        "conversation_ready": conversation_ready,
        "conversation_busy": conversation_busy,
        "conversation_lane": conversation_lane,
        "integrity": integrity_payload,
        "proof_readiness_healthy": proof_readiness_healthy,
        "certification_ready": bool(healthy and proof_readiness_healthy),
        "integrity_blockers": integrity_payload.get("proof_blockers", []),
    }
    return JSONResponse(heartbeat_payload, status_code=status_code)


# ── Hot Reload ────────────────────────────────────────────────

@router.post("/system/hot-reload", tags=["system"])
async def api_hot_reload(request: Request):
    """Reload Aura's cognitive modules without restarting the process.

    Query params:
        scope  – reload scope (phases, skills, consciousness, llm, affect,
                 memory, identity, resilience, orchestrator_mixins, learning,
                 agency, all). Defaults to "all", which is a curated live-safe
                 union rather than every loaded core module.
        file   – reload a single file by path (relative to project root).

    The kernel, ServiceContainer, event loop, loaded models, and
    conversation history are preserved.
    """
    _require_internal(request)

    try:
        from core.ops.hot_reload import get_hot_reloader

        reloader = get_hot_reloader()
        if ServiceContainer.get("hot_reloader", default=None) is None:
            ServiceContainer.register_instance("hot_reloader", reloader)

        filepath = request.query_params.get("file")
        scope = request.query_params.get("scope", "all")
        if filepath:
            result = await asyncio.to_thread(reloader.reload_file, filepath)
        else:
            result = await asyncio.to_thread(reloader.reload_scope, scope)

        status_code = 200 if result.ok else 207  # 207 Multi-Status for partial failure
        return JSONResponse(result.to_dict(), status_code=status_code)
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation('system', exc)
        logger.error("Hot reload failed: %s", exc, exc_info=True)
        return JSONResponse(
            {"ok": False, "error": str(exc)},
            status_code=500,
        )


@router.get("/system/hot-reload/status", tags=["system"])
async def api_hot_reload_status(request: Request):
    """Return the current state of the hot-reload engine."""
    _require_internal(request)

    try:
        from core.ops.hot_reload import get_hot_reloader

        reloader = get_hot_reloader()
        return JSONResponse(reloader.get_status())
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.error("Hot reload status failed: %s", exc, exc_info=True)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@router.get("/system/hot-reload/scopes", tags=["system"])
async def api_hot_reload_scopes(request: Request):
    """List all available reload scopes and their module prefixes."""
    _require_internal(request)

    try:
        from core.ops.hot_reload import PROTECTED_MODULES, PROTECTED_PREFIXES, RELOAD_SCOPES

        return JSONResponse({
            "scopes": {
                name: {"prefixes": prefixes}
                for name, prefixes in RELOAD_SCOPES.items()
            },
            "special_scopes": ["all"],
            "special_scope_details": {
                "all": "Curated live-safe union of reload scopes; excludes runtime-owned infrastructure that requires reboot."
            },
            "protected_modules": sorted(PROTECTED_MODULES),
            "protected_prefixes": sorted(PROTECTED_PREFIXES),
        })
    except _SYSTEM_RECOVERABLE_ERRORS as exc:
        record_degradation("system", exc)
        logger.error("Hot reload scope listing failed: %s", exc, exc_info=True)
        return JSONResponse({"ok": False, "error": str(exc), "scopes": {}}, status_code=500)
