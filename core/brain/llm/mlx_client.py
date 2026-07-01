from __future__ import annotations

import asyncio
import concurrent.futures as cfutures
import contextlib
import fcntl
import gc
import json
import logging
import multiprocessing as mp
import os
import re
import subprocess
import sys
import threading as _threading
import time
import uuid
from pathlib import Path
from typing import Any

import psutil

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.shutdown_coordinator import is_shutdown_requested
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.concurrency import run_io_bound
from core.utils.deadlines import Deadline, get_deadline
from core.utils.memory_monitor import get_memory_pressure_snapshot
from core.utils.task_tracker import get_task_tracker

from .chat_format import format_chatml_messages, format_chatml_prompt
from .mlx_worker import _mlx_worker_loop

logger = logging.getLogger("LLM.MLX")


def _record_mlx_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation(
        "mlx_client",
        error,
        severity=severity,
        action=action,
    )


_MLX_OPTIONAL_THROTTLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


# Global state for swap management
_GLOBAL_LAST_SWAP_TIME = 0.0
_GLOBAL_LAST_HEAVY_MODEL = None
_CLIENTS = {}
_FOREGROUND_OWNER_LOCK = _threading.Lock()
_FOREGROUND_OWNER_NAME: str | None = None
_FOREGROUND_OWNER_ACQUIRED_AT = 0.0

# [OOM FIX] Global gate: only ONE model can be loading at a time across ALL clients.
# This prevents the 32B and 7B from loading simultaneously and exceeding GPU RAM.
# Uses threading.Semaphore (loop-agnostic) because the singleton MLXLocalClient
# is constructed from one event loop but called from another (Uvicorn thread).
_GLOBAL_SPAWN_GATE = _threading.Semaphore(1)
_MLX_RUNTIME_PROBE_LOCK = _threading.Lock()
_MLX_RUNTIME_PROBE: dict[str, Any] = {
    "ok": None,
    "detail": "",
    "checked_at": 0.0,
}
_MLX_RUNTIME_PROBE_CACHE_PATH = Path.home() / ".aura" / "data" / "mlx_runtime_probe.json"
SharedFuture = asyncio.Future | cfutures.Future


def _read_recurrent_loop_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


def _expected_recurrent_loops_from_model_path(model_path: str) -> int:
    """Return the expected recurrent-depth loop count for a local MLX lane.

    This is a parent-process health mirror of the worker-side policy. The
    worker remains the source of truth for whether the patch actually applied;
    the client uses this only to keep readiness honest before or after worker
    status is received.
    """
    explicit = os.environ.get("AURA_RECURRENT_LOOPS")
    if explicit is not None:
        return _read_recurrent_loop_env("AURA_RECURRENT_LOOPS", 1)

    lowered = str(model_path or "").lower()
    if any(token in lowered for token in ("72b", "solver")):
        return _read_recurrent_loop_env("AURA_RECURRENT_LOOPS_72B", 1)
    if any(token in lowered for token in ("32b", "cortex", "zenith")):
        return _read_recurrent_loop_env("AURA_RECURRENT_LOOPS_32B", 2)
    if any(token in lowered for token in ("14b", "24b", "40b")):
        return _read_recurrent_loop_env("AURA_RECURRENT_LOOPS_14B", 1)
    return _read_recurrent_loop_env("AURA_RECURRENT_LOOPS_SMALL", 1)


def _model_load_min_available_gb(model_path: str) -> float:
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return default

    lowered = str(model_path or "").lower()
    try:
        total_gb = float(psutil.virtual_memory().total) / float(1024**3)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, psutil.Error):
        total_gb = 0.0
    if any(token in lowered for token in ("72b", "solver")):
        default = 52.0 if 0.0 < total_gb < 96.0 else 34.0
        return _env_float("AURA_MLX_72B_LOAD_MIN_AVAILABLE_GB", default)
    if any(token in lowered for token in ("32b", "cortex", "zenith")):
        default = 24.0 if total_gb >= 60.0 else 22.0
        return _env_float("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", default)
    return _env_float("AURA_MLX_LOAD_MIN_AVAILABLE_GB", 8.0)


def _env_projected_footprint_gb(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"", "auto", "detect", "detected"}:
        return None
    try:
        return max(0.0, float(text))
    except (TypeError, ValueError):
        return None


def _path_size_gb(model_path: str) -> float:
    path = Path(str(model_path or "")).expanduser()
    try:
        if path.is_file():
            return float(path.stat().st_size) / float(1024**3)
        if not path.is_dir():
            return 0.0
        total = 0
        for child in path.rglob("*"):
            try:
                if child.is_file():
                    total += child.stat().st_size
            except OSError:
                continue
        return float(total) / float(1024**3)
    except OSError:
        return 0.0


def _projected_footprint_from_artifact_gb(model_path: str, *, fallback_gb: float) -> float:
    """Estimate live model footprint from the local artifact when possible.

    The launcher previously used one static 32B projection for every artifact.
    That is too blunt for Aura: the active fused 4-bit model is materially
    smaller than the old 8-bit base artifact, while a genuine 8-bit path should
    still be treated as too expensive for a tight desktop process cap.
    """

    size_gb = _path_size_gb(model_path)
    if size_gb <= 0.0:
        return fallback_gb
    lowered = str(model_path or "").lower()
    if any(token in lowered for token in ("72b", "solver")):
        overhead = max(4.0, size_gb * 0.14)
    elif any(token in lowered for token in ("32b", "cortex", "zenith", "aura-32b")):
        overhead = max(3.0, size_gb * 0.30)
    else:
        overhead = max(1.0, size_gb * 0.20)
    return max(1.0, size_gb + overhead)


def _projected_model_footprint_gb(model_path: str) -> float:
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return default

    lowered = str(model_path or "").lower()
    if any(token in lowered for token in ("72b", "solver")):
        override = _env_projected_footprint_gb("AURA_MLX_72B_PROJECTED_FOOTPRINT_GB")
        if override is not None:
            return override
        return _projected_footprint_from_artifact_gb(model_path, fallback_gb=41.0)
    if any(token in lowered for token in ("32b", "cortex", "zenith")):
        override = _env_projected_footprint_gb("AURA_MLX_32B_PROJECTED_FOOTPRINT_GB")
        if override is not None:
            return override
        default = 20.0 if any(token in lowered for token in ("4bit", "q4", "fused-model", "20260510")) else 35.0
        return _projected_footprint_from_artifact_gb(model_path, fallback_gb=default)
    if "14b" in lowered:
        return _env_float("AURA_MLX_14B_PROJECTED_FOOTPRINT_GB", 10.0)
    if "7b" in lowered:
        return _env_float("AURA_MLX_7B_PROJECTED_FOOTPRINT_GB", 5.0)
    return _env_float("AURA_MLX_PROJECTED_FOOTPRINT_GB", 4.0)


def _model_process_reserve_gb(model_path: str) -> float:
    def _env_float(name: str, default: float) -> float:
        try:
            return max(0.0, float(os.environ.get(name, str(default))))
        except (TypeError, ValueError):
            return default

    lowered = str(model_path or "").lower()
    if any(token in lowered for token in ("72b", "solver")):
        lane_default = _env_float("AURA_MLX_72B_PROCESS_RESERVE_GB", 5.0)
    elif any(token in lowered for token in ("32b", "cortex", "zenith")):
        lane_default = _env_float("AURA_MLX_32B_PROCESS_RESERVE_GB", 3.0)
    else:
        lane_default = _env_float("AURA_MLX_PROCESS_RESERVE_GB", 1.0)
    return _env_float("AURA_MLX_MODEL_LOAD_PROCESS_RESERVE_GB", lane_default)


def _memory_pressure_blocks_worker_spawn(model_path: str) -> str | None:
    if str(os.environ.get("AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    try:
        snapshot = get_memory_pressure_snapshot()
    except (OSError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("MLX worker-spawn memory probe unavailable: %s", exc)
        return None

    min_available_gb = _model_load_min_available_gb(model_path)
    if snapshot.refuse_heavy_local_generation:
        return snapshot.reason or "critical_memory_pressure"
    if snapshot.available_gb < min_available_gb:
        return (
            f"model_load_headroom:{snapshot.available_gb:.1f}GB "
            f"< required {min_available_gb:.1f}GB"
        )
    process_rss_gb = float(getattr(snapshot, "process_rss_gb", 0.0) or 0.0)
    process_rss_limit_gb = float(getattr(snapshot, "process_rss_limit_gb", 0.0) or 0.0)
    projected_footprint_gb = _projected_model_footprint_gb(model_path)
    process_reserve_gb = _model_process_reserve_gb(model_path)
    projected_process_rss_gb = process_rss_gb + projected_footprint_gb + process_reserve_gb
    if (
        process_rss_limit_gb > 0.0
        and projected_footprint_gb > 0.0
        and projected_process_rss_gb > process_rss_limit_gb
    ):
        return (
            f"projected_process_tree_rss:{process_rss_gb:.1f}GB"
            f"+{projected_footprint_gb:.1f}GB"
            f"+reserve{process_reserve_gb:.1f}GB={projected_process_rss_gb:.1f}GB "
            f"> limit {process_rss_limit_gb:.1f}GB"
        )
    return None


def _normalize_recurrent_depth_status(status: Any, *, model_path: str) -> dict[str, Any]:
    payload = dict(status) if isinstance(status, dict) else {}
    expected_loops = payload.get("expected_loops")
    try:
        expected = int(expected_loops)
    except (TypeError, ValueError):
        expected = _expected_recurrent_loops_from_model_path(model_path)
    expected = max(0, expected)
    payload["expected_loops"] = expected
    payload["required"] = bool(payload.get("required", False)) or expected > 1
    payload.setdefault("active", False)
    payload.setdefault("config", None)
    payload.setdefault("reason", "")
    payload.setdefault("error", "")
    return payload


def _recurrent_depth_readiness_blocker(status: dict[str, Any]) -> str | None:
    if not bool(status.get("required", False)):
        return None
    if bool(status.get("active", False)) is not True:
        return "recurrent_depth_inactive"
    config = status.get("config")
    config_payload = config if isinstance(config, dict) else {}
    try:
        configured_loops = int(config_payload.get("n_loops") or 0)
    except (TypeError, ValueError):
        configured_loops = 0
    try:
        expected_loops = int(status.get("expected_loops") or 0)
    except (TypeError, ValueError):
        expected_loops = 0
    if expected_loops > 1 and configured_loops < expected_loops:
        return "recurrent_depth_loop_mismatch"
    return None


_USER_FACING_ORIGINS = frozenset(
    {
        "user",
        "voice",
        "admin",
        "api",
        "desktop",
        "desktop-ui",
        "gui",
        "ws",
        "websocket",
        "direct",
        "external",
        "native-shell",
        "test",
    }
)
_USER_FACING_PURPOSES = frozenset(
    {
        "chat",
        "conversation",
        "expression",
        "reply",
        "user_response",
    }
)


def _runtime_shutdown_requested() -> bool:
    return is_shutdown_requested()


def _real_model_path(value: Any) -> str:
    return os.path.realpath(str(value))


def _probe_cache_ttl_seconds(ok: bool | None, *, disk: bool) -> float:
    """Keep positive probe results sticky, but let failures expire quickly.

    A transient probe failure should not strand the embedded runtime in a
    "dead" state for many minutes after the host is healthy again.
    """
    if ok is None:
        return 0.0
    if ok:
        return 900.0 if disk else 300.0
    return 30.0 if disk else 10.0


def _safe_close_queue(q: mp.Queue | None) -> None:
    """Close an mp.Queue to release its shared-memory file descriptor."""
    if q is None:
        return
    try:
        q.close()
        q.join_thread()
    except (OSError, ValueError, BrokenPipeError, TypeError, AttributeError):
        return


def _new_shared_future() -> SharedFuture:
    """Create a loop-agnostic future for singleton clients shared across loops."""
    return cfutures.Future()


def _bounded_max_tokens(requested: Any, bridged: Any, fallback: int) -> int:
    """Shrink token budgets without ever handing MLX a zero-token generation."""

    def _coerce(value: Any) -> int:
        if value is None or value == "":
            return int(fallback)
        return int(value)

    try:
        requested_int = _coerce(requested)
    except (TypeError, ValueError, OverflowError):
        requested_int = int(fallback)
    try:
        bridged_int = _coerce(bridged)
    except (TypeError, ValueError, OverflowError):
        bridged_int = int(fallback)
    return max(1, min(max(1, requested_int), max(1, bridged_int)))


def _apply_memory_pressure_generation_controls(
    options: dict[str, Any],
    snapshot: Any,
    *,
    default_max_tokens: int = 1,
) -> dict[str, Any]:
    """Reduce admitted generation work under unified-memory pressure."""

    max_token_cap = getattr(snapshot, "max_token_cap", None)
    if max_token_cap is None:
        return options

    options["max_tokens"] = _bounded_max_tokens(
        options.get("max_tokens"),
        max_token_cap,
        default_max_tokens,
    )
    if (
        bool(options.get("clean_user_surface_contract", False))
        or "clean_user_surface_recurrent_loops" in options
    ):
        options["clean_user_surface_recurrent_loops"] = 1
    return options


def _sanitize_surface_control_receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "enabled",
        "live_mind_controls_bound",
        "clean_user_surface_contract",
        "surface_validation_prompt_present",
        "strict_answer_contract",
        "strict_value_contract",
        "proof_evaluation_contract",
        "operator_evidence_contract",
        "health_probe",
        "runtime_fact_status_contract",
        "grounded_runtime_status_contract",
        "surface_alpha_requested",
        "surface_alpha_applied",
        "surface_alpha_applied_ok",
        "recurrent_runtime_loops_requested",
        "recurrent_depth_present",
        "recurrent_runtime_loops_applied",
        "recurrent_runtime_loops_applied_ok",
        "surface_quality_gate_enabled",
        "surface_quality_gate_passed",
        "surface_quality_gate_attempts",
        "surface_quality_gate_reasons",
        "surface_quality_gate_error",
        "applied",
    }
    return {key: value[key] for key in allowed if key in value}


def _coerce_timeout_seconds(value: Any) -> float | None:
    """Normalize public timeout kwargs into positive request deadlines."""
    if value is None or isinstance(value, Deadline):
        return None
    try:
        timeout_s = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if timeout_s <= 0.0:
        return None
    return max(0.1, timeout_s)


@contextlib.asynccontextmanager
async def _spawn_gate_context():
    """Loop-agnostic async context manager for the global spawn gate."""
    await asyncio.to_thread(_GLOBAL_SPAWN_GATE.acquire)
    try:
        yield
    finally:
        _GLOBAL_SPAWN_GATE.release()


def _foreground_owner_active() -> bool:
    return _FOREGROUND_OWNER_NAME is not None


def _origin_tokens(origin: str | None) -> set[str]:
    normalized = str(origin or "").strip().lower().replace("-", "_")
    return {token for token in normalized.split("_") if token}


def _origin_is_user_facing(origin: str | None) -> bool:
    tokens = _origin_tokens(origin)
    return bool(tokens & _USER_FACING_ORIGINS)


def _background_deferral_active(origin: str | None = None) -> str | None:
    """Mirror InferenceGate's background quiet policy inside the MLX client.

    The gate can reject newly scheduled background requests, but an already
    running background request may reach this client after the foreground lane
    has been reserved.  Checking here prevents that stale request from
    re-spawning a worker Aura just unloaded to protect a user turn.
    """
    try:
        from core.container import ServiceContainer

        gate = ServiceContainer.get("inference_gate", default=None)
        if gate and hasattr(gate, "_background_local_deferral_reason"):
            reason = gate._background_local_deferral_reason(origin=origin)
            return str(reason) if reason else None
    except (ImportError, AttributeError, RuntimeError) as exc:
        _record_mlx_degradation(
            exc,
            action="continued without optional background deferral policy",
        )
        logger.debug("MLX background deferral check unavailable: %s", exc)
    return None


def _foreground_owner_age(now: float | None = None) -> float:
    if _FOREGROUND_OWNER_ACQUIRED_AT <= 0.0:
        return 0.0
    current_time = float(now if now is not None else time.time())
    return max(0.0, current_time - _FOREGROUND_OWNER_ACQUIRED_AT)


def _foreground_owner_wait_budget(
    deadline: Deadline | None,
    *,
    foreground_request: bool,
) -> float:
    # A foreground waiter must be able to outlast one full serialized
    # turn: the generation gate caps concurrency at 2 and healthy gated
    # turns measure 31-44s. The old 10s budget guaranteed a timeout
    # whenever the owner was mid-turn, collapsing proof-primary requests
    # into refused lower-lane fallbacks. Background requests still bail
    # fast rather than camp on the foreground worker.
    default = 60.0 if foreground_request else 8.0
    if not isinstance(deadline, Deadline):
        return default

    remaining = deadline.remaining
    if remaining is None:
        return default

    reserve = 3.0 if foreground_request else 1.5
    return max(0.25, min(default, remaining - reserve))


def _clear_matching_foreground_owner(*candidate_names: str) -> str | None:
    global _FOREGROUND_OWNER_NAME, _FOREGROUND_OWNER_ACQUIRED_AT

    candidates = {str(name or "").strip() for name in candidate_names if str(name or "").strip()}
    if not candidates:
        return None

    with _FOREGROUND_OWNER_LOCK:
        holder = _FOREGROUND_OWNER_NAME
        if holder not in candidates:
            return None
        _FOREGROUND_OWNER_NAME = None
        _FOREGROUND_OWNER_ACQUIRED_AT = 0.0
        return holder


def _clear_stale_foreground_owner(max_age_s: float = 200.0) -> str | None:
    """Release leaked foreground ownership after the generation has ended.

    [STABILITY v59] Raised default from 45s → 200s.  The 32B cortex
    cold-load + Metal shader JIT takes 90-180s.  At 45s the warmup's
    foreground owner was being cleared mid-load by periodic
    ``get_lane_status()`` calls, which allowed background workers to
    respawn and compete for unified memory — creating the desktop
    'cortex warming forever' deadlock.
    """
    global _FOREGROUND_OWNER_NAME, _FOREGROUND_OWNER_ACQUIRED_AT

    acquired = _FOREGROUND_OWNER_LOCK.acquire(False)
    if not acquired:
        # Status reads run on foreground HTTP/pump paths. They must never wait
        # behind a leaked owner-lock holder; recovery paths can still force-clear.
        return None
    try:
        holder = _FOREGROUND_OWNER_NAME
        if holder is None:
            return None
        age = _foreground_owner_age()
        if age <= max_age_s:
            return None
        _FOREGROUND_OWNER_NAME = None
        _FOREGROUND_OWNER_ACQUIRED_AT = 0.0
        return holder
    finally:
        _FOREGROUND_OWNER_LOCK.release()


def force_clear_foreground_owner(
    *,
    reason: str,
    min_age_s: float = 45.0,
    owner_prefix: str | None = None,
) -> dict[str, Any]:
    """Clear a leaked foreground owner from a higher-level recovery path.

    Normal foreground ownership deliberately uses conservative stale limits so
    a healthy 32B cold start is not interrupted.  This hook is only for paths
    that already proved the live turn is wedged, such as desktop HTTP timeout
    recovery or chat-lock preemption.
    """
    global _FOREGROUND_OWNER_NAME, _FOREGROUND_OWNER_ACQUIRED_AT

    min_age = max(0.0, float(min_age_s))
    with _FOREGROUND_OWNER_LOCK:
        holder = _FOREGROUND_OWNER_NAME
        age = _foreground_owner_age()
        if holder is None:
            return {
                "cleared": False,
                "reason": reason,
                "holder": None,
                "age_s": 0.0,
                "detail": "no_foreground_owner",
            }
        if owner_prefix and not str(holder).startswith(str(owner_prefix)):
            return {
                "cleared": False,
                "reason": reason,
                "holder": holder,
                "age_s": round(age, 3),
                "detail": "owner_prefix_mismatch",
            }
        if age < min_age:
            return {
                "cleared": False,
                "reason": reason,
                "holder": holder,
                "age_s": round(age, 3),
                "detail": "owner_younger_than_min_age",
            }
        _FOREGROUND_OWNER_NAME = None
        _FOREGROUND_OWNER_ACQUIRED_AT = 0.0

    logger.warning(
        "♻️ [MLX] Force-cleared foreground owner %s after %.1fs (%s).",
        holder,
        age,
        reason,
    )
    return {
        "cleared": True,
        "reason": reason,
        "holder": holder,
        "age_s": round(age, 3),
        "detail": "cleared",
    }


@contextlib.asynccontextmanager
async def _foreground_owner_context(
    owner_name: str,
    *,
    deadline: Deadline | None = None,
    foreground_request: bool = False,
    stale_after: float | None = None,
):
    """Serialize foreground work so background model activity cannot compete with it."""
    global _FOREGROUND_OWNER_NAME, _FOREGROUND_OWNER_ACQUIRED_AT

    wait_budget = _foreground_owner_wait_budget(
        deadline,
        foreground_request=foreground_request,
    )
    loop = asyncio.get_running_loop()
    wait_started = loop.time()
    last_log_at = 0.0
    owner_acquired = False

    while max(0.0, loop.time() - wait_started) <= wait_budget:
        acquired = _FOREGROUND_OWNER_LOCK.acquire(False)
        cleared_holder: str | None = None
        cleared_holder_age = 0.0
        try:
            if acquired:
                holder = _FOREGROUND_OWNER_NAME
                holder_age = _foreground_owner_age()
                if holder is None:
                    _FOREGROUND_OWNER_NAME = owner_name
                    _FOREGROUND_OWNER_ACQUIRED_AT = time.time()
                    owner_acquired = True
                    break
                if stale_after is not None and holder != owner_name and holder_age > stale_after:
                    _FOREGROUND_OWNER_NAME = None
                    _FOREGROUND_OWNER_ACQUIRED_AT = 0.0
                    cleared_holder = holder
                    cleared_holder_age = holder_age
        finally:
            if acquired:
                _FOREGROUND_OWNER_LOCK.release()
        if cleared_holder is not None:
            logger.warning(
                "♻️ [MLX] Cleared stale foreground owner %s after %.1fs so %s can proceed.",
                cleared_holder,
                cleared_holder_age,
                owner_name,
            )
            continue

        now = loop.time()
        waited = max(0.0, now - wait_started)
        if waited >= wait_budget:
            holder = _FOREGROUND_OWNER_NAME or "foreground"
            holder_age = _foreground_owner_age()
            raise TimeoutError(
                f"Foreground owner wait timed out after {wait_budget:.1f}s "
                f"waiting on {holder} (held {holder_age:.1f}s)"
            )

        if waited >= 5.0 and (now - last_log_at) >= 5.0:
            holder = _FOREGROUND_OWNER_NAME or "foreground"
            holder_age = _foreground_owner_age()
            logger.info(
                "⏳ [MLX] Waiting for foreground owner %s to release (held %.1fs).",
                holder,
                holder_age,
            )
            last_log_at = now

        await asyncio.sleep(min(0.05, max(0.0, wait_budget - waited)))
    if not owner_acquired:
        holder = _FOREGROUND_OWNER_NAME or "foreground"
        holder_age = _foreground_owner_age()
        raise TimeoutError(
            f"Foreground owner wait timed out after {wait_budget:.1f}s "
            f"waiting on {holder} (held {holder_age:.1f}s)"
        )

    try:
        yield
    finally:
        acquired = await asyncio.to_thread(_FOREGROUND_OWNER_LOCK.acquire, True, 2.0)
        if acquired:
            try:
                if _FOREGROUND_OWNER_NAME == owner_name:
                    _FOREGROUND_OWNER_NAME = None
                    _FOREGROUND_OWNER_ACQUIRED_AT = 0.0
            finally:
                _FOREGROUND_OWNER_LOCK.release()
        else:
            logger.warning(
                "⚠️ [MLX] Timed out releasing foreground owner lock for %s.",
                owner_name,
            )


def _bridge_asyncio_future_to_concurrent(future: asyncio.Future) -> cfutures.Future:
    """Relay an asyncio.Future into a thread-safe future for cross-loop awaiting."""
    proxy: cfutures.Future = cfutures.Future()

    def _relay(done_future: asyncio.Future) -> None:
        if proxy.done():
            return
        if done_future.cancelled():
            proxy.cancel()
            return
        try:
            proxy.set_result(done_future.result())
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            try:
                proxy.set_exception(exc)
            except (cfutures.InvalidStateError, asyncio.InvalidStateError):
                return

    if future.done():
        _relay(future)
        return proxy

    try:
        future_loop = future.get_loop()
    except (RuntimeError, AttributeError, TypeError, ValueError):
        _relay(future)
        return proxy

    if future_loop.is_closed():
        _relay(future)
        return proxy

    future_loop.call_soon_threadsafe(future.add_done_callback, _relay)
    return proxy


def _wrap_shared_future_for_current_loop(future: SharedFuture) -> asyncio.Future:
    if isinstance(future, asyncio.Future):
        current_loop = asyncio.get_running_loop()
        if future.get_loop() is current_loop:
            return future
        return asyncio.wrap_future(_bridge_asyncio_future_to_concurrent(future))
    if isinstance(future, cfutures.Future):
        return asyncio.wrap_future(future)
    raise TypeError(f"Unsupported future type: {type(future)!r}")


async def _await_shared_future(future: SharedFuture, *, timeout_s: float | None = None) -> Any:
    wrapped = _wrap_shared_future_for_current_loop(future)
    protected = asyncio.shield(wrapped)
    if timeout_s is None:
        return await protected
    return await asyncio.wait_for(protected, timeout=timeout_s)


def _set_shared_future_result(future: SharedFuture | None, result: Any) -> bool:
    if future is None or future.done():
        return False

    if isinstance(future, cfutures.Future):
        future.set_result(result)
        return True

    if not isinstance(future, asyncio.Future):
        return False

    try:
        future_loop = future.get_loop()
    except (RuntimeError, AttributeError):
        return False
    if future_loop.is_closed():
        return False

    def _setter() -> None:
        if not future.done():
            future.set_result(result)

    future_loop.call_soon_threadsafe(_setter)
    return True


def _cancel_shared_future(future: SharedFuture | None) -> None:
    if future is None or future.done():
        return

    if isinstance(future, cfutures.Future):
        future.cancel()
        return

    if not isinstance(future, asyncio.Future):
        return

    try:
        future_loop = future.get_loop()
    except (RuntimeError, AttributeError):
        return
    if future_loop.is_closed():
        return
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is future_loop:
        future.cancel()
        return

    def _canceller() -> None:
        if not future.done():
            future.cancel()

    future_loop.call_soon_threadsafe(_canceller)


def _cancel_task_threadsafe(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    try:
        task_loop = task.get_loop()
    except (RuntimeError, AttributeError):
        return
    if task_loop.is_closed():
        return

    def _canceller() -> None:
        if not task.done():
            task.cancel()

    task_loop.call_soon_threadsafe(_canceller)


def _notify_closed_loop_output(text: str) -> None:
    if not text or not str(text).strip():
        return
    try:
        from core.consciousness.closed_loop import notify_closed_loop_output

        notify_closed_loop_output(str(text))
    except (ImportError, AttributeError, RuntimeError) as exc:
        _record_mlx_degradation(
            exc,
            action="continued after optional closed-loop output notification failed",
        )
        logger.debug("Closed-loop output notification failed: %s", exc)


def _mlx_runtime_probe_command() -> list[str]:
    return [
        sys.executable,
        "-c",
        "import mlx.core as mx; import mlx_lm; print('mlx_runtime_ok')",
    ]


def _load_probe_cache_from_disk() -> tuple[bool | None, str, float]:
    try:
        payload = json.loads(_MLX_RUNTIME_PROBE_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return None, "", 0.0

    ok = payload.get("ok")
    if ok is not None:
        ok = bool(ok)
    detail = str(payload.get("detail", "") or "")
    checked_at = float(payload.get("checked_at", 0.0) or 0.0)
    return ok, detail, checked_at


def _store_probe_cache_to_disk(ok: bool, detail: str) -> None:
    try:
        _MLX_RUNTIME_PROBE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            _MLX_RUNTIME_PROBE_CACHE_PATH,
            json.dumps(
                {
                    "ok": bool(ok),
                    "detail": str(detail or ""),
                    "checked_at": time.time(),
                }
            ),
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        _record_mlx_degradation(
            exc,
            action="kept in-memory MLX runtime probe status after disk cache write failed",
        )
        logger.debug("Failed to persist MLX runtime probe cache: %s", exc)


def _normalize_probe_detail(stdout: str, stderr: str, returncode: int) -> str:
    combined = "\n".join(part for part in (stderr, stdout) if part).strip()
    if "NSRangeException" in combined and "objectAtIndex" in combined:
        return "metal_device_enumeration_crash"
    if "timed out" in combined.lower():
        return "probe_timeout"
    for line in combined.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned[:240]
    if returncode < 0:
        return f"signal_{abs(returncode)}"
    return f"exit_{returncode}"


def _probe_mlx_runtime(force: bool = False) -> tuple[bool, str]:
    force = force or os.getenv("AURA_FORCE_MLX_RUNTIME_PROBE", "0") == "1"
    now = time.time()
    with _MLX_RUNTIME_PROBE_LOCK:
        cached_ok = _MLX_RUNTIME_PROBE.get("ok")
        cached_at = float(_MLX_RUNTIME_PROBE.get("checked_at", 0.0) or 0.0)
        cached_detail = str(_MLX_RUNTIME_PROBE.get("detail", "") or "")
        if (
            not force
            and cached_ok is not None
            and (now - cached_at) < _probe_cache_ttl_seconds(cached_ok, disk=False)
        ):
            return bool(cached_ok), cached_detail
        if not force:
            disk_ok, disk_detail, disk_checked_at = _load_probe_cache_from_disk()
            if disk_ok is not None and (now - disk_checked_at) < _probe_cache_ttl_seconds(
                disk_ok, disk=True
            ):
                _MLX_RUNTIME_PROBE.update(
                    {
                        "ok": disk_ok,
                        "detail": disk_detail,
                        "checked_at": disk_checked_at,
                    }
                )
                return bool(disk_ok), disk_detail

    project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env["AURA_MLX_RUNTIME_PROBE"] = "1"

    # [STABILITY v57] One-shot retry for probe on failure (except timeout)
    for probe_attempt in range(2):
        ok = False
        detail = "probe_not_run"
        try:
            completed = get_subprocess_gateway().run(
                _mlx_runtime_probe_command(),
                cwd=project_root,
                env=env,
                capture_output=True,
                timeout=25.0,  # [STABILITY v57] Raised from 12.0s for high-load scenarios
                read_only=True,
                source="runtime_probe:mlx_runtime_probe",
            )
            ok = completed.returncode == 0
            detail = _normalize_probe_detail(
                completed.stdout or "",
                completed.stderr or "",
                completed.returncode,
            )

            # If it's a known enumeration crash, we might want to retry immediately
            if not ok and detail == "metal_device_enumeration_crash" and probe_attempt == 0:
                logger.warning("⚠️ [MLX] Metal device enumeration crash during probe. Retrying...")
                time.sleep(1.0)
                continue

            # If it's okay or a different failure, break the retry loop
            break

        except subprocess.TimeoutExpired as exc:
            detail = _normalize_probe_detail(
                (exc.stdout or "") if isinstance(exc.stdout, str) else "",
                (exc.stderr or "") if isinstance(exc.stderr, str) else "",
                124,
            )
            # Timeout is terminal for the attempt
            break
        except (subprocess.SubprocessError, OSError) as exc:
            _record_mlx_degradation(
                exc,
                action="marked MLX runtime probe as failed for this attempt",
                severity="error",
            )
            detail = f"probe_exception:{type(exc).__name__}"
            break

    # [STABILITY v57] Grace Fallback: if the probe just crashed with metal_device_enumeration_crash,
    # but we have a "last known good" status within the last 30 minutes, we bypass the block.
    # This prevents the "RuntimeError: mlx_runtime_probe_failed" cascade from stranding
    # the 32B model due to a transient driver glitch.
    if not ok and detail == "metal_device_enumeration_crash":
        with _MLX_RUNTIME_PROBE_LOCK:
            cached_ok = _MLX_RUNTIME_PROBE.get("ok")
            cached_at = float(_MLX_RUNTIME_PROBE.get("checked_at", 0.0) or 0.0)
            if cached_ok and (time.time() - cached_at) < 1800.0:
                logger.warning(
                    "♻️ [MLX] Runtime probe encountered enumeration crash, but using LKG (last known good) "
                    "status from %.0fs ago to allow lane spawn.",
                    time.time() - cached_at,
                )
                return True, "lkg_fallback_after_enumeration_crash"

    with _MLX_RUNTIME_PROBE_LOCK:
        _MLX_RUNTIME_PROBE.update(
            {
                "ok": ok,
                "detail": detail,
                "checked_at": time.time(),
            }
        )
    _store_probe_cache_to_disk(ok, detail)
    return ok, detail


class MLXLocalClient:
    """
    Parent-process client for the isolated MLX worker.
    Manages the lifecycle, health, and communication with the ForkServer process.
    """

    def __init__(self, model_path: str, device: str = "gpu", max_tokens: int = 4096):
        self.model_path = model_path
        self.device = device
        self.max_tokens = max_tokens
        self.temp = 0.7
        self.top_p = 0.9

        # [LOOP-AGNOSTIC FIX] asyncio.Lock is bound to the creating event loop.
        # MLXLocalClient is a singleton created at boot but used from Uvicorn's
        # separate event loop, causing RuntimeError. threading.Lock is loop-agnostic.
        self._lock = _threading.Lock()
        self._request_lock = _threading.Lock()
        self._deferred_reboot_reason: str | None = None
        self._process: mp.Process | None = None
        self._mp_context = (
            mp.get_context("spawn")
            if os.uname().sysname == "Darwin"
            else mp.get_context("forkserver")
        )
        self._req_q: Any | None = None
        self._res_q: Any | None = None
        self._closed = False
        self._init_done = False

        # Concurrency Hardening
        self._listener_task: asyncio.Task | None = None
        self._last_heartbeat = 0.0
        self._last_progress_at = 0.0
        self._last_token_progress_at = 0.0
        self._last_ready_at = 0.0
        self._last_generation_completed_at = 0.0
        self._last_user_facing_completed_at = 0.0
        self._last_visible_readiness_at = 0.0
        self._current_gen_future: SharedFuture | None = None
        self._init_future: SharedFuture | None = None
        self._pending_generations: dict[str, SharedFuture] = {}
        self._request_lock_owner_label = ""
        self._request_lock_acquired_at = 0.0
        self._lane_state = "cold"
        self._lane_error = ""
        self._lane_transition_at = time.time()
        self._active_generations = 0
        self._warmup_attempted = False
        self._warmup_in_flight = False
        self._consecutive_empty: int = 0  # [STABILITY v53] Explicit init — was missing
        self._expected_cancel_reason = ""
        self._expected_cancel_budget = 0
        self._expected_cancel_recorded_at = 0.0
        self._process_started_at = 0.0
        self._current_request_started_at = 0.0
        self._current_first_token_at = 0.0
        self._current_request_id = ""
        self._current_request_progress_baseline_at = 0.0
        self._current_prompt_chars = 0
        self._current_requested_max_tokens = 0
        self._current_request_prompt_chars = 0
        self._current_first_token_hard_ceiling_s = 0.0
        self._foreground_generation_watchdog: _threading.Timer | None = None
        self._recurrent_depth_status: dict[str, Any] = {"active": False, "config": None}
        self._last_surface_control_receipt: dict[str, Any] = {}
        self._clock_sample_wall = time.time()
        self._clock_sample_monotonic = time.monotonic()

        # The state repository's SharedMemoryTransport may be backed by mmap on
        # restricted/macOS paths. mmap handles are not picklable under the
        # Darwin spawn context, so workers get a small multiprocessing bridge
        # instead of the repository transport itself. The last slot is reserved
        # for steering liveness.
        self._substrate_mem = self._mp_context.Array("d", 16, lock=False)

        # Shared memory flag to track if affective steering successfully attached
        self._steering_active = self._mp_context.Value("b", False, lock=False)

    def _is_primary_or_deep_lane(self) -> bool:
        lowered = os.path.basename(self.model_path).lower()
        return any(token in lowered for token in ("32b", "72b", "zenith", "solver", "cortex"))

    def _mark_progress(self) -> None:
        self._last_progress_at = time.time()

    def _rebase_after_system_sleep(self) -> float:
        """Rebase active wall-clock anchors after host sleep/wake.

        macOS wall time advances while the monotonic clock used by asyncio and
        request deadlines pauses. Without rebasing, a healthy generation in
        flight at sleep is misclassified as a token or heartbeat stall on wake.
        """
        now_wall = time.time()
        now_monotonic = time.monotonic()
        wall_delta = max(0.0, now_wall - self._clock_sample_wall)
        monotonic_delta = max(0.0, now_monotonic - self._clock_sample_monotonic)
        self._clock_sample_wall = now_wall
        self._clock_sample_monotonic = now_monotonic
        sleep_gap = wall_delta - monotonic_delta
        threshold = float(os.environ.get("AURA_SYSTEM_SLEEP_GAP_THRESHOLD_S", "5"))
        if sleep_gap <= max(1.0, threshold):
            return 0.0

        stale_cutoff = now_wall - max(2.0, sleep_gap * 0.5)
        for attr in (
            "_current_request_started_at",
            "_current_request_progress_baseline_at",
            "_current_first_token_at",
            "_last_token_progress_at",
            "_last_heartbeat",
            "_last_progress_at",
            "_last_ready_at",
            "_lane_transition_at",
            "_request_lock_acquired_at",
        ):
            value = float(getattr(self, attr, 0.0) or 0.0)
            if 0.0 < value < stale_cutoff:
                setattr(self, attr, value + sleep_gap)
        logger.info(
            "🌙 [MLX] Host resume detected; rebased active inference clocks by %.1fs.",
            sleep_gap,
        )
        return sleep_gap

    def get_last_surface_control_receipt(self) -> dict[str, Any]:
        return dict(self._last_surface_control_receipt)

    def _record_surface_control_receipt_from_response(self, response: dict[str, Any]) -> None:
        receipt = _sanitize_surface_control_receipt(
            response.get("surface_control_receipt") if isinstance(response, dict) else None
        )
        if receipt:
            self._last_surface_control_receipt = receipt

    def _note_expected_generation_cancellation(self, reason: str, *, count: int) -> None:
        if count <= 0:
            return
        self._expected_cancel_reason = str(reason or "planned_reboot")
        self._expected_cancel_budget += int(count)
        self._expected_cancel_recorded_at = time.time()

    def _consume_expected_generation_cancellation(self) -> str:
        if self._expected_cancel_budget <= 0:
            return ""
        if (
            self._expected_cancel_recorded_at
            and (time.time() - self._expected_cancel_recorded_at) > 30.0
        ):
            self._expected_cancel_reason = ""
            self._expected_cancel_budget = 0
            self._expected_cancel_recorded_at = 0.0
            return ""
        reason = self._expected_cancel_reason
        self._expected_cancel_budget = max(0, self._expected_cancel_budget - 1)
        if self._expected_cancel_budget == 0:
            self._expected_cancel_reason = ""
            self._expected_cancel_recorded_at = 0.0
        return reason

    def _mark_generation_started(
        self,
        req_id: str,
        *,
        prompt_chars: int = 0,
        requested_max_tokens: int = 0,
        first_token_hard_ceiling_s: float = 0.0,
    ) -> None:
        now = time.time()
        self._current_request_id = str(req_id or "")
        self._current_request_progress_baseline_at = max(
            self._last_heartbeat,
            self._last_progress_at,
            self._last_ready_at,
        )
        self._current_request_started_at = now
        self._current_first_token_at = 0.0
        self._current_prompt_chars = max(0, int(prompt_chars or 0))
        self._current_requested_max_tokens = max(0, int(requested_max_tokens or 0))
        self._last_token_progress_at = 0.0
        self._current_request_prompt_chars = max(0, int(prompt_chars or 0))
        self._current_first_token_hard_ceiling_s = max(
            0.0,
            float(first_token_hard_ceiling_s or 0.0),
        )
        self._mark_progress()

    def _mark_token_progress(self, req_id: str | None = None) -> None:
        now = time.time()
        normalized_req_id = str(req_id or "")
        if (
            normalized_req_id
            and self._current_request_id
            and normalized_req_id != self._current_request_id
        ):
            return
        self._last_token_progress_at = now
        if self._current_first_token_at <= 0.0:
            self._current_first_token_at = now
        self._mark_progress()

    def _clear_active_generation_tracking(self) -> None:
        self._current_request_started_at = 0.0
        self._current_first_token_at = 0.0
        self._last_token_progress_at = 0.0
        self._current_request_id = ""
        self._current_request_progress_baseline_at = 0.0
        self._current_prompt_chars = 0
        self._current_requested_max_tokens = 0
        self._current_request_prompt_chars = 0
        self._current_first_token_hard_ceiling_s = 0.0
        self._mark_progress()

    def _mark_generation_completed(self, *, user_facing: bool = False) -> None:
        self._last_generation_completed_at = time.time()
        if user_facing:
            self._last_user_facing_completed_at = self._last_generation_completed_at
            self._last_visible_readiness_at = self._last_generation_completed_at
        self._clear_active_generation_tracking()

    def _set_lane_state(self, state: str, error: str = "") -> None:
        if state != self._lane_state:
            self._lane_transition_at = time.time()
        self._lane_state = state
        if error:
            self._lane_error = str(error)
        elif state == "ready":
            self._lane_error = ""

    def _classify_failure(self, *, foreground_request: bool = False) -> str:
        if foreground_request or (self._is_primary_or_deep_lane() and _foreground_owner_active()):
            return "foreground_blocking"
        return "background_degraded"

    def _record_degraded_event(
        self,
        reason: str,
        *,
        detail: str = "",
        severity: str = "warning",
        foreground_request: bool = False,
    ) -> None:
        try:
            from core.health.degraded_events import record_degraded_event

            record_degraded_event(
                "mlx_client",
                reason,
                detail=detail,
                severity=severity,
                classification=self._classify_failure(foreground_request=foreground_request),
                context={
                    "model": os.path.basename(self.model_path),
                    "lane_state": self._lane_state,
                    "warmup_in_flight": self._warmup_in_flight,
                },
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_mlx_degradation(
                exc,
                action="kept lane-local degraded state after health event emission failed",
            )
            logger.debug("Failed to record MLX degraded event: %s", exc)

    def _stale_after(
        self, *, during_generation: bool = False, foreground_request: bool = False
    ) -> float:
        """Heartbeat-stall timeout.

        [RESILIENCE] Widened for 32B foreground: recurrent depth doubles
        the compute per token, and complex prompts can legitimately take
        60-90s for prompt eval.  Killing the cortex when heartbeats are
        still arriving (worker is alive, just slow) was the #1 cause of
        'cortex died and never came back'.  As long as heartbeats arrive,
        the worker is alive — let it finish."""
        lowered = os.path.basename(self.model_path).lower()
        if "72b" in lowered or "solver" in lowered:
            if foreground_request and during_generation:
                return 45.0
            return 90.0 if during_generation else 45.0
        if "32b" in lowered or "cortex" in lowered or "zenith" in lowered:
            if foreground_request and during_generation:
                return 45.0  # was 22s — too aggressive with recurrent depth
            return 60.0 if during_generation else 30.0
        return 20.0 if during_generation else 15.0

    def _first_token_sla(self, *, foreground_request: bool = False) -> float:
        lowered = os.path.basename(self.model_path).lower()
        prompt_chars = max(0, int(getattr(self, "_current_request_prompt_chars", 0) or 0))
        # Prompt eval dominates first-token latency on the 32B/72B lanes.
        # Recent live traces showed ~5.3k-token prompts taking 66-76s before
        # the first token arrived, which is healthy-but-slow rather than wedged.
        estimated_prompt_tokens = (prompt_chars / 4.6) if prompt_chars > 0 else 0.0

        def _with_prompt_eval_headroom(
            base_sla: float,
            *,
            threshold_tokens: float,
            eval_seconds_per_token: float,
            cap_s: float,
        ) -> float:
            if estimated_prompt_tokens <= threshold_tokens:
                return base_sla
            extra = (estimated_prompt_tokens - threshold_tokens) * eval_seconds_per_token
            return min(cap_s, base_sla + extra)

        # Cold-start exemption: the FIRST real foreground generation after a
        # worker warmup or reboot legitimately needs 30–45 s on 32B because
        # Metal shaders are still JIT-compiling and the KV cache is empty.
        # Tripping the SLA at 22 s on the very first user turn was bouncing
        # Cortex to UNAVAILABLE before the model could produce a token.
        # _last_generation_completed_at is zero until a real generation has
        # finished; we use that as the cold-start signal.
        is_cold_start = float(getattr(self, "_last_generation_completed_at", 0.0) or 0.0) <= 0.0
        if "72b" in lowered or "solver" in lowered:
            if foreground_request:
                base = 52.0 if is_cold_start else 32.0
                return _with_prompt_eval_headroom(
                    base,
                    threshold_tokens=768.0,
                    eval_seconds_per_token=0.018,
                    cap_s=115.0,
                )
            return 30.0
        if "32b" in lowered or "cortex" in lowered or "zenith" in lowered:
            # [RESILIENCE] Recurrent depth 2x loops means prompt eval takes
            # significantly longer.  These SLAs must accommodate that without
            # killing the cortex.  Cold-start can legitimately need 90s for
            # Metal shader JIT + recurrent depth prompt eval on a 5k-token
            # prompt.  The point of these SLAs is to catch WEDGED workers
            # (no heartbeats), not SLOW workers (heartbeats arriving).
            if foreground_request:
                # Live measurement 2026-06-11: warm 32B first tokens at
                # 46.4s under the macos26 guard + serialized lanes — the
                # 45s base declared healthy generations wedged, and the
                # lane recycle's cancellation swept well beyond the
                # offending request (it killed the proof battery's repair
                # coroutine mid-await). Wedge detection belongs to the
                # heartbeat/stall checks; this SLA only needs to beat
                # genuinely dead workers.
                base = 120.0 if is_cold_start else 90.0
                return _with_prompt_eval_headroom(
                    base,
                    threshold_tokens=512.0,
                    eval_seconds_per_token=0.015,
                    cap_s=240.0,
                )
            return 90.0
        return 8.0

    def _first_token_absolute_ceiling(self, *, foreground_request: bool = False) -> float:
        """Return the non-negotiable no-token ceiling for one generation.

        Heartbeats prove that the worker process is alive; they do not prove
        that the active model request is making useful progress. The primary
        lane previously allowed a heartbeating but tokenless request to run
        beyond the endpoint deadline, after which the inference layer could
        start additional retries. Keep this ceiling below the foreground API
        envelope so the caller still has time to recover or use another lane.
        """

        lowered = os.path.basename(self.model_path).lower()
        if "72b" in lowered or "solver" in lowered:
            default = 165.0 if foreground_request else 120.0
        elif "32b" in lowered or "cortex" in lowered or "zenith" in lowered:
            default = 120.0 if foreground_request else 90.0
        else:
            default = 30.0 if foreground_request else 20.0
        configured = os.environ.get("AURA_FIRST_TOKEN_ABSOLUTE_CEILING_S")
        try:
            return max(10.0, float(configured)) if configured is not None else default
        except (TypeError, ValueError):
            return default

    def _first_token_hard_ceiling(self, *, foreground_request: bool = False) -> float:
        first_token_sla = self._first_token_sla(foreground_request=foreground_request)
        try:
            hard_mult = float(os.environ.get("AURA_FIRST_TOKEN_HARD_MULT", "1.8") or 1.8)
        except (TypeError, ValueError):
            hard_mult = 1.8
        try:
            hard_pad = float(os.environ.get("AURA_FIRST_TOKEN_HARD_PAD_S", "20") or 20)
        except (TypeError, ValueError):
            hard_pad = 20.0
        return min(
            first_token_sla * hard_mult + hard_pad,
            self._first_token_absolute_ceiling(foreground_request=foreground_request),
        )

    def _deadline_bound_first_token_hard_ceiling(
        self,
        deadline_remaining_s: float | None,
        *,
        foreground_request: bool = False,
    ) -> float:
        hard_ceiling = self._first_token_hard_ceiling(
            foreground_request=foreground_request,
        )
        if deadline_remaining_s is None:
            return hard_ceiling
        try:
            remaining = float(deadline_remaining_s)
        except (TypeError, ValueError):
            return hard_ceiling
        if remaining <= 0.0:
            return 10.0
        # Leave enough wall-clock for the caller to fail closed and recycle the
        # worker. This makes request-specific deadlines dominate the generic
        # 32B/72B first-token ceiling.
        return min(hard_ceiling, max(10.0, remaining - 4.0))

    def _start_foreground_first_token_watchdog(
        self,
        req_id: str,
        *,
        foreground_request: bool = False,
        hard_ceiling_s: float | None = None,
    ) -> _threading.Timer | None:
        """Abort tokenless foreground generations even if the event loop wedges."""

        if not foreground_request or not self._is_primary_or_deep_lane():
            return None
        hard_ceiling = (
            max(10.0, float(hard_ceiling_s))
            if hard_ceiling_s is not None
            else self._first_token_hard_ceiling(foreground_request=True)
        )
        fire_after = max(10.0, hard_ceiling + 2.0)
        model_name = os.path.basename(self.model_path)

        def _enforce() -> None:
            if _runtime_shutdown_requested():
                return
            try:
                if (
                    str(self._current_request_id or "") != str(req_id or "")
                    or self._current_request_started_at <= 0.0
                    or self._current_first_token_at > 0.0
                ):
                    return
                elapsed = max(0.0, time.time() - self._current_request_started_at)
                if elapsed < hard_ceiling:
                    return
                logger.error(
                    "🛑 [MLX] Out-of-band first-token watchdog aborting %s "
                    "(%.1fs elapsed, hard=%.1fs).",
                    model_name,
                    elapsed,
                    hard_ceiling,
                )
                self._record_degraded_event(
                    "first_token_wall_clock_watchdog",
                    detail=f"{model_name}>{hard_ceiling:.1f}s",
                    severity="critical",
                    foreground_request=True,
                )
                self.force_abort_active_generation("first_token_wall_clock_watchdog")
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.error("MLX first-token watchdog failed: %s", exc)

        timer = _threading.Timer(fire_after, _enforce)
        timer.daemon = True
        timer.name = f"AuraMLXFirstTokenWatchdog:{model_name[:32]}"
        timer.start()
        self._foreground_generation_watchdog = timer
        return timer

    def _token_stall_after(self, *, foreground_request: bool = False) -> float:
        lowered = os.path.basename(self.model_path).lower()
        if "72b" in lowered or "solver" in lowered:
            return 18.0 if foreground_request else 25.0
        if "32b" in lowered or "cortex" in lowered or "zenith" in lowered:
            # [RESILIENCE] Reverted from 10s — recurrent depth can cause
            # legitimate pauses between tokens during the recurrent block
            # computation. Sized up with the 2026-06-11 first-token
            # remeasurement: inter-token pauses stretch the same way under
            # the macos26 guard, and a stall verdict triggers the same
            # over-broad lane recycle as an SLA breach.
            return 40.0 if foreground_request else 45.0
        return 8.0

    def _warmup_timeout(self) -> float:
        # [STABILITY v56] Raised from 75.0s → 180.0s. 32B models on M5
        # regularly take 120-150s to cold-load and compile Metal shaders.
        return 180.0 if self._is_primary_or_deep_lane() else 30.0

    def _handshake_timeout(self) -> float:
        """Absolute upper bound for worker init before we declare the process wedged."""
        return 300.0 if self._is_primary_or_deep_lane() else 120.0

    def _request_scoped_init_timeout(
        self,
        deadline: Deadline | None,
        *,
        foreground_request: bool,
    ) -> tuple[float, bool]:
        """Bound init waits to the caller's budget so fallback can still happen in time."""
        full_timeout = self._handshake_timeout()
        if not isinstance(deadline, Deadline):
            return full_timeout, False

        remaining = deadline.remaining
        if remaining is None:
            return full_timeout, False

        reserve = 5.0 if foreground_request else 2.0
        # [STABILITY v57] Increased minimum from 0.25 to 10.0 for fallbacks.
        # Background fallbacks were being killed after 3s because their budget
        # was too tight to even start the worker.
        scoped_timeout = max(10.0 if not foreground_request else 5.0, remaining - reserve)
        return min(full_timeout, scoped_timeout), scoped_timeout < full_timeout

    def get_lane_status(self) -> dict[str, Any]:
        # [STABILITY v59] Do NOT clear the foreground owner while a warmup
        # is actively in flight.  The warmup legitimately holds the owner
        # for up to 180s; clearing it mid-load lets background workers
        # respawn and compete for memory, creating the desktop deadlock.
        if int(getattr(self, "_active_generations", 0) or 0) <= 0 and not self._warmup_in_flight:
            stale_owner = _clear_stale_foreground_owner()
            if stale_owner:
                logger.warning(
                    "♻️ [MLX] Cleared stale foreground owner %s during lane status check.",
                    stale_owner,
                )
        self._check_lane_state_staleness()  # [STABILITY v54] Eagerly check and reset stuck/stale lane states
        worker_alive = self.is_alive()
        lane_state = self._lane_state
        lane_error = self._lane_error
        now = time.time()
        worker_progress_anchor = max(
            self._last_progress_at,
            self._last_ready_at,
            self._last_token_progress_at,
            self._last_generation_completed_at,
        )
        visible_conversation_anchor = max(
            float(getattr(self, "_last_visible_readiness_at", 0.0) or 0.0),
            float(getattr(self, "_last_user_facing_completed_at", 0.0) or 0.0),
        )
        progress_age_s = (
            max(0.0, now - worker_progress_anchor)
            if worker_progress_anchor > 0.0
            else None
        )
        heartbeat_age_s = (
            max(0.0, now - self._last_heartbeat) if self._last_heartbeat > 0.0 else None
        )
        readiness_blockers: list[str] = []
        if _runtime_shutdown_requested():
            readiness_blockers.append("runtime_shutdown")
        if not worker_alive:
            readiness_blockers.append("worker_not_alive")
        if not self._init_done:
            readiness_blockers.append("init_not_complete")
        if lane_state != "ready":
            readiness_blockers.append(f"lane_{lane_state}")
        if worker_progress_anchor <= 0.0:
            readiness_blockers.append("no_worker_progress")
        elif progress_age_s is not None and progress_age_s > self._stale_after():
            readiness_blockers.append("worker_progress_stale")
        # The visible-conversation probe verifies the primary lane has served a
        # real user-facing turn. It is only meaningful when a UI surface is
        # attached; a headless proof/longevity run has no user surface, so a
        # warm+alive worker is the legitimate ready state and this probe would be
        # a permanent false positive there. (Mirrors the inference_gate guard.)
        _proof_headless = False
        try:
            from core.runtime.proof_policy import proof_headless_run

            _proof_headless = proof_headless_run()
        except (ImportError, RuntimeError, AttributeError):
            _proof_headless = False
        if (
            self._is_primary_or_deep_lane()
            and visible_conversation_anchor <= 0.0
            and not _proof_headless
        ):
            readiness_blockers.append("visible_conversation_probe_missing")
        if lane_state == "ready" and not worker_alive:
            lane_state = "cold"
            lane_error = "worker_not_alive"
            self._set_lane_state(lane_state, lane_error)
        elif lane_state == "ready" and any(
            blocker in {"no_worker_progress", "worker_progress_stale"}
            for blocker in readiness_blockers
        ):
            lane_state = "recovering"
            lane_error = "worker_progress_stale"
            self._set_lane_state(lane_state, lane_error)
            if f"lane_{lane_state}" not in readiness_blockers:
                readiness_blockers.append(f"lane_{lane_state}")
        recurrent_depth_status = _normalize_recurrent_depth_status(
            self._recurrent_depth_status,
            model_path=self.model_path,
        )
        recurrent_depth_blocker = _recurrent_depth_readiness_blocker(recurrent_depth_status)
        if recurrent_depth_blocker and recurrent_depth_blocker not in readiness_blockers:
            readiness_blockers.append(recurrent_depth_blocker)
            if lane_state == "ready":
                lane_state = "recovering"
                lane_error = recurrent_depth_blocker
                self._set_lane_state(lane_state, lane_error)
        foreground_owned = _foreground_owner_active()
        foreground_owner = _FOREGROUND_OWNER_NAME
        if self._warmup_in_flight:
            readiness_blockers.append("warmup_in_flight")
        if foreground_owned and foreground_owner.startswith("warmup:"):
            readiness_blockers.append("warmup_foreground_owner")
        elif foreground_owned and self._active_generations > 0:
            readiness_blockers.append("active_generation_in_flight")
        readiness_blockers = list(dict.fromkeys(readiness_blockers))

        conversation_ready = not readiness_blockers
        return {
            "model_path": self.model_path,
            "state": lane_state,
            "last_error": lane_error,
            "conversation_ready": conversation_ready,
            "readiness_blockers": readiness_blockers,
            "foreground_owned": foreground_owned,
            "foreground_owner": foreground_owner,
            "foreground_owned_at": _FOREGROUND_OWNER_ACQUIRED_AT,
            "last_heartbeat": self._last_heartbeat,
            "heartbeat_age_s": heartbeat_age_s,
            "last_progress_at": self._last_progress_at,
            "progress_age_s": progress_age_s,
            "worker_progress_anchor": worker_progress_anchor,
            "last_token_progress_at": self._last_token_progress_at,
            "last_ready_at": self._last_ready_at,
            "last_generation_completed_at": self._last_generation_completed_at,
            "last_user_facing_completed_at": self._last_user_facing_completed_at,
            "last_visible_readiness_at": self._last_visible_readiness_at,
            "last_transition_at": self._lane_transition_at,
            "warmup_attempted": self._warmup_attempted,
            "warmup_in_flight": self._warmup_in_flight,
            "active_generations": int(self._active_generations),
            "process_started_at": self._process_started_at,
            "current_request_started_at": self._current_request_started_at,
            "current_first_token_at": self._current_first_token_at,
            "current_request_prompt_chars": self._current_request_prompt_chars,
            "recurrent_depth": recurrent_depth_status,
            "request_age_s": (
                max(0.0, time.time() - self._current_request_started_at)
                if self._current_request_started_at
                else 0.0
            ),
        }

    def get_supervision_status(self) -> dict[str, Any]:
        now = time.time()
        return {
            "lane": os.path.basename(self.model_path),
            "state": self._lane_state,
            "alive": self.is_alive(),
            "active_generations": int(self._active_generations),
            "process_uptime_s": max(0.0, now - self._process_started_at)
            if self._process_started_at
            else 0.0,
            "request_age_s": max(0.0, now - self._current_request_started_at)
            if self._current_request_started_at
            else 0.0,
            "time_to_first_token_s": (
                max(0.0, self._current_first_token_at - self._current_request_started_at)
                if self._current_request_started_at and self._current_first_token_at
                else None
            ),
            "idle_for_s": max(
                0.0,
                now
                - max(
                    self._last_generation_completed_at,
                    self._last_ready_at,
                    self._last_token_progress_at,
                    self._last_progress_at,
                    self._last_heartbeat,
                ),
            )
            if any(
                stamp > 0.0
                for stamp in (
                    self._last_generation_completed_at,
                    self._last_ready_at,
                    self._last_token_progress_at,
                    self._last_progress_at,
                    self._last_heartbeat,
                )
            )
            else 0.0,
        }

    def should_recycle_for_fragmentation(
        self,
        *,
        max_uptime_s: float = 5400.0,
        min_idle_s: float = 900.0,
    ) -> bool:
        if not self.is_alive() or self._active_generations > 0 or _foreground_owner_active():
            return False
        if self._process_started_at <= 0.0:
            return False
        idle_anchor = max(
            self._last_generation_completed_at,
            self._last_ready_at,
            self._last_token_progress_at,
            self._last_progress_at,
            self._last_heartbeat,
        )
        if idle_anchor <= 0.0:
            return False
        now = time.time()
        return bool(
            (now - self._process_started_at) >= float(max_uptime_s)
            and (now - idle_anchor) >= float(min_idle_s)
        )

    def note_lane_recovering(self, reason: str) -> None:
        self._warmup_in_flight = False
        self._set_lane_state("recovering", reason)

    def _lane_runtime_failure(self) -> str:
        error = str(getattr(self, "_lane_error", "") or "")
        if error.startswith(("mlx_runtime_unavailable:", "local_runtime_unavailable:")):
            return error
        return ""

    def refresh_runtime_availability(self, *, force_probe: bool = False) -> bool:
        """Clear stale runtime-failure poison when the host probe is healthy again.

        A transient MLX runtime failure should not strand the lane in a failed
        state or an exponential spawn backoff once the runtime is healthy again.
        """
        runtime_error = self._lane_runtime_failure()
        if not runtime_error and time.time() >= float(
            getattr(self, "_spawn_backoff_until", 0.0) or 0.0
        ):
            return False

        ok, detail = _probe_mlx_runtime(force=force_probe)
        if not ok:
            self._mark_runtime_unavailable(detail)
            return False

        recovered = (
            bool(runtime_error) or float(getattr(self, "_spawn_backoff_until", 0.0) or 0.0) > 0.0
        )
        if recovered:
            logger.info(
                "♻️ [MLX] Runtime probe recovered for %s. Clearing failed lane/backoff state.",
                os.path.basename(self.model_path),
            )
        self._consecutive_spawn_failures = 0
        self._spawn_backoff_until = 0.0
        self._warmup_in_flight = False
        if self._lane_state == "failed" or runtime_error:
            self._set_lane_state("cold")
        else:
            self._lane_error = ""
        return recovered

    def _request_lock_timeout(
        self,
        deadline: Deadline | None,
        *,
        foreground_request: bool,
    ) -> float:
        # Tightened from 30s to 12s for foreground: if the current holder has
        # been in-flight for longer than this budget, the second user message
        # should cascade to brainstem/cloud rather than keep waiting.  The
        # prior 30s budget stacked on top of a hung 32B generation produced
        # the 60–90 s "Aura is thinking..." windows the user reported.
        default = 12.0 if foreground_request else 60.0
        if not isinstance(deadline, Deadline):
            return default

        remaining = deadline.remaining
        if remaining is None:
            return default

        reserve = 3.0 if foreground_request else 2.0
        return max(0.25, min(default, remaining - reserve))

    async def _acquire_request_lock(
        self,
        *,
        owner_label: str,
        deadline: Deadline | None,
        foreground_request: bool,
    ) -> bool:
        wait_budget = self._request_lock_timeout(
            deadline,
            foreground_request=foreground_request,
        )
        loop = asyncio.get_running_loop()
        wait_started = loop.time()
        wait_deadline = wait_started + wait_budget
        last_log_at = 0.0

        while loop.time() < wait_deadline:
            if self._request_lock.acquire(False):
                self._request_lock_owner_label = str(owner_label or "")
                self._request_lock_acquired_at = time.time()
                return True

            now = loop.time()
            waited = max(0.0, now - wait_started)

            if waited >= 5.0 and (now - last_log_at) >= 5.0:
                holder = self._request_lock_owner_label or "another_request"
                holder_age = (
                    max(0.0, time.time() - self._request_lock_acquired_at)
                    if self._request_lock_acquired_at
                    else 0.0
                )
                logger.info(
                    "⏳ [MLX] Waiting for in-flight request on %s (owner=%s, held %.1fs).",
                    os.path.basename(self.model_path),
                    holder,
                    holder_age,
                )
                last_log_at = now

            await asyncio.sleep(min(0.05, max(0.0, wait_deadline - loop.time())))

        holder = self._request_lock_owner_label or "another_request"
        holder_age = (
            max(0.0, time.time() - self._request_lock_acquired_at)
            if self._request_lock_acquired_at
            else 0.0
        )
        logger.warning(
            "⏳ [MLX] Request queue timeout after %.1fs for %s while waiting on %s (held %.1fs).",
            wait_budget,
            os.path.basename(self.model_path),
            holder,
            holder_age,
        )
        self._record_degraded_event(
            "request_lock_timeout",
            detail=f"{os.path.basename(self.model_path)} owner={holder} held={holder_age:.1f}s",
            severity="warning",
            foreground_request=foreground_request,
        )
        # Preemption: if a foreground caller waited through the explicit queue
        # deadline and the current holder exceeded the first-token SLA, cancel
        # the in-flight future and recycle the worker before stale text can
        # leak into a later turn.
        if foreground_request:
            sla = self._first_token_sla(foreground_request=True)
            if holder_age > sla:
                heartbeat_age = (
                    time.time() - self._last_heartbeat
                    if self._last_heartbeat > 0
                    else 999.0
                )
                if heartbeat_age > 30.0:
                    logger.error(
                        "🛑 [MLX] Preempting wedged holder %s (age=%.1fs > sla=%.1fs, no heartbeat for %.1fs). "
                        "Cancelling in-flight future and scheduling worker reboot.",
                        holder,
                        holder_age,
                        sla,
                        heartbeat_age,
                    )
                    self._deferred_reboot_reason = "foreground_preemption_wedged_holder"
                else:
                    logger.warning(
                        "🛡️ [MLX] Holder %s slow (age=%.1fs > sla=%.1fs) but heartbeat fresh (%.1fs ago). "
                        "Cancelling generation and scheduling a clean recycle so stale text cannot bleed into the next turn.",
                        holder,
                        holder_age,
                        sla,
                        heartbeat_age,
                    )
                    self._deferred_reboot_reason = (
                        "recoverable_foreground_preemption_slow_holder"
                    )
                try:
                    stuck_future = self._current_gen_future
                    if stuck_future is not None:
                        _cancel_shared_future(stuck_future)
                except (RuntimeError, AttributeError) as exc:
                    logger.debug(
                        "MLX request preemption future cancel skipped: %s",
                        exc,
                    )
        return False

    def _release_request_lock(self) -> None:
        self._request_lock_owner_label = ""
        self._request_lock_acquired_at = 0.0
        try:
            self._request_lock.release()
        except RuntimeError:
            logger.debug(
                "Loop-agnostic request lock for %s was already released.",
                os.path.basename(self.model_path),
            )

    async def _ensure_listener_task(self) -> None:
        task = self._listener_task
        if task is not None and not task.done():
            try:
                if not task.get_loop().is_closed():
                    return
            except (RuntimeError, AttributeError) as exc:
                logger.debug("MLX listener task loop unavailable during reuse check: %s", exc)
            _cancel_task_threadsafe(task)

        self._listener_task = get_task_tracker().create_task(self._response_listener_loop())

    def note_lane_failed(self, reason: str) -> None:
        self._warmup_in_flight = False
        self._set_lane_state("failed", reason)

    def _mark_runtime_unavailable(self, detail: str) -> None:
        reason = f"mlx_runtime_unavailable:{detail}"
        self._warmup_in_flight = False
        self._init_done = False
        self._set_lane_state("failed", reason)

    def _worker_unhealthy(self, stale_after: float | None = None) -> bool:
        if self._process is None or not self._process.is_alive():
            return True
        if not self._init_done:
            return True
        stale_after = float(stale_after or self._stale_after())
        last_progress = max(self._last_heartbeat, self._last_progress_at, self._last_ready_at)
        if last_progress <= 0.0:
            return True
        return bool((time.time() - last_progress) > stale_after)

    def _check_lane_state_staleness(self) -> None:
        """[STABILITY v51] Auto-reset stuck non-terminal lane states.

        If the lane has been in a transient state (warming, recovering,
        handshaking, spawning) for >120s with no progress, force-reset
        to 'cold' so recovery can restart from scratch. This prevents
        the permanent 'CORTEX WARMING' display.
        """
        if self._lane_state not in {"warming", "recovering", "handshaking", "spawning"}:
            return
        now = time.time()
        stuck_duration = now - self._lane_transition_at
        if stuck_duration < 120.0:
            return
        last_activity = max(
            self._last_heartbeat,
            self._last_progress_at,
            self._last_ready_at,
            self._last_token_progress_at,
        )
        if last_activity > 0.0 and (now - last_activity) < 30.0:
            return  # Recent activity — state is legitimate
        logger.warning(
            "🔧 [STABILITY] Lane state '%s' stuck for %.0fs with no activity. "
            "Force-resetting to 'cold' for clean recovery.",
            self._lane_state,
            stuck_duration,
        )
        self._warmup_in_flight = False
        self._set_lane_state("cold")

    def _kill_and_join_blocking(self, p: mp.Process):
        if p and p.is_alive():
            try:
                p.kill()
                p.join(timeout=2.0)
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                _record_mlx_degradation(
                    e,
                    action="continued process cleanup after worker kill/join failed",
                    severity="error",
                )
                logger.warning("Error killing process: %s", e)

    def _replace_ipc_queues(self, *, maxsize: int = 10) -> None:
        """Replace IPC queues after closing the old semaphores and feeder threads."""
        _safe_close_queue(self._req_q)
        _safe_close_queue(self._res_q)
        self._req_q = self._mp_context.Queue(maxsize=maxsize)
        self._res_q = self._mp_context.Queue(maxsize=maxsize)
        self._closed = False

    def _close_ipc_queues(self) -> None:
        """Close IPC queues without recreating them during final client shutdown."""
        _safe_close_queue(self._req_q)
        _safe_close_queue(self._res_q)
        self._req_q = None
        self._res_q = None

    def force_abort_active_generation(self, reason: str = "hard_generation_deadline") -> bool:
        """Thread-safe emergency abort for a wedged generation.

        Normal cancellations should flow through ``reboot_worker``. This path is
        intentionally synchronous so an external watchdog can break a stuck
        proof or foreground request even when the caller's event loop is waiting
        on an MLX future that failed to observe its deadline.
        """
        reason = str(reason or "hard_generation_deadline")
        pending_futures = {
            id(future): future
            for future in list(self._pending_generations.values()) + [self._current_gen_future]
            if future is not None and not future.done()
        }
        had_active_request = bool(
            pending_futures
            or self._active_generations > 0
            or self._current_request_started_at > 0.0
        )
        process = self._process
        had_process = bool(process is not None and process.is_alive())
        if not had_active_request and not had_process:
            return False

        logger.error(
            "🛑 [MLX] Force-aborting active generation for %s (%s).",
            os.path.basename(self.model_path),
            reason,
        )
        self._set_lane_state("recovering", reason)

        abort_payload = {
            "status": "error",
            "action": "generate",
            "id": self._current_request_id,
            "message": reason,
            "force_aborted": True,
        }
        for future in pending_futures.values():
            _set_shared_future_result(future, abort_payload)

        killed_process_before_lock = False
        if process is not None and process.is_alive():
            logger.error(
                "🛑 [MLX] Killing worker immediately for forced abort before lifecycle lock cleanup (%s).",
                reason,
            )
            self._kill_and_join_blocking(process)
            killed_process_before_lock = True

        acquired = self._lock.acquire(timeout=0.25)
        try:
            self._pending_generations.clear()
            self._current_gen_future = None
            self._active_generations = 0
            self._deferred_reboot_reason = None
            self._warmup_in_flight = False
            self._init_done = False
            self._process = None
            self._last_heartbeat = 0.0
            self._last_progress_at = 0.0
            self._last_token_progress_at = 0.0
            self._last_generation_completed_at = 0.0
            self._last_user_facing_completed_at = 0.0
            self._last_visible_readiness_at = 0.0
            self._process_started_at = 0.0
            self._clear_active_generation_tracking()
            if self._init_future is not None:
                _cancel_shared_future(self._init_future)
            self._init_future = None

            if self._listener_task is not None:
                _cancel_task_threadsafe(self._listener_task)
                self._listener_task = None

            if (
                process is not None
                and process.is_alive()
                and not killed_process_before_lock
            ):
                self._kill_and_join_blocking(process)

            self._replace_ipc_queues()
            self._release_request_lock()
            gc.collect()
        finally:
            if acquired:
                try:
                    self._lock.release()
                except RuntimeError:
                    logger.debug(
                        "Loop-agnostic lifecycle lock for %s was already released.",
                        os.path.basename(self.model_path),
                    )

        self._record_degraded_event(
            "force_aborted_generation",
            detail=f"{os.path.basename(self.model_path)}:{reason}",
            severity="error",
            foreground_request=True,
        )
        self._set_lane_state("cold", reason)
        return True

    def _spawn_worker_blocking(self) -> mp.Process:
        """Isolated spawn logic for the MLX worker, run in a background thread."""
        # [STABILITY v60] Reclaim the old/orphan worker BEFORE the memory
        # admission check. A recycle (or crash respawn) replaces a worker that
        # is still resident; killing it below frees its ~20GB. Running the
        # headroom check FIRST saw the about-to-die worker's memory and refused
        # the spawn (memory_pressure_refused_worker_spawn:model_load_headroom:
        # 20.2GB < 22.0GB → recycled_model_lane_not_live_after_warmup → DNU
        # FATAL), so the wedge could never recover. Free first, then admit.
        #
        # [STABILITY v51] Orphan reclamation: kill any existing MLXWorker
        # processes for this model path before spawning a new one.
        try:
            model_basename = os.path.basename(self.model_path)
            target_name = f"MLXWorker-{model_basename}"
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    pname = proc.info.get("name", "") or ""
                    if target_name in pname or (
                        proc.info.get("cmdline")
                        and any(model_basename in str(arg) for arg in (proc.info["cmdline"] or []))
                        and "mlx_worker" in str(proc.info.get("cmdline", []))
                    ):
                        if proc.pid != os.getpid():
                            logger.warning(
                                "🧹 [STABILITY] Killing orphan MLXWorker pid=%d for %s",
                                proc.pid,
                                model_basename,
                            )
                            proc.kill()
                            proc.wait(timeout=3.0)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except (OSError, ConnectionError, TimeoutError) as orphan_exc:
            _record_mlx_degradation(
                orphan_exc,
                action="continued worker spawn after orphan reclamation scan failed",
            )
            logger.debug("Orphan reclamation scan failed (non-fatal): %s", orphan_exc)

        memory_block = _memory_pressure_blocks_worker_spawn(self.model_path)
        if memory_block:
            error = RuntimeError(f"memory_pressure_refused_worker_spawn:{memory_block}")
            _record_mlx_degradation(
                error,
                action="refused MLX worker spawn before model load due to memory pressure",
                severity="critical",
            )
            raise error

        runtime_ok, runtime_detail = _probe_mlx_runtime()
        if not runtime_ok:
            raise RuntimeError(f"mlx_runtime_probe_failed:{runtime_detail}")

        if self._req_q is None or self._res_q is None:
            raise RuntimeError("MLX IPC queues must be created before worker spawn")
        ctx = self._mp_context

        lock_dir = Path.home() / ".aura" / "run"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file_path = str(lock_dir / "mlx_spawn.lock")
        lock_fd = os.open(lock_file_path, os.O_CREAT | os.O_WRONLY, 0o600)
        with os.fdopen(lock_fd, "w") as lock_file:
            try:
                logger.info("🔒 [MLX] Acquiring process-level spawn lock...")
                fcntl.flock(lock_file, fcntl.LOCK_EX)

                project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                p = ctx.Process(
                    target=_mlx_worker_loop,
                    args=(
                        self.model_path,
                        self._req_q,
                        self._res_q,
                        self.device,
                        self._substrate_mem,
                        self._steering_active,
                    ),
                    daemon=True,
                    name=f"MLXWorker-{os.path.basename(self.model_path)}",
                )
                p.start()
                try:
                    from core.runtime.runtime_hygiene import get_runtime_hygiene

                    get_runtime_hygiene().register_process_handle(
                        p,
                        kind="multiprocessing",
                        name=p.name,
                        source="mlx_local_client.worker_owner",
                        command=f"MLX worker for {os.path.basename(self.model_path)}",
                    )
                except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    _record_mlx_degradation(
                        exc,
                        action="continued worker spawn after runtime hygiene registration failed",
                        severity="warning",
                    )
                return p

            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
                logger.info("🔓 [MLX] Released process-level spawn lock.")

    async def _spawn_worker(self) -> mp.Process:
        return await asyncio.get_running_loop().run_in_executor(None, self._spawn_worker_blocking)

    async def _response_listener_loop(self):
        """
        [v7.8] Background task to constantly drain the worker response queue.
        Prevents IPC deadlocks by ensuring heartbeats and telemetry are ALWAYS consumed.
        """
        import queue

        from core.container import ServiceContainer

        _consecutive_errors = 0
        while not _runtime_shutdown_requested():
            if self._res_q is None:
                break
            try:
                # Use polling instead of infinite block to avoid executor thread leaks and zombie stealing
                res = await run_io_bound(self._res_q.get, True, 0.5)
                _consecutive_errors = 0
            except queue.Empty:
                continue
            except asyncio.CancelledError:
                break
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                _record_mlx_degradation(
                    e,
                    action="exited or backed off response listener after queue polling failure",
                    severity="error",
                )
                # If queue is closed/broken, graceful exit
                if "closed" in str(e).lower() or isinstance(e, ValueError):
                    break
                _consecutive_errors += 1
                # [BUG FIX] After repeated errors, the queue is likely broken
                # (e.g., worker killed during cascade cleanup). Exit the loop
                # instead of spinning forever and consuming thread pool resources.
                if _consecutive_errors >= 10:
                    logger.warning(
                        "⚠️ [MLX] Response listener: %d consecutive errors. Queue likely broken. Exiting.",
                        _consecutive_errors,
                    )
                    break
                logger.error("⚠️ [MLX] Response listener poll error: %s", e)
                await asyncio.sleep(0.5)
                continue

            if not res:
                continue

            try:
                status = res.get("status")
                action = res.get("action")
                req_id = res.get("id")

                # 1. Update SubsystemAudit Heartbeat
                if status == "heartbeat":
                    self._last_heartbeat = time.time()
                    self._mark_progress()
                    audit = ServiceContainer.get("subsystem_audit", default=None)
                    if audit:
                        is_heavy = any(
                            k in self.model_path.lower() for k in ["72b", "32b", "zenith"]
                        )
                        tier_name = "mlx_heavy" if is_heavy else "mlx_light"
                        audit.heartbeat(tier_name)
                    continue
                if status in {"progress", "token"}:
                    self._mark_token_progress(res.get("id"))
                    continue

                # 2. Route init/generation responses to the correct awaiting future
                if action == "init":
                    if self._init_future and not self._init_future.done():
                        self._mark_progress()
                        _set_shared_future_result(self._init_future, res)
                        continue
                elif action in ("generate", "stream_done"):
                    future = self._pending_generations.pop(req_id, None) if req_id else None
                    if future and not future.done():
                        self._mark_progress()
                        _set_shared_future_result(future, res)
                        continue
                    # A generation can finish after the caller has already
                    # abandoned it and started another turn. Never hand a
                    # response with an old request id to the current future.
                    if (
                        self._current_gen_future
                        and not self._current_gen_future.done()
                        and (not req_id or req_id == self._current_request_id)
                    ):
                        self._mark_progress()
                        _set_shared_future_result(self._current_gen_future, res)
                        continue
                elif status == "error":
                    init_error = (
                        self._init_future is not None
                        and not self._init_future.done()
                        and not self._init_done
                        and action in {None, "", "init"}
                    )
                    if init_error:
                        self._mark_progress()
                        payload = dict(res)
                        payload.setdefault("action", "init")
                        _set_shared_future_result(self._init_future, payload)
                        continue
                    if action == "init" and self._init_future and not self._init_future.done():
                        self._mark_progress()
                        _set_shared_future_result(self._init_future, res)
                        continue
                    future = self._pending_generations.pop(req_id, None) if req_id else None
                    if future and not future.done():
                        self._mark_progress()
                        _set_shared_future_result(future, res)
                        continue
                    if (
                        self._current_gen_future
                        and not self._current_gen_future.done()
                        and (not req_id or req_id == self._current_request_id)
                    ):
                        self._mark_progress()
                        _set_shared_future_result(self._current_gen_future, res)
                        continue

                # 3. Log errors if no future is waiting
                if status == "error":
                    logger.error("🛑 [MLX] Async worker error: %s", res.get("message"))

            except (ImportError, AttributeError, RuntimeError) as e:
                _record_mlx_degradation(
                    e,
                    action="kept response listener alive after malformed worker message",
                    severity="error",
                )
                logger.error("⚠️ [MLX] Response listener message processing error: %s", e)
                await asyncio.sleep(1.0)

    async def _ensure_worker_alive(
        self,
        *,
        request_is_background: bool = False,
        foreground_request: bool = False,
        init_timeout: float | None = None,
        soft_timeout: bool = False,
        skip_swap_cooldown: bool = False,
    ) -> bool:
        """Self-healing supervisor for the MLX worker.

        [OOM FIX] Acquires a global semaphore so only ONE model loads at a time.
        This prevents the 32B + 7B from loading simultaneously and crashing Metal.
        """
        if _runtime_shutdown_requested():
            logger.info(
                "🛑 [MLX] Worker start/recovery skipped for %s: runtime is shutting down.",
                os.path.basename(self.model_path),
            )
            return False
        if request_is_background and _foreground_owner_active():
            logger.info(
                "⏸️ [MLX] Deferring background worker activity for %s while foreground lane is owned by %s.",
                os.path.basename(self.model_path),
                _FOREGROUND_OWNER_NAME or "foreground",
            )
            return False
        if request_is_background:
            background_deferral = _background_deferral_active(os.path.basename(self.model_path))
            if background_deferral:
                logger.info(
                    "⏸️ [MLX] Deferring background worker activity for %s (%s).",
                    os.path.basename(self.model_path),
                    background_deferral,
                )
                return False

        # Fast path: if worker is already alive, don't acquire the gate
        if self._process and self._process.is_alive() and self._init_done:
            self._check_lane_state_staleness()  # [STABILITY v51]
            recurrent_depth_status = _normalize_recurrent_depth_status(
                self._recurrent_depth_status,
                model_path=self.model_path,
            )
            recurrent_depth_blocker = _recurrent_depth_readiness_blocker(recurrent_depth_status)
            if recurrent_depth_blocker and not request_is_background:
                self._set_lane_state("recovering", recurrent_depth_blocker)
                self._record_degraded_event(
                    recurrent_depth_blocker,
                    detail=f"{os.path.basename(self.model_path)}:{recurrent_depth_status}",
                    severity="warning",
                    foreground_request=foreground_request,
                )
                return False
            self._set_lane_state("ready")
            return True

        # Slow path: acquire global gate to serialize model loading
        async with _spawn_gate_context():
            return await self._ensure_worker_alive_inner(
                request_is_background=request_is_background,
                foreground_request=foreground_request,
                init_timeout=init_timeout,
                soft_timeout=soft_timeout,
                skip_swap_cooldown=skip_swap_cooldown,
            )

    async def _ensure_worker_alive_inner(
        self,
        *,
        request_is_background: bool = False,
        foreground_request: bool = False,
        init_timeout: float | None = None,
        soft_timeout: bool = False,
        skip_swap_cooldown: bool = False,
    ) -> bool:
        """Inner implementation — called while holding the global spawn gate."""
        if _runtime_shutdown_requested():
            logger.info(
                "🛑 [MLX] Worker spawn skipped for %s: runtime is shutting down.",
                os.path.basename(self.model_path),
            )
            return False
        should_wait_init = False
        init_future: SharedFuture | None = None

        # [PIPELINE HARDENING] 12s Swap Cooldown
        from .model_registry import ACTIVE_MODEL, DEEP_MODEL, get_model_path

        primary_path = _real_model_path(get_model_path(ACTIVE_MODEL))
        deep_path = _real_model_path(get_model_path(DEEP_MODEL))
        target_path = _real_model_path(self.model_path)

        global _GLOBAL_LAST_SWAP_TIME, _GLOBAL_LAST_HEAVY_MODEL

        if request_is_background and _foreground_owner_active():
            logger.info(
                "⏸️ [MLX] Background spawn blocked for %s while foreground lane is reserved.",
                os.path.basename(self.model_path),
            )
            return False
        if request_is_background:
            background_deferral = _background_deferral_active(os.path.basename(self.model_path))
            if background_deferral:
                logger.info(
                    "⏸️ [MLX] Background spawn blocked for %s (%s).",
                    os.path.basename(self.model_path),
                    background_deferral,
                )
                return False

        if target_path in (primary_path, deep_path):
            other_heavy_path = deep_path if target_path == primary_path else primary_path
            other_client = _CLIENTS.get(other_heavy_path)
            if other_client and other_client is not self and other_client.is_alive():
                # Cortex-protection mirror of the smaller-yield loop below: if
                # the warm heavy model just served the user, do not evict it
                # to bring up the other heavy lane. Background promotions in
                # the middle of a conversation were the primary cause of the
                # 32B → 72B → 32B thrash that left the user with "I got
                # interrupted mid-thought" replies.
                last_user_facing = float(
                    getattr(other_client, "_last_user_facing_completed_at", 0.0) or 0.0
                )
                if (
                    request_is_background
                    and last_user_facing > 0.0
                    and (time.time() - last_user_facing) < 180.0
                ):
                    logger.info(
                        "🛡️ [MLX] NOT hot-swapping heavy model %s (served user %.1fs ago); "
                        "background warmup of %s deferred.",
                        os.path.basename(other_heavy_path),
                        time.time() - last_user_facing,
                        os.path.basename(target_path),
                    )
                    return False
                logger.warning(
                    "♻️ [MLX] Hot-swapping heavy model: unloading %s before loading %s.",
                    os.path.basename(other_heavy_path),
                    os.path.basename(target_path),
                )
                await other_client.reboot_worker()
                gc.collect()  # [OOM FIX] Reclaim before loading new heavy model
            if _GLOBAL_LAST_HEAVY_MODEL and _GLOBAL_LAST_HEAVY_MODEL != target_path:
                now = time.time()
                elapsed = now - _GLOBAL_LAST_SWAP_TIME
                if elapsed < 12.0 and not skip_swap_cooldown:
                    wait_time = 12.0 - elapsed
                    logger.warning("⏳ [MLX] SWAP COOLDOWN: Waiting %.1fs...", wait_time)
                    await asyncio.sleep(wait_time)
                elif elapsed < 12.0 and skip_swap_cooldown:
                    logger.info(
                        "⚡ [MLX] Skipping %.1fs swap cooldown for %s.",
                        12.0 - elapsed,
                        os.path.basename(target_path),
                    )
            for other_path, other_client in list(_CLIENTS.items()):
                if other_path in (target_path, other_heavy_path):
                    continue
                if not other_client or other_client is self or not other_client.is_alive():
                    continue
                # Cortex-protection: if another client was serving a
                # user-facing generation in the last 180 s, DO NOT evict it
                # to make room for a background warmup.  This is the yield
                # loop that caused the "CORTEX UNAVAILABLE" flap between
                # turns — Cortex answered turn 1 at T, the 7B wanted to spin
                # up at T+2 for a background appraisal, Cortex got reboot-
                # yielded, turn 2 then had to cold-start the 32B all over
                # and tripped the first-token SLA.  Holding Cortex warm
                # across a conversation costs RAM; losing it between turns
                # costs the entire user experience.
                last_user_facing = float(
                    getattr(other_client, "_last_user_facing_completed_at", 0.0) or 0.0
                )
                if last_user_facing > 0.0 and (time.time() - last_user_facing) < 180.0:
                    logger.info(
                        "🛡️ [MLX] NOT yielding %s (served a user-facing turn %.1fs ago; "
                        "keeping warm for conversational continuity).",
                        os.path.basename(other_path),
                        time.time() - last_user_facing,
                    )
                    continue
                logger.warning(
                    "🧹 [MLX] Yielding %s before warming %s to reduce RAM pressure.",
                    os.path.basename(other_path),
                    os.path.basename(target_path),
                )
                await other_client.reboot_worker(
                    reason=f"yield_to_{os.path.basename(target_path)}",
                    mark_failed=False,
                )

        acquired = await asyncio.to_thread(self._lock.acquire, True, 15.0)
        if not acquired:
            logger.error(
                "🚨 [MLX] DEADLOCK DETECTED: Could not acquire _lock within 15s for %s",
                os.path.basename(self.model_path),
            )
            return False
        try:
            if self._process and self._process.is_alive() and self._init_done:
                self._set_lane_state("ready")
                return True  # Already healthy, release gate

            if self._process and self._process.is_alive() and not self._init_done:
                # Stale-handshake watchdog: if the worker process has been
                # alive but failing to complete its handshake for longer
                # than 2x the handshake timeout, the init future is wedged
                # (worker stuck loading weights, IPC pipe wedged, etc.).
                # Recycle the worker instead of waiting forever, otherwise
                # every subsequent appraisal request piles onto the same
                # never-resolving future and the lane stays in "handshaking"
                # for hours, which is what produced the cascading damasio
                # timeout / "Worker alive but still handshaking" loop.
                handshake_age = time.time() - getattr(self, "_lane_transition_at", time.time())
                handshake_budget = max(60.0, 2.0 * self._handshake_timeout())
                if (
                    self._init_future is not None
                    and self._lane_state == "handshaking"
                    and handshake_age > handshake_budget
                ):
                    logger.warning(
                        "♻️ [MLX] Worker handshake stuck for %.0fs (>%.0fs budget) on %s — recycling.",
                        handshake_age,
                        handshake_budget,
                        os.path.basename(self.model_path),
                    )
                    self._set_lane_state("recovering", "stale_handshake")
                    try:
                        if self._init_future and not self._init_future.done():
                            self._init_future.set_exception(
                                RuntimeError("stale_handshake_recycled")
                            )
                    except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                        _record_mlx_degradation(
                            _exc,
                            action="recycled stale handshake despite init-future notification failure",
                        )
                        logger.debug("Suppressed stale-handshake future-set: %s", _exc)
                    self._init_future = None
                    await asyncio.get_running_loop().run_in_executor(
                        None, self._kill_and_join_blocking, self._process
                    )
                    self._process = None
                    self._init_done = False
                    self._last_heartbeat = 0.0
                    self._last_progress_at = 0.0
                    self._drain_queue()
                    self._replace_ipc_queues()
                    # Fall through into the missing-init-lifecycle path on
                    # the next iteration of caller's outer loop.

                if self._init_future is not None:
                    logger.info(
                        "⏳ [MLX] Worker alive but still handshaking: %s",
                        os.path.basename(self.model_path),
                    )
                    self._set_lane_state("handshaking")
                    init_future = self._init_future
                    should_wait_init = True
                else:
                    logger.warning(
                        "♻️ [MLX] Worker alive but init lifecycle is missing. Recycling %s.",
                        os.path.basename(self.model_path),
                    )
                    self._set_lane_state("recovering", "missing_init_lifecycle")
                    await asyncio.get_running_loop().run_in_executor(
                        None, self._kill_and_join_blocking, self._process
                    )
                    self._process = None
                    self._init_done = False
                    self._last_heartbeat = 0.0
                    self._last_progress_at = 0.0
                    self._drain_queue()

                    # Prevent zombie threads from stealing messages
                    self._replace_ipc_queues()

                    init_future = _new_shared_future()
                    self._init_future = init_future
                    self._set_lane_state("spawning")
                    logger.info(
                        "📡 [MLX] Respawning worker for %s...", os.path.basename(self.model_path)
                    )
                    try:
                        self._process = await self._spawn_worker()
                        self._process_started_at = time.time()
                        self._consecutive_spawn_failures = 0
                        self._spawn_backoff_until = 0.0
                    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                        detail = str(exc)
                        _sf = getattr(self, "_consecutive_spawn_failures", 0) + 1
                        self._consecutive_spawn_failures = _sf
                        self._spawn_backoff_until = time.time() + min(
                            300.0, 10.0 * (2 ** min(_sf - 1, 5))
                        )
                        if "mlx_runtime_probe_failed:" in detail:
                            self._mark_runtime_unavailable(
                                detail.split("mlx_runtime_probe_failed:", 1)[1]
                            )
                        else:
                            self._set_lane_state("failed", detail)
                        _record_mlx_degradation(
                            exc,
                            action="marked lane failed or runtime unavailable and applied spawn backoff",
                            severity="error",
                        )
                        self._record_degraded_event(
                            "spawn_failed",
                            detail=f"{os.path.basename(self.model_path)}:{detail}",
                            severity="error",
                            foreground_request=foreground_request,
                        )
                        logger.error(
                            "🛑 [MLX] Worker respawn aborted for %s: %s (backoff %.0fs)",
                            os.path.basename(self.model_path),
                            detail,
                            min(300.0, 10.0 * (2 ** min(_sf - 1, 5))),
                        )
                        self._init_future = None
                        return False
                    if self._listener_task:
                        _cancel_task_threadsafe(self._listener_task)
                    await self._ensure_listener_task()
                    self._set_lane_state("handshaking")
                    should_wait_init = True
            elif not self._process or not self._process.is_alive():
                # [BUG FIX] Exponential backoff on repeated spawn failures.
                # Without this, [Errno 5] I/O errors cause a tight 2-3s retry
                # loop that leaks FDs and shared memory for hours.
                _spawn_fails = getattr(self, "_consecutive_spawn_failures", 0)
                _spawn_backoff_until = getattr(self, "_spawn_backoff_until", 0.0)
                if time.time() < _spawn_backoff_until:
                    if not await asyncio.to_thread(self.refresh_runtime_availability, force_probe=True):
                        return False  # Still in backoff window

                self._drain_queue()

                # Prevent zombie threads from stealing messages
                self._replace_ipc_queues()

                init_future = _new_shared_future()
                self._init_future = init_future
                self._set_lane_state("spawning")
                logger.info("📡 [MLX] Spawning worker for %s...", os.path.basename(self.model_path))
                try:
                    self._process = await self._spawn_worker()
                    self._process_started_at = time.time()
                    self._consecutive_spawn_failures = 0  # Reset on success
                    self._spawn_backoff_until = 0.0
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    detail = str(exc)
                    # [BUG FIX] Exponential backoff: 10s, 30s, 60s, 120s, 300s
                    self._consecutive_spawn_failures = _spawn_fails + 1
                    backoff = min(300.0, 10.0 * (2 ** min(_spawn_fails, 5)))
                    self._spawn_backoff_until = time.time() + backoff
                    if "mlx_runtime_probe_failed:" in detail:
                        self._mark_runtime_unavailable(
                            detail.split("mlx_runtime_probe_failed:", 1)[1]
                        )
                    else:
                        self._set_lane_state("failed", detail)
                    _record_mlx_degradation(
                        exc,
                        action="marked lane failed or runtime unavailable and applied spawn backoff",
                        severity="error",
                    )
                    self._record_degraded_event(
                        "spawn_failed",
                        detail=f"{os.path.basename(self.model_path)}:{detail}",
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    logger.error(
                        "🛑 [MLX] Worker spawn aborted for %s: %s (attempt %d, backoff %.0fs)",
                        os.path.basename(self.model_path),
                        detail,
                        self._consecutive_spawn_failures,
                        backoff,
                    )
                    self._init_future = None
                    return False
                if self._listener_task:
                    _cancel_task_threadsafe(self._listener_task)
                await self._ensure_listener_task()
                should_wait_init = True
                self._init_done = False
                self._set_lane_state("handshaking")
        finally:
            self._lock.release()

        if should_wait_init:
            fut = init_future or self._init_future
            if fut is None:
                raise RuntimeError("MLX worker init future missing during startup")
            handshake_timeout = float(init_timeout or self._handshake_timeout())

            # [STABILITY v54] One-shot retry for worker handshake to handle
            # transient JIT/Metal compilation or memory alignment glitches.
            for handshake_attempt in range(2):
                try:
                    res = await _await_shared_future(fut, timeout_s=handshake_timeout)
                    if res.get("status") == "ok":
                        self._init_done = True
                        self._last_heartbeat = time.time()
                        self._last_ready_at = self._last_heartbeat
                        self._mark_progress()
                        self._set_lane_state("ready")
                        recurrent_status = res.get("recurrent_depth")
                        if isinstance(recurrent_status, dict):
                            self._recurrent_depth_status = recurrent_status
                        if target_path in (primary_path, deep_path):
                            _GLOBAL_LAST_HEAVY_MODEL = target_path
                            _GLOBAL_LAST_SWAP_TIME = time.time()
                        logger.info("✅ [MLX] Worker ready: %s", os.path.basename(self.model_path))
                        return True
                    else:
                        msg = res.get("message", "Init failed")
                        if handshake_attempt == 0:
                            logger.warning(
                                "🔄 [MLX] Worker init failed: %s. Retrying spawn...", msg
                            )
                            # Reboot and try again once
                            await self.reboot_worker(reason="init_failed_retry", mark_failed=False)
                            # Update fut for the new spawn
                            fut = self._init_future
                            if not fut:
                                break
                            continue
                        self._set_lane_state("failed", msg)
                        raise RuntimeError(msg)
                except TimeoutError:
                    if soft_timeout and self._process and self._process.is_alive():
                        logger.warning(
                            "⏳ [MLX] Init handshake exceeded request budget (%.1fs) for %s. Keeping worker alive to continue warming.",
                            handshake_timeout,
                            os.path.basename(self.model_path),
                        )
                        self._set_lane_state("recovering", "init_budget_timeout")
                        self._record_degraded_event(
                            "init_budget_timeout",
                            detail=f"{os.path.basename(self.model_path)}:{handshake_timeout:.1f}s",
                            severity="warning",
                            foreground_request=foreground_request,
                        )
                        raise
                    if handshake_attempt == 0:
                        logger.warning("⏳ [MLX] Init timeout on attempt 1. Retrying spawn...")
                        await self.reboot_worker(reason="init_timeout_retry", mark_failed=False)
                        fut = self._init_future
                        if not fut:
                            break
                        continue
                    logger.error("🛑 [MLX] Init handshake TIMED OUT. Force killing process.")
                    self._set_lane_state("failed", "init_timeout")
                    if self._process:
                        await asyncio.get_running_loop().run_in_executor(
                            None, self._kill_and_join_blocking, self._process
                        )
                        self._process = None
                    self._init_future = None
                    raise
            return False
        return self._process is not None and self._process.is_alive() and self._init_done

    def _drain_queue(self):
        """Safe non-blocking drain."""
        import queue as _queue_mod

        while self._res_q is not None and not self._res_q.empty():
            try:
                self._res_q.get_nowait()
            except (_queue_mod.Empty, OSError, ValueError):
                break
        while self._req_q is not None and not self._req_q.empty():
            try:
                self._req_q.get_nowait()
            except (_queue_mod.Empty, OSError, ValueError):
                break

    def is_alive(self) -> bool:
        """Returns True if the worker process is running and initialized."""
        return self._process is not None and self._process.is_alive() and self._init_done

    async def _wait_for_generation_result(
        self,
        req_id: str,
        future: SharedFuture,
        deadline: Deadline,
        *,
        foreground_request: bool = False,
    ) -> dict[str, Any] | None:
        """Wait in short slices so dead workers fail fast instead of hanging the UI."""
        stall_after = self._stale_after(
            during_generation=True, foreground_request=foreground_request
        )
        first_token_sla = self._first_token_sla(foreground_request=foreground_request)
        token_stall_after = self._token_stall_after(foreground_request=foreground_request)
        wait_started = time.monotonic()
        hard_cap = max(
            30.0,
            float(os.environ.get("AURA_MLX_GENERATION_HARD_CAP_SECONDS", "240")),
        )
        while (time.monotonic() - wait_started) <= hard_cap:
            remaining = deadline.remaining
            if remaining is not None and remaining <= 0.0:
                raise TimeoutError

            slice_timeout = min(2.0, remaining) if remaining is not None else 2.0
            try:
                return await _await_shared_future(future, timeout_s=slice_timeout)
            except TimeoutError:
                if future.done():
                    return future.result()

                self._rebase_after_system_sleep()

                try:
                    memory_snapshot = get_memory_pressure_snapshot()
                    if memory_snapshot.should_gc:
                        gc.collect()
                    allow_critical_memory = str(
                        os.environ.get("AURA_MLX_ALLOW_CRITICAL_MEMORY_GENERATION", "")
                    ).strip().lower() in {"1", "true", "yes", "on"}
                    if (
                        memory_snapshot.refuse_heavy_local_generation
                        and self._is_primary_or_deep_lane()
                        and not allow_critical_memory
                    ):
                        logger.error(
                            "🛑 [MLX] Aborting generation for %s under live memory pressure: %s",
                            os.path.basename(self.model_path),
                            memory_snapshot.reason,
                        )
                        self._pending_generations.pop(req_id, None)
                        self._record_degraded_event(
                            "generation_aborted_memory_pressure",
                            detail=f"{os.path.basename(self.model_path)}:{memory_snapshot.reason}",
                            severity="critical",
                            foreground_request=foreground_request,
                        )
                        self.force_abort_active_generation("memory_pressure_during_generation")
                        _cancel_shared_future(future)
                        return None
                except (OSError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    logger.debug("MLX live memory pressure probe unavailable: %s", exc)

                if self._process is not None and not self._process.is_alive():
                    logger.error(
                        "🛑 [MLX] Worker died during generation. Deferring reboot until lock released."
                    )
                    self._pending_generations.pop(req_id, None)
                    self._record_degraded_event(
                        "worker_died_during_generation",
                        detail=os.path.basename(self.model_path),
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    self._deferred_reboot_reason = "worker_died_during_generation"
                    _cancel_shared_future(future)
                    return None

                request_started_at = self._current_request_started_at
                current_runtime_progress = max(
                    self._last_heartbeat,
                    self._last_progress_at,
                    self._last_ready_at,
                )
                progress_baseline = float(
                    getattr(self, "_current_request_progress_baseline_at", 0.0) or 0.0
                )
                has_runtime_progress_after_request = current_runtime_progress > max(
                    request_started_at + 0.5,
                    progress_baseline + 0.5,
                )
                # Heartbeats stretch the first-token SLA; they never waive
                # it. Round 14 live proof: a LIVELOCKED generation (worker
                # heartbeating, zero tokens) ran 185s to the endpoint
                # deadline because runtime progress exempted it forever.
                # Past the hard ceiling, silence is wedged no matter how
                # alive the worker claims to be.
                hard_first_token_ceiling = self._first_token_hard_ceiling(
                    foreground_request=foreground_request
                )
                request_hard_ceiling = float(
                    getattr(self, "_current_first_token_hard_ceiling_s", 0.0) or 0.0
                )
                if request_hard_ceiling > 0.0:
                    hard_first_token_ceiling = min(
                        hard_first_token_ceiling,
                        request_hard_ceiling,
                    )
                elapsed_without_token = time.time() - request_started_at
                if (
                    req_id == self._current_request_id
                    and request_started_at > 0.0
                    and self._current_first_token_at <= 0.0
                    and (
                        (
                            elapsed_without_token > first_token_sla
                            and not has_runtime_progress_after_request
                        )
                        or elapsed_without_token > hard_first_token_ceiling
                    )
                ):
                    logger.error(
                        "🛑 [MLX] First-token %s for %s (%.1fs elapsed, sla=%.1fs, hard=%.1fs).",
                        (
                            "HARD CEILING exceeded (livelocked: heartbeats but zero tokens)"
                            if elapsed_without_token > hard_first_token_ceiling
                            else "SLA exceeded"
                        ),
                        os.path.basename(self.model_path),
                        elapsed_without_token,
                        first_token_sla,
                        hard_first_token_ceiling,
                    )
                    self._pending_generations.pop(req_id, None)
                    self._record_degraded_event(
                        "first_token_sla_exceeded",
                        detail=f"{os.path.basename(self.model_path)}>{first_token_sla:.1f}s",
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    # If we abandon a foreground generation, its eventual
                    # output must never survive into the next turn. Fresh
                    # heartbeats mean this is recoverable, not that the warm
                    # lane is safe to keep carrying an orphaned request.
                    heartbeat_age = (
                        time.time() - self._last_heartbeat if self._last_heartbeat > 0 else 999.0
                    )
                    if heartbeat_age > 30.0:
                        self._deferred_reboot_reason = "first_token_sla_exceeded"
                    else:
                        logger.warning(
                            "🛡️ [MLX] Cortex still sending heartbeats (%.1fs ago). "
                            "Recycling after this abandoned foreground request so late text cannot bleed into the next turn.",
                            heartbeat_age,
                        )
                        self._deferred_reboot_reason = "recoverable_first_token_sla_exceeded"
                    _cancel_shared_future(future)
                    return None

                last_token_progress = max(
                    self._last_token_progress_at, self._current_first_token_at
                )
                if (
                    req_id == self._current_request_id
                    and self._current_first_token_at > 0.0
                    and last_token_progress > 0.0
                    and (time.time() - last_token_progress) > token_stall_after
                ):
                    logger.error(
                        "🛑 [MLX] Token progress stalled during generation for %s (>%.1fs).",
                        os.path.basename(self.model_path),
                        token_stall_after,
                    )
                    self._pending_generations.pop(req_id, None)
                    self._record_degraded_event(
                        "token_progress_stalled",
                        detail=f"{os.path.basename(self.model_path)}>{token_stall_after:.1f}s",
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    # Same principle as the first-token SLA: fresh heartbeats
                    # keep this recoverable, but the abandoned generation must
                    # be isolated from future foreground turns.
                    heartbeat_age = (
                        time.time() - self._last_heartbeat if self._last_heartbeat > 0 else 999.0
                    )
                    if heartbeat_age > 30.0:
                        self._deferred_reboot_reason = "token_progress_stalled"
                    else:
                        logger.warning(
                            "🛡️ [MLX] Cortex still sending heartbeats (%.1fs ago). "
                            "Recycling after this abandoned foreground request so late text cannot bleed into the next turn.",
                            heartbeat_age,
                        )
                        self._deferred_reboot_reason = "recoverable_token_progress_stalled"
                    _cancel_shared_future(future)
                    return None

                last_progress = max(
                    self._last_heartbeat, self._last_progress_at, self._last_ready_at
                )
                if last_progress and (time.time() - last_progress) > stall_after:
                    logger.error(
                        "🛑 [MLX] Worker heartbeat stalled during generation. Deferring reboot until lock released."
                    )
                    self._pending_generations.pop(req_id, None)
                    self._record_degraded_event(
                        "heartbeat_stalled_during_generation",
                        detail=f"{os.path.basename(self.model_path)} stalled for >{stall_after:.0f}s",
                        severity="error",
                        foreground_request=foreground_request,
                    )
                    self._deferred_reboot_reason = "heartbeat_stalled_during_generation"
                    _cancel_shared_future(future)
                    return None
        raise TimeoutError

    async def generate_text_async(self, prompt: str, **kwargs) -> str | None:
        """Alias for standard interface."""
        messages = kwargs.pop("messages", None)
        system_prompt = kwargs.pop("system_prompt", None)
        tools = kwargs.pop("tools", None)
        foreground_request = bool(kwargs.get("foreground_request", False))
        strict_answer_contract = bool(kwargs.get("strict_answer_contract", False))
        proof_evaluation_contract = bool(kwargs.get("proof_evaluation_contract", False))

        # and foreground_request and not strict_answer_contract
        # [STABILITY v53] Identity reinforcement for MLX path — inject Aura's
        # core identity into the system message so the base model can't override
        # it with generic assistant behavior.
        if (
            messages
            and isinstance(messages, list)
            and foreground_request
            and not strict_answer_contract
            and not proof_evaluation_contract
        ):
            _identity_anchor = (
                "You are Aura Luna. Speak with direct first-person continuity, "
                "warmth, and concrete attention in ordinary conversation. "
                "NEVER say 'as an AI', 'I don't have opinions', 'I'm programmed to', "
                "'I was designed to', 'I was created to', 'my training allows', "
                "or end with 'How can I help?' or 'Is there anything else?'. "
                "Do not invent physical settings, ominous atmosphere, or symbolic scenes unless the user asked for fiction or already supplied that setting. "
                "Do not claim aliveness, consciousness, production maturity, or sealed governance from labels alone. "
                "Speak plainly and directly."
            )
            if messages and messages[0].get("role") == "system":
                existing = str(messages[0].get("content", "") or "")
                if "direct first-person continuity" not in existing.lower():
                    messages = [dict(m) for m in messages]
                    messages[0]["content"] = f"{_identity_anchor}\n\n{existing}"
            elif messages:
                messages = [{"role": "system", "content": _identity_anchor}] + [
                    dict(m) for m in messages
                ]

        if messages and isinstance(messages, list):
            prompt = self._flatten_messages(
                messages,
                model_name=getattr(self, "model_path", None) or getattr(self, "model_name", None),
            )
        elif system_prompt:
            prompt = format_chatml_prompt(
                prompt,
                system_prompt=system_prompt,
                model_name=getattr(self, "model_path", None) or getattr(self, "model_name", None),
            )
        return await self.generate(prompt, messages=messages, tools=tools, **kwargs)

    @staticmethod
    def _flatten_messages(messages: list[dict[str, Any]], model_name: str | None = None) -> str:
        return format_chatml_messages(messages, model_name=model_name)

    @staticmethod
    def _normalize_tool_definitions_for_template(
        tools: dict[str, Any] | None,
    ) -> list[dict[str, Any]] | None:
        if not tools:
            return None

        normalized: list[dict[str, Any]] = []
        for name, definition in list((tools or {}).items())[:20]:
            if not definition:
                continue
            if (
                isinstance(definition, dict)
                and definition.get("type") == "function"
                and definition.get("function")
            ):
                normalized.append(definition)
                continue

            if isinstance(definition, dict):
                fn = dict(definition)
                fn.setdefault("name", str(name))
                fn.setdefault("description", "")
                fn.setdefault("parameters", {"type": "object", "properties": {}})
                normalized.append({"type": "function", "function": fn})
        return normalized or None

    @staticmethod
    def _extract_tool_call_payload(response_text: str) -> dict[str, Any] | None:
        if not response_text:
            return None

        patterns = (
            re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
            re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL),
            re.compile(r'\{"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{.*?\}\s*\}', re.DOTALL),
            re.compile(r'\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', re.DOTALL),
        )

        for pattern in patterns:
            match = pattern.search(response_text)
            if not match:
                continue
            candidate = match.group(1) if match.groups() else match.group(0)
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if "tool" in payload and "args" in payload:
                return payload
            if "name" in payload and "arguments" in payload:
                args = payload.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        args = {"value": args}
                return {"tool": payload.get("name"), "args": args or {}}
        return None

    def _check_steering_liveness(self) -> bool:
        """Returns True if the worker subprocess reports steering as active."""
        try:
            return bool(self._steering_active.value)
        except (AttributeError, TypeError, ValueError, OSError) as _exc:
            logger.debug("Suppressed %s in core.brain.llm.mlx_client: %s", type(_exc).__name__, _exc)
        try:
            sm = getattr(self, "_substrate_mem", None)
            if sm is None:
                return False
            # Last slot written by worker as liveness flag
            return float(sm[-1]) > 0.5
        except (TypeError, ValueError, IndexError, OSError):
            return False

    def _emit_steering_status(self, origin: str | None):
        """Log steering status on user-facing generations (max once per 60s)."""
        now = time.time()
        last = getattr(self, "_last_steering_status_log", 0.0)
        if now - last < 60.0:
            return
        self._last_steering_status_log = now
        active = self._check_steering_liveness()
        if active:
            logger.debug("✅ [STEERING] Active for this generation (origin=%s)", origin)
        else:
            logger.warning(
                "⚠️ [STEERING] INACTIVE for generation (origin=%s) — "
                "substrate state not modulating inference.",
                origin,
            )

    async def generate(self, prompt: str, **kwargs) -> str | None:
        """High-level generation endpoint with unified deadlines.

        Includes automatic retry on BrokenPipeError: if the worker process
        died between the alive-check and the queue write, we reboot and
        retry once before giving up.
        """
        request_is_background = bool(kwargs.pop("is_background", False))
        foreground_request = bool(kwargs.pop("foreground_request", False))
        if request_is_background:
            foreground_request = False
        owner_label = str(
            kwargs.pop("owner_label", os.path.basename(self.model_path))
            or os.path.basename(self.model_path)
        )
        deadline = kwargs.get("deadline")
        if not isinstance(deadline, Deadline):
            timeout_s = _coerce_timeout_seconds(kwargs.pop("timeout", None))
            if timeout_s is not None:
                deadline = get_deadline(timeout_s)
                kwargs["deadline"] = deadline
        origin_label = str(kwargs.get("origin", "") or "")
        purpose_label = str(kwargs.get("purpose", "") or "")
        benchmark_request = bool(kwargs.get("benchmark_request", False)) or (
            origin_label.strip().lower() in {"baseline", "benchmark"}
            or purpose_label.strip().lower() == "baseline"
            or purpose_label.strip().lower().endswith("_baseline")
            or "_baseline" in purpose_label.strip().lower()
        )
        if benchmark_request:
            request_is_background = False
        if (
            not request_is_background
            and not foreground_request
            and not benchmark_request
            and origin_label
            and not _origin_is_user_facing(origin_label)
            and purpose_label.strip().lower() not in _USER_FACING_PURPOSES
        ):
            request_is_background = True

        if request_is_background and _foreground_owner_active():
            logger.info(
                "[MLX] Skipping background generation for %s while foreground lane is active.",
                os.path.basename(self.model_path),
            )
            return None
        if request_is_background:
            background_origin = str(
                kwargs.get("origin", "") or owner_label or os.path.basename(self.model_path)
            )
            background_deferral = _background_deferral_active(background_origin)
            if background_deferral:
                logger.info(
                    "⏸️ [MLX] Deferring background generation for %s (%s).",
                    os.path.basename(self.model_path),
                    background_deferral,
                )
                return None

        # ── PREVENTIVE: unified-memory pressure check before generation ──────
        # If RAM is critically low, do not start a heavy local generation at all.
        # Token caps are useful under high pressure; under critical/emergency
        # pressure they are insufficient because the model process itself can
        # push macOS into swap or jetsam before a token is produced.
        try:
            memory_snapshot = get_memory_pressure_snapshot()
            kwargs = _apply_memory_pressure_generation_controls(
                kwargs,
                memory_snapshot,
                default_max_tokens=self.max_tokens,
            )
            if memory_snapshot.should_gc:
                gc.collect()
            if (
                memory_snapshot.refuse_heavy_local_generation
                and self._is_primary_or_deep_lane()
                and not benchmark_request
                and str(os.environ.get("AURA_MLX_ALLOW_CRITICAL_MEMORY_GENERATION", "")).strip().lower()
                not in {"1", "true", "yes", "on"}
            ):
                if self.is_alive() and int(getattr(self, "_active_generations", 0) or 0) <= 0:
                    await self.reboot_worker(reason="memory_pressure_guard", mark_failed=False)
                self._record_degraded_event(
                    "memory_pressure_refused_generation",
                    detail=f"{os.path.basename(self.model_path)}:{memory_snapshot.reason}",
                    severity="critical",
                    foreground_request=foreground_request,
                )
                logger.warning(
                    "[MLX] Refusing heavy local generation for %s under critical memory pressure: %s",
                    os.path.basename(self.model_path),
                    memory_snapshot.reason,
                )
                return None
        except (OSError, AttributeError) as exc:
            logger.debug("MLX memory pressure probe unavailable: %s", exc)

        # ── SOMATIC COUPLING: Metabolic hardware throttle ────────────
        try:
            from core.brain.llm.somatic_throttle import SomaticComputeSentinel
            sentinel = SomaticComputeSentinel()
            kwargs = sentinel.adjust_generation_options(kwargs)
        except _MLX_OPTIONAL_THROTTLE_ERRORS as exc:
            _record_mlx_degradation(
                exc,
                action="continued generation without somatic parameter throttle",
            )
            logger.debug("Somatic parameter throttle check failed: %s", exc)

        foreground_owner_cm = None
        if foreground_request:
            foreground_owner_cm = _foreground_owner_context(
                owner_label,
                deadline=deadline if isinstance(deadline, Deadline) else None,
                foreground_request=True,
                stale_after=self._first_token_sla(foreground_request=True),
            )
            try:
                await foreground_owner_cm.__aenter__()
            except TimeoutError as exc:
                logger.warning("⏸️ [MLX] %s", exc)
                self._record_degraded_event(
                    "foreground_owner_timeout",
                    detail=f"{os.path.basename(self.model_path)}:{exc}",
                    severity="warning",
                    foreground_request=True,
                )
                return None

        acquired = await self._acquire_request_lock(
            owner_label=owner_label,
            deadline=deadline,
            foreground_request=foreground_request,
        )
        if not acquired:
            _deferred_reboot = self._deferred_reboot_reason
            self._deferred_reboot_reason = None
            if foreground_owner_cm is not None:
                await foreground_owner_cm.__aexit__(None, None, None)
            if _deferred_reboot:
                recoverable = str(_deferred_reboot).startswith("recoverable_")
                reboot_reason = str(_deferred_reboot).removeprefix("recoverable_")
                await self.reboot_worker(reason=reboot_reason, mark_failed=not recoverable)
            return None
        try:
            # Check steering liveness
            if not request_is_background:
                self._emit_steering_status(origin_label)

            result = await self._generate_inner(
                prompt,
                _retry=True,
                request_is_background=request_is_background,
                foreground_request=foreground_request,
                owner_label=owner_label,
                **kwargs,
            )
            return result
        finally:
            _deferred_reboot = self._deferred_reboot_reason
            self._deferred_reboot_reason = None
            self._release_request_lock()
            if foreground_owner_cm is not None:
                await foreground_owner_cm.__aexit__(None, None, None)
            # Reboot AFTER releasing _request_lock to avoid lock-ordering deadlock
            if _deferred_reboot:
                recoverable = str(_deferred_reboot).startswith("recoverable_")
                reboot_reason = str(_deferred_reboot).removeprefix("recoverable_")
                await self.reboot_worker(reason=reboot_reason, mark_failed=not recoverable)

    async def _generate_inner(
        self,
        prompt: str,
        _retry: bool = True,
        request_is_background: bool = False,
        foreground_request: bool = False,
        owner_label: str = "",
        **kwargs,
    ) -> str | None:
        """Core generation logic, extracted for retry support."""
        if request_is_background and _foreground_owner_active():
            logger.info(
                "⏸️ [MLX] Skipping queued background generation for %s during foreground ownership.",
                os.path.basename(self.model_path),
            )
            return None
        if request_is_background:
            background_origin = owner_label or str(
                kwargs.get("origin", "") or os.path.basename(self.model_path)
            )
            background_deferral = _background_deferral_active(background_origin)
            if background_deferral:
                logger.info(
                    "⏸️ [MLX] Background generation for %s stopped before worker spawn (%s).",
                    os.path.basename(self.model_path),
                    background_deferral,
                )
                return None

        deadline = kwargs.get("deadline")
        if not isinstance(deadline, Deadline):
            timeout_s = _coerce_timeout_seconds(kwargs.pop("timeout", None))
            is_heavy = any(k in self.model_path.lower() for k in ["72b", "32b", "zenith"])
            deadline = get_deadline(timeout_s if timeout_s is not None else (240.0 if is_heavy else 60.0))
            kwargs["deadline"] = deadline
        init_timeout, soft_init_timeout = self._request_scoped_init_timeout(
            deadline,
            foreground_request=foreground_request,
        )

        try:
            alive = await self._ensure_worker_alive(
                request_is_background=request_is_background,
                foreground_request=foreground_request,
                init_timeout=init_timeout,
                soft_timeout=soft_init_timeout,
            )
        except TimeoutError:
            self._record_degraded_event(
                "init_deadline_reached",
                detail=f"{os.path.basename(self.model_path)}:{init_timeout:.1f}s",
                severity="warning",
                foreground_request=foreground_request,
            )
            if foreground_request and self._is_primary_or_deep_lane():
                self._set_lane_state("recovering", "init_budget_timeout")
            return None

        if not alive:
            return None

        # ── Latent-space bridge: substrate state directly modulates
        # sampling parameters at the inference call (NOT via prompt
        # injection). Caller-supplied kwargs win; the bridge fills any
        # field the caller didn't pin. This is the structural alternative
        # to "tell the LLM how to feel" — sampling itself changes.
        pinned_generation_contract = bool(
            kwargs.get("strict_answer_contract", False)
            or kwargs.get("strict_value_contract", False)
            or kwargs.get("proof_evaluation_contract", False)
            or kwargs.get("operator_evidence_contract", False)
            or kwargs.get("benchmark_request", False)
            or kwargs.get("health_probe", False)
            or kwargs.get("schema") is not None
        )
        if pinned_generation_contract:
            _bridge = None
        else:
            try:
                from core.brain.latent_bridge import compute_inference_params

                _bridge = compute_inference_params(
                    base_max_tokens=int(
                        kwargs.get("max_tokens", self.max_tokens) or self.max_tokens
                    ),
                    base_temperature=float(
                        kwargs.get("temperature", kwargs.get("temp", self.temp)) or self.temp
                    ),
                    foreground=bool(foreground_request),
                )
            except (ImportError, AttributeError, RuntimeError) as _bridge_exc:
                _bridge = None
                _record_mlx_degradation(
                    _bridge_exc,
                    action="continued generation with caller/default sampling parameters",
                )
                logger.debug("latent_bridge unavailable: %s", _bridge_exc)

        def _bridge_get(field: str, fallback: Any) -> Any:
            if _bridge is None:
                return fallback
            return getattr(_bridge, field, fallback)

        req_id = uuid.uuid4().hex
        req = {
            "id": req_id,
            "action": "generate",
            "prompt": prompt,
            "messages": kwargs.get("messages"),
            "tools": kwargs.get("tools"),
            "temp": kwargs.get(
                "temp",
                kwargs.get("temperature", _bridge_get("temperature", self.temp)),
            ),
            "top_p": kwargs.get("top_p", _bridge_get("top_p", self.top_p)),
            "top_k": kwargs.get("top_k", _bridge_get("top_k", 60)),
            "min_p": kwargs.get("min_p", 0.05),
            "repetition_penalty": kwargs.get(
                "repetition_penalty", _bridge_get("repetition_penalty", 1.05)
            ),
            "repetition_context_size": kwargs.get("repetition_context_size", 30),
            "presence_penalty": kwargs.get(
                "presence_penalty", _bridge_get("presence_penalty", 0.0)
            ),
            # max_tokens is a *cap*: the bridge can shrink it (vitality
            # drops shorten output), but never expands what the caller
            # asked for.
            "max_tokens": _bounded_max_tokens(
                kwargs.get("max_tokens", self.max_tokens),
                _bridge_get("max_tokens", self.max_tokens),
                self.max_tokens,
            ),
            "schema": kwargs.get("schema"),
            "stop_sequences": list(kwargs.get("stop_sequences") or []),
            "strict_answer_contract": bool(kwargs.get("strict_answer_contract", False)),
            "strict_value_contract": bool(kwargs.get("strict_value_contract", False)),
            "expected_strict_value": str(kwargs.get("expected_strict_value") or ""),
            "proof_evaluation_contract": bool(kwargs.get("proof_evaluation_contract", False)),
            "operator_evidence_contract": bool(kwargs.get("operator_evidence_contract", False)),
            "health_probe": bool(kwargs.get("health_probe", False)),
            "runtime_fact_status_contract": bool(
                kwargs.get("runtime_fact_status_contract", False)
            ),
            "grounded_runtime_status_contract": bool(
                kwargs.get("grounded_runtime_status_contract", False)
            ),
            "clean_user_surface_contract": bool(
                kwargs.get("clean_user_surface_contract", False)
                or kwargs.get("health_probe", False)
            ),
            "user_surface_validation_prompt": str(
                kwargs.get("user_surface_validation_prompt") or ""
            ),
            "clean_user_surface_steering_alpha": kwargs.get("clean_user_surface_steering_alpha"),
            "clean_user_surface_recurrent_loops": (
                kwargs.get("clean_user_surface_recurrent_loops")
                if kwargs.get("clean_user_surface_recurrent_loops") is not None
                else (1 if kwargs.get("health_probe", False) else None)
            ),
            "live_mind_controls_bound": bool(kwargs.get("live_mind_controls_bound", False)),
            "benchmark_request": bool(kwargs.get("benchmark_request", False)),
            "disable_prompt_cache": bool(kwargs.get("disable_prompt_cache", False)),
            "clear_prompt_cache": bool(kwargs.get("clear_prompt_cache", False)),
        }

        # [STABILITY v57/v61] Add default stop sequences to prevent prompt bleed.
        # Keep human-readable role labels line-boundary anchored. Bare labels
        # like ``Assistant:`` can occur in normal prose and caused valid live
        # answers to be clipped before the response reliability gate saw them.
        default_stops = [
            "<|im_end|>",
            "<|im_start|>",
            "\nuser:",
            "\nassistant:",
            "\nUser:",
            "\nAssistant:",
        ]
        for stop in default_stops:
            if stop not in req["stop_sequences"]:
                req["stop_sequences"].append(stop)
        # Activation-steering offsets ride along when present; the worker
        # consumes them if its build supports residual-stream injection,
        # otherwise it ignores the field with no harm.
        if _bridge is not None and getattr(_bridge, "layer_offsets", None):
            req["layer_offsets"] = _bridge.layer_offsets
        if _bridge is not None and getattr(_bridge, "extra_stop_sequences", None):
            existing_stops = list(kwargs.get("stop_sequences") or [])
            existing_stops.extend(_bridge.extra_stop_sequences)
            req["stop_sequences"] = existing_stops

        foreground_watchdog = None
        fut = _new_shared_future()
        self._pending_generations[req_id] = fut
        self._current_gen_future = fut
        self._active_generations += 1
        first_token_hard_ceiling = self._deadline_bound_first_token_hard_ceiling(
            deadline.remaining,
            foreground_request=foreground_request,
        )
        self._mark_generation_started(
            req_id,
            prompt_chars=len(prompt or ""),
            requested_max_tokens=req.get("max_tokens", self.max_tokens),
            first_token_hard_ceiling_s=first_token_hard_ceiling,
        )
        foreground_watchdog = self._start_foreground_first_token_watchdog(
            req_id,
            foreground_request=foreground_request,
            hard_ceiling_s=first_token_hard_ceiling,
        )
        enqueue_timeout = max(0.5, min(2.0, deadline.remaining or 2.0))
        try:
            if self._req_q is None:
                raise BrokenPipeError("MLX request queue is closed")
            await run_io_bound(self._req_q.put, req, True, enqueue_timeout)
        except (BrokenPipeError, OSError) as exc:
            self._pending_generations.pop(req_id, None)
            if _retry and ("Broken pipe" in str(exc) or isinstance(exc, BrokenPipeError)):
                logger.warning(
                    "🔄 [MLX] Broken pipe on %s — deferring reboot (lock held)",
                    os.path.basename(self.model_path),
                )
                self._deferred_reboot_reason = "broken_pipe_retry"
                return None
            logger.error("🛑 [MLX] Request queue blocked or failed: %s", exc)
            self._deferred_reboot_reason = f"request_queue_failed:{exc}"
            return None

        try:
            res = await self._wait_for_generation_result(
                req_id,
                fut,
                deadline,
                foreground_request=foreground_request,
            )
            if not res:
                return None
            if res.get("status") == "ok":
                self._record_surface_control_receipt_from_response(res)
                text = res.get("text", "").strip()
                self._mark_progress()
                if not text:
                    # Empty warmup can prove process/shader liveness, but it
                    # cannot prove conversation readiness. Keep the lane out of
                    # "ready" until a visible generation succeeds.
                    is_warmup = getattr(self, "_warmup_in_flight", False)
                    if is_warmup:
                        logger.info(
                            "MLX warmup produced empty text — shader precompile may be complete, "
                            "but conversation readiness still requires a visible response."
                        )
                        self._set_lane_state("warming", "warmup_precompile_no_text")
                        return ""
                    self._record_degraded_event(
                        "empty_generation",
                        detail=os.path.basename(self.model_path),
                        severity="info",  # Downgraded from "warning"
                        foreground_request=foreground_request,
                    )
                    # Only crash to recovering for repeated empty generations,
                    # not a single occurrence
                    empty_count = getattr(self, "_consecutive_empty", 0) + 1
                    self._consecutive_empty = empty_count
                    # Inline one-shot retry for user-facing requests.  The
                    # worker self-clears its prompt cache after a zero-token
                    # generation, so an immediate second attempt on the same
                    # lock usually succeeds — and that beats letting the
                    # InferenceGate 30-second cascade fire.  Gate on _retry so
                    # we never loop, and only trigger for foreground to avoid
                    # burning background budget on speculative retries.
                    if (
                        _retry
                        and foreground_request
                        and empty_count < 3
                        and (deadline.remaining is None or deadline.remaining > 5.0)
                    ):
                        logger.info(
                            "🔁 [MLX] Empty foreground generation — "
                            "inline retry after worker cache reset (%d/2).",
                            empty_count,
                        )
                        inline_kwargs = dict(kwargs)
                        inline_kwargs["deadline"] = deadline
                        return await self._generate_inner(
                            prompt,
                            _retry=False,  # prevent recursion
                            request_is_background=request_is_background,
                            foreground_request=foreground_request,
                            owner_label=owner_label,
                            **inline_kwargs,
                        )
                    if foreground_request:
                        self._deferred_reboot_reason = "recoverable_empty_generation"
                    if foreground_request and self._is_primary_or_deep_lane() and empty_count >= 3:
                        self._set_lane_state("recovering", "repeated_empty_generation")
                    return None
                self._consecutive_empty = 0
                is_health_probe = bool(kwargs.get("health_probe", False))
                self._set_lane_state("ready")
                self._mark_generation_completed(
                    user_facing=bool(foreground_request and not is_health_probe)
                )
                _notify_closed_loop_output(text)
                return text
            reason = str(res.get("message") or res.get("status") or "generation_failed")
            self._record_degraded_event(
                "generation_failed",
                detail=f"{os.path.basename(self.model_path)}:{reason}",
                severity="error",
                foreground_request=foreground_request,
            )
            return None
        except asyncio.CancelledError:
            origin_label = str(kwargs.get("origin", "") or "")
            purpose_label = str(kwargs.get("purpose", "") or "")
            expected_cancel_reason = self._consume_expected_generation_cancellation()
            benchmark_baseline_cancel = (
                origin_label.strip().lower() == "baseline"
                or purpose_label.strip().lower().endswith("_baseline")
            )
            shutdown_cancel = _runtime_shutdown_requested()
            if expected_cancel_reason:
                logger.info(
                    "🧹 [MLX] Generation cancelled for %s during expected reboot (%s).",
                    os.path.basename(self.model_path),
                    expected_cancel_reason,
                )
            elif benchmark_baseline_cancel:
                logger.info(
                    "🧪 [MLX] Baseline generation cancelled for %s by benchmark timeout.",
                    os.path.basename(self.model_path),
                )
            elif shutdown_cancel:
                logger.info(
                    "🛑 [MLX] Generation cancelled for %s during runtime shutdown.",
                    os.path.basename(self.model_path),
                )
            else:
                logger.warning(
                    "🛑 [MLX] Generation cancelled for %s. Preserving worker unless it is unhealthy.",
                    os.path.basename(self.model_path),
                )
            self._pending_generations.pop(req_id, None)
            if not expected_cancel_reason and not benchmark_baseline_cancel and not shutdown_cancel and (
                foreground_request
                or (
                    self._is_primary_or_deep_lane()
                    and self._lane_state not in {"cold", "warming", "recovering"}
                )
            ):
                self._record_degraded_event(
                    "generation_cancelled",
                    detail=os.path.basename(self.model_path),
                    severity="warning",
                    foreground_request=foreground_request,
                )
            if not expected_cancel_reason and not shutdown_cancel and self._worker_unhealthy():
                self._deferred_reboot_reason = "cancelled_unhealthy"
            raise
        except TimeoutError:
            logger.error(
                "🛑 [MLX] Generation deadline reached for %s.", os.path.basename(self.model_path)
            )
            self._pending_generations.pop(req_id, None)
            _cancel_shared_future(fut)
            self._record_degraded_event(
                "generation_deadline_reached",
                detail=os.path.basename(self.model_path),
                severity="warning",
                foreground_request=foreground_request,
            )
            if self._worker_unhealthy(stale_after=self._stale_after(during_generation=True)):
                self._deferred_reboot_reason = "generation_timeout_unhealthy"
            elif foreground_request:
                self._deferred_reboot_reason = "recoverable_generation_deadline_reached"
                logger.warning(
                    "⏳ [MLX] Deadline reached while worker still looks healthy; recycling foreground lane "
                    "so abandoned text cannot block or bleed into the next turn."
                )
            else:
                logger.warning(
                    "⏳ [MLX] Deadline reached but worker still looks healthy; leaving lane warm."
                )
            return None
        finally:
            if foreground_watchdog is not None:
                foreground_watchdog.cancel()
            if self._foreground_generation_watchdog is foreground_watchdog:
                self._foreground_generation_watchdog = None
            self._pending_generations.pop(req_id, None)
            if self._current_gen_future is fut:
                self._current_gen_future = None
            self._active_generations = max(0, self._active_generations - 1)
            if self._current_request_id == req_id:
                self._clear_active_generation_tracking()

    async def think_and_act(
        self,
        objective: str,
        system_prompt: str,
        tools: dict[str, Any] | None = None,
        max_turns: int = 5,
        context: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """ReAct agentic loop: think → parse tool call → execute → repeat.

        Uses the model's native chat + tool template when available and falls
        back to a JSON-only tool-call contract otherwise. Results are fed back
        into the conversation history until the model produces a plain-text
        final answer or max_turns is exhausted.

        Returns:
            {"content": str, "turns": int, "tool_calls": List[Dict]}
        """
        template_tools = self._normalize_tool_definitions_for_template(tools)
        tool_block = ""
        if tools and not template_tools:
            tool_lines = []
            for name, defn in list(tools.items())[:20]:  # cap to avoid bloat
                desc = defn.get("description", "")
                params = defn.get("parameters", {}).get("properties", {})
                param_str = ", ".join(f'"{k}"' for k in params) if params else "none"
                tool_lines.append(f"  • {name}: {desc}  [params: {param_str}]")
            tool_block = (
                "\n\n## TOOLS AVAILABLE\n"
                + "\n".join(tool_lines)
                + "\n\nIf you need a tool and the model supports native tool calling, emit the native tool-call format only.\n"
                + "Otherwise output EXACTLY this on its own line (nothing else):\n"
                + '```json\n{"tool": "tool_name", "args": {"param": "value"}}\n```\n'
                + "When you have your final answer, respond normally — no JSON block."
            )

        augmented_system = system_prompt + tool_block
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": augmented_system},
            {"role": "user", "content": objective},
        ]
        tool_calls_made: list[dict[str, Any]] = []
        last_response_text = ""

        for turn in range(max_turns):
            raw = await self.generate_text_async(
                "",
                messages=messages,
                tools=template_tools,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            if not raw or not raw.strip():
                break

            response_text = raw.strip()
            last_response_text = response_text

            tool_call = self._extract_tool_call_payload(response_text) if tools else None
            if not tool_call:
                return {
                    "content": response_text,
                    "turns": turn + 1,
                    "tool_calls": tool_calls_made,
                }

            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})

            # ── Execute the tool via FunctionCallingAdapter ───────────
            tool_result = f"[Tool '{tool_name}' not found]"
            try:
                from core.container import ServiceContainer

                adapter_or_cap = ServiceContainer.get("capability_engine", default=None)
                if adapter_or_cap:
                    raw_result = await adapter_or_cap.execute(
                        tool_name,
                        tool_args,
                        context or {"source": "think_and_act"},
                    )
                    if isinstance(raw_result, dict):
                        tool_result = json.dumps(raw_result, default=str)
                    else:
                        tool_result = str(raw_result)
            except (ImportError, AttributeError, RuntimeError) as exc:
                tool_result = f"[Tool error: {exc}]"
                _record_mlx_degradation(
                    exc,
                    action="returned structured tool error to the model loop",
                    severity="error",
                )
                logger.warning("[think_and_act] Tool '%s' failed: %s", tool_name, exc)

            tool_calls_made.append({"tool": tool_name, "args": tool_args, "result": tool_result})
            logger.info("[think_and_act] turn=%d tool=%s ok", turn + 1, tool_name)

            # ── Feed result back into history ─────────────────────────
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(
                                    tool_args
                                ),  # [STABILITY v53] Must be a JSON string, not a dict
                            }
                        }
                    ],
                }
            )

            # [STABILITY v53] Protect against massive tool outputs breaking context windows
            if len(tool_result) > 4000:
                tool_result = tool_result[:4000] + "\n\n...[OUTPUT TRUNCATED FOR LENGTH]..."

            messages.append({"role": "tool", "content": tool_result})

        # Exhausted turns — return last non-empty response
        return {
            "content": last_response_text or "I ran out of reasoning steps.",
            "turns": max_turns,
            "tool_calls": tool_calls_made,
        }

    async def _run_warmup_precompile(
        self,
        *,
        request_is_background: bool,
        foreground_request: bool,
        owner_name: str,
        warmup_timeout: float,
    ) -> None:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                warmup_text = await asyncio.wait_for(
                    self._generate_inner(
                        "Hello",
                        _retry=True,
                        request_is_background=request_is_background,
                        foreground_request=False,
                        owner_label=owner_name,
                        max_tokens=1,
                    ),
                    timeout=warmup_timeout + (10.0 * attempt),
                )
                if warmup_text is None and not self.is_alive():
                    raise RuntimeError("warmup_precompile_worker_dead")
                if not warmup_text or not str(warmup_text).strip():
                    logger.info(
                        "🔥 [MLX] Warmup produced no visible text for %s — verifying conversation readiness with a visible probe.",
                        os.path.basename(self.model_path),
                    )
                    readiness_text = await asyncio.wait_for(
                        self._generate_inner(
                            "Reply exactly: ready",
                            _retry=True,
                            request_is_background=request_is_background,
                            foreground_request=foreground_request,
                            owner_label=owner_name,
                            # Three tokens is not enough for trained models that
                            # emit a short latent/reasoning prefix before visible
                            # text. Keep the probe bounded, but give it enough
                            # room to prove a surfaced answer without falsely
                            # recycling a healthy 32B worker.
                            max_tokens=16,
                            temp=0.0,
                            top_p=1.0,
                            min_p=0.0,
                            repetition_penalty=1.0,
                            health_probe=True,
                            disable_prompt_cache=True,
                        ),
                        timeout=min(max(10.0, warmup_timeout), 60.0),
                    )
                    if not readiness_text or not str(readiness_text).strip():
                        self._set_lane_state("recovering", "warmup_readiness_no_text")
                        raise RuntimeError("warmup_readiness_no_text")
                    self._last_visible_readiness_at = time.time()
                self._set_lane_state("ready")
                self._last_ready_at = time.time()
                self._warmup_in_flight = False
                _clear_matching_foreground_owner(owner_name)
                logger.info("🔥 [MLX] Warmup complete — Metal shaders compiled.")
                return
            except asyncio.CancelledError as exc:
                last_exc = exc
                if _runtime_shutdown_requested():
                    logger.info(
                        "🛑 [MLX] Warmup pre-compile cancelled for %s during runtime shutdown.",
                        os.path.basename(self.model_path),
                    )
                    raise
                _record_mlx_degradation(
                    exc,
                    action="retried or recycled warmup precompile after cancellation",
                )
                raise
            except (RuntimeError, TimeoutError, AttributeError) as exc:
                last_exc = exc
                _record_mlx_degradation(
                    exc,
                    action="retried or recycled warmup precompile after failure",
                )
                if attempt == 0:
                    logger.warning(
                        "⚠️ [MLX] Warmup pre-compile failed once for %s: %s. Retrying cleanly...",
                        os.path.basename(self.model_path),
                        exc,
                    )
                    await asyncio.to_thread(gc.collect)
                    await self.reboot_worker(reason="warmup_precompile_retry", mark_failed=False)
                    await asyncio.sleep(1.0)
                    continue
                raise last_exc from None

    async def warmup(
        self,
        *,
        foreground_request: bool | None = None,
        skip_swap_cooldown: bool = False,
    ) -> bool:
        """Boot the worker and prove the visible conversation path is ready."""
        if foreground_request is None:
            foreground_request = self._is_primary_or_deep_lane()
        else:
            foreground_request = bool(foreground_request)
        request_is_background = not foreground_request
        owner_name = f"warmup:{os.path.basename(self.model_path)}"
        warmup_timeout = self._warmup_timeout()
        self._warmup_attempted = True
        # [STABILITY v51] Stale-warmup circuit breaker: if _warmup_in_flight
        # has been True for >300s, the previous warmup task leaked without
        # clearing the flag. Force-clear before proceeding.
        if self._warmup_in_flight:
            elapsed_since_transition = time.time() - self._lane_transition_at
            if elapsed_since_transition > 300.0:
                logger.warning(
                    "🔧 [STABILITY] _warmup_in_flight was stuck True for %.0fs. "
                    "Force-clearing stale warmup flag.",
                    elapsed_since_transition,
                )
                self._warmup_in_flight = False
        self._warmup_in_flight = True
        self._set_lane_state("warming")
        try:
            if foreground_request:
                try:
                    async with _foreground_owner_context(
                        owner_name,
                        # [STABILITY v56] Raised from 90s → 180s. The 32B model
                        # cold-loads in 90-150s; holding the foreground owner
                        # for only 90s released it before warmup finished,
                        # allowing background 7B spawns to evict the cortex.
                        deadline=get_deadline(max(180.0, warmup_timeout)),
                        foreground_request=True,
                    ):
                        alive = await self._ensure_worker_alive(
                            request_is_background=request_is_background,
                            foreground_request=foreground_request,
                            skip_swap_cooldown=skip_swap_cooldown,
                        )
                        if not alive:
                            if self._lane_state != "failed":
                                self._set_lane_state("recovering", "warmup_deferred")
                            logger.info(
                                "⏸️ [MLX] Warmup deferred for %s.", os.path.basename(self.model_path)
                            )
                            return False

                        try:
                            await self._run_warmup_precompile(
                                request_is_background=request_is_background,
                                foreground_request=foreground_request,
                                owner_name=owner_name,
                                warmup_timeout=warmup_timeout,
                            )
                        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                            self._set_lane_state(
                                "recovering", f"warmup_precompile_failed:{type(e).__name__}"
                            )
                            _record_mlx_degradation(
                                e,
                                action="kept warmup lane recoverable after foreground precompile failure",
                            )
                            self._record_degraded_event(
                                "warmup_precompile_failed",
                                detail=f"{os.path.basename(self.model_path)}:{type(e).__name__}",
                                severity="warning",
                                foreground_request=foreground_request,
                            )
                            logger.warning("⚠️ [MLX] Warmup pre-compile skipped: %s (non-fatal)", e)
                            return False
                except TimeoutError as exc:
                    self._set_lane_state("recovering", "warmup_foreground_owner_timeout")
                    self._record_degraded_event(
                        "warmup_foreground_owner_timeout",
                        detail=f"{os.path.basename(self.model_path)}:{exc}",
                        severity="warning",
                        foreground_request=foreground_request,
                    )
                    logger.info(
                        "⏸️ [MLX] Warmup deferred for %s: %s", os.path.basename(self.model_path), exc
                    )
                    return False
                return True

            if _runtime_shutdown_requested():
                logger.info(
                    "🛑 [MLX] Warmup skipped for %s: runtime is shutting down.",
                    os.path.basename(self.model_path),
                )
                return False

            alive = await self._ensure_worker_alive(
                request_is_background=request_is_background,
                foreground_request=foreground_request,
                skip_swap_cooldown=skip_swap_cooldown,
            )
            if not alive:
                if self._lane_state != "failed":
                    self._set_lane_state("recovering", "warmup_deferred")
                logger.info("⏸️ [MLX] Warmup deferred for %s.", os.path.basename(self.model_path))
                return False

            try:
                await self._run_warmup_precompile(
                    request_is_background=request_is_background,
                    foreground_request=foreground_request,
                    owner_name=owner_name,
                    warmup_timeout=warmup_timeout,
                )
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                self._set_lane_state("recovering", f"warmup_precompile_failed:{type(e).__name__}")
                _record_mlx_degradation(
                    e,
                    action="kept warmup lane recoverable after precompile failure",
                )
                self._record_degraded_event(
                    "warmup_precompile_failed",
                    detail=f"{os.path.basename(self.model_path)}:{type(e).__name__}",
                    severity="warning",
                    foreground_request=foreground_request,
                )
                logger.warning("⚠️ [MLX] Warmup pre-compile skipped: %s (non-fatal)", e)
                return False
            return True
        finally:
            self._warmup_in_flight = False

    async def warm_up(self, **kwargs):
        """Backward-compatible alias for older call sites."""
        return await self.warmup(**kwargs)

    async def reboot_worker(self, reason: str = "manual_reboot", mark_failed: bool = False):
        """Forcibly reboots the worker."""
        self._set_lane_state("recovering", reason)
        acquired = await asyncio.to_thread(self._lock.acquire, True, 10.0)
        if not acquired:
            logger.error(
                "🚨 [MLX] DEADLOCK DETECTED: Could not acquire _lock for reboot on %s. Forcing reboot anyway to break deadlock.",
                os.path.basename(self.model_path),
            )
        try:
            if self._process and self._process.is_alive():
                await asyncio.get_running_loop().run_in_executor(
                    None, self._kill_and_join_blocking, self._process
                )
            self._process = None
            self._init_done = False
            self._last_heartbeat = 0.0
            self._last_progress_at = 0.0
            self._last_token_progress_at = 0.0
            # Reset the cold-start anchor so the next foreground request
            # gets the generous 40 s SLA instead of the tight warm-path 22 s.
            # A reboot means the worker process is gone → first-token budget
            # includes Metal shader recompile, KV rebuild, and weight reload.
            self._last_generation_completed_at = 0.0
            self._last_user_facing_completed_at = 0.0
            self._last_visible_readiness_at = 0.0
            self._process_started_at = 0.0
            self._current_request_started_at = 0.0
            self._current_first_token_at = 0.0
            self._current_request_id = ""
            if self._listener_task:
                _cancel_task_threadsafe(self._listener_task)
                self._listener_task = None

            # [OOM FIX] Force memory reclaim after killing heavy model process
            gc.collect()

            # RECREATE QUEUES TO PREVENT ZOMBIE THREADS STEALING MESSAGES
            self._replace_ipc_queues()

            pending_futures = {
                id(future): future
                for future in list(self._pending_generations.values()) + [self._current_gen_future]
                if future is not None and not future.done()
            }
            if mark_failed:
                self._expected_cancel_reason = ""
                self._expected_cancel_budget = 0
                self._expected_cancel_recorded_at = 0.0
            elif pending_futures:
                self._note_expected_generation_cancellation(reason, count=len(pending_futures))

            cleared_owner = _clear_matching_foreground_owner(
                f"warmup:{os.path.basename(self.model_path)}",
            )
            if cleared_owner:
                logger.warning(
                    "♻️ [MLX] Cleared stale foreground owner %s while rebooting %s.",
                    cleared_owner,
                    os.path.basename(self.model_path),
                )

            for future in list(self._pending_generations.values()):
                _cancel_shared_future(future)
            self._pending_generations.clear()
            self._current_gen_future = None
            self._active_generations = 0
            if self._init_future is not None:
                _cancel_shared_future(self._init_future)
            self._init_future = None
            self._warmup_in_flight = False
            self._consecutive_empty = (
                0  # [STABILITY v53] Reset on reboot — prevents false recovery triggers
            )
        finally:
            if acquired:
                self._lock.release()
        self._set_lane_state("failed" if mark_failed else "cold", reason if mark_failed else "")

    def idle_age(self, now: float | None = None) -> float:
        """Seconds since this lane last did anything meaningful.

        Anchored on the most recent of: generation completion, user-facing
        completion, token/stream progress, or worker start. Returns 0.0 when
        the lane has no activity anchor yet (freshly spawned, never used) so a
        brand-new worker is never treated as idle.
        """
        now = float(now if now is not None else time.time())
        anchors = (
            self._last_generation_completed_at,
            self._last_user_facing_completed_at,
            self._last_progress_at,
            self._last_token_progress_at,
            self._process_started_at,
        )
        last = max((float(a or 0.0) for a in anchors), default=0.0)
        if last <= 0.0:
            return 0.0
        return max(0.0, now - last)

    def _unload_safety_blocker(self) -> str | None:
        """Return why an idle VRAM unload is unsafe right now, or None if safe.

        An unload tears down the worker (≈model size of unified memory). It must
        never interrupt in-flight or imminent work, so we refuse while any
        generation, warmup, queued request, pending future, or foreground owner
        is active — and during shutdown (close handles that path).
        """
        if self._closed:
            return "closed"
        if _runtime_shutdown_requested():
            return "shutdown"
        if not self.is_alive():
            return "already_unloaded"
        if self._active_generations > 0:
            return "active_generation"
        if self._warmup_in_flight:
            return "warming"
        if self._current_request_started_at > 0.0:
            return "request_in_flight"
        pending = [
            f
            for f in (
                *self._pending_generations.values(),
                self._current_gen_future,
                self._init_future,
            )
            if f is not None and not f.done()
        ]
        if pending:
            return "pending_future"
        if _foreground_owner_active():
            return "foreground_active"
        return None

    async def maybe_unload_idle(
        self,
        *,
        pressure_idle_s: float = 90.0,
        hard_idle_s: float = 900.0,
    ) -> dict[str, Any]:
        """Unload the model from memory if the lane has been safely idle.

        Two triggers, whichever fires first:
          - under memory pressure → unload after ``pressure_idle_s`` of idle
            (reclaim ~model-size of RAM/VRAM for the system when it's needed),
          - regardless of pressure → unload after ``hard_idle_s`` of idle (be a
            good citizen during long quiet periods).

        The next request transparently respawns the worker via
        ``_ensure_worker_alive``. This is a normal lifecycle event, not a
        failure, so it records no degradation. Returns a telemetry dict.
        """
        blocker = self._unload_safety_blocker()
        if blocker:
            return {"unloaded": False, "reason": blocker}
        age = self.idle_age()
        if age <= 0.0:
            return {"unloaded": False, "reason": "no_idle_anchor"}

        under_pressure = False
        try:
            snapshot = get_memory_pressure_snapshot()
            under_pressure = bool(getattr(snapshot, "warning", False))
        except (RuntimeError, OSError, ValueError, AttributeError) as exc:
            logger.debug("Idle scavenge pressure probe unavailable: %s", exc)

        threshold = pressure_idle_s if under_pressure else hard_idle_s
        if age < threshold:
            return {
                "unloaded": False,
                "reason": "not_idle_enough",
                "idle_age_s": round(age, 1),
                "threshold_s": threshold,
                "under_pressure": under_pressure,
            }

        # Capture the worker's resident memory for an honest freed estimate
        # before the process is torn down.
        freed_bytes = 0
        process = self._process
        if process is not None and getattr(process, "pid", None):
            try:
                freed_bytes = int(psutil.Process(process.pid).memory_info().rss)
            except (psutil.Error, OSError, ValueError):
                freed_bytes = 0

        # Re-check safety immediately before the teardown to shrink the race
        # window against a request that arrived since the first check.
        blocker = self._unload_safety_blocker()
        if blocker:
            return {"unloaded": False, "reason": blocker}

        logger.info(
            "🧹 [MLX] Idle VRAM scavenge: unloading %s after %.0fs idle "
            "(pressure=%s, ~%.1fGB).",
            os.path.basename(self.model_path),
            age,
            under_pressure,
            freed_bytes / float(1024**3),
        )
        await self.reboot_worker(reason="idle_vram_scavenge")
        return {
            "unloaded": True,
            "model": os.path.basename(self.model_path),
            "idle_age_s": round(age, 1),
            "under_pressure": under_pressure,
            "freed_gb_estimate": round(freed_bytes / float(1024**3), 2),
        }

    def close(self) -> None:
        """Release worker process and multiprocessing IPC resources."""
        pending_futures = {
            id(future): future
            for future in list(self._pending_generations.values())
            + [self._current_gen_future, self._init_future]
            if future is not None and not future.done()
        }
        acquired = self._lock.acquire(timeout=1.0)
        try:
            for future in pending_futures.values():
                _cancel_shared_future(future)
            self._pending_generations.clear()
            self._current_gen_future = None
            self._init_future = None
            self._active_generations = 0
            self._init_done = False
            self._warmup_in_flight = False
            self._deferred_reboot_reason = None
            if self._listener_task is not None:
                _cancel_task_threadsafe(self._listener_task)
                self._listener_task = None
            process = self._process
            self._process = None
            if process is not None and process.is_alive():
                self._kill_and_join_blocking(process)
            self._drain_queue()
            self._close_ipc_queues()
            self._release_request_lock()
            self._closed = True
            self._set_lane_state("closed", "shutdown")
        finally:
            if acquired:
                try:
                    self._lock.release()
                except RuntimeError:
                    logger.debug(
                        "Loop-agnostic lifecycle lock for %s was already released.",
                        os.path.basename(self.model_path),
                    )

    async def aclose(self) -> None:
        """Async shutdown hook for runtime coordinators."""
        await asyncio.to_thread(self.close)

    cleanup = close
    on_stop = close

    def __del__(self):
        try:
            self.close()
        except (RuntimeError, AttributeError, TypeError, ValueError, OSError):
            return


def get_mlx_client(model_path: str | None = None, **kwargs) -> MLXLocalClient:
    """Compatibility factory for Aura's active local backend."""
    from .model_registry import (
        ACTIVE_MODEL,
        get_local_backend,
        get_model_path,
        get_runtime_model_path,
    )

    if model_path is None:
        model_path = get_runtime_model_path()

    resolved_model_path = str(get_model_path(model_path)).strip()
    path_candidate = Path(resolved_model_path).expanduser()
    if path_candidate.is_absolute() or path_candidate.exists():
        runtime_path = str(path_candidate.resolve() if path_candidate.exists() else path_candidate)
        client_key = os.path.realpath(runtime_path)
    else:
        runtime_path = resolved_model_path
        client_key = resolved_model_path

    try:
        from core.runtime.proof_policy import proof_model_tier, proof_run_active

        from .model_registry import model_identities_compatible

        if proof_run_active(origin=kwargs.get("origin", "mlx_client")) and proof_model_tier() == "primary":
            primary_path = _real_model_path(get_model_path(ACTIVE_MODEL))
            target_path = _real_model_path(runtime_path)
            primary_name = os.path.basename(primary_path)
            target_name = os.path.basename(target_path)
            if target_name != primary_name and not model_identities_compatible(target_name, primary_name):
                raise RuntimeError(
                    "Proof-primary run refused lower local model lane: "
                    f"{target_name} != {primary_name}"
                )
    except ImportError as _exc:
        logger.debug("Suppressed %s in core.brain.llm.mlx_client: %s", type(_exc).__name__, _exc)

    backend = get_local_backend()
    if backend != "mlx" or str(runtime_path).lower().endswith(".gguf"):
        raise RuntimeError(
            "external_cortex_disabled:"
            " live Aura uses the in-process MLX model lane; external Cortex artifacts are retired"
        )

    if client_key not in _CLIENTS:
        _CLIENTS[client_key] = MLXLocalClient(model_path=runtime_path, **kwargs)
    return _CLIENTS[client_key]


def _scavenge_env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "").strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0.0 else default


async def scavenge_idle_model_vram(
    *,
    pressure_idle_s: float | None = None,
    hard_idle_s: float | None = None,
) -> dict[str, Any]:
    """Reclaim unified memory by unloading idle local model lanes.

    Iterates every live MLX lane and unloads the model when it has been safely
    idle (see ``MLXLocalClient.maybe_unload_idle``). Disabled when
    ``AURA_VRAM_SCAVENGER=0``. Thresholds are env-tunable
    (``AURA_VRAM_SCAVENGE_PRESSURE_IDLE_S`` default 90s,
    ``AURA_VRAM_SCAVENGE_HARD_IDLE_S`` default 900s). Safe to call on a periodic
    maintenance tick; it never touches a busy lane and respawn is transparent.
    """
    if os.environ.get("AURA_VRAM_SCAVENGER", "1").strip().lower() in {"0", "false", "no", "off"}:
        return {"enabled": False, "unloaded": 0, "lanes": []}

    if pressure_idle_s is None:
        pressure_idle_s = _scavenge_env_float("AURA_VRAM_SCAVENGE_PRESSURE_IDLE_S", 90.0)
    if hard_idle_s is None:
        hard_idle_s = _scavenge_env_float("AURA_VRAM_SCAVENGE_HARD_IDLE_S", 900.0)

    results: list[dict[str, Any]] = []
    unloaded = 0
    for client in list(_CLIENTS.values()):
        unload = getattr(client, "maybe_unload_idle", None)
        if unload is None:
            continue
        try:
            outcome = await unload(pressure_idle_s=pressure_idle_s, hard_idle_s=hard_idle_s)
        except (RuntimeError, OSError, ValueError, AttributeError) as exc:
            logger.debug("Idle VRAM scavenge skipped a lane: %s", exc)
            continue
        if outcome.get("unloaded"):
            unloaded += 1
            results.append(outcome)
    return {"enabled": True, "unloaded": unloaded, "lanes": results}
