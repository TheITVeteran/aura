from __future__ import annotations

import asyncio
import concurrent.futures as cfutures
import contextlib
import fcntl
import gc
import json
import logging
import math
import multiprocessing as mp
import os
import queue
import re
import stat
import subprocess
import sys
import threading as _threading
import time
import uuid
from collections.abc import AsyncIterator
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.runtime import resource_psutil as psutil

if TYPE_CHECKING:
    from core.brain.lane_admission import ActiveLane

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, declare
from core.runtime.resource_observation import get_resource_observer
from core.runtime.shutdown_coordinator import (
    is_shutdown_requested,
    record_shutdown_admission_event,
)
from core.runtime.shutdown_execution import run_sync_shutdown_callable_blocking
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.concurrency import run_io_bound
from core.utils.deadlines import Deadline, get_deadline
from core.utils.memory_monitor import get_memory_pressure_snapshot
from core.utils.task_tracker import get_task_tracker

from .chat_format import format_chatml_messages, format_chatml_prompt
from .mlx_worker import _mlx_worker_loop

logger = logging.getLogger("LLM.MLX")
_AURA_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def _observed_process_rss_bytes(pid: int) -> int:
    try:
        process = get_resource_observer().process(int(pid))
        return int(process.rss_bytes) if process is not None else 0
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return 0

_MODEL_LOAD_FOREGROUND_ADMISSION_TIMEOUT_FLAG = declare(
    "AURA_FOREGROUND_MODEL_LOAD_ADMISSION_TIMEOUT_S",
    kind=FlagKind.FLOAT,
    default=30.0,
    description="Maximum foreground wait for the canonical model-load lease",
    owner="core.brain.llm.mlx_client",
)
_MODEL_LOAD_BACKGROUND_ADMISSION_TIMEOUT_FLAG = declare(
    "AURA_BACKGROUND_MODEL_LOAD_ADMISSION_TIMEOUT_S",
    kind=FlagKind.FLOAT,
    default=0.0,
    description="Maximum background wait for the canonical model-load lease",
    owner="core.brain.llm.mlx_client",
)


_HEAVY_LANE_NAME_TOKENS = ("32b", "72b", "zenith", "solver", "cortex")


# Ceiling applied when the somatic throttle cannot report. Not a refusal —
# this is a throttle, and refusing generation for a metabolic hiccup would
# take conversation down — but not the wide-open default either.
_UNTHROTTLED_FALLBACK_MAX_TOKENS = 1024


def _apply_unthrottled_fallback_ceiling(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Damp generation when body pressure could not be consulted.

    Mutates and returns the SAME mapping rather than a copy. The throttle's
    success path rebinds kwargs, but its failure path historically left the
    caller's object untouched, and callers downstream rely on that identity;
    swapping in a copy here silently detached their later mutations.
    """
    if not isinstance(kwargs, dict):
        return kwargs
    requested = kwargs.get("max_tokens")
    if requested is None:
        # IMPOSE NOTHING. Several internal paths — the warmup precompile
        # probe above all — deliberately omit max_tokens and rely on their
        # own budgeting; putting a number there changes what those paths do
        # and, in the warmup case, left a durable owner unreleased. This is
        # a ceiling on an over-large request, not a default for callers who
        # never asked for one.
        return kwargs
    try:
        current = int(requested)
    except (TypeError, ValueError):
        # A malformed budget is not a budget; clamp it to something sane.
        kwargs["max_tokens"] = _UNTHROTTLED_FALLBACK_MAX_TOKENS
        return kwargs
    if current > _UNTHROTTLED_FALLBACK_MAX_TOKENS:
        kwargs["max_tokens"] = _UNTHROTTLED_FALLBACK_MAX_TOKENS
    return kwargs


def _model_is_heavy_lane(model_path: str | None) -> bool:
    """True when a path names one of the big resident lanes.

    Measured first: get_model_artifact_profile reads the artifact's own
    config/weight index and derives a parameter count, so a renamed or
    aliased checkpoint is still classified by what it IS. The historical
    name tokens are unioned in rather than replaced — this predicate gates
    memory admission, and for a safety guard the fail-safe direction is to
    over-include. A profile that cannot be read falls back to the tokens
    alone, which is exactly the old behaviour.
    """
    path = str(model_path or "")
    if not path:
        return False
    measured = False
    try:
        from core.brain.llm.model_artifact_profile import model_is_heavy

        measured = bool(model_is_heavy(path))
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError, TypeError) as exc:
        _record_mlx_degradation(
            exc,
            action="model artifact profile unavailable; lane class fell back to naming",
            severity="debug",
        )
    lowered = os.path.basename(path).lower()
    named = any(token in lowered for token in _HEAVY_LANE_NAME_TOKENS)
    return measured or named


_DEEP_SOLVER_NAME_TOKENS = ("72b", "solver")


def _model_is_deep_solver_lane(model_path: str | None) -> bool:
    """True for the optional local deep Solver lane (the 72B class).

    Same measured-first, union-with-naming rule as _model_is_heavy_lane: the
    artifact's own parameter count is the authority, and the historical name
    tokens are kept so a rename can only ever ADD caution, never remove it.
    """
    path = str(model_path or "")
    if not path:
        return False
    measured = False
    try:
        from core.brain.llm.model_artifact_profile import model_size_class

        measured = model_size_class(path) == "72b"
    except (ImportError, AttributeError, OSError, RuntimeError, ValueError, TypeError) as exc:
        _record_mlx_degradation(
            exc,
            action="model artifact profile unavailable; solver class fell back to naming",
            severity="debug",
        )
    lowered = os.path.basename(path).lower()
    return measured or any(token in lowered for token in _DEEP_SOLVER_NAME_TOKENS)


def _model_path_is_deep_solver(model_path: str | None) -> bool:
    """Return whether a model path names Aura's optional local deep Solver lane."""

    lowered = os.path.basename(str(model_path or "")).lower()
    return any(token in lowered for token in ("72b", "solver"))


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
_GLOBAL_LAST_HEAVY_MODEL: str | None = None
_CLIENTS: dict[str, Any] = {}
_FOREGROUND_OWNER_LOCK = _threading.Lock()
_FOREGROUND_OWNER_NAME: str | None = None
_FOREGROUND_OWNER_ACQUIRED_AT = 0.0
# The budget the CURRENT holder declared for itself when it took ownership.
# Eviction is judged against this, never against a newcomer's budget.
_FOREGROUND_OWNER_STALE_AFTER: float | None = None
# No foreground owner may be evicted before this, whatever anyone declares.
# A newcomer with a 5s budget must not be able to steal a lane from a turn
# that is legitimately still working.
_FOREGROUND_OWNER_MIN_EVICTION_S = 30.0

# [OOM FIX] Global gate: only ONE model can be loading at a time across ALL clients.
# This prevents the 32B and 7B from loading simultaneously and exceeding GPU RAM.
# Uses threading.Semaphore (loop-agnostic) because the singleton MLXLocalClient
# is constructed from one event loop but called from another (Uvicorn thread).
_GLOBAL_SPAWN_GATE = _threading.Semaphore(1)
# Gate holders may legitimately spend minutes loading the 32B, but waiters must
# defer quickly. A waiter is not the load owner and must never consume a whole
# foreground turn merely waiting for the mechanical single-spawn mutex.
try:
    _SPAWN_GATE_ACQUIRE_TIMEOUT_S = max(
        0.05, float(os.environ.get("AURA_SPAWN_GATE_ACQUIRE_TIMEOUT_S", "5"))
    )
except (TypeError, ValueError):
    _SPAWN_GATE_ACQUIRE_TIMEOUT_S = 5.0
_GLOBAL_SPAWN_GATE_STATE_LOCK = _threading.Lock()
_GLOBAL_SPAWN_GATE_TOKEN = ""
_GLOBAL_SPAWN_GATE_OWNER = ""
_GLOBAL_SPAWN_GATE_ACQUIRED_AT = 0.0
_MLX_RUNTIME_PROBE_LOCK = _threading.Lock()
_MLX_RUNTIME_PROBE: dict[str, Any] = {
    "ok": None,
    "detail": "",
    "checked_at": 0.0,
}
_MLX_RUNTIME_PROBE_CACHE_PATH = Path.home() / ".aura" / "data" / "mlx_runtime_probe.json"

# Visible conversation-readiness probe. The lane may only claim "ready" after
# this exact question comes back with an answer that actually responds to it.
_READINESS_PROBE_PROMPT = "Reply exactly: ready"
_READINESS_EXPECTED_TOKEN = "ready"
_READINESS_ANSWER_MAX_CHARS = 200
# A warmup still running after this long is stuck: cancel it (and prove it
# ended) before a replacement starts.
_WARMUP_STALE_AFTER_S = 300.0
# Reboot lock discipline: wait this much longer after the first 10s before
# treating contention as anything other than a live lifecycle operation, and
# only force an unsynchronized reboot after this many consecutive failures.
_REBOOT_LOCK_ESCALATED_WAIT_S = 35.0
_REBOOT_LOCK_FORCE_AFTER = 3
# close() is terminal, so it always completes — but it waits properly first
# instead of racing a live lifecycle operation after one second.
_CLOSE_LOCK_WAIT_S = 10.0


def _readiness_answer_accepted(text: Any) -> bool:
    """Does the readiness probe's answer actually respond to what was asked?

    CP126 b6439433: the probe asked the model to reply exactly ``ready`` and
    then accepted ANY nonblank text, so hallucinated, garbled, stale, or
    prompt-echo output proved the lane ready.

    Exact equality would be too brittle in the other direction — trained lanes
    can emit a short latent/reasoning prefix before visible text, and falsely
    recycling a healthy 32B is its own outage. Requiring the expected token to
    appear in a bounded answer that is not merely the prompt echoed back is a
    real check that the failure modes above cannot pass.
    """
    answer = str(text or "").strip()
    if not answer or len(answer) > _READINESS_ANSWER_MAX_CHARS:
        return False
    lowered = answer.lower()
    if _READINESS_EXPECTED_TOKEN not in lowered:
        return False
    # An echo of the instruction is not an answer to it.
    return "reply exactly" not in lowered
SharedFuture = asyncio.Future[Any] | cfutures.Future[Any]


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


def _finite_env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    """Env float with a fail-safe contract: malformed, NaN, or infinite
    values fall back to the default instead of poisoning admission math."""
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    if minimum is not None and value < minimum:
        return default
    return value


def _model_load_min_available_gb(model_path: str) -> float:
    def _env_float(name: str, default: float) -> float:
        return _finite_env_float(name, default, minimum=0.0)

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
        value = float(text)
    except (TypeError, ValueError):
        return None
    # A zero/negative override makes a real multi-GB worker appear free and
    # NaN/inf poisons every downstream admission sum — ignore such overrides.
    if not math.isfinite(value) or value <= 0.0:
        logger.warning(
            "Ignoring invalid projected-footprint override %s=%r.", name, raw
        )
        return None
    return value


# Model artifacts are immutable while the runtime holds them (fusion
# publishes a NEW directory), so their size is computed once per
# (path, mtime) and reused. Uncached, this rglob+stat walk ran on the
# EVENT LOOP inside model-load admission while 20GB of safetensors reads
# saturated the disk — the 5.5-8.6s loop stalls captured in
# data/error_logs/stalls/stall_1784673149 / stall_1784675621 bottom out
# exactly here (pathlib stat under _projected_footprint_from_artifact_gb).
_PATH_SIZE_CACHE: dict[tuple[str, int], float] = {}


def _path_size_gb(model_path: str) -> float:
    path = Path(str(model_path or "")).expanduser()
    try:
        if path.is_file():
            return float(path.stat().st_size) / float(1024**3)
        if not path.is_dir():
            return 0.0
        cache_key = (str(path), path.stat().st_mtime_ns)
        cached = _PATH_SIZE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        total = 0
        for child in path.rglob("*"):
            try:
                if child.is_file():
                    total += child.stat().st_size
            except OSError:
                continue
        size_gb = float(total) / float(1024**3)
        if len(_PATH_SIZE_CACHE) > 64:
            _PATH_SIZE_CACHE.clear()
        _PATH_SIZE_CACHE[cache_key] = size_gb
        return size_gb
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
        return _finite_env_float(name, default, minimum=0.0)

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
        return _finite_env_float(name, default, minimum=0.0)

    lowered = str(model_path or "").lower()
    if any(token in lowered for token in ("72b", "solver")):
        lane_default = _env_float("AURA_MLX_72B_PROCESS_RESERVE_GB", 5.0)
    elif any(token in lowered for token in ("32b", "cortex", "zenith")):
        lane_default = _env_float("AURA_MLX_32B_PROCESS_RESERVE_GB", 3.0)
    else:
        lane_default = _env_float("AURA_MLX_PROCESS_RESERVE_GB", 1.0)
    return _env_float("AURA_MLX_MODEL_LOAD_PROCESS_RESERVE_GB", lane_default)


def _declared_mlx_worker_footprint_gb(model_path: str) -> float:
    """Declared peak for the main worker plus optional in-worker model owners."""

    declared = _projected_model_footprint_gb(model_path) + _model_process_reserve_gb(
        model_path
    )
    from core.runtime.flags import FlagKind, declare

    contrastive_enabled = bool(declare(
        "AURA_CONTRASTIVE_DECODING", kind=FlagKind.BOOL, default=False,
        description="Enable contrastive decoding with an amateur model",
        owner="core.brain.llm.mlx_client",
    ).value())
    amateur_path = str(declare(
        "AURA_CONTRASTIVE_AMATEUR_MODEL", kind=FlagKind.STRING, default="",
        description="Amateur model path for contrastive decoding",
        owner="core.brain.llm.mlx_client",
    ).value() or "").strip()
    if contrastive_enabled and amateur_path and _real_model_path(amateur_path) != _real_model_path(
        model_path
    ):
        declared += _projected_model_footprint_gb(amateur_path) + 1.0
    return declared


def _memory_pressure_blocks_worker_spawn(model_path: str) -> str | None:
    # The operator bypass stays available for recovery, but it is a DECISION,
    # not a setting: it is time-bounded, use-bounded and receipted, so a flag
    # left in a launch profile cannot silently disable spawn admission for the
    # life of a deployment. When the window closes the guard re-arms itself.
    from core.brain.llm.emergency_override import consume_override

    decision = consume_override(
        "AURA_FORCE_CORTEX_WARMUP_UNDER_PRESSURE",
        guard="memory_pressure_spawn_admission",
        observed=f"spawn of {os.path.basename(model_path)}",
    )
    if decision.active:
        _record_mlx_degradation(
            RuntimeError(decision.as_detail()),
            action="bypassed memory-pressure spawn admission via governed operator override",
            severity="warning",
        )
        return None
    try:
        snapshot = get_memory_pressure_snapshot()
    except (OSError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        # Fail CLOSED: spawning a 20-40GB worker with NO capacity observation
        # is exactly the moment conservative admission matters. The caller
        # treats this like any other transient spawn blocker and retries.
        _record_mlx_degradation(
            exc,
            action="refused worker spawn while the memory probe was unavailable",
            severity="error",
        )
        return "memory_probe_unavailable"

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
    process_reserve_gb = _model_process_reserve_gb(model_path)
    projected_footprint_gb = (
        _declared_mlx_worker_footprint_gb(model_path) - process_reserve_gb
    )
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


def _observed_active_lanes(exclude_client: Any = None) -> list[ActiveLane]:
    """Snapshot every live model lane as a declared-footprint ActiveLane.

    Pull-model observation over _CLIENTS: no bookkeeping to desync. The
    candidate's own client is excluded so a worker recycle never counts its
    old footprint against its own respawn.
    """
    from core.brain.lane_admission import ActiveLane, classify_lane

    lanes: list[ActiveLane] = []
    for path, client in list(_CLIENTS.items()):
        if client is None or client is exclude_client:
            continue
        try:
            if not client.is_alive():
                continue
        except (AttributeError, RuntimeError, OSError, ValueError):
            continue
        lane, qos = classify_lane(path)
        last = float(getattr(client, "_last_user_facing_completed_at", 0.0) or 0.0)
        lanes.append(
            ActiveLane(
                lane=lane,
                qos=qos,
                footprint_gb=_declared_mlx_worker_footprint_gb(path),
                model_path=path,
                last_user_facing_age_s=(time.time() - last) if last > 0.0 else None,
            )
        )
    return lanes


def _model_lane_owner_id(client: Any) -> str:
    existing = str(getattr(client, "_model_lane_owner_id", "") or "")
    if existing:
        return existing
    model_path = _real_model_path(getattr(client, "model_path", ""))
    owner_id = f"mlx:{os.getpid()}:{model_path}"
    try:
        client._model_lane_owner_id = owner_id
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return owner_id


def _observed_model_lane_owners(exclude_client: Any = None) -> list[Any]:
    """Return process-identified MLX owners for durable lane accounting."""

    from core.brain.lane_admission import QoSClass, classify_lane
    from core.runtime.model_lane_control import (
        LaneOwnerObservation,
        process_identity_for_pid,
    )

    owners: list[LaneOwnerObservation] = []
    for path, client in list(_CLIENTS.items()):
        if client is None or client is exclude_client:
            continue
        try:
            process = getattr(client, "_process", None)
            if process is None or not process.is_alive() or not client.is_alive():
                continue
            pid = int(getattr(process, "pid", 0) or 0)
            if pid <= 0:
                continue
            lane, qos = classify_lane(path)
            last = float(getattr(client, "_last_user_facing_completed_at", 0.0) or 0.0)
            observed_gb = float(_observed_process_rss_bytes(pid)) / float(1024**3)
            owners.append(
                LaneOwnerObservation(
                    owner_id=_model_lane_owner_id(client),
                    model_path=path,
                    declared_gb=_declared_mlx_worker_footprint_gb(path),
                    observed_gb=observed_gb,
                    process=process_identity_for_pid(pid),
                    priority=10 if qos is QoSClass.GUARANTEED else 50,
                    preemptible=not bool(
                        int(getattr(client, "_active_generations", 0) or 0) > 0
                        or (
                            getattr(client, "_current_gen_future", None) is not None
                            and not client._current_gen_future.done()
                        )
                    ),
                    last_user_facing_age_s=(time.time() - last) if last > 0.0 else None,
                    metadata={
                        "runtime_pid": os.getpid(),
                        "lane": lane,
                        "fencing_token": int(
                            getattr(client, "_model_lane_fencing_token", 0) or 0
                        ),
                    },
                )
            )
        except (AttributeError, RuntimeError, OSError, TypeError, ValueError):
            continue
    return owners


async def _evict_model_lane_owner(owner: Any, reason: str) -> bool:
    """Evict one exact local MLX owner and prove its worker is dead."""

    target = next(
        (
            client
            for client in list(_CLIENTS.values())
            if client is not None and _model_lane_owner_id(client) == owner.owner_id
        ),
        None,
    )
    if target is None:
        from core.runtime.model_lane_control import evict_managed_process_owner

        return await evict_managed_process_owner(owner, reason)
    active = bool(
        int(getattr(target, "_active_generations", 0) or 0) > 0
        or getattr(target, "_warmup_in_flight", False)
        or getattr(target, "_current_request_started_at", 0.0) > 0.0
        or any(
            future is not None and not future.done()
            for future in (
                *getattr(target, "_pending_generations", {}).values(),
                getattr(target, "_current_gen_future", None),
                getattr(target, "_init_future", None),
            )
        )
    )
    if active:
        logger.info(
            "MLX model-lane preemption refused during active work owner=%s reason=%s",
            owner.owner_id,
            reason,
        )
        return False
    await target.reboot_worker(
        reason=f"yield_to_lane_transaction:{reason}",
        mark_failed=False,
    )
    try:
        alive = bool(target.is_alive())
    except (AttributeError, RuntimeError, OSError, ValueError):
        alive = True
    return not alive


async def _reclaim_model_lane_capacity(claim: Any) -> bool:
    from core.runtime.flags import FlagKind, declare
    """Wait boundedly for killed model memory to leave the observed envelope."""

    try:
        timeout_s = max(
            0.0,
            float(declare(
                "AURA_MODEL_LANE_RECLAIM_TIMEOUT_S", kind=FlagKind.FLOAT, default=20.0,
                description="Budget for reclaiming a model lane before spawn",
                owner="core.brain.llm.mlx_client",
            ).value()),
        )
    except (TypeError, ValueError):
        timeout_s = 20.0
    deadline = time.monotonic() + timeout_s
    max_observations = max(1, int(timeout_s / 0.5) + 2)
    blocker: str | None = "capacity_not_observed"
    for _attempt in range(max_observations):
        blocker = await asyncio.to_thread(
            _memory_pressure_blocks_worker_spawn,
            claim.model_path,
        )
        if blocker is None:
            return True
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            break
        await asyncio.sleep(min(0.5, remaining_s))
    logger.warning(
        "Model-lane reservation could not observe reclaimed capacity for %s: %s",
        os.path.basename(claim.model_path),
        blocker,
    )
    return False


async def _compensate_model_lane_owner(owner: Any, reason: str) -> bool:
    """Restore an owner displaced by a candidate that did not commit."""

    target = next(
        (
            client
            for client in list(_CLIENTS.values())
            if client is not None and _model_lane_owner_id(client) == owner.owner_id
        ),
        None,
    )
    if target is None:
        return False

    timeout_s = max(60.0, float(target._warmup_timeout()) + 30.0)
    try:
        restored = bool(
            await asyncio.wait_for(
                target.warmup(skip_swap_cooldown=True),
                timeout=timeout_s,
            )
        )
        ready = bool(restored and target.is_alive())
        if not ready:
            logger.warning(
                "Compensation could not restore model lane owner=%s reason=%s",
                owner.owner_id,
                reason,
            )
        return ready
    except asyncio.CancelledError:
        raise
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
        _record_mlx_degradation(
            exc,
            action="recorded failed restoration of an evicted model lane",
            severity="error",
        )
        return False


def _note_lane_worker_death(client: Any, reason: str) -> None:
    """Report a worker death to the crash-loop breaker (roadmap K4).

    Lifetime is measured from the spawn timestamp; the breaker itself
    decides whether the death counts (young + non-deliberate). Never
    throws — death accounting must not break a recovery path.
    """
    try:
        started = float(getattr(client, "_process_started_at", 0.0) or 0.0)
        if started <= 0.0:
            return
        from core.runtime.lane_reconciler import get_crash_loop_breaker

        get_crash_loop_breaker().note_death(
            _real_model_path(client.model_path),
            lifetime_s=max(0.0, time.time() - started),
            reason=reason,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Crash-loop death report skipped: %s", exc)
    # The dead worker's MODEL_LOAD admission lease must die with it: a
    # MODEL_LOAD lease conflicts with every other MODEL_LOAD lease, so an
    # unreleased lease walls every recovery load behind its TTL while each
    # retry burns to resource_timeout — the 2026-07-15 soak P0 (cortex
    # never loaded all night while RAM sat at 40%). Same seam as the K4
    # report, same never-throws contract.
    try:
        from core.brain.lane_admission import classify_lane
        from core.runtime.control_plane import WorkClass, get_runtime_control_plane

        lane, _qos = classify_lane(client.model_path)
        reaped = get_runtime_control_plane().admission.reap_dead_holder_leases_sync(
            lane=lane,
            work_class=WorkClass.MODEL_LOAD,
            reason=reason,
        )
        if reaped:
            logger.warning(
                "🧹 Reaped %d orphaned model-load admission lease(s) for dead %s worker (%s).",
                reaped,
                lane,
                reason,
            )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Dead-holder lease reap skipped: %s", exc)


def _lane_is_last_warm(client: Any) -> bool:
    """K5 disruption budget: is this client the ONLY live model lane?

    Voluntary disruptions (yields for background warmups) must never
    remove the last warm lane — a cold gap with nothing warm is strictly
    worse than deferring a background spawn.
    """
    try:
        if client is None or not client.is_alive():
            return False
        for other in list(_CLIENTS.values()):
            if other is None or other is client:
                continue
            try:
                if other.is_alive():
                    return False
            except (AttributeError, RuntimeError, OSError, ValueError):
                continue
        return True
    except (AttributeError, RuntimeError, OSError, ValueError):
        return False


def _crash_loop_blocks_worker_spawn(client: Any) -> str | None:
    """Consult the K4 crash-loop breaker before a (re)spawn. Never throws."""
    try:
        from core.runtime.lane_reconciler import get_crash_loop_breaker
    except ImportError as exc:
        # Breaker module absent from this build: proceed, but visibly.
        _record_mlx_degradation(
            exc,
            action="spawned without crash-loop breaker (module unavailable)",
        )
        return None
    try:
        blocked = get_crash_loop_breaker().blocked(_real_model_path(client.model_path))
        return str(blocked) if blocked else None
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        # Fail CLOSED: the breaker exists but is broken — during an active
        # crash storm this is precisely when unchecked respawns do damage.
        _record_mlx_degradation(
            exc,
            action="refused worker spawn while the crash-loop breaker was unavailable",
            severity="error",
        )
        return "crash_loop_breaker_unavailable"


class _ModelLoadAdmissionDeniedError(RuntimeError):
    def __init__(self, reason: str, *, receipt_id: str = "") -> None:
        self.reason = str(reason or "resource_admission_denied")
        self.receipt_id = str(receipt_id or "")
        super().__init__(
            f"model_load_admission_denied:{self.reason}"
            + (f":receipt={self.receipt_id}" if self.receipt_id else "")
        )


def _model_load_admission_timeout_s(*, foreground_request: bool) -> float:
    flag = (
        _MODEL_LOAD_FOREGROUND_ADMISSION_TIMEOUT_FLAG
        if foreground_request
        else _MODEL_LOAD_BACKGROUND_ADMISSION_TIMEOUT_FLAG
    )
    return max(0.0, float(flag.value()))


def _model_load_lease_ttl_s(client: Any) -> float:
    """Cover the complete worker handshake, not only warmup precompile.

    The primary lane allows a 300-second init handshake. The old 240-second
    lease expired while that live load still held the spawn gate, admitting a
    second load attempt behind it and creating the observed recovery cascade.
    """

    return max(
        180.0,
        float(client._warmup_timeout()) + 120.0,
        float(client._handshake_timeout()) + 120.0,
    )


@contextlib.asynccontextmanager
async def _model_load_admission_context(
    client: Any,
    *,
    foreground_request: bool,
) -> AsyncIterator[Any]:
    """Hold scheduling and durable capacity reservations through handshake.

    Required evictions complete while the durable reservation remains counted.
    The candidate is committed as a process-identified owner only after its
    worker handshake succeeds; every other exit cancels and compensates.
    """

    try:
        from core.brain.lane_admission import classify_lane
        from core.runtime.control_plane import (
            AdmissionPriority,
            AdmissionRequest,
            WorkClass,
            get_runtime_control_plane,
        )
        from core.runtime.model_lane_control import (
            LaneClaim,
            get_model_lane_controller,
            process_identity_for_pid,
        )
    except ImportError as exc:
        _record_mlx_degradation(
            exc,
            action="refused model load because canonical resource admission could not import",
            severity="critical",
        )
        raise _ModelLoadAdmissionDeniedError("resource_admission_unavailable") from exc

    lane, qos = classify_lane(client.model_path)
    # Off-loop: the footprint projection stats the whole model directory,
    # and doing that on the event loop during a concurrent 20GB model read
    # produced the recorded 5.5-8.6s admission stalls. The walk is also
    # memoized, so this thread hop is cold-path only.
    request_gb = await asyncio.to_thread(
        _declared_mlx_worker_footprint_gb, client.model_path
    )
    timeout_s = _model_load_admission_timeout_s(
        foreground_request=foreground_request
    )
    from core.brain.lane_admission import QoSClass

    # The PRIMARY cortex (GUARANTEED QoS) always loads at FOREGROUND priority,
    # even when a background prewarm task triggered it. It is the user-facing
    # default model — background priority (80) meant the fairness gate blocked
    # its load behind every continuous foreground fallback inference (priority
    # 10), forever: the cortex could never load while the fallback answered,
    # and the fallback answered because the cortex never loaded (2026-07-15
    # soak deadlock, resource_timeout). At equal priority the load and the
    # fallback inference interleave FIFO, so the cortex finally comes up.
    is_primary_cortex = qos is QoSClass.GUARANTEED
    model_load_lease_ttl_s = _model_load_lease_ttl_s(client)
    request = AdmissionRequest(
        owner=f"mlx.model_load:{os.path.basename(client.model_path)}",
        work_class=WorkClass.MODEL_LOAD,
        lane=lane,
        priority=(
            AdmissionPriority.FOREGROUND
            if (foreground_request or is_primary_cortex)
            else AdmissionPriority.BACKGROUND
        ),
        timeout_s=timeout_s,
        lease_ttl_s=model_load_lease_ttl_s,
        receipt_required=True,
        estimated_memory_mb=request_gb * 1024.0,
        metadata={
            "model_path": str(client.model_path),
            "lane_qos": str(qos),
            "foreground_request": bool(foreground_request),
            "declared_request_gb": request_gb,
        },
    )
    try:
        admission = get_runtime_control_plane().admission
        decision = await admission.acquire(request)
    except asyncio.CancelledError:
        raise
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_mlx_degradation(
            exc,
            action="refused model load because canonical resource admission failed",
            severity="critical",
        )
        raise _ModelLoadAdmissionDeniedError("resource_admission_failed") from exc
    if not decision.admitted:
        raise _ModelLoadAdmissionDeniedError(
            decision.reason,
            receipt_id=decision.receipt_id,
        )
    clear_admission_backoff = getattr(
        client,
        "_clear_model_load_admission_backoff",
        None,
    )
    if callable(clear_admission_backoff):
        clear_admission_backoff()

    lease_released = False

    async def _release_schedule_lease(reason: str) -> None:
        nonlocal lease_released
        if lease_released:
            return
        lease_released = True
        try:
            await admission.release(decision.lease_id, reason=reason)
        except KeyError:
            logger.warning(
                "Model-load admission lease expired before release lane=%s lease=%s",
                lane,
                decision.lease_id,
            )
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="model load completed but canonical admission release failed",
                severity="warning",
            )

    lane_controller = get_model_lane_controller()
    disruptive_deep_handoff = bool(
        foreground_request
        and callable(getattr(client, "_is_deep_solver_lane", None))
        and client._is_deep_solver_lane()
    )
    lane_claim = LaneClaim(
        owner_id=_model_lane_owner_id(client),
        model_path=str(client.model_path),
        request_gb=request_gb,
        priority=(
            int(AdmissionPriority.FOREGROUND)
            if (foreground_request or is_primary_cortex)
            else int(AdmissionPriority.BACKGROUND)
        ),
        foreground=foreground_request,
        allow_disruptive_eviction=disruptive_deep_handoff,
        allow_last_warm_eviction=disruptive_deep_handoff,
        reservation_ttl_s=model_load_lease_ttl_s,
        request_id=f"model-lane-{request.request_id}",
        metadata={
            "scheduling_lease_id": decision.lease_id,
            "scheduling_receipt_id": decision.receipt_id,
            "model_path": str(client.model_path),
            "lane_qos": str(qos),
            "foreground_request": bool(foreground_request),
            "disruptive_deep_handoff": disruptive_deep_handoff,
            "compensation_strategy": "mlx_warmup_exact_owner",
        },
    )
    lane_decision = None
    try:
        lane_decision = await lane_controller.reserve(
            lane_claim,
            observations=_observed_model_lane_owners(exclude_client=client),
        )
        if not lane_decision.admitted:
            await _release_schedule_lease("model_lane_reservation_refused")
            raise _ModelLoadAdmissionDeniedError(
                lane_decision.reason,
                receipt_id=lane_decision.receipt_id,
            )
        if not lane_decision.ready_to_spawn:
            lane_decision = await lane_controller.prepare(
                lane_decision,
                evict=_evict_model_lane_owner,
                observe=lambda: _observed_model_lane_owners(exclude_client=client),
                reclaim=_reclaim_model_lane_capacity,
                compensate=_compensate_model_lane_owner,
            )
        if not lane_decision.ready_to_spawn:
            await _release_schedule_lease("model_lane_eviction_or_reclamation_failed")
            raise _ModelLoadAdmissionDeniedError(
                lane_decision.reason,
                receipt_id=lane_decision.receipt_id,
            )

        yield decision
        process = getattr(client, "_process", None)
        pid = int(getattr(process, "pid", 0) or 0) if process is not None else 0
        worker_ready = bool(
            process is not None
            and process.is_alive()
            and getattr(client, "_init_done", False)
        )
        if not worker_ready:
            await _release_schedule_lease("model_load_did_not_reach_ready")
            cancelled = await lane_controller.cancel(
                lane_decision,
                reason="candidate_worker_not_ready",
                compensate=_compensate_model_lane_owner,
            )
            raise _ModelLoadAdmissionDeniedError(
                cancelled.reason,
                receipt_id=cancelled.receipt_id,
            )
        observed_gb = float(_observed_process_rss_bytes(pid)) / float(1024**3)
        try:
            committed = await lane_controller.commit(
                lane_decision,
                process=process_identity_for_pid(pid),
                observed_gb=observed_gb,
                metadata={"worker_name": str(getattr(process, "name", ""))},
            )
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="killed candidate worker because durable lane commit failed",
                severity="critical",
            )
            await client.reboot_worker(reason="lane_commit_failed", mark_failed=True)
            await _release_schedule_lease("model_lane_commit_failed")
            cancelled = await lane_controller.cancel(
                lane_decision,
                reason=f"candidate_commit_failed:{type(exc).__name__}",
                compensate=_compensate_model_lane_owner,
            )
            raise _ModelLoadAdmissionDeniedError(
                cancelled.reason,
                receipt_id=cancelled.receipt_id,
            ) from exc
        adopt_owner = getattr(client, "_adopt_durable_model_lane_owner", None)
        if callable(adopt_owner):
            adopt_owner(
                fencing_token=committed.fencing_token,
                receipt_id=committed.receipt_id,
            )
        else:
            client._model_lane_fencing_token = committed.fencing_token
            client._model_lane_terminal_receipt_id = committed.receipt_id
            from core.runtime.model_lane_control import register_model_lane_owner_adapter

            register_model_lane_owner_adapter(
                committed.owner_id,
                evict=_evict_model_lane_owner,
                compensate=_compensate_model_lane_owner,
            )
    except asyncio.CancelledError:
        await asyncio.shield(_release_schedule_lease("model_load_cancelled"))
        if lane_decision is not None and lane_decision.admitted:
            await asyncio.shield(
                lane_controller.cancel(
                    lane_decision,
                    reason="candidate_load_cancelled",
                    compensate=_compensate_model_lane_owner,
                )
            )
        raise
    except (OSError, RuntimeError, AttributeError, TypeError, ValueError):
        if lane_decision is not None and lane_decision.admitted:
            await _release_schedule_lease("model_load_failed")
            try:
                await lane_controller.cancel(
                    lane_decision,
                    reason="candidate_load_failed",
                    compensate=_compensate_model_lane_owner,
                )
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                _record_mlx_degradation(
                    exc,
                    action="model load failed and lane reservation cancellation also failed",
                    severity="critical",
                )
        raise
    finally:
        await _release_schedule_lease("model_load_finished")


def _normalize_recurrent_depth_status(status: Any, *, model_path: str) -> dict[str, Any]:
    payload = dict(status) if isinstance(status, dict) else {}
    expected_loops = payload.get("expected_loops")
    try:
        expected = (
            int(expected_loops)
            if expected_loops is not None
            else _expected_recurrent_loops_from_model_path(model_path)
        )
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
    return bool(is_shutdown_requested())


def _shutdown_blocks_model_work(model_path: str, *, action: str) -> bool:
    """Return true when shutdown has latched and model work must not start.

    This guard intentionally lives at the MLX boundary, not only in callers:
    recovery, prewarm, health, and chat paths all converge here. Once the
    process-wide shutdown latch is set, no worker spawn, warmup, or recovery
    admission may create new model work.
    """

    if not _runtime_shutdown_requested():
        return False
    record_shutdown_admission_event(
        f"mlx:{action}:{os.path.basename(str(model_path or '')) or 'unknown-model'}",
        resource_kind="mlx_worker",
        outcome="suppressed",
        detail="shutdown_latch",
    )
    logger.info(
        "🛑 [MLX] %s skipped for %s: runtime shutdown is latched.",
        action,
        os.path.basename(str(model_path or "")) or "unknown-model",
    )
    return True


def _open_spawn_lock_file(lock_file_path: str):
    """Open the spawn lock as a verified regular file we own.

    CP126 cb05a61b. The lock lived at a fixed path under the user's home and
    was opened with O_CREAT|O_WRONLY and no O_NOFOLLOW, then wrapped in write
    mode — which TRUNCATES whatever the path resolves to. Any other component
    running as the same user could replace mlx_spawn.lock with a symlink and
    have a worker spawn destroy an unrelated writable file.

    O_NOFOLLOW refuses a symlink at the final component; the fstat checks
    then confirm we are holding a regular file we own, with no extra hard
    links pointing at it. Opened r+ rather than w: a lock file is held, never
    written, so there is nothing to truncate.
    """
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        lock_fd = os.open(lock_file_path, flags, 0o600)
    except OSError as exc:
        # ELOOP/EMLINK here means the path IS a symlink: a tampering signal,
        # not a transient error.
        raise RuntimeError(
            f"mlx_spawn_lock_unsafe:{lock_file_path}:{exc.__class__.__name__}"
        ) from exc
    try:
        st = os.fstat(lock_fd)
        if not stat.S_ISREG(st.st_mode):
            raise RuntimeError(f"mlx_spawn_lock_not_regular_file:{lock_file_path}")
        if st.st_uid != os.getuid():
            raise RuntimeError(
                f"mlx_spawn_lock_foreign_owner:{lock_file_path}:uid={st.st_uid}"
            )
        if st.st_nlink != 1:
            raise RuntimeError(
                f"mlx_spawn_lock_hardlinked:{lock_file_path}:links={st.st_nlink}"
            )
        if st.st_mode & 0o077:
            # Group/other permissions on a lock another user could then hold.
            os.fchmod(lock_fd, 0o600)
    except Exception:
        os.close(lock_fd)
        raise
    return os.fdopen(lock_fd, "r+")


def _acquire_spawn_file_lock(lock_file: Any, *, model_path: str) -> None:
    """Acquire the cross-process spawn lock with timeout and shutdown polling."""

    try:
        timeout_s = max(
            1.0,
            float(os.environ.get("AURA_MLX_SPAWN_FILE_LOCK_TIMEOUT_S", "90") or 90.0),
        )
    except (TypeError, ValueError):
        timeout_s = 90.0
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _shutdown_blocks_model_work(model_path, action="spawn lock wait"):
            raise RuntimeError("runtime_shutdown")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            time.sleep(0.1)
    raise TimeoutError(
        f"mlx_spawn_file_lock_timeout:{os.path.basename(model_path)}:{timeout_s:.1f}s"
    )


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
    def _close_and_join() -> None:
        q.close()
        q.join_thread()

    try:
        run_sync_shutdown_callable_blocking(
            _close_and_join,
            timeout_s=1.0,
            name="mlx-queue-close",
        )
    except (OSError, ValueError, BrokenPipeError, TypeError, AttributeError, TimeoutError) as exc:
        logger.debug("MLX queue cleanup did not complete: %s", exc)
    else:
        try:
            from core.runtime.runtime_hygiene import get_runtime_hygiene

            get_runtime_hygiene().unregister_shutdown_resource(q)
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            pass


def _register_runtime_queue(q: mp.Queue, *, name: str) -> None:
    from core.runtime.runtime_hygiene import get_runtime_hygiene

    get_runtime_hygiene().register_shutdown_resource(
        q,
        kind="multiprocessing_queue",
        name=name,
        source="core.brain.llm.mlx_client",
        timeout_s=1.0,
    )


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


def _bounded_generation_max_tokens(
    requested: Any,
    bridged: Any,
    hard_output_ceiling: Any,
    fallback: int,
    requested_output_contract: Any = None,
) -> int:
    """Apply adaptive shrinkage without making a typed contract impossible."""

    bounded = _bounded_max_tokens(requested, bridged, fallback)
    if hard_output_ceiling is not None and hard_output_ceiling != "":
        bounded = _bounded_max_tokens(bounded, hard_output_ceiling, fallback)

    contract_floor = _requested_output_contract_generation_floor(
        requested_output_contract
    )
    if contract_floor <= 0:
        return bounded

    try:
        caller_cap = max(1, int(requested))
    except (TypeError, ValueError, OverflowError):
        caller_cap = max(1, int(fallback))
    admitted_cap = caller_cap
    if hard_output_ceiling is not None and hard_output_ceiling != "":
        try:
            admitted_cap = min(admitted_cap, max(1, int(hard_output_ceiling)))
        except (TypeError, ValueError, OverflowError):
            pass
    return max(bounded, min(contract_floor, admitted_cap))


def _requested_output_contract_generation_floor(contract: Any) -> int:
    """Return a conservative native-generation floor for a typed user contract."""

    if not isinstance(contract, dict) or not contract:
        return 0
    if bool(contract.get("exact_reply", False)):
        try:
            utf8_bytes = max(1, int(contract.get("exact_reply_utf8_bytes") or 0))
        except (TypeError, ValueError, OverflowError):
            utf8_bytes = 0
        if utf8_bytes > 0:
            # Any supported tokenizer needs no more content tokens than UTF-8
            # bytes, plus one slot for EOS/stop termination.
            return utf8_bytes + 1
    try:
        return max(0, int(contract.get("semantic_token_cap") or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


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
        "generation_max_tokens",
        "caller_requested_max_tokens",
        "adaptive_suggested_max_tokens",
        "output_contract_generation_floor",
        "generated_tokens",
        "semantic_output_token_cap",
        "hard_output_token_ceiling",
        "instruction_shape_repair_applied",
        "deterministic_repair_applied",
        "text_mutation_count",
        "exact_reply_token_count",
        "exact_reply_required_termination_headroom",
        "exact_reply_available_termination_headroom",
        "exact_reply_content_capacity_sufficient",
        "exact_reply_termination_headroom_sufficient",
        "exact_reply_token_ceiling_valid",
        "exact_reply_native_capacity_sufficient",
        "applied",
    }
    receipt = {key: value[key] for key in allowed if key in value}
    contract = value.get("requested_output_contract")
    if isinstance(contract, dict):
        contract_allowed = {
            "kind",
            "word_min",
            "word_max",
            "sentence_count",
            "explicit_brevity",
            "exact_reply",
            "exact_reply_chars",
            "exact_reply_utf8_bytes",
            "semantic_token_cap",
            "hard_token_ceiling",
            "confidence",
        }
        receipt["requested_output_contract"] = {
            key: contract[key] for key in contract_allowed if key in contract
        }
    mutations = value.get("text_mutations")
    if isinstance(mutations, list):
        from core.brain.live_mind_contract import normalize_text_mutations

        receipt["text_mutations"] = normalize_text_mutations(mutations)
        receipt["text_mutation_count"] = len(receipt["text_mutations"])
    return receipt


def _surface_quality_rejection_reasons(value: Any) -> tuple[str, ...]:
    """Identify an intentional worker quality rejection, not an empty decode."""

    receipt = value if isinstance(value, dict) else {}
    if not bool(receipt.get("surface_quality_gate_enabled")):
        return ()
    if bool(receipt.get("surface_quality_gate_passed")):
        return ()
    raw_reasons = receipt.get("surface_quality_gate_reasons")
    if not isinstance(raw_reasons, (list, tuple)):
        return ()
    return tuple(
        str(reason).strip()[:120]
        for reason in raw_reasons
        if str(reason).strip()
    )[:8]


def _coerce_timeout_seconds(value: Any) -> float | None:
    """Normalize public timeout kwargs into positive request deadlines.

    None means "caller supplied no timeout" (defaults apply downstream).
    A MALFORMED value must not silently erase the caller's intent to be
    bounded — it becomes a conservative bounded default instead.
    """
    if value is None or isinstance(value, Deadline):
        return None
    try:
        timeout_s = float(value)
    except (TypeError, ValueError, OverflowError):
        _record_mlx_degradation(
            ValueError(f"malformed generation timeout: {value!r}"),
            action="replaced malformed timeout with a bounded 120s deadline",
        )
        return 120.0
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        _record_mlx_degradation(
            ValueError(f"non-finite or non-positive generation timeout: {timeout_s!r}"),
            action="replaced invalid timeout with a bounded 120s deadline",
        )
        return 120.0
    return max(0.1, timeout_s)


def _spawn_gate_snapshot() -> dict[str, Any]:
    with _GLOBAL_SPAWN_GATE_STATE_LOCK:
        owner = _GLOBAL_SPAWN_GATE_OWNER
        acquired_at = _GLOBAL_SPAWN_GATE_ACQUIRED_AT
        token = _GLOBAL_SPAWN_GATE_TOKEN
    return {
        "held": bool(token),
        "owner": owner,
        "acquired_at_monotonic": acquired_at,
        "age_s": (
            max(0.0, time.monotonic() - acquired_at)
            if token and acquired_at > 0.0
            else 0.0
        ),
    }


@contextlib.asynccontextmanager
async def _spawn_gate_context(
    *, owner: str = "unknown"
) -> AsyncIterator[dict[str, Any]]:
    """Cancellation-safe, bounded ownership of the global spawn gate.

    A blocking ``Semaphore.acquire`` delegated with ``asyncio.to_thread`` is
    not cancellation-safe: cancelling the coroutine does not stop its thread.
    That abandoned thread can later acquire the semaphore with no surviving
    context manager to release it. Foreground recovery's 15-second deadline
    exercised exactly that path and leaked the gate for every later warmup.

    Nonblocking acquisition on the event-loop thread is constant-time. Bounded
    polling preserves cross-loop/thread compatibility while guaranteeing a
    cancelled waiter can never acquire after its caller is gone.
    """

    deadline = time.monotonic() + float(_SPAWN_GATE_ACQUIRE_TIMEOUT_S)
    acquired = False
    lease_token = uuid.uuid4().hex
    while not acquired:
        acquired = _GLOBAL_SPAWN_GATE.acquire(blocking=False)
        if acquired:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            holder = _spawn_gate_snapshot()
            raise TimeoutError(
                f"spawn_gate_timeout:{_SPAWN_GATE_ACQUIRE_TIMEOUT_S:.3f}s:"
                f"holder={holder['owner'] or 'unknown'}:age={holder['age_s']:.3f}s"
            )
        await asyncio.sleep(min(0.05, remaining))

    with _GLOBAL_SPAWN_GATE_STATE_LOCK:
        global _GLOBAL_SPAWN_GATE_TOKEN
        global _GLOBAL_SPAWN_GATE_OWNER
        global _GLOBAL_SPAWN_GATE_ACQUIRED_AT
        _GLOBAL_SPAWN_GATE_TOKEN = lease_token
        _GLOBAL_SPAWN_GATE_OWNER = str(owner or "unknown")[:160]
        _GLOBAL_SPAWN_GATE_ACQUIRED_AT = time.monotonic()
    try:
        yield _spawn_gate_snapshot()
    finally:
        with _GLOBAL_SPAWN_GATE_STATE_LOCK:
            if _GLOBAL_SPAWN_GATE_TOKEN == lease_token:
                _GLOBAL_SPAWN_GATE_TOKEN = ""
                _GLOBAL_SPAWN_GATE_OWNER = ""
                _GLOBAL_SPAWN_GATE_ACQUIRED_AT = 0.0
            else:
                logger.critical(
                    "Spawn gate ownership metadata changed before release owner=%s",
                    owner,
                )
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
    # Ownership is released, but the wedged generation may still be decoding
    # and holding the GPU. Ask it to stop between tokens so the incoming turn
    # gets compute within one decode step — without killing the warm worker.
    soft_cancel = soft_cancel_active_generations(reason=f"owner_cleared:{reason}")
    return {
        "cleared": True,
        "reason": reason,
        "holder": holder,
        "age_s": round(age, 3),
        "detail": "cleared",
        "soft_cancel": soft_cancel,
    }


def soft_cancel_active_generations(*, reason: str) -> list[dict[str, Any]]:
    """Request cooperative cancel on every client with an active generation.

    Returns the per-client receipts for the clients that accepted a cancel
    request; clients with nothing running are skipped.
    """
    receipts: list[dict[str, Any]] = []
    for client in list(_CLIENTS.values()):
        try:
            receipt = client.soft_cancel_active_generation(reason)
        except (AttributeError, OSError, RuntimeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="skipped one client during cooperative cancel sweep",
                severity="warning",
            )
            continue
        if receipt.get("requested"):
            receipt["model"] = os.path.basename(getattr(client, "model_path", "") or "")
            receipts.append(receipt)
    return receipts


def _foreground_owner_eviction_after() -> float | None:
    """How long the CURRENT holder may hold before it may be evicted.

    None when the holder declared no budget: an owner that never said how
    long it needs is not evictable on age alone, because any number we
    invented for it would be a guess used to cancel real work.
    """
    declared = _FOREGROUND_OWNER_STALE_AFTER
    if declared is None:
        return None
    return max(float(declared), _FOREGROUND_OWNER_MIN_EVICTION_S)


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
    global _FOREGROUND_OWNER_STALE_AFTER

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
                    _FOREGROUND_OWNER_STALE_AFTER = stale_after
                    owner_acquired = True
                    break
                # CP126 4cb6a1a0. Eviction used to compare the holder's age
                # against the NEWCOMER's stale_after, which is normalized from
                # a caller-selected timeout to as little as 5 seconds. A short
                # request could therefore declare a legitimately-working owner
                # stale by its own budget and steal foreground authority.
                #
                # An owner is stale only by ITS OWN declared contract, floored
                # so that no declared budget can make a live turn instantly
                # evictable. A holder that declared nothing is never evicted
                # on age alone.
                eviction_after = _foreground_owner_eviction_after()
                if (
                    eviction_after is not None
                    and holder != owner_name
                    and holder_age > eviction_after
                ):
                    _FOREGROUND_OWNER_NAME = None
                    _FOREGROUND_OWNER_ACQUIRED_AT = 0.0
                    _FOREGROUND_OWNER_STALE_AFTER = None
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
                    _FOREGROUND_OWNER_STALE_AFTER = None
            finally:
                _FOREGROUND_OWNER_LOCK.release()
        else:
            # Leaving our finished ownership registered blocks every later
            # foreground turn until a stale-clear heuristic happens to fire.
            # Self-clear WITHOUT the lock as a last resort: we only remove
            # our own entry, so the worst race (another waiter observing the
            # cleared slot a moment early) is strictly better than a leak.
            if _FOREGROUND_OWNER_NAME == owner_name:
                _FOREGROUND_OWNER_NAME = None
                _FOREGROUND_OWNER_ACQUIRED_AT = 0.0
                _FOREGROUND_OWNER_STALE_AFTER = None
            _record_mlx_degradation(
                TimeoutError("foreground owner release lock timeout"),
                action="self-cleared finished foreground ownership without the owner lock",
                severity="error",
            )
            logger.warning(
                "⚠️ [MLX] Timed out releasing foreground owner lock for %s — self-cleared.",
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
        # ANY failure type must be relayed: the old narrow tuple let e.g.
        # OSError/TimeoutError escape the callback, leaving cross-loop
        # waiters on the proxy unresolved forever.
        try:
            result = done_future.result()
        except BaseException as exc:  # noqa: BLE001 - relay every failure to the waiter
            try:
                proxy.set_exception(exc)
            except (cfutures.InvalidStateError, asyncio.InvalidStateError):
                pass
            return
        try:
            proxy.set_result(result)
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
        try:
            future.set_result(result)
        except cfutures.InvalidStateError:
            # Another thread completed/cancelled it between the done() check
            # and here — response delivery must not break on that race.
            return False
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
    # The probe must prove USABLE inference plumbing, not just importability:
    # allocate on the default device, run a small matmul, force evaluation,
    # and check the numeric result. Import-only probes passed on hosts whose
    # Metal device could not actually evaluate a tensor.
    return [
        sys.executable,
        "-c",
        (
            "import mlx.core as mx; import mlx_lm; "
            "a = mx.ones((8, 8)); s = (a @ a).sum(); mx.eval(s); "
            "assert abs(float(s) - 512.0) < 1e-3, float(s); "
            "print('mlx_runtime_ok')"
        ),
    ]


def _load_probe_cache_from_disk() -> tuple[bool | None, str, float]:
    try:
        payload = json.loads(_MLX_RUNTIME_PROBE_CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return None, "", 0.0
    if not isinstance(payload, dict):
        return None, "", 0.0

    # STRICT boolean: json.loads never produces the string "false" for a
    # well-formed writer, so any non-bool here is a malformed or tampered
    # cache — treat it as absent, not as bool("false") == True.
    ok = payload.get("ok")
    if not isinstance(ok, bool):
        return None, "", 0.0
    detail = str(payload.get("detail", "") or "")
    try:
        checked_at = float(payload.get("checked_at", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None, "", 0.0
    now = time.time()
    # A future-dated timestamp would stay "fresh" until wall time caught up;
    # allow only small clock skew.
    if not math.isfinite(checked_at) or checked_at <= 0.0 or checked_at > now + 60.0:
        return None, "", 0.0
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
    except (json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
        # OSError included: a completed health probe must never raise out of
        # cache persistence (disk-full/permissions) and disrupt its caller.
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


# How long a real probe success may bridge a CURRENT enumeration crash, and
# how many consecutive spawns it may cover. A crash that outlives this is not
# a driver glitch; it is a broken runtime, and spawning onto it wastes a
# 20-40GB load and strands the lane anyway.
_LKG_PROBE_WINDOW_S = _finite_env_float(
    "AURA_MLX_LKG_PROBE_WINDOW_S", 300.0, minimum=0.0
)
_LKG_PROBE_MAX_CONSECUTIVE = 2


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
            ok = completed.returncode == 0 and "mlx_runtime_ok" in (
                completed.stdout or ""
            )
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

    # [STABILITY v57] Grace Fallback for a TRANSIENT enumeration crash.
    #
    # CP126 5f02bc9d. This used to accept any in-memory success younger than
    # 30 minutes, with no bound on how many times it could fire. The probe we
    # just ran said the selected runtime is broken RIGHT NOW — and the retry
    # above already gave it a second chance — so a half-hour-old success was
    # being allowed to certify a currently-failing Metal stack indefinitely,
    # spawn after spawn, with nothing louder than a log line.
    #
    # It stays a bridge over a driver glitch, but a bounded one: a short
    # window anchored to the last REAL success (LKG never refreshes the
    # cache, so it cannot extend itself), a cap on consecutive uses, and a
    # degradation record every time, because spawning a 20-40GB worker onto
    # an unconfirmed runtime is a risk someone should be able to see.
    if not ok and detail == "metal_device_enumeration_crash":
        with _MLX_RUNTIME_PROBE_LOCK:
            cached_ok = _MLX_RUNTIME_PROBE.get("ok")
            cached_at = float(_MLX_RUNTIME_PROBE.get("checked_at", 0.0) or 0.0)
            age_s = time.time() - cached_at
            lkg_uses = int(_MLX_RUNTIME_PROBE.get("lkg_uses", 0) or 0)
            within_window = bool(cached_ok) and age_s < _LKG_PROBE_WINDOW_S
            budget_left = lkg_uses < _LKG_PROBE_MAX_CONSECUTIVE
            if within_window and budget_left:
                _MLX_RUNTIME_PROBE["lkg_uses"] = lkg_uses + 1
        if within_window and budget_left:
            _record_mlx_degradation(
                RuntimeError(
                    f"metal_device_enumeration_crash bridged by last-known-good "
                    f"status from {age_s:.0f}s ago "
                    f"(use {lkg_uses + 1}/{_LKG_PROBE_MAX_CONSECUTIVE})"
                ),
                action="allowed worker spawn on an unconfirmed MLX runtime",
                severity="warning",
            )
            logger.warning(
                "♻️ [MLX] Runtime probe hit an enumeration crash; bridging with a "
                "last-known-good status from %.0fs ago (use %d/%d).",
                age_s, lkg_uses + 1, _LKG_PROBE_MAX_CONSECUTIVE,
            )
            return True, "lkg_fallback_after_enumeration_crash"
        if cached_ok:
            # The bridge is out: either the last real success aged out or the
            # crash has repeated past the point where "transient" is credible.
            _record_mlx_degradation(
                RuntimeError(
                    f"metal_device_enumeration_crash not bridged: "
                    f"lkg_age={age_s:.0f}s uses={lkg_uses}"
                ),
                action="refused worker spawn until a live runtime probe succeeds",
                severity="error",
            )

    with _MLX_RUNTIME_PROBE_LOCK:
        _MLX_RUNTIME_PROBE.update(
            {
                "ok": ok,
                "detail": detail,
                "checked_at": time.time(),
                # A probe that actually ran resets the bridge: consecutive
                # means consecutive, and a confirmed-good runtime has earned
                # its grace back.
                "lkg_uses": 0 if ok else int(_MLX_RUNTIME_PROBE.get("lkg_uses", 0) or 0),
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
        self._expert_adapter_path: str | None = None
        self._process: mp.Process | None = None
        self._model_lane_owner_id = f"mlx:{os.getpid()}:{_real_model_path(model_path)}"
        self._model_lane_state_lock = _threading.RLock()
        # Partial-failure receipt for a durable owner we could not release.
        # None means the lane holds no stranded fence.
        self._lane_release_failure: dict[str, Any] | None = None
        self._model_lane_fencing_token = 0
        self._model_lane_terminal_receipt_id = ""
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
        # Once-per-episode reporting latches for worker self-reported health
        # evidence (heartbeat loop_stalled / ipc_broken frames).
        self._worker_loop_stall_reported = False
        self._worker_ipc_broken_reported = False
        self._last_progress_at = 0.0
        self._last_token_progress_at = 0.0
        # Per-spawn key authorizing privileged output-contract selection.
        # Empty until a worker is spawned; a client with no worker has
        # nothing to authorize.
        self._contract_key: bytes = b""
        self._latent_progress_by_request: dict[str, dict[str, Any]] = {}
        # Explicit drop accounting for the latent progress channel: state that
        # was refused (uncorrelated id) and state that aged out (window
        # eviction) are different failures and are counted separately.
        self._latent_progress_dropped_unknown = 0
        self._latent_progress_evicted = 0
        self._latent_progress_drop_reported = False
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
        # Singleflight handle + its OWN start timestamp (CP126 4d8a7d6b):
        # staleness must be measured against the warmup, not against
        # _lane_transition_at, which any other lane transition refreshes.
        self._warmup_inflight: asyncio.Future | None = None
        self._warmup_started_at: float = 0.0
        # Consecutive reboot lock-acquisition failures (CP126 ec341dfa).
        self._reboot_lock_failures = 0
        self._model_load_admission_state_lock = _threading.Lock()
        self._model_load_admission_backoff_until = 0.0
        self._model_load_admission_backoff_until_unix = 0.0
        self._model_load_admission_denial_reason = ""
        self._model_load_admission_denial_receipt_id = ""
        self._model_load_admission_denial_count = 0
        self._model_load_admission_suppressed_count = 0
        self._model_load_admission_denied_at = 0.0
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
        self._worker_identity: dict[str, Any] = {}
        self._last_surface_control_receipt: dict[str, Any] = {}
        self._surface_control_receipt_context: ContextVar[dict[str, Any] | None] = (
            ContextVar(
                f"aura_mlx_surface_receipt_{id(self)}",
                default=None,
            )
        )
        self._last_interoception: dict[str, Any] = {}
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
        self._steering_liveness_observed = False

        # Cooperative preemption channel: the parent writes the ACTIVE job's
        # numeric sequence here to ask the worker to stop between tokens.
        # Cancel latency is one decode step and the model stays warm — unlike
        # force-abort, which kills the worker and pays a full model reload.
        self._cancel_seq = self._mp_context.Value("Q", 0, lock=False)
        self._job_seq_counter = 0
        self._current_request_seq = 0

    def _is_primary_or_deep_lane(self) -> bool:
        """Whether this lane is one of the big resident models.

        CP126 24aaa654. This was decided purely by searching the path for
        "32b"/"72b"/"zenith"/"solver"/"cortex", and it gates the memory
        guards that stand between a 20-40GB allocation and jetsam. A renamed,
        aliased or nonstandard checkpoint therefore walked straight past
        them, while an unrelated path containing one of those tokens was
        treated as heavy.

        Measured artifact evidence (parameter count from the model's own
        config/index) is the authority. The name tokens are kept as a UNION,
        never a replacement: for a safety guard the fail-safe direction is to
        over-include, so a model that either measures heavy or is named heavy
        is treated as heavy.
        """
        return _model_is_heavy_lane(self.model_path)

    def _is_primary_lane(self) -> bool:
        """The serving cortex lane specifically — deep/solver excluded."""
        return self._is_primary_or_deep_lane() and not self._is_deep_solver_lane()

    def _can_run_resident_background_health_probe(
        self,
        deferral_reason: str,
        *,
        health_probe: bool,
    ) -> bool:
        """Allow one bounded readiness probe without weakening spawn admission."""
        return bool(
            health_probe
            and deferral_reason == "foreground_headroom_reserved"
            and self._is_primary_lane()
            and self.is_alive()
        )

    def _is_deep_solver_lane(self) -> bool:
        return _model_path_is_deep_solver(self.model_path)

    @staticmethod
    def _model_load_admission_backoff_seconds(reason: str, count: int) -> float:
        normalized = str(reason or "").lower()
        attempt = max(1, int(count))
        if normalized.startswith("event_loop_lag_") or normalized == "event_loop_signal_unavailable":
            base_s, cap_s = 3.0, 30.0
        elif "memory_pressure" in normalized or "thermal_pressure" in normalized:
            base_s, cap_s = 15.0, 300.0
        elif normalized in {
            "runtime_shutdown_requested",
            "background_capability_suspended",
            "large_model_capability_suspended",
        }:
            base_s, cap_s = 30.0, 300.0
        else:
            base_s, cap_s = 10.0, 120.0
        return min(cap_s, base_s * (2 ** min(attempt - 1, 5)))

    def _model_load_admission_backoff_active(self) -> bool:
        with self._model_load_admission_state_lock:
            active = time.monotonic() < float(
                self._model_load_admission_backoff_until or 0.0
            )
            if active:
                self._model_load_admission_suppressed_count += 1
            return active

    def _note_model_load_admission_denial(
        self,
        reason: str,
        *,
        receipt_id: str,
    ) -> float:
        with self._model_load_admission_state_lock:
            normalized = str(reason or "resource_admission_denied")
            if normalized == self._model_load_admission_denial_reason:
                self._model_load_admission_denial_count += 1
            else:
                self._model_load_admission_denial_reason = normalized
                self._model_load_admission_denial_count = 1
            backoff_s = self._model_load_admission_backoff_seconds(
                normalized,
                self._model_load_admission_denial_count,
            )
            now_monotonic = time.monotonic()
            now_unix = time.time()
            self._model_load_admission_backoff_until = now_monotonic + backoff_s
            self._model_load_admission_backoff_until_unix = now_unix + backoff_s
            self._model_load_admission_denial_receipt_id = str(receipt_id or "")
            self._model_load_admission_denied_at = now_unix
            self._model_load_admission_suppressed_count = 0
            return backoff_s

    def _clear_model_load_admission_backoff(self) -> None:
        with self._model_load_admission_state_lock:
            self._model_load_admission_backoff_until = 0.0
            self._model_load_admission_backoff_until_unix = 0.0
            self._model_load_admission_denial_reason = ""
            self._model_load_admission_denial_receipt_id = ""
            self._model_load_admission_denial_count = 0
            self._model_load_admission_suppressed_count = 0
            self._model_load_admission_denied_at = 0.0

    def _model_load_admission_status(self) -> dict[str, Any]:
        with self._model_load_admission_state_lock:
            return {
                "backing_off": time.monotonic()
                < float(self._model_load_admission_backoff_until or 0.0),
                "retry_at_unix": self._model_load_admission_backoff_until_unix,
                "reason": self._model_load_admission_denial_reason,
                "receipt_id": self._model_load_admission_denial_receipt_id,
                "denial_count": self._model_load_admission_denial_count,
                "suppressed_calls": self._model_load_admission_suppressed_count,
                "last_denied_at_unix": self._model_load_admission_denied_at,
            }

    def _adopt_durable_model_lane_owner(
        self,
        *,
        fencing_token: int,
        receipt_id: str,
    ) -> None:
        """Publish one committed owner token atomically to worker listeners."""

        token = int(fencing_token or 0)
        if token <= 0:
            raise ValueError("model-lane fencing token must be positive")
        with self._model_lane_state_lock:
            from core.runtime.model_lane_control import register_model_lane_owner_adapter

            register_model_lane_owner_adapter(
                self._model_lane_owner_id,
                evict=_evict_model_lane_owner,
                compensate=_compensate_model_lane_owner,
            )
            self._model_lane_fencing_token = token
            self._model_lane_terminal_receipt_id = str(receipt_id or "")

    def _durable_model_lane_owner_snapshot(self) -> tuple[str, int, str]:
        with self._model_lane_state_lock:
            return (
                str(self._model_lane_owner_id or ""),
                int(self._model_lane_fencing_token or 0),
                str(self._model_lane_terminal_receipt_id or ""),
            )

    def lane_recovery_required(self) -> dict[str, Any] | None:
        """The unreleased durable owner blocking this lane, if any.

        None means the lane holds no stranded fence. A dict is a partial-
        failure receipt: it names the owner and fencing token that must be
        released before this lane can be admitted again, so the dependency is
        actionable rather than an unexplained admission refusal later.
        """
        pending = getattr(self, "_lane_release_failure", None)
        return dict(pending) if pending else None

    def _note_lane_release_failure(self, exc: BaseException, *, reason: str) -> None:
        """Record a durable-owner release that could not be confirmed."""
        with self._model_lane_state_lock:
            owner_id = str(self._model_lane_owner_id or "")
            fencing_token = int(self._model_lane_fencing_token or 0)
        self._lane_release_failure = {
            "owner_id": owner_id,
            "fencing_token": fencing_token,
            "reason": reason,
            "error": f"{type(exc).__name__}: {exc}"[:200],
            "at_unix": time.time(),
        }
        _record_mlx_degradation(
            exc,
            action=(
                f"lane left FENCED: durable owner {owner_id or '<unknown>'} "
                f"token={fencing_token} could not be released during {reason}; "
                "admission stays blocked until it is"
            ),
            severity="critical",
        )
        self._record_degraded_event(
            "durable_owner_release_failed",
            detail=f"{os.path.basename(self.model_path)}:{owner_id}:token={fencing_token}",
            severity="critical",
            foreground_request=True,
        )
        self._set_lane_state("fenced", f"durable_owner_release_failed:{reason}")

    def _clear_lane_release_failure(self) -> None:
        """A confirmed release retires the fence."""
        self._lane_release_failure = None

    def _release_durable_model_lane_owner_sync(self, *, reason: str) -> bool:
        """Release the exact committed owner before another worker may spawn.

        The state lock spans the controller operation so a concurrent commit
        cannot publish a newer token while lifecycle cleanup is releasing the
        old one. A missing owner is already a settled release and still clears
        the local token; controller failures retain it so respawn can refuse
        rather than heartbeat a stale fence.
        """

        with self._model_lane_state_lock:
            owner_id = str(self._model_lane_owner_id or "")
            fencing_token = int(self._model_lane_fencing_token or 0)
            if not owner_id or fencing_token <= 0:
                self._model_lane_fencing_token = 0
                self._model_lane_terminal_receipt_id = ""
                return True

            from core.runtime.model_lane_control import (
                get_model_lane_controller,
                unregister_model_lane_owner_adapter,
            )

            released = get_model_lane_controller().release_owner_sync(
                owner_id,
                fencing_token=fencing_token,
                reason=str(reason or "worker_stopped"),
            )
            if not released:
                # CP126 158ed09e. The controller ALSO returns False for a
                # FENCING-TOKEN MISMATCH, i.e. a NEWER durable owner is
                # registered. Unregistering the adapter and discarding the
                # token + terminal receipt on that path threw away the only
                # handles able to reconcile that owner, while returning True
                # told every caller the lane was cleanly released.
                #
                # Keep the claim state and the fence recorded so respawn
                # refuses rather than heartbeating a stale fence, and report
                # the truth.
                self._note_lane_release_failure(
                    RuntimeError(
                        f"durable_owner_release_not_confirmed:{owner_id}:token={fencing_token}"
                    ),
                    reason=str(reason or "worker_stopped"),
                )
                return False

            unregister_model_lane_owner_adapter(owner_id)
            self._model_lane_fencing_token = 0
            self._model_lane_terminal_receipt_id = ""
            self._clear_lane_release_failure()
            return True

    async def _release_durable_model_lane_owner(self, *, reason: str) -> bool:
        return await asyncio.to_thread(
            self._release_durable_model_lane_owner_sync,
            reason=reason,
        )

    def _mark_progress(self) -> None:
        self._last_progress_at = time.time()

    def latent_progress_counters(self) -> dict[str, int]:
        """Drop accounting for the latent progress channel.

        Exposed so a refused stream is visible to health surfaces rather than
        only to whoever reads the logs: dropped_unknown counts progress for
        request ids this client never issued (a broken or hostile child),
        evicted counts entries aged out of the bounded window (normal churn).
        """
        return {
            "tracked": len(self._latent_progress_by_request),
            "dropped_unknown": self._latent_progress_dropped_unknown,
            "evicted": self._latent_progress_evicted,
        }

    def _authorize_job(self, job: Any, *, principal: str) -> Any:
        """Sign a job's privileged contract selection before submission.

        Single choke point: every path that puts work on the request queue
        goes through here, so a privileged contract cannot reach the worker
        without the authority of the lane that owns it. Jobs selecting
        nothing privileged are returned untouched.
        """
        if not isinstance(job, dict) or not self._contract_key:
            return job
        try:
            from core.brain.llm.contract_authority import sign_job

            return sign_job(job, self._contract_key, principal=principal)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="could not sign a privileged contract selection",
                severity="error",
            )
            return job

    def _record_latent_progress(self, response: dict[str, Any]) -> None:
        """Retain bounded parent-side evidence for the active latent stage."""

        request_id = str(response.get("id") or "")
        if not request_id:
            return
        # Only track ids that belong to a PENDING or current request — a
        # broken or compromised child streaming unique ids must not grow
        # parent-side state without bound.
        if (
            request_id not in self._pending_generations
            and request_id != self._current_request_id
            and request_id not in self._latent_progress_by_request
        ):
            # Counted, not merely ignored: a child streaming ids the parent
            # never issued is either broken or compromised, and a silent drop
            # makes that indistinguishable from a healthy stream.
            self._latent_progress_dropped_unknown += 1
            if not self._latent_progress_drop_reported:
                self._latent_progress_drop_reported = True
                _record_mlx_degradation(
                    RuntimeError(
                        f"latent progress for unknown request id {request_id!r} "
                        f"from {os.path.basename(self.model_path)}"
                    ),
                    action="dropped uncorrelated latent progress from the worker",
                    severity="warning",
                )
            return
        allowed = {
            "stage",
            "stage_duration_s",
            "elapsed_s",
            "spent_layer_apps",
            "input_tokens",
            "branches",
            "slots",
            "max_branch_steps",
            "exchanges",
            "selected_branch",
            "steps_taken",
            "attempts",
            "accepted",
            "wrapped_layers",
            "generated_tokens",
            "termination",
        }
        snapshot = {
            key: response.get(key)
            for key in allowed
            if key in response
        }
        snapshot.update(
            {
                "request_id": request_id,
                "received_at_unix": time.time(),
            }
        )
        self._latent_progress_by_request[request_id] = snapshot
        # Bounded: evict the oldest entries beyond a small window.
        if len(self._latent_progress_by_request) > 64:
            for stale_id in sorted(
                self._latent_progress_by_request,
                key=lambda rid: float(
                    self._latent_progress_by_request[rid].get("received_at_unix", 0.0)
                ),
            )[: len(self._latent_progress_by_request) - 64]:
                self._latent_progress_by_request.pop(stale_id, None)
                self._latent_progress_evicted += 1

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
        threshold = _finite_env_float("AURA_SYSTEM_SLEEP_GAP_THRESHOLD_S", 5.0, minimum=1.0)
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

    def _surface_control_receipt_slot(self) -> ContextVar[dict[str, Any] | None]:
        slot = getattr(self, "_surface_control_receipt_context", None)
        if slot is None:
            slot = ContextVar(
                f"aura_mlx_surface_receipt_{id(self)}",
                default=None,
            )
            self._surface_control_receipt_context = slot
        return slot

    def _set_task_surface_control_receipt(self, receipt: dict[str, Any]) -> None:
        self._surface_control_receipt_slot().set(dict(receipt))

    def get_last_surface_control_receipt(self) -> dict[str, Any]:
        task_receipt = self._surface_control_receipt_slot().get()
        if task_receipt is not None:
            return dict(task_receipt)
        return {}

    def get_diagnostic_last_surface_control_receipt(self) -> dict[str, Any]:
        """Return process-wide last-call telemetry, never request proof."""

        return dict(getattr(self, "_last_surface_control_receipt", {}) or {})

    def _record_surface_control_receipt_from_response(self, response: dict[str, Any]) -> None:
        receipt = _sanitize_surface_control_receipt(
            response.get("surface_control_receipt") if isinstance(response, dict) else None
        )
        if isinstance(response, dict) and "tokens_used" in response:
            try:
                receipt["generated_tokens"] = max(0, int(response.get("tokens_used") or 0))
            except (TypeError, ValueError, OverflowError):
                pass
        self._set_task_surface_control_receipt(receipt)
        if receipt:
            self._last_surface_control_receipt = receipt

    def _preserve_lane_after_surface_quality_rejection(self) -> None:
        """Clear empty-decode pressure while keeping the healthy worker resident."""

        self._consecutive_empty = 0
        self._set_lane_state("ready")

    def get_last_interoception(self) -> dict[str, Any]:
        """The substrate interoception payload of the most recent completed
        generation on this lane (worker-measured; see interoception_tap)."""
        return dict(self._last_interoception) if self._last_interoception else {}

    def _record_interoception_from_response(
        self,
        response: dict[str, Any],
        *,
        foreground_request: bool,
        owner_label: str,
    ) -> None:
        """Capture the worker's felt-thought measurements and hand them to the
        thought-interoception organ. Observational only — never raises into the
        generation path."""
        try:
            payload = response.get("interoception") if isinstance(response, dict) else None
            if not isinstance(payload, dict) or not payload:
                return
            from core.being.thought_interoception import (
                get_thought_interoception,
                text_fingerprint,
            )

            # Fingerprint the payload to the text it measured, so consumers
            # (e.g. unified_inference feedback) can prove they are pairing the
            # right trace with the right response even under concurrent lanes.
            stored = dict(payload)
            stored["_text_fingerprint"] = text_fingerprint(str(response.get("text") or ""))

            # Ingest FIRST: the engine is the validator. Storing beforehand
            # exposed malformed/unbounded worker data through the public
            # getter as the "most recent measurement" even when the engine
            # rejected it.
            get_thought_interoception().ingest(
                payload,
                origin=owner_label or "mlx",
                foreground=bool(foreground_request),
                response_text=str(response.get("text") or ""),
            )
            self._last_interoception = stored
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            record_degradation(
                "mlx_client_interoception", exc, severity="warning",
                action="continued generation return after interoception ingest failed",
            )

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
        request_seq: int = 0,
    ) -> None:
        now = time.time()
        self._current_request_id = str(req_id or "")
        self._current_request_seq = max(0, int(request_seq or 0))
        # A new generation supersedes any stale cooperative-cancel request.
        cancel_seq = getattr(self, "_cancel_seq", None)
        if cancel_seq is not None and int(getattr(cancel_seq, "value", 0)) not in (
            0,
            self._current_request_seq,
        ):
            cancel_seq.value = 0
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
        if not normalized_req_id and self._current_request_id:
            # Id-less progress cannot be ATTRIBUTED to the active request:
            # crediting it set first-token timestamps from unrelated or
            # malformed messages. It still proves the worker is alive.
            self._mark_progress()
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
        self._current_request_seq = 0
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

    def _classify_failure(
        self,
        *,
        foreground_request: bool = False,
        reason: str = "",
        classification: str | None = None,
    ) -> str:
        if classification:
            return classification
        normalized_reason = str(reason or "").lower()
        if (
            self._is_deep_solver_lane()
            and (
                "memory_pressure_refused_worker_spawn" in normalized_reason
                or "optional_deep_solver_memory_refusal" in normalized_reason
            )
        ):
            return "non_critical_fallback"
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
        classification: str | None = None,
    ) -> None:
        try:
            from core.health.degraded_events import record_degraded_event

            record_degraded_event(
                "mlx_client",
                reason,
                detail=detail,
                severity=severity,
                classification=self._classify_failure(
                    foreground_request=foreground_request,
                    reason=f"{reason}:{detail}",
                    classification=classification,
                ),
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

    def _is_optional_deep_solver_memory_refusal(self, detail: str) -> bool:
        return self._is_deep_solver_lane() and "memory_pressure_refused_worker_spawn:" in str(detail)

    def _handle_optional_deep_solver_memory_refusal(self, detail: str) -> bool:
        """Treat a refused 72B load as an unavailable optional lane, not a live-system failure."""

        if not self._is_optional_deep_solver_memory_refusal(detail):
            return False
        self._set_lane_state("cold")
        self._init_future = None
        self._consecutive_spawn_failures = 0
        # Short backoff prevents repeated oversized 72B attempts from flooding the
        # neural stream while keeping the optional lane available after pressure falls.
        self._spawn_backoff_until = time.time() + 60.0
        self._record_degraded_event(
            "optional_deep_solver_memory_refusal",
            detail=f"{os.path.basename(self.model_path)}:{detail}",
            severity="warning",
            foreground_request=False,
            classification="non_critical_fallback",
        )
        logger.warning(
            "🛡️ [MLX] Optional deep Solver worker refused by memory guard for %s: %s. "
            "Keeping primary Cortex authoritative.",
            os.path.basename(self.model_path),
            detail,
        )
        return True

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
        if _model_is_deep_solver_lane(self.model_path):
            if foreground_request and during_generation:
                return 45.0
            return 90.0 if during_generation else 45.0
        if _model_is_heavy_lane(self.model_path):
            if foreground_request and during_generation:
                return 45.0  # was 22s — too aggressive with recurrent depth
            return 60.0 if during_generation else 30.0
        return 20.0 if during_generation else 15.0

    def _pressure_adaptive_stretch(self) -> tuple[float, str]:
        """Bounded stretch for token-progress budgets under live memory pressure.

        A RESIDENT heavy model's first token slows under unified-memory
        contention because prompt eval competes for bandwidth — the worker is
        starved, not wedged. Killing it answers a bandwidth problem with a
        ~20GB reload that deepens the contention (the Jul 7 soak doom loop:
        stall → force-kill → cold reload → next turn stalls under the same
        pressure). Only token-progress budgets stretch, and only within
        bounds: heartbeat wedge detection is untouched, caller deadlines
        still dominate, and the emergency tier still refuses generation
        outright before this is consulted.
        """
        if str(
            os.environ.get("AURA_FIRST_TOKEN_PRESSURE_ADAPT", "1")
        ).strip().lower() not in {"1", "true", "yes", "on"}:
            return 1.0, ""
        if not _model_is_heavy_lane(self.model_path):
            return 1.0, ""
        try:
            snapshot = get_memory_pressure_snapshot()
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError):
            return 1.0, ""
        if snapshot.emergency:
            return 1.0, ""  # the refuse-generation path owns emergencies
        if snapshot.critical:
            return 1.5, "memory_pressure_critical"
        if snapshot.high:
            return 1.35, "memory_pressure_high"
        if snapshot.warning:
            return 1.2, "memory_pressure_warning"
        return 1.0, ""

    def _pressure_receipt_suffix(self) -> str:
        """Name the memory-pressure tier on stall receipts.

        A stall verdict under contention is a different incident than a stall
        on an idle machine; the narrator (and anyone reading the degradation
        ledger) should not have to correlate timestamps to know which one
        happened.
        """
        try:
            snapshot = get_memory_pressure_snapshot()
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError):
            return ""
        return f":memory={snapshot.level}" if snapshot.warning else ""

    def _first_token_sla(self, *, foreground_request: bool = False) -> float:
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
        if _model_is_deep_solver_lane(self.model_path):
            if foreground_request:
                base = 52.0 if is_cold_start else 32.0
                return _with_prompt_eval_headroom(
                    base,
                    threshold_tokens=768.0,
                    eval_seconds_per_token=0.018,
                    cap_s=115.0,
                )
            return 30.0
        if _model_is_heavy_lane(self.model_path):
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

        if _model_is_deep_solver_lane(self.model_path):
            default = 165.0 if foreground_request else 120.0
        elif _model_is_heavy_lane(self.model_path):
            default = 120.0 if foreground_request else 90.0
        else:
            default = 30.0 if foreground_request else 20.0
        stretch, _ = self._pressure_adaptive_stretch()
        default *= stretch
        if os.environ.get("AURA_FIRST_TOKEN_ABSOLUTE_CEILING_S") is not None:
            configured = _finite_env_float(
                "AURA_FIRST_TOKEN_ABSOLUTE_CEILING_S", default, minimum=10.0
            )
            # Bounded above too: an absurd ceiling disables the watchdog.
            return min(3600.0, max(10.0, configured))
        return default

    def _first_token_hard_ceiling(self, *, foreground_request: bool = False) -> float:
        first_token_sla = self._first_token_sla(foreground_request=foreground_request)
        # Finite-range validation: negative or non-finite multipliers/padding
        # previously produced premature aborts, an unbounded watchdog, or a
        # non-finite timer value.
        hard_mult = min(10.0, _finite_env_float("AURA_FIRST_TOKEN_HARD_MULT", 1.8, minimum=1.0))
        hard_pad = min(600.0, _finite_env_float("AURA_FIRST_TOKEN_HARD_PAD_S", 20.0, minimum=0.0))
        # The hard ceiling exists to kill LIVELOCKED generations (heartbeats,
        # zero tokens). Under live memory pressure a starved-but-healthy heavy
        # lane looks exactly like that livelock from outside — stretch the
        # verdict boundary (bounded, never past the caller's deadline) so
        # contention gets time to clear instead of triggering a 20GB reload.
        stretch, _ = self._pressure_adaptive_stretch()
        return min(
            first_token_sla * hard_mult * stretch + hard_pad,
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
        stretch, _ = self._pressure_adaptive_stretch()
        if _model_is_deep_solver_lane(self.model_path):
            return (18.0 if foreground_request else 25.0) * stretch
        if _model_is_heavy_lane(self.model_path):
            # [RESILIENCE] Reverted from 10s — recurrent depth can cause
            # legitimate pauses between tokens during the recurrent block
            # computation. Sized up with the 2026-06-11 first-token
            # remeasurement: inter-token pauses stretch the same way under
            # the macos26 guard, and a stall verdict triggers the same
            # over-broad lane recycle as an SLA breach.
            return (40.0 if foreground_request else 45.0) * stretch
        return 8.0

    def _confirm_worker_reported_loop_stall(
        self,
        payload: dict[str, Any],
    ) -> tuple[bool, float]:
        """Apply request-aware budgets to the worker's coarse progress alarm.

        The child only knows that no token activity has occurred for 30s. On a
        resident 32B request that is normal during prompt evaluation, so the
        parent confirms the signal against the request's first-token or
        inter-token budget before recording a runtime fault.
        """

        request_id = str(payload.get("request_id") or "")
        current_request_id = str(getattr(self, "_current_request_id", "") or "")
        if not request_id or not current_request_id or request_id != current_request_id:
            return False, 0.0
        try:
            age_s = max(0.0, float(payload.get("job_age_s") or 0.0))
        except (TypeError, ValueError):
            return False, 0.0

        first_token_at = float(getattr(self, "_current_first_token_at", 0.0) or 0.0)
        if first_token_at <= 0.0:
            threshold_s = float(
                getattr(self, "_current_first_token_hard_ceiling_s", 0.0) or 0.0
            )
            if threshold_s <= 0.0:
                threshold_s = self._first_token_hard_ceiling(
                    foreground_request=self._is_primary_or_deep_lane(),
                )
        else:
            threshold_s = self._token_stall_after(
                foreground_request=self._is_primary_or_deep_lane(),
            )
        threshold_s = max(30.0, float(threshold_s))
        return age_s > threshold_s, threshold_s

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
            "model_load_admission": self._model_load_admission_status(),
            "spawn_gate": _spawn_gate_snapshot(),
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
        # A foreground warmup can be refused by the unified-memory guard even
        # though the primary worker is already alive and initialized. Marking
        # that as recovering strands the live desktop lane behind
        # lane_recovering + visible_conversation_probe_missing. In that case the
        # correct next step is to let the foreground turn prove visible
        # readiness, not spawn or recycle another model process.
        if str(reason or "") == "foreground_warmup_deferred_memory_pressure":
            try:
                worker_ready = bool(self.is_alive() and self._init_done)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                worker_ready = False
            if worker_ready:
                now = time.time()
                self._last_ready_at = max(float(self._last_ready_at or 0.0), now)
                self._last_progress_at = max(float(self._last_progress_at or 0.0), now)
                self._set_lane_state("ready")
                return
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
        maintenance_preempt_requested = False

        while loop.time() < wait_deadline:
            if self._request_lock.acquire(False):
                self._request_lock_owner_label = str(owner_label or "")
                self._request_lock_acquired_at = time.time()
                return True

            now = loop.time()
            waited = max(0.0, now - wait_started)
            holder = self._request_lock_owner_label or "another_request"

            if (
                foreground_request
                and not maintenance_preempt_requested
                and holder == "reasoning_nonparametric_ingest"
            ):
                receipt = self.soft_cancel_active_generation(
                    reason="foreground_preemption_nonparametric_ingest"
                )
                maintenance_preempt_requested = bool(receipt.get("requested"))
                if maintenance_preempt_requested:
                    logger.info(
                        "⏭️ [MLX] Foreground request preempted bounded "
                        "non-parametric maintenance on %s.",
                        os.path.basename(self.model_path),
                    )

            if waited >= 5.0 and (now - last_log_at) >= 5.0:
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
            # Reusable only if its loop is genuinely serving AND the task is
            # not already being cancelled. Respawn paths cancel the old
            # listener and immediately call this method — treating the
            # still-cancelling task as reusable left the fresh worker with
            # NO response consumer. A stopped-but-unclosed foreign loop is
            # equally dead for our purposes.
            cancelling = task.cancelled() or bool(
                getattr(task, "cancelling", lambda: 0)()
            )
            loop_serving = False
            try:
                task_loop = task.get_loop()
                loop_serving = (not task_loop.is_closed()) and task_loop.is_running()
            except (RuntimeError, AttributeError) as exc:
                logger.debug("MLX listener task loop unavailable during reuse check: %s", exc)
            if loop_serving and not cancelling:
                return
            _cancel_task_threadsafe(task)
            # CP126 bd5dea11: cancellation is ASYNCHRONOUS. Creating the
            # replacement immediately left two listeners briefly draining the
            # SAME response queue, so the old one could steal the new worker's
            # init/generation frames. Prove the old listener is gone (bounded —
            # a wedged listener must not block worker recovery forever) and
            # drop the handle before installing a replacement.
            if task.get_loop() is asyncio.get_running_loop():
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
                except (asyncio.CancelledError, TimeoutError):
                    pass
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    logger.debug("Prior MLX listener ended with %s", type(exc).__name__)
                if not task.done():
                    _record_mlx_degradation(
                        TimeoutError("listener_cancel_unconfirmed"),
                        action=(
                            "installed a replacement response listener before the "
                            "prior one confirmed termination"
                        ),
                        severity="warning",
                    )
            self._listener_task = None

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
        # State-aware budget: a heavy-lane spawn/handshake legitimately runs
        # for minutes while 20-40GB of weights load. The old flat 120s reset
        # cancelled LIVE handshakes from a mere status poll, well inside the
        # 300s the handshake itself was granted.
        if self._lane_state in {"spawning", "handshaking"}:
            allowed = max(120.0, float(self._handshake_timeout()) + 30.0)
        elif self._lane_state == "warming":
            allowed = max(120.0, float(self._warmup_timeout()) + 30.0)
        else:
            allowed = 120.0
        if stuck_duration < allowed:
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
        # The reset must not orphan a live worker: declaring the lane cold
        # while the old process survives lets the next spawn stack a second
        # multi-GB worker beside it.
        process = self._process
        if process is not None and process.is_alive():
            _record_mlx_degradation(
                RuntimeError(f"stale_lane_reset_killed_live_worker:{self._lane_state}"),
                action="killed unresponsive worker during stale lane-state reset",
                severity="error",
            )
            _note_lane_worker_death(self, "lane_state_stale_reset")
            self._process = None
            self._kill_and_join_blocking(process)
        self._warmup_in_flight = False
        self._set_lane_state("cold")

    def _kill_and_join_blocking(self, p: mp.Process) -> bool:
        """Kill and join a worker, PROVING termination. Returns True when dead.

        The old helper swallowed failures and never re-checked ``is_alive``,
        so callers replaced queues and respawned while the accelerator-owning
        child could still be running.
        """
        if not p:
            return True
        if not p.is_alive():
            return True
        try:
            p.kill()
            p.join(timeout=2.0)
            if p.is_alive():
                # One more attempt before reporting a survivor.
                p.kill()
                p.join(timeout=2.0)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            _record_mlx_degradation(
                e,
                action="continued process cleanup after worker kill/join failed",
                severity="error",
            )
            logger.warning("Error killing process: %s", e)
        try:
            still_alive = bool(p.is_alive())
        except (RuntimeError, AttributeError, ValueError):
            still_alive = False
        if still_alive:
            _record_mlx_degradation(
                RuntimeError(f"worker_survived_kill:pid={getattr(p, 'pid', '?')}"),
                action="worker process survived kill+join; caller state may briefly double-count it",
                severity="critical",
            )
        return not still_alive

    def _replace_ipc_queues(self, *, maxsize: int = 10) -> None:
        """Replace IPC queues after closing the old semaphores and feeder threads."""
        _safe_close_queue(self._req_q)
        _safe_close_queue(self._res_q)
        self._req_q = self._mp_context.Queue(maxsize=maxsize)
        self._res_q = self._mp_context.Queue(maxsize=maxsize)
        try:
            _register_runtime_queue(self._req_q, name="mlx.request_queue")
            _register_runtime_queue(self._res_q, name="mlx.response_queue")
        except (ImportError, RuntimeError, AttributeError, TypeError, ValueError):
            self._close_ipc_queues()
            raise
        self._closed = False

    def _close_ipc_queues(self) -> None:
        """Close IPC queues without recreating them during final client shutdown."""
        _safe_close_queue(self._req_q)
        _safe_close_queue(self._res_q)
        self._req_q = None
        self._res_q = None

    async def _generate_batch_response_async(
        self,
        prompt: str,
        *,
        n: int = 4,
        max_tokens: int = 512,
        temperature: float = 0.8,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Return one task-local batched worker response without global state."""
        if self._req_q is None or self._closed:
            return {}
        # Memory-pressure admission: a 16-candidate 2048-token resident batch
        # is heavy generation. The serial and latent paths refuse under
        # critical pressure — the batch path previously dispatched anyway.
        try:
            snapshot = get_memory_pressure_snapshot()
            if snapshot.refuse_heavy_local_generation:
                _record_mlx_degradation(
                    RuntimeError(snapshot.reason or "critical_memory_pressure"),
                    action="refused batched generation under critical memory pressure",
                    severity="warning",
                )
                return {}
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="refused batched generation while the memory probe was unavailable",
                severity="warning",
            )
            return {}
        alive = await self._ensure_worker_alive(request_is_background=True)
        if not alive:
            return {}
        try:
            admitted_n = max(1, min(16, int(n)))
        except (TypeError, ValueError, OverflowError):
            admitted_n = 4
        admitted_max_tokens = min(
            2048,
            _bounded_max_tokens(max_tokens, max_tokens, 512),
        )
        try:
            admitted_temperature = float(temperature)
        except (TypeError, ValueError, OverflowError):
            admitted_temperature = 0.8
        if admitted_temperature != admitted_temperature or not (
            float("-inf") < admitted_temperature < float("inf")
        ):
            admitted_temperature = 0.8
        admitted_temperature = max(0.0, min(2.0, admitted_temperature))
        try:
            admitted_timeout = float(timeout_s)
        except (TypeError, ValueError, OverflowError):
            admitted_timeout = 180.0
        if not math.isfinite(admitted_timeout):
            # Infinity previously created an UNBOUNDED wait on the future.
            admitted_timeout = 180.0
        admitted_timeout = min(600.0, max(10.0, admitted_timeout))
        req_id = uuid.uuid4().hex
        req = {
            "id": req_id,
            "action": "generate_batch",
            "prompt": str(prompt or ""),
            "n": admitted_n,
            "max_tokens": admitted_max_tokens,
            "temperature": admitted_temperature,
        }
        fut = _new_shared_future()
        self._pending_generations[req_id] = fut
        timed_out = False
        try:
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(req, principal="mlx_client.health_probe"),
                True,
                2.0,
            )
            res = await _await_shared_future(fut, timeout_s=admitted_timeout)
        except (TimeoutError, BrokenPipeError, OSError, queue.Full) as exc:
            # queue.Full included: queue saturation is expected load
            # contention and belongs inside the documented empty-response
            # fallback envelope, not raised to the caller.
            timed_out = isinstance(exc, TimeoutError)
            _record_mlx_degradation(
                exc,
                action="returned empty batch after batched generation failed; caller falls back to serial",
                severity="warning",
            )
            return {}
        finally:
            # ALWAYS unregister — caller cancellation previously left the
            # worker command live and the future registered indefinitely.
            self._pending_generations.pop(req_id, None)
            if timed_out:
                # The queued decode continues invisibly after a timeout;
                # ask the worker to yield instead of burning the lane.
                with contextlib.suppress(Exception):
                    self.soft_cancel_active_generation(
                        reason=f"batch_timeout:{req_id[:12]}"
                    )
        if not res or res.get("status") != "ok":
            return {}
        raw_texts_value = res.get("texts")
        # A malformed worker payload (a plain string iterates as characters)
        # must not fabricate hundreds of one-character candidates.
        if not isinstance(raw_texts_value, (list, tuple)):
            _record_mlx_degradation(
                TypeError(f"batch texts payload was {type(raw_texts_value).__name__}"),
                action="dropped malformed batch response payload",
            )
            return {}
        raw_texts = [str(t or "") for t in raw_texts_value][:admitted_n]
        raw_candidate_tokens = list(res.get("tokens_used_by_candidate") or [])
        texts: list[str] = []
        tokens_used_by_candidate: list[int] = []
        for index, text in enumerate(raw_texts):
            if not text.strip():
                continue
            texts.append(text)
            try:
                candidate_tokens = max(0, int(raw_candidate_tokens[index] or 0))
            except (IndexError, TypeError, ValueError, OverflowError):
                candidate_tokens = 0
            tokens_used_by_candidate.append(candidate_tokens)
        try:
            tokens_used = max(0, int(res.get("tokens_used") or 0))
        except (TypeError, ValueError, OverflowError):
            tokens_used = 0
        return {
            "texts": texts,
            "request_id": req_id,
            "max_tokens": admitted_max_tokens,
            "temperature": admitted_temperature,
            "tokens_used": tokens_used,
            "tokens_used_by_candidate": tokens_used_by_candidate,
        }

    async def generate_batch_async(
        self,
        prompt: str,
        *,
        n: int = 4,
        max_tokens: int = 512,
        temperature: float = 0.8,
        timeout_s: float = 180.0,
    ) -> list[str]:
        """Decode raw verifier candidates in one batched worker pass."""

        response = await self._generate_batch_response_async(
            prompt,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
        return list(response.get("texts") or [])

    async def generate_batch_with_metadata_async(
        self,
        prompt: str,
        *,
        n: int = 4,
        max_tokens: int = 512,
        temperature: float = 0.8,
        timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Return batched candidates with one truthful shared decode receipt."""

        response = await self._generate_batch_response_async(
            prompt,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )
        texts = list(response.get("texts") or [])
        if not texts:
            return {}
        model_name = os.path.basename(str(self.model_path or "")) or "unknown"
        candidate_tokens = list(response.get("tokens_used_by_candidate") or [])
        return {
            "texts": texts,
            "generation_metadata": {
                "endpoint": f"MLX-BATCH:{model_name}",
                "provider": "mlx",
                "model": model_name,
                "is_local": True,
                "provider_verified": True,
                "batch_request_id": response.get("request_id"),
                "surface_control_receipt": {
                    "enabled": False,
                    "applied": False,
                    "generation_required": True,
                    "application_status": "raw_batch_requires_parent_verification",
                    "clean_user_surface_contract": False,
                    "surface_quality_gate_enabled": False,
                    "surface_quality_gate_passed": False,
                    "generation_max_tokens": response.get("max_tokens"),
                    "batch_generated_tokens_total": response.get("tokens_used"),
                    "batch_candidate_count": len(texts),
                    "source": "mlx_batch_worker",
                },
            },
            "candidate_generation_metadata": [
                {
                    "generated_tokens": (
                        max(0, int(candidate_tokens[index] or 0))
                        if index < len(candidate_tokens)
                        else 0
                    )
                }
                for index in range(len(texts))
            ],
        }

    async def ingest_nonparametric_async(
        self,
        *,
        max_pairs: int = 1,
        scan_limit: int = 16,
        max_positions: int = 96,
        max_sequence_tokens: int = 192,
        timeout_s: float = 20.0,
    ) -> dict[str, Any]:
        """Run bounded trusted-memory ingestion on a resident worker only.

        This maintenance command never spawns or loads a model.  It shares the
        worker request lock, advertises active ownership to the lane controller,
        and cooperatively cancels before recycling a worker that exceeds its
        deadline.
        """

        base = {
            "schema": "aura.nonparametric_ingest.worker.v1",
            "spawned_worker": False,
        }
        if self._closed:
            return {**base, "status": "skipped_client_closed"}
        if _foreground_owner_active():
            return {**base, "status": "skipped_foreground_active"}
        if self._active_generations > 0 or self._warmup_in_flight:
            return {**base, "status": "skipped_worker_busy"}
        if (
            self._req_q is None
            or not self._init_done
            or self._process is None
            or not self._process.is_alive()
        ):
            return {**base, "status": "skipped_worker_not_resident"}
        try:
            if get_memory_pressure_snapshot().refuse_heavy_local_generation:
                return {**base, "status": "skipped_memory_pressure"}
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError):
            return {**base, "status": "skipped_memory_unobservable"}

        try:
            bounded_timeout_s = max(2.0, min(35.0, float(timeout_s)))
            bounded_max_pairs = max(1, min(4, int(max_pairs)))
            bounded_scan_limit = max(1, min(64, int(scan_limit)))
            bounded_max_positions = max(1, min(256, int(max_positions)))
            bounded_max_sequence_tokens = max(
                8,
                min(512, int(max_sequence_tokens)),
            )
        except (TypeError, ValueError, OverflowError):
            return {**base, "status": "invalid_maintenance_budget"}
        deadline = get_deadline(bounded_timeout_s)
        acquired = await self._acquire_request_lock(
            owner_label="reasoning_nonparametric_ingest",
            deadline=deadline,
            foreground_request=False,
        )
        if not acquired:
            return {**base, "status": "skipped_request_lane_busy"}

        future: SharedFuture | None = None
        request_id = ""
        deferred_reboot = ""
        try:
            if (
                self._req_q is None
                or not self._init_done
                or self._process is None
                or not self._process.is_alive()
            ):
                return {**base, "status": "skipped_worker_not_resident"}
            if not await self._set_durable_lane_preemptible(False):
                return {**base, "status": "skipped_lane_fence_lost"}

            request_id = uuid.uuid4().hex
            self._job_seq_counter += 1
            request_seq = self._job_seq_counter
            request = {
                "id": request_id,
                "seq": request_seq,
                "action": "nonparametric_ingest",
                "max_pairs": bounded_max_pairs,
                "scan_limit": bounded_scan_limit,
                "max_positions": bounded_max_positions,
                "max_sequence_tokens": bounded_max_sequence_tokens,
                "deadline_s": max(1.0, bounded_timeout_s - 2.0),
            }
            future = _new_shared_future()
            self._pending_generations[request_id] = future
            self._current_gen_future = future
            self._active_generations += 1
            self._mark_generation_started(
                request_id,
                first_token_hard_ceiling_s=bounded_timeout_s,
                request_seq=request_seq,
            )
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(request, principal="mlx_client.structured_request"),
                True,
                2.0,
            )
            try:
                response = await _await_shared_future(
                    future,
                    timeout_s=bounded_timeout_s,
                )
            except TimeoutError:
                self.soft_cancel_active_generation(
                    reason="nonparametric_ingest_deadline"
                )
                try:
                    response = await _await_shared_future(future, timeout_s=3.0)
                except TimeoutError:
                    deferred_reboot = "nonparametric_ingest_deadline"
                    return {**base, "status": "timed_out_worker_recycled"}
            if not isinstance(response, dict):
                return {**base, "status": "invalid_worker_response"}
            if response.get("status") != "ok":
                return {
                    **base,
                    "status": "worker_error",
                    "reason": str(response.get("message") or "unknown"),
                }
            return {
                **base,
                "status": str(response.get("state") or "complete"),
                "pairs_considered": int(response.get("pairs_considered") or 0),
                "pairs_scanned": int(response.get("pairs_scanned") or 0),
                "pairs_ingested": int(response.get("pairs_ingested") or 0),
                "positions_ingested": int(
                    response.get("positions_ingested") or 0
                ),
            }
        except asyncio.CancelledError:
            if future is not None:
                self.soft_cancel_active_generation(
                    reason="nonparametric_ingest_caller_cancelled"
                )
                try:
                    await asyncio.shield(
                        _await_shared_future(future, timeout_s=3.0)
                    )
                except (asyncio.CancelledError, TimeoutError):
                    deferred_reboot = "nonparametric_ingest_cancel_drain_failed"
            raise
        except (BrokenPipeError, OSError, TimeoutError, queue.Full) as exc:
            _record_mlx_degradation(
                exc,
                action=(
                    "kept non-parametric maintenance bounded after resident "
                    "worker IPC failed"
                ),
                severity="warning",
            )
            return {**base, "status": f"ipc_failed:{type(exc).__name__}"}
        finally:
            try:
                if future is not None:
                    await asyncio.shield(
                        self._finish_generation_ownership(
                            request_id,
                            future,
                            None,
                        )
                    )
            finally:
                self._release_request_lock()
                if deferred_reboot:
                    await self.reboot_worker(
                        reason=deferred_reboot,
                        mark_failed=False,
                    )

    async def set_expert_adapter(
        self, adapter_path: str | None, *, timeout_s: float = 90.0
    ) -> dict[str, Any]:
        """Attach/detach a domain-specialist LoRA on the RESIDENT worker model.

        The expert-LoRA library's live seam: the adapter (~40MB) is wrapped
        onto the loaded model inside the worker — no model reload, seconds not
        minutes. ``None``/"" detaches. Refuses while a generation is active
        (weights must never change mid-decode) and never spawns a worker just
        to attach — an adapter is worthless without a resident model.
        """
        path = str(adapter_path or "").strip()
        if self._closed:
            return {"ok": False, "reason": "client_closed"}
        if (
            self._req_q is None
            or not (self._process and self._process.is_alive() and self._init_done)
        ):
            return {"ok": False, "reason": "worker_not_ready"}
        if int(getattr(self, "_active_generations", 0) or 0) > 0 or self._warmup_in_flight:
            return {"ok": False, "reason": "generation_active"}
        adapter_exists = (
            await asyncio.to_thread(lambda: Path(path).expanduser().is_dir())
            if path
            else True
        )
        if not adapter_exists:
            return {"ok": False, "reason": f"adapter_missing:{path}"}

        req_id = uuid.uuid4().hex
        fut = _new_shared_future()
        self._pending_generations[req_id] = fut
        try:
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(
                    {"id": req_id, "action": "set_expert_adapter", "path": path},
                    principal="mlx_client.expert_adapter",
                ),
                True,
                2.0,
            )
            res = await _await_shared_future(fut, timeout_s=max(10.0, float(timeout_s)))
        except (TimeoutError, BrokenPipeError, OSError) as exc:
            self._pending_generations.pop(req_id, None)
            _record_mlx_degradation(
                exc,
                action="left resident model unchanged after expert adapter swap timed out",
                severity="warning",
            )
            return {"ok": False, "reason": f"swap_timeout:{type(exc).__name__}"}

        if res and res.get("status") == "ok":
            self._expert_adapter_path = str(res.get("resident") or "") or None
            return {
                "ok": True,
                "resident": self._expert_adapter_path,
                "wrapped_layers": int(res.get("wrapped_layers") or 0),
                "detached_layers": int(res.get("detached_layers") or 0),
            }
        return {
            "ok": False,
            "reason": str((res or {}).get("message") or "swap_failed"),
        }

    @property
    def expert_adapter_resident(self) -> str | None:
        return getattr(self, "_expert_adapter_path", None)

    def _init_receipt_errors(self, res: dict[str, Any]) -> list[str]:
        """Every reason this init receipt must NOT be trusted as READY.

        Checks the exact model/worker identity and the recurrent-depth
        invariants the lane declares it needs. Returns an empty list only when
        the receipt positively establishes both.
        """
        errors: list[str] = []

        identity = res.get("worker_identity")
        try:
            from core.brain.llm.latent_cortex.runtime_identity import (
                worker_identity_errors,
            )

            errors.extend(worker_identity_errors(identity))
        except ImportError as exc:
            # No validator means no proof of identity. Absence of a check is
            # not a passed check.
            _record_mlx_degradation(
                exc,
                action="worker identity validator unavailable during handshake",
                severity="error",
            )
            errors.append("worker_identity_validator_unavailable")

        # The worker must be serving the model THIS client asked for.
        if isinstance(identity, dict):
            reported_path = str(identity.get("worker_model_path") or "")
            if reported_path and _real_model_path(reported_path) != _real_model_path(
                self.model_path
            ):
                errors.append("worker_model_path_mismatch")

        # Recurrence: if this lane requires depth, the receipt must prove it.
        required_loops = _expected_recurrent_loops_from_model_path(self.model_path)
        recurrent_status = res.get("recurrent_depth")
        if required_loops > 1:
            if not isinstance(recurrent_status, dict):
                errors.append("missing_recurrent_depth_receipt")
            else:
                if not bool(recurrent_status.get("active")):
                    errors.append("recurrent_depth_inactive")
                reported_loops = recurrent_status.get("loops")
                if (
                    isinstance(reported_loops, int)
                    and reported_loops != required_loops
                ):
                    errors.append(
                        f"recurrent_depth_mismatch:{reported_loops}!={required_loops}"
                    )
        return errors

    def get_worker_identity_snapshot(self) -> dict[str, Any]:
        """Return immutable identity evidence for resident-scale policy decisions."""

        identity = getattr(self, "_worker_identity", None)
        return dict(identity) if isinstance(identity, dict) else {}

    def _clean_latent_cancel_ack(
        self,
        response: Any,
        *,
        expected_request_id: str = "",
        expected_request_sha256: str = "",
    ) -> bool:
        """Whether this acknowledgement proves a CLEAN cancel of THIS episode.

        CP126 07d62d51. The check used to accept a reason string and a couple
        of worker-supplied booleans, bound to nothing. Anything shaped like
        {"message": "soft_cancelled", "receipt": {"params_unchanged": True}}
        could therefore certify that model parameters were untouched and
        ephemeral weights erased — for a different request, a different
        worker, or a previous episode entirely. That certification is what
        lets the lane keep serving without a reboot, so a stale or replayed
        ack was a path to serving on weights nobody had proven clean.

        The receipt already carries the identity needed to bind it; nothing
        was reading it. An acknowledgement now has to name this request, this
        payload, and this worker.
        """
        if not isinstance(response, dict):
            return False
        reason = str(response.get("message") or response.get("reason") or "")
        receipt = response.get("receipt")
        if reason != "soft_cancelled" or not isinstance(receipt, dict):
            return False

        # Bound to THIS request.
        if expected_request_id:
            if str(response.get("id") or "") != expected_request_id:
                self._record_cancel_ack_rejection("request_id_mismatch")
                return False
        # Bound to THIS payload.
        if expected_request_sha256:
            if str(receipt.get("request_payload_sha256") or "") != expected_request_sha256:
                self._record_cancel_ack_rejection("request_payload_sha256_mismatch")
                return False
        # Bound to THIS worker: a receipt from a previous boot describes a
        # process whose weights are no longer the ones we are about to keep
        # serving on.
        identity = getattr(self, "_worker_identity", None)
        if isinstance(identity, dict) and identity:
            expected_boot = str(identity.get("worker_boot_id") or "")
            if expected_boot and str(receipt.get("worker_boot_id") or "") != expected_boot:
                self._record_cancel_ack_rejection("worker_boot_id_mismatch")
                return False
            expected_pid = identity.get("worker_pid")
            if (
                isinstance(expected_pid, int)
                and receipt.get("worker_pid") not in (None, expected_pid)
            ):
                self._record_cancel_ack_rejection("worker_pid_mismatch")
                return False
        reported_path = str(receipt.get("worker_model_path") or "")
        if reported_path and _real_model_path(reported_path) != _real_model_path(
            self.model_path
        ):
            self._record_cancel_ack_rejection("worker_model_path_mismatch")
            return False

        if receipt.get("params_unchanged") is not True:
            return False
        if (
            receipt.get("fast_weights_applied") is True
            and receipt.get("fast_weights_erased") is not True
        ):
            return False
        return True

    def _record_cancel_ack_rejection(self, why: str) -> None:
        """An ack that failed to bind is evidence, not noise.

        A worker sending unbindable cancellation receipts is either buggy or
        replaying, and either way the lane must reboot rather than trust the
        clean-cancel claim.
        """
        _record_mlx_degradation(
            RuntimeError(f"latent_cancel_ack_unbound:{why}"),
            action="refused a latent cancellation acknowledgement it could not bind",
            severity="error",
        )

    async def latent_reason_async(
        self,
        prompt: str | None = None,
        *,
        messages: list | None = None,
        config: dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
        runtime_controls: dict[str, Any] | None = None,
        domain: str = "general",
        timeout_s: float = 300.0,
        foreground_request: bool = True,
        verifier_guidance: bool = False,
        facet_reliability: dict[str, float] | None = None,
        cognitive_context: list | None = None,
        operation_authority: dict[str, Any] | None = None,
        action_policy_evidence: dict[str, Any] | None = None,
        response_contract: str | None = None,
    ) -> dict[str, Any]:
        """Run a Recursive Latent Cortex episode on the RESIDENT worker model.

        Workspace recurrence + virtual-width branches over the frozen
        checkpoint (docs/RECURSIVE_LATENT_CORTEX.md). Refuses while a
        generation is in flight (the episode needs exclusive weights/KV) and
        never spawns a worker just to think — no resident model, no episode.
        Returns ``{"ok": bool, "text": str, "receipt": {...}, "reason": str}``.
        """
        base = {"ok": False, "text": "", "receipt": {}}
        if self._closed:
            return {**base, "reason": "client_closed"}
        if not (isinstance(prompt, str) and prompt.strip()) and not (
            isinstance(messages, list) and messages
        ):
            return {**base, "reason": "empty_prompt"}
        if type(foreground_request) is not bool:
            return {**base, "reason": "invalid_foreground_request"}
        if config is not None and not isinstance(config, dict):
            return {**base, "reason": "invalid_config"}
        if budget is not None and not isinstance(budget, dict):
            return {**base, "reason": "invalid_budget"}
        if runtime_controls is not None and not isinstance(runtime_controls, dict):
            return {**base, "reason": "invalid_runtime_controls"}
        if response_contract is not None:
            if not isinstance(response_contract, str) or not response_contract.strip():
                return {**base, "reason": "invalid_response_contract"}
            try:
                from core.brain.llm.latent_cortex.response_contracts import (
                    parse_response_contract,
                )

                parse_response_contract(response_contract)
            except ValueError:
                return {**base, "reason": "invalid_response_contract"}
        wire_cognitive_context: list[dict[str, Any]] | None = None
        try:
            from core.brain.llm.latent_cortex.cognitive_context import (
                normalize_cognitive_context,
            )

            wire_cognitive_context = normalize_cognitive_context(cognitive_context) or None
        except (TypeError, ValueError):
            return {**base, "reason": "invalid_cognitive_context"}
        wire_config = dict(config or {})
        wire_budget = dict(budget or {})
        wire_runtime_controls = dict(runtime_controls or {})
        wire_action_policy_evidence: dict[str, Any] | None = None
        if action_policy_evidence is not None:
            try:
                from core.brain.llm.latent_cortex.value_of_computation import (
                    validate_evidence_snapshot,
                )

                wire_action_policy_evidence = validate_evidence_snapshot(
                    action_policy_evidence
                )
            except (ImportError, TypeError, ValueError):
                return {**base, "reason": "invalid_action_policy_evidence"}
        wire_operation_authority: dict[str, Any] | None = None
        if operation_authority is not None:
            try:
                from core.brain.llm.latent_cortex.epistemic_runtime import (
                    validate_runtime_operation_authority,
                )

                wire_operation_authority = validate_runtime_operation_authority(
                    operation_authority,
                    prompt=prompt,
                    messages=messages,
                    config=wire_config,
                    budget=wire_budget,
                    cognitive_context=wire_cognitive_context,
                    action_policy_evidence=wire_action_policy_evidence,
                )
            except (ImportError, TypeError, ValueError):
                return {**base, "reason": "invalid_runtime_operation_authority"}
        if runtime_controls is not None:
            required_controls = {
                "clean_user_surface_recurrent_loops",
                "clean_user_surface_steering_alpha",
            }
            if set(wire_runtime_controls) != required_controls:
                return {**base, "reason": "invalid_runtime_controls"}
            recurrent_loops = wire_runtime_controls.get(
                "clean_user_surface_recurrent_loops"
            )
            steering_alpha = wire_runtime_controls.get(
                "clean_user_surface_steering_alpha"
            )
            if (
                type(recurrent_loops) is not int
                or not 1 <= recurrent_loops <= 2
                or isinstance(steering_alpha, bool)
                or not isinstance(steering_alpha, (int, float))
                or not math.isfinite(float(steering_alpha))
                or not 0.01 <= float(steering_alpha) <= 1.0
            ):
                return {**base, "reason": "invalid_runtime_controls"}
        # CP126 9721b1be. These are semantic inputs to the episode, so they
        # must be normalized ONCE here and bound into the request digest —
        # building them only at job-construction time left two episodes with
        # different verifier behavior sharing one expected request identity.
        wire_verifier_guidance = True if verifier_guidance else None
        wire_facet_reliability: dict[str, float] | None = None
        if verifier_guidance and isinstance(facet_reliability, dict) and facet_reliability:
            # Held-out facet calibration rides only alongside the verifier it
            # calibrates; worker revalidates the shape.
            wire_facet_reliability = {
                str(name): float(value)
                for name, value in list(facet_reliability.items())[:8]
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            } or None
        try:
            from core.brain.llm.latent_cortex.runtime_identity import (
                latent_request_payload_sha256,
            )

            expected_request_sha256 = latent_request_payload_sha256(
                prompt=str(prompt) if prompt is not None else None,
                messages=list(messages) if messages is not None else None,
                domain=str(domain or "general"),
                config=wire_config if config is not None else None,
                budget=wire_budget if budget is not None else None,
                runtime_controls=(
                    wire_runtime_controls if runtime_controls is not None else None
                ),
                cognitive_context=wire_cognitive_context,
                operation_authority=wire_operation_authority,
                action_policy_evidence=wire_action_policy_evidence,
                response_contract=response_contract,
                verifier_guidance=wire_verifier_guidance,
                facet_reliability=wire_facet_reliability,
            )
        except (TypeError, ValueError, OverflowError):
            return {**base, "reason": "invalid_request_payload"}
        try:
            bounded_timeout_s = float(timeout_s)
        except (TypeError, ValueError, OverflowError):
            return {**base, "reason": "invalid_timeout"}
        if not math.isfinite(bounded_timeout_s) or bounded_timeout_s <= 0.0:
            return {**base, "reason": "invalid_timeout"}
        bounded_timeout_s = min(900.0, max(5.0, bounded_timeout_s))
        if (
            self._req_q is None
            or not (self._process and self._process.is_alive() and self._init_done)
        ):
            return {**base, "reason": "worker_not_ready"}
        try:
            if get_memory_pressure_snapshot().refuse_heavy_local_generation:
                return {**base, "reason": "memory_pressure"}
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError):
            return {**base, "reason": "memory_pressure_unobservable"}

        deadline = get_deadline(bounded_timeout_s)
        owner_label = "latent_cortex_foreground" if foreground_request else "latent_cortex_lab"
        foreground_owner_cm = None
        if foreground_request:
            foreground_owner_cm = _foreground_owner_context(
                owner_label,
                deadline=deadline,
                foreground_request=True,
                stale_after=bounded_timeout_s,
            )
            try:
                await foreground_owner_cm.__aenter__()
            except TimeoutError:
                return {**base, "reason": "foreground_owner_busy"}

        try:
            acquired = await self._acquire_request_lock(
                owner_label=owner_label,
                deadline=deadline,
                foreground_request=foreground_request,
            )
        except BaseException:  # noqa: BLE001 - cancellation must release foreground ownership
            if foreground_owner_cm is not None:
                await asyncio.shield(foreground_owner_cm.__aexit__(*sys.exc_info()))
            raise
        if not acquired:
            if foreground_owner_cm is not None:
                await foreground_owner_cm.__aexit__(None, None, None)
            return {**base, "reason": "request_lane_busy"}

        fut: SharedFuture | None = None
        req_id = ""
        deferred_reboot = ""
        lane_fenced = False
        try:
            if (
                self._req_q is None
                or not (self._process and self._process.is_alive() and self._init_done)
            ):
                return {**base, "reason": "worker_not_ready"}
            if self._warmup_in_flight or self._active_generations > 0:
                return {**base, "reason": "generation_active"}
            if not await self._set_durable_lane_preemptible(False):
                return {**base, "reason": "lane_fence_lost"}
            lane_fenced = True

            req_id = uuid.uuid4().hex
            self._job_seq_counter += 1
            request_seq = self._job_seq_counter
            job: dict[str, Any] = {
                "id": req_id,
                "seq": request_seq,
                "action": "latent_reason",
                "domain": str(domain or "general"),
            }
            # Exactly the values bound into expected_request_sha256 above.
            if wire_verifier_guidance:
                job["verifier_guidance"] = True
                if wire_facet_reliability:
                    job["facet_reliability"] = dict(wire_facet_reliability)
            if prompt is not None:
                job["prompt"] = str(prompt)
            if messages is not None:
                job["messages"] = list(messages)
            if config is not None:
                job["config"] = wire_config
            if budget is not None:
                job["budget"] = wire_budget
            if runtime_controls is not None:
                job["runtime_controls"] = wire_runtime_controls
                job["clean_user_surface_contract"] = True
                job["live_mind_controls_bound"] = True
                job.update(wire_runtime_controls)
            else:
                # Latent episodes without explicit surface-parity controls are
                # the experiment lane: they keep historical full governor
                # steering. Every OTHER worker job now defaults to the surface
                # clamp (fail-safe inversion after the July 2026 coherence
                # incident) — this opt-out is deliberately scoped to episodes.
                job["allow_full_affective_steering"] = True
            if wire_cognitive_context is not None:
                job["cognitive_context"] = wire_cognitive_context
            if wire_operation_authority is not None:
                job["operation_authority"] = wire_operation_authority
            if wire_action_policy_evidence is not None:
                job["action_policy_evidence"] = wire_action_policy_evidence
            if response_contract is not None:
                job["response_contract"] = response_contract

            fut = _new_shared_future()
            self._pending_generations[req_id] = fut
            self._latent_progress_by_request[req_id] = {
                "request_id": req_id,
                "stage": "submitted",
                "received_at_unix": time.time(),
            }
            self._current_gen_future = fut
            self._active_generations += 1
            requested_tokens_raw = wire_config.get("decode_max_tokens", 0)
            requested_tokens = (
                requested_tokens_raw
                if type(requested_tokens_raw) is int and requested_tokens_raw > 0
                else 0
            )
            prompt_chars = len(prompt or "") + sum(
                len(str(message.get("content") or ""))
                for message in (messages or [])
                if isinstance(message, dict)
            )
            self._mark_generation_started(
                req_id,
                prompt_chars=prompt_chars,
                requested_max_tokens=requested_tokens,
                first_token_hard_ceiling_s=bounded_timeout_s,
                request_seq=request_seq,
            )
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(job, principal="mlx_client.latent_reason"),
                True,
                min(2.0, max(0.5, deadline.remaining or 2.0)),
            )
            try:
                res = await _await_shared_future(fut, timeout_s=bounded_timeout_s)
            except TimeoutError:
                self.soft_cancel_active_generation("latent_reason_deadline")
                try:
                    cancel_ack = await _await_shared_future(
                        fut,
                        timeout_s=min(
                            12.0,
                            max(3.0, bounded_timeout_s * 0.10),
                        ),
                    )
                except (TimeoutError, BrokenPipeError, OSError):
                    cancel_ack = None
                if self._clean_latent_cancel_ack(
                    cancel_ack,
                    expected_request_id=req_id,
                    expected_request_sha256=expected_request_sha256,
                ):
                    receipt = dict(cancel_ack.get("receipt") or {})
                    progress = dict(
                        self._latent_progress_by_request.get(req_id) or {}
                    )
                    logger.warning(
                        "Latent owner deadline reached cleanly: stage=%s "
                        "input_tokens=%s elapsed=%s timings=%s",
                        receipt.get("last_stage")
                        or progress.get("stage")
                        or "unknown",
                        receipt.get("input_token_count")
                        or progress.get("input_tokens")
                        or "unknown",
                        progress.get("elapsed_s") or "unknown",
                        receipt.get("stage_timings_s") or {},
                    )
                    return {
                        **base,
                        "receipt": receipt,
                        "progress": progress,
                        "reason": "latent_timeout:cooperative_cancelled",
                    }
                deferred_reboot = "latent_reason_deadline_unacknowledged"
                return {**base, "reason": "latent_timeout:TimeoutError"}

            if not isinstance(res, dict):
                return {**base, "reason": "invalid_worker_response"}
            raw_receipt = res.get("receipt")
            if raw_receipt is not None and not isinstance(raw_receipt, dict):
                return {**base, "reason": "invalid_worker_receipt"}
            receipt = dict(raw_receipt or {})
            reason = str(res.get("message") or res.get("reason") or "")
            if reason in {
                "checkpoint_invariant_violated",
                "fast_weight_cleanup_unproven",
            }:
                deferred_reboot = f"latent_integrity:{reason}"
            if res.get("status") == "ok":
                from core.brain.llm.latent_cortex.runtime_identity import (
                    collect_latent_runtime_identity,
                    worker_identity_errors,
                )

                identity_errors = worker_identity_errors(
                    receipt,
                    expected=getattr(self, "_worker_identity", {}),
                )
                if receipt.get("request_payload_sha256") != expected_request_sha256:
                    identity_errors.append("request_payload_sha256_mismatch")
                if identity_errors:
                    deferred_reboot = "latent_integrity:worker_identity_mismatch"
                    return {
                        **base,
                        "receipt": receipt,
                        "reason": "worker_identity_failed:" + ",".join(identity_errors),
                    }
                try:
                    identity_remaining = deadline.remaining
                    if identity_remaining is not None and identity_remaining <= 0.0:
                        return {
                            **base,
                            "receipt": receipt,
                            "reason": "runtime_identity_deadline_exhausted",
                        }
                    identity_timeout = min(
                        15.0,
                        max(0.1, float(identity_remaining or 15.0)),
                    )
                    runtime_identity = await asyncio.wait_for(
                        run_io_bound(
                            collect_latent_runtime_identity,
                            _AURA_SOURCE_ROOT,
                        ),
                        timeout=identity_timeout,
                    )
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    _record_mlx_degradation(
                        exc,
                        action="refused latent success whose runtime identity could not be captured",
                        severity="degraded",
                    )
                    return {
                        **base,
                        "receipt": receipt,
                        "reason": f"runtime_identity_failed:{type(exc).__name__}",
                    }
                receipt["runtime_identity"] = dict(runtime_identity)
                if runtime_identity.get("identity_bound") is not True:
                    return {
                        **base,
                        "receipt": receipt,
                        "reason": "runtime_identity_unbound",
                    }
                # CP126 d78cbfa4: a status=ok response used to be coerced with
                # str(value or "") — a missing, empty, list, or mapping answer
                # became ok=true with empty or stringified-container text and
                # bypassed fallback entirely. An episode is successful only
                # when it produced an actual nonempty STRING answer.
                answer = res.get("text")
                if not isinstance(answer, str) or not answer.strip():
                    _record_mlx_degradation(
                        TypeError(
                            "latent_answer_invalid:"
                            f"{type(answer).__name__}:{len(answer) if isinstance(answer, str) else 'n/a'}"
                        ),
                        action="refused latent success for a missing, empty, or non-string answer",
                        severity="degraded",
                    )
                    return {
                        **base,
                        "receipt": receipt,
                        "progress": dict(
                            self._latent_progress_by_request.get(req_id) or {}
                        ),
                        "reason": "latent_answer_invalid",
                    }
                self._mark_progress()
                return {
                    "ok": True,
                    "text": answer,
                    "receipt": receipt,
                    "progress": dict(
                        self._latent_progress_by_request.get(req_id) or {}
                    ),
                    "reason": str(res.get("reason") or ""),
                }
            return {
                **base,
                "receipt": receipt,
                "progress": dict(
                    self._latent_progress_by_request.get(req_id) or {}
                ),
                "reason": reason or "latent_reason_failed",
            }
        except asyncio.CancelledError:
            if fut is not None:
                self.soft_cancel_active_generation("latent_reason_caller_cancelled")
                deferred_reboot = "latent_reason_caller_cancelled"
            raise
        except (BrokenPipeError, OSError, TimeoutError, queue.Full) as exc:
            deferred_reboot = f"latent_ipc_failed:{type(exc).__name__}"
            _record_mlx_degradation(
                exc,
                action="recycled resident worker after latent_reason IPC failure",
                severity="warning",
            )
            return {**base, "reason": f"latent_ipc_failed:{type(exc).__name__}"}
        finally:
            try:
                try:
                    if fut is not None:
                        await asyncio.shield(
                            self._finish_generation_ownership(
                                req_id,
                                fut,
                                None,
                                release_lane=not bool(deferred_reboot),
                            )
                        )
                finally:
                    if deferred_reboot:
                        await asyncio.shield(
                            self.reboot_worker(
                                reason=deferred_reboot,
                                mark_failed=False,
                            )
                        )
                    elif lane_fenced and fut is None and self._active_generations <= 0:
                        await asyncio.shield(
                            self._set_durable_lane_preemptible(True)
                        )
            finally:
                self._latent_progress_by_request.pop(req_id, None)
                self._release_request_lock()
                if foreground_owner_cm is not None:
                    await foreground_owner_cm.__aexit__(None, None, None)

    async def reload_model_artifact(self, model_path: str) -> dict[str, Any]:
        """Serve a newly published fused artifact by re-pointing this lane.

        The model lives in the WORKER process, so the only correct swap is a
        worker recycle with the new path. (This replaces a retired
        live_learner monkey-patch that loaded a second full copy of the model
        into the ORCHESTRATOR process — ~20GB of wired memory on the 32B lane
        — while generations kept flowing through the worker's old weights.)
        Busy lanes defer the recycle until the active request finishes; the
        respawn path re-resolves the fused manifest, so crash recovery after
        the swap also serves the promoted artifact.
        """
        resolved = await asyncio.to_thread(
            lambda: Path(str(model_path or "")).expanduser()
        )
        if not await asyncio.to_thread(resolved.is_dir):
            return {"ok": False, "reason": f"artifact_missing:{resolved}"}
        previous = self.model_path
        self.model_path = str(resolved)
        self._expert_adapter_path = None  # adapters belong to the old weights
        if (
            int(getattr(self, "_active_generations", 0) or 0) > 0
            or self._current_request_started_at > 0.0
        ):
            self._deferred_reboot_reason = "promoted_artifact_swap"
            logger.info(
                "🧬 [MLX] Promoted artifact staged for %s; recycling after the active request.",
                resolved.name,
            )
            return {"ok": True, "mode": "deferred", "previous": previous}
        await self.reboot_worker(reason="promoted_artifact_swap", mark_failed=False)
        logger.info("🧬 [MLX] Promoted artifact live: %s", resolved.name)
        return {"ok": True, "mode": "recycled", "previous": previous}

    def soft_cancel_active_generation(self, reason: str = "foreground_preemption") -> dict[str, Any]:
        """Ask the ACTIVE generation to stop between tokens (cooperative).

        Writes the active job's sequence number into shared memory; the worker
        token loop polls it each decode step and finishes the job early with a
        ``soft_cancelled`` response. Cancel latency is roughly one decode step
        and the worker (and its loaded model) stays warm — the cheap first
        rung on the preemption ladder before ``force_abort_active_generation``.

        Returns a receipt; ``requested`` is False when there is nothing to
        cancel or the cancel channel is unavailable.
        """
        reason = str(reason or "foreground_preemption")
        cancel_seq = getattr(self, "_cancel_seq", None)
        active_seq = int(getattr(self, "_current_request_seq", 0) or 0)
        generation_active = bool(
            self._current_request_started_at > 0.0 and active_seq > 0
        )
        if cancel_seq is None or not generation_active:
            return {
                "requested": False,
                "reason": reason,
                "active_seq": active_seq,
                "detail": "no_active_generation" if cancel_seq is not None else "cancel_channel_unavailable",
            }
        try:
            cancel_seq.value = active_seq
        except (OSError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="fell back to force-abort ladder after soft-cancel write failed",
                severity="warning",
            )
            return {
                "requested": False,
                "reason": reason,
                "active_seq": active_seq,
                "detail": f"cancel_write_failed:{type(exc).__name__}",
            }
        logger.info(
            "✋ [MLX] Soft-cancel requested for job seq=%d on %s (%s).",
            active_seq,
            os.path.basename(self.model_path),
            reason,
        )
        return {"requested": True, "reason": reason, "active_seq": active_seq}

    # Deferred-reboot reasons (after the recoverable_ prefix is stripped) that
    # follow a soft-cancel and are therefore eligible for warm-lane
    # preservation when the worker acknowledges the cancel.
    _SOFT_CANCEL_PRESERVABLE_REASONS = frozenset(
        {
            "first_token_sla_exceeded",
            "token_progress_stalled",
            "generation_deadline_reached",
        }
    )

    async def _soft_cancel_acknowledged(self, timeout_s: float | None = None) -> bool:
        """Wait (bounded) for the worker to acknowledge a soft-cancel.

        Acknowledgement = the worker cleared the shared cancel flag (it
        demonstrably passed through its token loop) while staying alive with
        fresh heartbeats. When this returns True the orphaned generation has
        already been dropped worker-side — late text cannot bleed into the
        next turn because its request id is no longer pending — so the warm
        model can be preserved instead of paying a ~60-90s reload.
        """
        if timeout_s is None:
            timeout_s = _finite_env_float("AURA_MLX_SOFT_CANCEL_ACK_WAIT_S", 12.0, minimum=0.5)
        try:
            timeout_s = float(timeout_s)
        except (TypeError, ValueError):
            timeout_s = 12.0
        if not math.isfinite(timeout_s):
            # An infinite ack wait would wedge cleanup forever.
            timeout_s = 12.0
        cancel_seq = getattr(self, "_cancel_seq", None)
        if cancel_seq is None:
            return False
        deadline = time.monotonic() + min(120.0, max(0.5, timeout_s))
        while time.monotonic() < deadline:
            process = self._process
            if process is None or not process.is_alive():
                return False
            try:
                cancel_cleared = int(getattr(cancel_seq, "value", 0)) == 0
            except (OSError, ValueError):
                return False
            heartbeat_fresh = (
                self._last_heartbeat > 0.0
                and (time.time() - self._last_heartbeat) < 20.0
            )
            if cancel_cleared and heartbeat_fresh:
                return True
            await asyncio.sleep(0.25)
        return False

    async def _resolve_deferred_reboot(self, deferred: str) -> None:
        """Resolve an abandoned-request verdict: preserve the warm lane or reboot.

        Historically EVERY abandoned request recycled the worker ("so late
        text cannot bleed into the next turn") — a full model reload during
        which arriving turns died, observed live as soak death-clusters. The
        soft-cancel channel already isolates the orphaned output; what was
        missing is verifying the worker actually observed the cancel. Now:
        recoverable abandons keep the warm worker when the cancel is
        acknowledged, and only unacknowledged (truly wedged) workers reboot.
        """
        recoverable = deferred.startswith("recoverable_")
        reason = deferred.removeprefix("recoverable_")
        if recoverable and reason in self._SOFT_CANCEL_PRESERVABLE_REASONS:
            if await self._soft_cancel_acknowledged():
                logger.info(
                    "♻️✅ [MLX] Worker acknowledged soft-cancel after %s — warm lane preserved, no reboot.",
                    reason,
                )
                self._record_degraded_event(
                    "warm_lane_preserved_after_soft_cancel",
                    detail=f"{os.path.basename(self.model_path)}:{reason}",
                    severity="warning",
                    foreground_request=False,
                )
                return
            logger.warning(
                "🛑 [MLX] No soft-cancel acknowledgement after %s — worker presumed wedged; rebooting.",
                reason,
            )
        # A staged artifact promotion is maintenance, not a failure verdict.
        await self.reboot_worker(
            reason=reason,
            mark_failed=not recoverable and reason != "promoted_artifact_swap",
        )

    def force_abort_active_generation(self, reason: str = "hard_generation_deadline") -> bool:
        """Thread-safe emergency abort for a wedged generation.

        Normal cancellations should flow through ``reboot_worker``. This path is
        intentionally synchronous so an external watchdog can break a stuck
        proof or foreground request even when the caller's event loop is waiting
        on an MLX future that failed to observe its deadline.
        """
        reason = str(reason or "hard_generation_deadline")
        pending_by_request = {
            str(req_id): future
            for req_id, future in list(self._pending_generations.items())
            if future is not None and not future.done()
        }
        current_future = self._current_gen_future
        had_active_request = bool(
            pending_by_request
            or (current_future is not None and not current_future.done())
            or self._active_generations > 0
            or self._current_request_started_at > 0.0
        )
        process = self._process
        had_process = bool(process is not None and process.is_alive())
        if not had_active_request and not had_process:
            return False
        if not had_active_request:
            # An idle-but-alive worker is being killed on an arbitrary reason
            # string — legitimate for emergencies, but it must be visible.
            _record_mlx_degradation(
                RuntimeError(f"force_abort_without_active_request:{reason}"),
                action="force-aborted an idle worker with no pending request",
                severity="warning",
            )

        logger.error(
            "🛑 [MLX] Force-aborting active generation for %s (%s).",
            os.path.basename(self.model_path),
            reason,
        )
        self._set_lane_state("recovering", reason)

        # Each future receives ITS OWN request identity: completing every
        # pending future with the current request id let callers receive an
        # abort receipt for another request.
        seen_future_ids: set[int] = set()
        for req_id, future in pending_by_request.items():
            seen_future_ids.add(id(future))
            _set_shared_future_result(
                future,
                {
                    "status": "error",
                    "action": "generate",
                    "id": req_id,
                    "message": reason,
                    "force_aborted": True,
                },
            )
        if (
            current_future is not None
            and not current_future.done()
            and id(current_future) not in seen_future_ids
        ):
            _set_shared_future_result(
                current_future,
                {
                    "status": "error",
                    "action": "generate",
                    "id": self._current_request_id,
                    "message": reason,
                    "force_aborted": True,
                },
            )

        killed_process_before_lock = False
        if process is not None and process.is_alive():
            logger.error(
                "🛑 [MLX] Killing worker immediately for forced abort before lifecycle lock cleanup (%s).",
                reason,
            )
            _note_lane_worker_death(self, reason)
            # Consume the lifetime anchor so no later seam double-counts
            # this death (reboot_worker / the self-died respawn branch).
            self._process_started_at = 0.0
            self._kill_and_join_blocking(process)
            killed_process_before_lock = True

        acquired = self._lock.acquire(timeout=0.25)
        if not acquired:
            # Emergency semantics: proceed anyway (the wedge may BE the lock
            # holder), but leave a visible receipt that lifecycle state was
            # mutated without ownership so a racing spawn/reboot can be
            # diagnosed instead of silently corrupted.
            _record_mlx_degradation(
                RuntimeError("force_abort_without_lifecycle_lock"),
                action="force-abort mutated lifecycle state without the lifecycle lock",
                severity="error",
            )
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
                _note_lane_worker_death(self, reason)
                self._process_started_at = 0.0
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
        try:
            self._release_durable_model_lane_owner_sync(reason=reason)
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            # CP126 35eefee4. This recorded a critical degradation and then
            # marked the lane COLD and returned success. The fencing token was
            # NOT cleared (the release raised part-way), so later admission
            # would be blocked by a fence nobody had been told about — a
            # terminal recovery dependency reported as a clean abort.
            #
            # The abort itself did happen, so the caller is still told the
            # generation was aborted; what changes is that the lane is left in
            # a NAMED fenced state carrying the owner and token that must be
            # released before this lane can serve again.
            self._note_lane_release_failure(exc, reason=reason)
            return True
        self._set_lane_state("cold", reason)
        return True

    def _spawn_worker_blocking(self) -> mp.Process:
        """Isolated spawn logic for the MLX worker, run in a background thread."""
        if _shutdown_blocks_model_work(self.model_path, action="worker spawn"):
            raise RuntimeError("runtime_shutdown")
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
            for observed_process in get_resource_observer().processes():
                if _shutdown_blocks_model_work(self.model_path, action="orphan scan"):
                    raise RuntimeError("runtime_shutdown")
                try:
                    pname = observed_process.name
                    command = observed_process.cmdline
                    if target_name in pname or (
                        command
                        and any(model_basename in str(arg) for arg in command)
                        and "mlx_worker" in str(command)
                    ):
                        ancestor_pids = set(observed_process.ancestor_pids)
                        if (
                            observed_process.pid != os.getpid()
                            and os.getpid() in ancestor_pids
                        ):
                            logger.warning(
                                "🧹 [STABILITY] Killing orphan MLXWorker pid=%d for %s",
                                observed_process.pid,
                                model_basename,
                            )
                            action_process = psutil.Process(observed_process.pid)
                            action_process.kill()
                            action_process.wait(timeout=3.0)
                        elif observed_process.pid != os.getpid():
                            logger.info(
                                "Model-path match pid=%d for %s belongs to another root; "
                                "durable lane accounting will arbitrate it without cross-root kill.",
                                observed_process.pid,
                                model_basename,
                            )
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except (OSError, ConnectionError, TimeoutError) as orphan_exc:
            _record_mlx_degradation(
                orphan_exc,
                action="continued worker spawn after orphan reclamation scan failed",
            )
            logger.debug("Orphan reclamation scan failed (non-fatal): %s", orphan_exc)

        memory_block = _memory_pressure_blocks_worker_spawn(self.model_path)
        if memory_block and not self._is_deep_solver_lane():
            # A worker we just killed (orphan reclamation above, or a prior
            # generation-timeout force-abort) frees ~18GB, but the OS reclaim of
            # wired Metal memory lags process exit. Checking headroom instantly
            # sees the pre-reclaim number and refuses — which takes the whole
            # conversation lane COLD even though the memory is about to be free.
            # Observed live during the 200-turn soak (2026-07-06): a Cortex
            # generation timed out, the worker was killed, respawn was refused
            # at 20.3GB < 24GB while the killed worker's 18.6GB had not yet been
            # reclaimed, and a cluster of turns died until pressure eased. Wait
            # (bounded) for reclaim and re-check before refusing. Runs in
            # _spawn_worker_blocking's executor thread, so the sleep does not
            # block the event loop; the deep-solver lane still refuses instantly.
            try:
                reclaim_wait_s = float(
                    os.environ.get("AURA_MLX_SPAWN_RECLAIM_WAIT_S", "15") or 15.0
                )
            except (TypeError, ValueError):
                reclaim_wait_s = 15.0
            reclaim_deadline = time.monotonic() + max(0.0, reclaim_wait_s)
            waited = False
            while memory_block and time.monotonic() < reclaim_deadline:
                if _shutdown_blocks_model_work(self.model_path, action="memory reclaim wait"):
                    raise RuntimeError("runtime_shutdown")
                waited = True
                time.sleep(1.5)
                if _shutdown_blocks_model_work(self.model_path, action="memory reclaim retry"):
                    raise RuntimeError("runtime_shutdown")
                memory_block = _memory_pressure_blocks_worker_spawn(self.model_path)
            if waited and not memory_block:
                logger.info(
                    "🟢 [MLX] Headroom recovered after worker reclaim; proceeding with spawn."
                )
        if memory_block:
            error = RuntimeError(f"memory_pressure_refused_worker_spawn:{memory_block}")
            if self._is_deep_solver_lane():
                logger.warning(
                    "🛡️ [MLX] Refusing optional deep Solver spawn before model load: %s",
                    memory_block,
                )
                raise error
            _record_mlx_degradation(
                error,
                action="refused MLX worker spawn before model load due to memory pressure",
                severity="critical",
            )
            raise error

        runtime_ok, runtime_detail = _probe_mlx_runtime()
        if not runtime_ok:
            raise RuntimeError(f"mlx_runtime_probe_failed:{runtime_detail}")
        if _shutdown_blocks_model_work(self.model_path, action="post-runtime-probe spawn"):
            raise RuntimeError("runtime_shutdown")

        if self._req_q is None or self._res_q is None:
            raise RuntimeError("MLX IPC queues must be created before worker spawn")
        ctx = self._mp_context

        lock_dir = Path.home() / ".aura" / "run"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_file_path = str(lock_dir / "mlx_spawn.lock")
        lock_file = _open_spawn_lock_file(lock_file_path)
        with lock_file:
            try:
                logger.info("🔒 [MLX] Acquiring process-level spawn lock...")
                _acquire_spawn_file_lock(lock_file, model_path=self.model_path)
                if _shutdown_blocks_model_work(self.model_path, action="locked worker spawn"):
                    raise RuntimeError("runtime_shutdown")

                project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                # CP126 841bf5f7. A fresh contract-signing key per spawn: it
                # is handed to the child at fork, never persisted, and is
                # meaningless to any other worker. Privileged output
                # contracts must be signed with it to take effect.
                from core.brain.llm.contract_authority import new_contract_key

                self._contract_key = new_contract_key()
                p = ctx.Process(
                    target=_mlx_worker_loop,
                    args=(
                        self.model_path,
                        self._req_q,
                        self._res_q,
                        self.device,
                        self._substrate_mem,
                        self._steering_active,
                        self._cancel_seq,
                        self._contract_key,
                    ),
                    daemon=True,
                    name=f"MLXWorker-{os.path.basename(self.model_path)}",
                )
                if _shutdown_blocks_model_work(self.model_path, action="worker process start"):
                    raise RuntimeError("runtime_shutdown")
                p.start()
                if _runtime_shutdown_requested():
                    record_shutdown_admission_event(
                        f"mlx:worker_start:{os.path.basename(self.model_path)}",
                        resource_kind="mlx_worker",
                        outcome="crossed",
                        detail=f"pid={p.pid}",
                    )
                    logger.warning(
                        "🛑 [MLX] Shutdown crossed worker start for %s; terminating pid=%s.",
                        os.path.basename(self.model_path),
                        p.pid,
                    )
                    self._kill_and_join_blocking(p)
                    record_shutdown_admission_event(
                        f"mlx:worker_start:{os.path.basename(self.model_path)}",
                        resource_kind="mlx_worker",
                        outcome="reaped" if not p.is_alive() else "survived",
                        detail=f"pid={p.pid}",
                    )
                    raise RuntimeError("runtime_shutdown")
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
        if _shutdown_blocks_model_work(self.model_path, action="async worker spawn"):
            raise RuntimeError("runtime_shutdown")
        return await asyncio.get_running_loop().run_in_executor(None, self._spawn_worker_blocking)

    async def _response_listener_loop(self):
        """
        [v7.8] Background task to constantly drain the worker response queue.
        Prevents IPC deadlocks by ensuring heartbeats and telemetry are ALWAYS consumed.
        """
        import queue

        from core.container import ServiceContainer

        _consecutive_errors = 0
        # Fresh pipe, fresh reporting state: a new worker must not inherit a
        # previous worker's stall/broken-pipe report latches.
        self._worker_loop_stall_reported = False
        self._worker_ipc_broken_reported = False
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
                    # Worker-reported progress evidence: the heartbeat now
                    # carries the inference-loop's own stall verdict, so a
                    # wedged decode loop is visible BEFORE the worker-side
                    # watchdog (360s) or a caller timeout fires. Surface it
                    # once per stall episode; liveness semantics stay as-is.
                    worker_reported_stall = bool(res.get("loop_stalled"))
                    stalled, stall_threshold_s = self._confirm_worker_reported_loop_stall(res)
                    if worker_reported_stall and stalled and not self._worker_loop_stall_reported:
                        self._worker_loop_stall_reported = True
                        _record_mlx_degradation(
                            RuntimeError(
                                "worker_loop_stalled:"
                                f"request={res.get('request_id') or '<unknown>'}:"
                                f"age_s={res.get('job_age_s')}:"
                                f"budget_s={stall_threshold_s:.1f}"
                            ),
                            action="worker heartbeat exceeded the active request's progress budget",
                            severity="error",
                        )
                        self.soft_cancel_active_generation("worker_loop_stalled")
                        if not self._deferred_reboot_reason:
                            self._deferred_reboot_reason = "recoverable_token_progress_stalled"
                    elif not worker_reported_stall or not stalled:
                        self._worker_loop_stall_reported = False
                    if bool(res.get("ipc_broken")) and not self._worker_ipc_broken_reported:
                        self._worker_ipc_broken_reported = True
                        _record_mlx_degradation(
                            RuntimeError("worker_response_pipe_broken"),
                            action="worker heartbeat reports a broken response pipe; expecting worker exit",
                            severity="critical",
                        )
                    owner_id, fencing_token, _receipt_id = (
                        self._durable_model_lane_owner_snapshot()
                    )
                    if fencing_token > 0 and owner_id:
                        try:
                            from core.runtime.model_lane_control import (
                                get_model_lane_controller,
                            )

                            lease_alive = await get_model_lane_controller().heartbeat_owner(
                                owner_id,
                                fencing_token=fencing_token,
                            )
                            if not lease_alive:
                                raise RuntimeError("model_lane_fence_lost")
                        except (
                            OSError,
                            RuntimeError,
                            AttributeError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            _record_mlx_degradation(
                                exc,
                                action="stopped MLX worker after durable lane heartbeat failed",
                                severity="critical",
                            )
                            self._deferred_reboot_reason = "model_lane_fence_lost"
                            process, self._process = self._process, None
                            if process is not None:
                                await asyncio.to_thread(self._kill_and_join_blocking, process)
                            from core.runtime.model_lane_control import (
                                unregister_model_lane_owner_adapter,
                            )

                            unregister_model_lane_owner_adapter(owner_id)
                            with self._model_lane_state_lock:
                                if self._model_lane_fencing_token == fencing_token:
                                    self._model_lane_fencing_token = 0
                                    self._model_lane_terminal_receipt_id = ""
                            # Fail every waiter BEFORE exiting the listener:
                            # killing the worker and returning stranded all
                            # pending generation/init futures until their own
                            # timeouts fired (the waiter fast-path only trips
                            # while _process is non-None).
                            for req_id, pending in list(self._pending_generations.items()):
                                if pending is not None and not pending.done():
                                    _set_shared_future_result(
                                        pending,
                                        {
                                            "status": "error",
                                            "action": "generate",
                                            "id": str(req_id),
                                            "message": "model_lane_fence_lost",
                                        },
                                    )
                            current_fut = self._current_gen_future
                            if current_fut is not None and not current_fut.done():
                                _set_shared_future_result(
                                    current_fut,
                                    {
                                        "status": "error",
                                        "action": "generate",
                                        "id": self._current_request_id,
                                        "message": "model_lane_fence_lost",
                                    },
                                )
                            if self._init_future is not None and not self._init_future.done():
                                _cancel_shared_future(self._init_future)
                            self._set_lane_state("cold", "model_lane_fence_lost")
                            return
                    audit = ServiceContainer.get("subsystem_audit", default=None)
                    if audit:
                        tier_name = (
                            "mlx_heavy" if _model_is_heavy_lane(self.model_path)
                            else "mlx_light"
                        )
                        audit.heartbeat(tier_name)
                    continue
                if status in {"progress", "token"}:
                    if action == "latent_reason" and isinstance(res, dict):
                        self._record_latent_progress(res)
                    self._mark_token_progress(res.get("id"))
                    live_intero = res.get("interoception_live")
                    if isinstance(live_intero, dict) and live_intero:
                        try:
                            from core.being.thought_interoception import (
                                get_thought_interoception,
                            )

                            get_thought_interoception().pulse_live(live_intero)
                        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
                            logger.debug("Live interoception pulse dropped.")
                    continue

                # 2. Route init/generation responses to the correct awaiting future
                if action == "init":
                    if self._init_future and not self._init_future.done():
                        self._mark_progress()
                        _set_shared_future_result(self._init_future, res)
                        continue
                elif action in (
                    "generate",
                    "generate_batch",
                    "stream_done",
                    "set_expert_adapter",
                    "nonparametric_ingest",
                    "latent_reason",
                ):
                    future = self._pending_generations.pop(req_id, None) if req_id else None
                    if future and not future.done():
                        self._mark_progress()
                        _set_shared_future_result(future, res)
                        continue
                    # A generation can finish after the caller has already
                    # abandoned it and started another turn. Never hand a
                    # response with an old request id to the current future.
                    #
                    # CP126 49d694a1: the id-less fallback (`not req_id or ...`)
                    # let a stale or malformed terminal frame COMPLETE the
                    # current request with another turn's content. The worker
                    # stamps every response with its job id, so an id-less
                    # terminal frame is malformed by construction — the error
                    # route already rejects it, and so does this one now.
                    if (
                        self._current_gen_future
                        and not self._current_gen_future.done()
                        and req_id
                        and req_id == self._current_request_id
                    ):
                        self._mark_progress()
                        _set_shared_future_result(self._current_gen_future, res)
                        continue
                    if not req_id:
                        _record_mlx_degradation(
                            RuntimeError(
                                f"uncorrelated_worker_response:{action or 'unknown'}"
                            ),
                            action=(
                                "dropped an id-less worker response instead of "
                                "completing the active request with it"
                            ),
                            severity="warning",
                        )
                        continue
                elif status == "degraded":
                    # Worker self-reported health frames (e.g. the memory
                    # sentinel going blind/recovering). Not terminal, never
                    # correlated to a request — surface them as degradations
                    # instead of ignoring them.
                    _record_mlx_degradation(
                        RuntimeError(
                            f"worker_degraded:{action or 'unknown'}:{res.get('message', '')}"
                        ),
                        action=f"worker self-reported degradation ({action or 'unknown'})",
                        severity=(
                            "critical"
                            if action == "memory_sentinel_degraded"
                            else "warning"
                        ),
                    )
                    logger.warning(
                        "⚠️ [MLX] Worker degradation frame (%s): %s",
                        action,
                        res.get("message"),
                    )
                    continue
                elif status == "error" and action == "memory_fuse":
                    # The worker's memory sentinel is about to hard-exit the
                    # process. This frame is intentionally id-less (it is not
                    # a request result) — attribute the imminent death to
                    # every in-flight request NOW instead of letting each one
                    # discover it via timeout against a dead process.
                    fuse_message = str(res.get("message") or "worker_memory_fuse")
                    _record_mlx_degradation(
                        RuntimeError(f"worker_memory_fuse:{fuse_message}"),
                        action="worker memory fuse tripped; failing in-flight requests with attribution",
                        severity="critical",
                    )
                    logger.critical("🛑 [MLX] %s", fuse_message)
                    for pending_id, pending in list(self._pending_generations.items()):
                        if pending is not None and not pending.done():
                            _set_shared_future_result(
                                pending,
                                {
                                    "status": "error",
                                    "action": "generate",
                                    "id": str(pending_id),
                                    "message": f"worker_memory_fuse:{fuse_message}",
                                    "memory_pressure": res.get("memory_pressure") or {},
                                },
                            )
                    self._pending_generations.clear()
                    current_fut = self._current_gen_future
                    if current_fut is not None and not current_fut.done():
                        _set_shared_future_result(
                            current_fut,
                            {
                                "status": "error",
                                "action": "generate",
                                "id": self._current_request_id,
                                "message": f"worker_memory_fuse:{fuse_message}",
                                "memory_pressure": res.get("memory_pressure") or {},
                            },
                        )
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
                        and req_id
                        and req_id == self._current_request_id
                    ):
                        self._mark_progress()
                        _set_shared_future_result(self._current_gen_future, res)
                        continue
                    if not req_id and status in {"ok", "error"}:
                        # The worker stamps every response with its job id —
                        # an id-less terminal message is stale or malformed
                        # and must NOT complete the current request with
                        # someone else's content or error state.
                        _record_mlx_degradation(
                            RuntimeError("id_less_worker_response_dropped"),
                            action="dropped id-less terminal worker message instead of completing current request",
                        )
                        continue

                # 3. Log errors if no future is waiting
                if status == "error":
                    logger.error("🛑 [MLX] Async worker error: %s", res.get("message"))

            except Exception as e:  # noqa: BLE001 - the listener is the sole response
                # consumer: ANY escaping processing error (TypeError, ValueError,
                # OSError, future completion races, callback failures) would leave
                # a live worker with no one draining its queue.
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
        if _shutdown_blocks_model_work(self.model_path, action="worker start/recovery"):
            return False
        if (
            request_is_background
            and _foreground_owner_active()
            and not self._is_primary_lane()
        ):
            # Same inversion as the warmup guard (2026-07-10): the Reflex
            # fallback serving turns OWNED the foreground, which deferred
            # cortex recovery here — the primary could never come back while
            # its own fallback was answering for it. The primary lane's
            # recovery is exempt; other background lanes still yield.
            logger.info(
                "⏸️ [MLX] Deferring background worker activity for %s while foreground lane is owned by %s.",
                os.path.basename(self.model_path),
                _FOREGROUND_OWNER_NAME or "foreground",
            )
            return False
        if request_is_background and not self._is_primary_lane():
            # Every reason the gate's quiet policy returns (foreground_
            # reserved, headroom, cortex_startup_quiet, quiet window)
            # protects the user's turn from BACKGROUND COMPETITION. The
            # primary lane's own revival is not competition — it is the
            # thing the user's turn is waiting for, so it is exempt here
            # exactly as at the owner guard above.
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
            self._clear_model_load_admission_backoff()
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

        # Slow path: admission owns whether model loading may proceed; the
        # spawn gate remains the mechanical single-spawn mutex beneath it.
        if request_is_background and self._model_load_admission_backoff_active():
            return False
        if int(self._model_lane_fencing_token or 0) > 0:
            try:
                await self._release_durable_model_lane_owner(
                    reason="dead_worker_before_respawn",
                )
            except (
                ImportError,
                OSError,
                RuntimeError,
                AttributeError,
                TypeError,
                ValueError,
            ) as exc:
                self._set_lane_state("recovering", "stale_model_lane_owner_release_failed")
                _record_mlx_degradation(
                    exc,
                    action=(
                        "refused worker respawn until the dead worker's durable "
                        "model-lane owner can be released"
                    ),
                    severity="critical",
                )
                return False
        try:
            async with _model_load_admission_context(
                self,
                foreground_request=foreground_request,
            ):
                async with _spawn_gate_context(
                    owner=f"{os.path.basename(self.model_path)}:"
                    f"{'foreground' if foreground_request else 'background'}"
                ):
                    return await self._ensure_worker_alive_inner(
                        request_is_background=request_is_background,
                        foreground_request=foreground_request,
                        init_timeout=init_timeout,
                        soft_timeout=soft_timeout,
                        skip_swap_cooldown=skip_swap_cooldown,
                    )
        except _ModelLoadAdmissionDeniedError as admission_exc:
            # The inner spawn path can establish a specific terminal failure
            # (for example, a failed Metal runtime probe) before the durable
            # transaction observes that no candidate reached READY.  Preserve
            # that causal state instead of replacing it with the less useful
            # outer transaction consequence.
            if self._lane_state != "failed" or not str(self._lane_error or ""):
                self._set_lane_state("recovering", admission_exc.reason)
            backoff_s = self._note_model_load_admission_denial(
                admission_exc.reason,
                receipt_id=admission_exc.receipt_id,
            )
            if foreground_request:
                self._record_degraded_event(
                    "model_load_admission_denied",
                    detail=(
                        f"{os.path.basename(self.model_path)}:{admission_exc.reason}:"
                        f"receipt={admission_exc.receipt_id or 'none'}"
                    ),
                    severity="warning",
                    foreground_request=True,
                )
            admission_logger = logger.warning if foreground_request else logger.info
            admission_logger(
                "⏸️ [MLX] Model-load admission deferred for %s: %s "
                "(receipt=%s, recheck_in=%.1fs)",
                os.path.basename(self.model_path),
                admission_exc.reason,
                admission_exc.receipt_id or "none",
                backoff_s,
            )
            return False
        except TimeoutError as gate_exc:
            # Another lane's spawn is wedged holding the global gate. Defer
            # honestly instead of joining the pileup — the warmup's finally
            # still clears its flag, admission stays unblocked, and the
            # watchdog handles the wedged holder.
            self._set_lane_state("recovering", "spawn_gate_timeout")
            self._record_degraded_event(
                "spawn_gate_timeout",
                detail=f"{os.path.basename(self.model_path)}:{gate_exc}",
                severity="warning",
                foreground_request=foreground_request,
            )
            logger.warning(
                "⏸️ [MLX] Spawn gate held too long by another lane; deferring %s spawn (%s).",
                os.path.basename(self.model_path), gate_exc,
            )
            return False

    async def _ensure_worker_alive_inner(
        self,
        *,
        request_is_background: bool = False,
        foreground_request: bool = False,
        init_timeout: float | None = None,
        soft_timeout: bool = False,
        skip_swap_cooldown: bool = False,
        _init_retry: bool = False,
    ) -> bool:
        """Inner implementation — called while holding the global spawn gate."""
        if _shutdown_blocks_model_work(self.model_path, action="worker spawn"):
            return False
        # K4 crash-loop backoff: a lane whose workers keep dying young must
        # not respawn on demand. Refuse fast with a named reason — the
        # escalation ladder answers while the backoff drains. A healthy
        # worker passing through is never disturbed.
        if not (self._process and self._process.is_alive() and self._init_done):
            crash_blocked = _crash_loop_blocks_worker_spawn(self)
            if crash_blocked:
                self._set_lane_state("recovering", crash_blocked)
                self._record_degraded_event(
                    "crash_loop_backoff",
                    detail=f"{os.path.basename(self.model_path)}:{crash_blocked}",
                    severity="warning",
                    foreground_request=foreground_request,
                )
                logger.warning(
                    "⛔ [MLX] Respawn refused for %s: %s",
                    os.path.basename(self.model_path),
                    crash_blocked,
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

        if (
            request_is_background
            and _foreground_owner_active()
            and not self._is_primary_lane()
        ):
            # Primary-lane exemption (2026-07-10 inversion family): the
            # reconciler's prewarm arrives here as background work; blocking
            # it while the Reflex fallback owns the foreground kept the
            # cortex dead exactly while users waited on it.
            logger.info(
                "⏸️ [MLX] Background spawn blocked for %s while foreground lane is reserved.",
                os.path.basename(self.model_path),
            )
            return False
        if request_is_background and not self._is_primary_lane():
            background_deferral = _background_deferral_active(os.path.basename(self.model_path))
            if background_deferral:
                logger.info(
                    "⏸️ [MLX] Background spawn blocked for %s (%s).",
                    os.path.basename(self.model_path),
                    background_deferral,
                )
                return False

        if target_path in (primary_path, deep_path):
            # Required yields are owned by the durable lane transaction before
            # this inner spawn path is entered.  This block now owns only the
            # anti-thrash cooldown; it must never evict outside the reservation
            # fence or claim admission before process death is verified.
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
                    except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                        detail = str(exc)
                        if self._handle_optional_deep_solver_memory_refusal(detail):
                            return False
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
                if self._process is not None:
                    # The worker died on its own (OS OOM kill, segfault): no
                    # kill path saw it, so account for it here — then drop
                    # the dead handle so the death is counted exactly once.
                    _note_lane_worker_death(self, "process_died_unexpectedly")
                    self._process = None
                    self._process_started_at = 0.0
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
                except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                    # OSError included: queue creation, lock files, and
                    # multiprocessing start raise it — it previously escaped
                    # with the lane stuck in "spawning" and a pending init.
                    detail = str(exc)
                    if self._handle_optional_deep_solver_memory_refusal(detail):
                        return False
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
                self._worker_identity = {}
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
                        # READINESS IS EARNED, NOT ANNOUNCED. CP126 34c42774:
                        # any dict with status=ok used to set init_done,
                        # heartbeats and lane=ready, and only THEN copy the
                        # recurrence receipt and worker identity. A worker
                        # that never reported recurrence, or reported a
                        # malformed identity, was already serving by the time
                        # anyone looked. The invariants are checked first and
                        # the handshake fails if they do not hold — which
                        # feeds the existing one-shot retry.
                        readiness_errors = self._init_receipt_errors(res)
                        if readiness_errors:
                            _record_mlx_degradation(
                                ValueError(
                                    "init_receipt_invalid:" + ",".join(readiness_errors)
                                ),
                                action="refused READY on an unvalidated worker init receipt",
                                severity="error",
                            )
                            self._init_done = False
                            self._worker_identity = {}
                            self._recurrent_depth_status = {}
                            self._set_lane_state(
                                "failed", "init_receipt_invalid",
                            )
                            if handshake_attempt == 0:
                                continue
                            return False
                        self._init_done = True
                        self._last_heartbeat = time.time()
                        self._last_ready_at = self._last_heartbeat
                        self._mark_progress()
                        self._set_lane_state("ready")
                        recurrent_status = res.get("recurrent_depth")
                        # Always REPLACE: preserving the previous worker's
                        # status when the new receipt is absent/malformed let
                        # an old active=true certify a new worker that never
                        # reported recurrence.
                        self._recurrent_depth_status = (
                            recurrent_status if isinstance(recurrent_status, dict) else {}
                        )
                        if not isinstance(recurrent_status, dict):
                            _record_mlx_degradation(
                                ValueError("missing_recurrent_depth_receipt"),
                                action="cleared stale recurrence status after init receipt omitted it",
                            )
                        worker_identity = res.get("worker_identity")
                        self._worker_identity = (
                            dict(worker_identity)
                            if isinstance(worker_identity, dict)
                            else {}
                        )
                        raw_steering = res.get("steering_active")
                        if raw_steering is not None:
                            try:
                                if isinstance(raw_steering, bool):
                                    steering_active = raw_steering
                                else:
                                    # A malformed string "false" is truthy —
                                    # never bool() an untyped IPC value into
                                    # the shared steering channels.
                                    _record_mlx_degradation(
                                        TypeError(
                                            f"non-bool steering receipt: {raw_steering!r}"
                                        ),
                                        action="treated malformed steering receipt as inactive",
                                    )
                                    steering_active = False
                                self._steering_active.value = steering_active
                                self._substrate_mem[-1] = 1.0 if steering_active else 0.0
                                self._steering_liveness_observed = True
                            except (TypeError, ValueError, IndexError, AttributeError) as steering_receipt_exc:
                                _record_mlx_degradation(
                                    steering_receipt_exc,
                                    action="kept worker ready after steering liveness receipt write failed",
                                    severity="warning",
                                )
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
                                # Reboot tears the lifecycle down WITHOUT
                                # spawning a replacement, so the falsy check
                                # silently skipped the advertised one-shot
                                # retry. Re-enter the spawn path once (the
                                # spawn gate is already held by our caller).
                                if not _init_retry:
                                    return await self._ensure_worker_alive_inner(
                                        request_is_background=request_is_background,
                                        foreground_request=foreground_request,
                                        init_timeout=init_timeout,
                                        soft_timeout=soft_timeout,
                                        skip_swap_cooldown=True,
                                        _init_retry=True,
                                    )
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
                            # CP126 0c91528f (timeout half). reboot_worker is a
                            # TEARDOWN: it clears _init_future and does NOT
                            # spawn a replacement, so this falsy check used to
                            # `break` and silently skip the advertised one-shot
                            # retry — the same defect already closed on the
                            # init-error branch above. Re-enter the spawn
                            # transaction so the retry actually happens.
                            if not _init_retry:
                                return await self._ensure_worker_alive_inner(
                                    request_is_background=request_is_background,
                                    foreground_request=foreground_request,
                                    init_timeout=init_timeout,
                                    soft_timeout=soft_timeout,
                                    skip_swap_cooldown=True,
                                    _init_retry=True,
                                )
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
        # Finite-bounded: a malformed value previously RAISED through the
        # generation wait path, and infinity disabled the hard cap entirely.
        hard_cap = max(
            30.0,
            _finite_env_float("AURA_MLX_GENERATION_HARD_CAP_SECONDS", 240.0, minimum=30.0),
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

                # OBSERVATION and ENFORCEMENT are separated. They used to share
                # one try block, so a failure while ABORTING (queue cleanup,
                # future cancellation) was reported as "probe unavailable" and
                # the loop kept waiting with lifecycle state half-cleared —
                # the request neither aborted nor honestly failed.
                memory_snapshot = None
                try:
                    memory_snapshot = get_memory_pressure_snapshot()
                    if memory_snapshot.should_gc:
                        gc.collect()
                except (OSError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    # Unobserved pressure is not observed headroom. Heavy lanes
                    # are the allocation that pushes this host over, so a blind
                    # probe is recorded rather than shrugged off at debug.
                    if self._is_primary_or_deep_lane():
                        _record_mlx_degradation(
                            exc,
                            action=(
                                "live memory-pressure probe unavailable during heavy "
                                "generation; abort decision could not be made"
                            ),
                            severity="warning",
                        )
                    else:
                        logger.debug("MLX live memory pressure probe unavailable: %s", exc)

                if (
                    memory_snapshot is not None
                    and memory_snapshot.refuse_heavy_local_generation
                    and self._is_primary_or_deep_lane()
                ):
                    from core.brain.llm.emergency_override import consume_override

                    live_override = consume_override(
                        "AURA_MLX_ALLOW_CRITICAL_MEMORY_GENERATION",
                        guard="live_memory_pressure_abort",
                        observed=(
                            f"{os.path.basename(self.model_path)}:{memory_snapshot.reason}"
                        ),
                    )
                    if not live_override.active:
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
                        try:
                            self.force_abort_active_generation(
                                "memory_pressure_during_generation"
                            )
                            _cancel_shared_future(future)
                        except (
                            OSError, AttributeError, RuntimeError, TypeError, ValueError,
                        ) as abort_exc:
                            # The abort itself failed. Critical pressure WAS
                            # observed and cleanup cannot be proven, so the
                            # request ends terminally with that on the record
                            # instead of quietly resuming the wait.
                            _record_mlx_degradation(
                                abort_exc,
                                action=(
                                    "memory-pressure abort failed; generation state "
                                    "could not be proven clean"
                                ),
                                severity="critical",
                            )
                            self._record_degraded_event(
                                "generation_abort_failed_memory_pressure",
                                detail=(
                                    f"{os.path.basename(self.model_path)}:"
                                    f"{type(abort_exc).__name__}"
                                ),
                                severity="critical",
                                foreground_request=foreground_request,
                            )
                        return None

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
                        detail=(
                            f"{os.path.basename(self.model_path)}>{first_token_sla:.1f}s"
                            f"{self._pressure_receipt_suffix()}"
                        ),
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
                    # Ask the worker to drop the orphaned generation between
                    # tokens — the abandoned output then never arrives at all,
                    # instead of relying solely on a worker recycle.
                    self.soft_cancel_active_generation("abandoned_first_token_sla")
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
                        detail=(
                            f"{os.path.basename(self.model_path)}>{token_stall_after:.1f}s"
                            f"{self._pressure_receipt_suffix()}"
                        ),
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
                    self.soft_cancel_active_generation("abandoned_token_stall")
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
                    self.soft_cancel_active_generation("abandoned_heartbeat_stall")
                    _cancel_shared_future(future)
                    return None
        raise TimeoutError

    async def generate_text_async(self, prompt: str, **kwargs) -> str | None:
        """Alias for standard interface."""
        messages = kwargs.pop("messages", None)
        system_prompt = kwargs.pop("system_prompt", None)
        tools = kwargs.pop("tools", None)
        if isinstance(messages, list) and messages:
            # Harden the public boundary: a malformed (non-mapping) element
            # previously raised AttributeError below, outside the normal
            # generation failure contract.
            well_formed = [m for m in messages if isinstance(m, dict)]
            if len(well_formed) != len(messages):
                _record_mlx_degradation(
                    TypeError("non-mapping chat message dropped"),
                    action="dropped malformed message entries before flattening",
                )
            messages = well_formed or None
        if messages and system_prompt:
            # A separate system prompt alongside conversation history was
            # silently DISCARDED (the messages branch below skips it) —
            # callers lost policy/schema/safety instructions. Merge it.
            merged = [dict(m) for m in messages]
            if merged[0].get("role") == "system":
                existing = str(merged[0].get("content", "") or "")
                if str(system_prompt) not in existing:
                    merged[0]["content"] = f"{system_prompt}\n\n{existing}".strip()
            else:
                merged.insert(0, {"role": "system", "content": str(system_prompt)})
            messages = merged
        foreground_request = bool(kwargs.get("foreground_request", False))
        strict_answer_contract = bool(kwargs.get("strict_answer_contract", False))
        proof_evaluation_contract = bool(kwargs.get("proof_evaluation_contract", False))
        web_interlocutor_contract = bool(kwargs.get("web_interlocutor_contract", False))

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
            and not web_interlocutor_contract
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
    def _extract_tool_call_payload(
        response_text: str,
        *,
        allowed_tools: set[str] | None = None,
    ) -> dict[str, Any] | None:
        """Extract a tool call ONLY when the model actually intended one.

        CP126 5a924075 + 0da5db2e. This used to scan anywhere in model prose
        and accept any fenced JSON object with name/arguments — so a
        quotation, a worked example, or user-supplied text that the model
        merely repeated became an EFFECT REQUEST in the agent loop. It also
        returned arbitrary tool names, accepted non-dict args, and invented a
        ``{"value": ...}`` wrapper for strings that failed to parse.

        Now:
          * ``<tool_call>`` (the model's native structured channel) is trusted
            anywhere, because emitting it IS the tool intent.
          * A bare JSON object (fenced or not) counts only as a WHOLE-RESPONSE
            envelope — i.e. it is the entire reply, not a snippet embedded in
            prose that discusses it.
          * The tool name must be in this turn's advertised allowlist.
          * Arguments must be a real JSON object; nothing is invented.
        """
        if not response_text:
            return None

        stripped = response_text.strip()

        def _normalize(payload: Any) -> dict[str, Any] | None:
            if not isinstance(payload, dict):
                return None
            if "tool" in payload and "args" in payload:
                name, args = payload.get("tool"), payload.get("args")
            elif "name" in payload and "arguments" in payload:
                name, args = payload.get("name"), payload.get("arguments")
            else:
                return None
            if not isinstance(name, str) or not name.strip():
                return None
            name = name.strip()
            if allowed_tools is not None and name not in allowed_tools:
                _record_mlx_degradation(
                    PermissionError(f"tool_not_advertised:{name[:64]}"),
                    action="refused a parsed tool call naming a tool not offered this turn",
                    severity="warning",
                )
                return None
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError, ValueError):
                    # Never INVENT an argument shape for text that failed to
                    # parse — an unparseable argument string is not a call.
                    return None
            if args is None:
                args = {}
            if not isinstance(args, dict):
                return None
            return {"tool": name, "args": args}

        # 1. Native structured channel — an explicit tool-intent envelope.
        native = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", stripped, re.DOTALL)
        if native:
            try:
                call = _normalize(json.loads(native.group(1)))
            except json.JSONDecodeError:
                call = None
            if call is not None:
                return call

        # 2. Whole-response JSON envelope only. A fenced block must BE the
        #    response; prose wrapped around it means the model was talking
        #    about a call, not making one.
        candidate: str | None = None
        fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
        if fenced:
            candidate = fenced.group(1)
        elif stripped.startswith("{") and stripped.endswith("}"):
            candidate = stripped
        if candidate is None:
            return None
        try:
            return _normalize(json.loads(candidate))
        except json.JSONDecodeError:
            return None

    def _check_steering_liveness(self) -> bool | None:
        """Return steering liveness once the worker has reported it.

        ``False`` is a real fault signal after the first worker receipt. Before
        that receipt, treating the default shared-memory zero as "inactive"
        creates a misleading neural-stream warning during first foreground
        generations and web-interlocutor bootstraps.
        """
        if not bool(getattr(self, "_steering_liveness_observed", False)):
            try:
                sm = getattr(self, "_substrate_mem", None)
                if sm is not None and float(sm[-1]) > 0.5:
                    self._steering_liveness_observed = True
                    return True
            except (TypeError, ValueError, IndexError, OSError):
                pass
            return None
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
        if active is None:
            logger.debug(
                "⏳ [STEERING] Liveness pending first worker receipt (origin=%s)",
                origin,
            )
        elif active:
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
        self._set_task_surface_control_receipt({})
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
        # SCHEDULING classification may be inferred from labels — treating a
        # baseline run as foreground is harmless. SAFETY exemption may not:
        # benchmark_request also waives the critical memory-pressure refusal,
        # and inferring it from any free-form purpose containing "_baseline"
        # let any caller self-authorize that waiver. The explicit kwarg is the
        # only thing that can lift a safety guard.
        benchmark_request_explicit = bool(kwargs.get("benchmark_request", False))
        benchmark_request = benchmark_request_explicit or (
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
            # Consume the override only when the refusal it would bypass is
            # actually about to fire; a guard that never triggers must not
            # spend the emergency budget.
            override_applies = (
                memory_snapshot.refuse_heavy_local_generation
                and self._is_primary_or_deep_lane()
                and not benchmark_request_explicit
            )
            override_decision = None
            if override_applies:
                from core.brain.llm.emergency_override import consume_override

                override_decision = consume_override(
                    "AURA_MLX_ALLOW_CRITICAL_MEMORY_GENERATION",
                    guard="critical_memory_generation_refusal",
                    observed=(
                        f"{os.path.basename(self.model_path)}:{memory_snapshot.reason}"
                    ),
                )
            critical_override = bool(
                override_decision is not None and override_decision.active
            )
            if override_applies and critical_override:
                # The override disables a refusal made AFTER critical pressure
                # was positively observed, i.e. the last guard before the model
                # process can push macOS into swap or jetsam. It stays
                # available for recovery, but a stale deployment flag must not
                # be able to do this silently — the bypass is now as loud as
                # the refusal it replaces.
                self._record_degraded_event(
                    "memory_pressure_generation_override",
                    detail=(
                        f"{os.path.basename(self.model_path)}:{memory_snapshot.reason}:"
                        f"{override_decision.as_detail()}"
                    ),
                    severity="critical",
                    foreground_request=foreground_request,
                )
                logger.critical(
                    "[MLX] Proceeding with heavy local generation for %s DESPITE critical "
                    "memory pressure (%s) because AURA_MLX_ALLOW_CRITICAL_MEMORY_GENERATION "
                    "is set. This bypasses the last guard before swap/jetsam.",
                    os.path.basename(self.model_path),
                    memory_snapshot.reason,
                )
            if override_applies and not critical_override:
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
        except (OSError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            # The guard exists because critical pressure can trigger swap or
            # jetsam before a single token is produced. A probe that cannot
            # answer is NOT evidence of headroom, and logging at debug made an
            # unobservable memory state indistinguishable from a healthy one.
            # Heavy primary/deep generation is refused while blind; smaller
            # lanes continue, since they are not the allocation that pushes
            # the host over.
            _record_mlx_degradation(
                exc,
                action="refused heavy local generation because memory pressure could not be observed",
                severity="critical",
            )
            if self._is_primary_or_deep_lane() and not benchmark_request_explicit:
                self._record_degraded_event(
                    "memory_pressure_unobservable_refused_generation",
                    detail=f"{os.path.basename(self.model_path)}:{type(exc).__name__}",
                    severity="critical",
                    foreground_request=foreground_request,
                )
                logger.warning(
                    "[MLX] Refusing heavy local generation for %s: memory pressure probe "
                    "unavailable (%s), so headroom cannot be established.",
                    os.path.basename(self.model_path),
                    exc,
                )
                return None
            logger.debug("MLX memory pressure probe unavailable: %s", exc)

        # ── SOMATIC COUPLING: Metabolic hardware throttle ────────────
        #
        # CP126 4e95a54c. A failed throttle check was recorded and then
        # generation proceeded with UNTHROTTLED parameters — the body-pressure
        # control vanished exactly when its state could not be established.
        # On a host that has been driven into swap and jetsam before, the
        # unthrottled path is the expensive one.
        #
        # A failure now applies a conservative floor instead of nothing: the
        # generation still runs (this is a throttle, not an admission gate,
        # and refusing here would take conversation down for a metabolic
        # hiccup) but it runs damped rather than wide open.
        try:
            from core.brain.llm.somatic_throttle import SomaticComputeSentinel
            sentinel = SomaticComputeSentinel()
            kwargs = sentinel.adjust_generation_options(kwargs)
        except _MLX_OPTIONAL_THROTTLE_ERRORS as exc:
            kwargs = _apply_unthrottled_fallback_ceiling(kwargs)
            _record_mlx_degradation(
                exc,
                action=(
                    "somatic throttle unavailable; applied a conservative "
                    "generation ceiling instead of running unthrottled"
                ),
                severity="warning",
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

        try:
            acquired = await self._acquire_request_lock(
                owner_label=owner_label,
                deadline=deadline,
                foreground_request=foreground_request,
            )
        except BaseException:
            # Cancellation or an unexpected acquisition failure must not
            # leave the global foreground owner entered — that blocked every
            # background lane until a stale-clear heuristic fired.
            if foreground_owner_cm is not None:
                with contextlib.suppress(Exception):
                    await asyncio.shield(foreground_owner_cm.__aexit__(*sys.exc_info()))
            raise
        if not acquired:
            # A request that did NOT acquire the lane must not consume the
            # shared deferred-reboot verdict — that stole reboots requested
            # by the actual lane owner (the owning request's cleanup below
            # resolves it).
            if foreground_owner_cm is not None:
                await foreground_owner_cm.__aexit__(None, None, None)
            return None
        try:
            # Check steering liveness
            if not request_is_background:
                self._emit_steering_status(origin_label)

            # Reliability tracing: inference nests under the HTTP root span
            # (contextvars), so a slow turn reads as one connected trace.
            try:
                from core.observability.tracing import get_tracer
                _span_cm = get_tracer().span(
                    "inference.generate",
                    attributes={
                        "model": os.path.basename(self.model_path),
                        "origin": origin_label,
                        "purpose": purpose_label,
                        "background": request_is_background,
                    },
                )
            except (ImportError, AttributeError, RuntimeError):
                _span_cm = contextlib.nullcontext(None)
            with _span_cm as _span:
                result = await self._generate_inner(
                    prompt,
                    _retry=True,
                    request_is_background=request_is_background,
                    foreground_request=foreground_request,
                    owner_label=owner_label,
                    **kwargs,
                )
                if _span is not None:
                    _span.set_attribute("result_chars", len(result) if result else 0)
                if not result:
                    self._set_task_surface_control_receipt({})
                return result
        finally:
            _deferred_reboot = self._deferred_reboot_reason
            self._deferred_reboot_reason = None
            self._release_request_lock()
            # Each cleanup step is independently protected: an owner-exit
            # failure previously REPLACED the generation's own exception with
            # lifecycle noise and skipped deferred-reboot resolution.
            if foreground_owner_cm is not None:
                try:
                    await foreground_owner_cm.__aexit__(None, None, None)
                except Exception as _owner_exit_exc:  # noqa: BLE001
                    _record_mlx_degradation(
                        _owner_exit_exc,
                        action="continued generation cleanup after foreground owner exit failed",
                        severity="error",
                    )
            # Resolve AFTER releasing _request_lock to avoid lock-ordering deadlock
            if _deferred_reboot:
                try:
                    await self._resolve_deferred_reboot(str(_deferred_reboot))
                except Exception as _reboot_exc:  # noqa: BLE001
                    _record_mlx_degradation(
                        _reboot_exc,
                        action="failed to resolve deferred reboot during generation cleanup",
                        severity="error",
                    )

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
                if not self._can_run_resident_background_health_probe(
                    background_deferral,
                    health_probe=bool(kwargs.get("health_probe")),
                ):
                    logger.info(
                        "⏸️ [MLX] Background generation for %s stopped before worker spawn (%s).",
                        os.path.basename(self.model_path),
                        background_deferral,
                    )
                    return None
                logger.info(
                    "🩺 [MLX] Running bounded readiness probe on resident primary worker "
                    "despite background headroom reservation."
                )

        deadline = kwargs.get("deadline")
        if not isinstance(deadline, Deadline):
            timeout_s = _coerce_timeout_seconds(kwargs.pop("timeout", None))
            # CP126 24aaa654: a request budget inferred from path substrings
            # gave renamed or aliased resident checkpoints a 60s deadline
            # meant for small models, and handed unrelated paths containing
            # "32b" an inflated one. Measured artifact evidence decides.
            is_heavy = _model_is_heavy_lane(self.model_path)
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
            or kwargs.get("web_interlocutor_contract", False)
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

        requested_output_contract = kwargs.get("requested_output_contract")
        if not isinstance(requested_output_contract, dict):
            requested_output_contract = {}
        hard_output_token_ceiling = kwargs.get("hard_output_token_ceiling")
        adaptive_suggested_max_tokens = _bridge_get("max_tokens", self.max_tokens)
        contract_generation_floor = _requested_output_contract_generation_floor(
            requested_output_contract
        )
        generation_max_tokens = _bounded_generation_max_tokens(
            kwargs.get("max_tokens", self.max_tokens),
            adaptive_suggested_max_tokens,
            hard_output_token_ceiling,
            self.max_tokens,
            requested_output_contract,
        )

        req_id = uuid.uuid4().hex
        self._job_seq_counter += 1
        req = {
            "id": req_id,
            "seq": self._job_seq_counter,
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
            # max_tokens is a cap: both the latent bridge and the typed visible
            # output contract may shrink it, but neither can expand the caller.
            "max_tokens": generation_max_tokens,
            "caller_requested_max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "adaptive_suggested_max_tokens": adaptive_suggested_max_tokens,
            "output_contract_generation_floor": contract_generation_floor,
            "requested_output_contract": dict(requested_output_contract),
            "semantic_output_token_cap": kwargs.get("semantic_output_token_cap"),
            "hard_output_token_ceiling": hard_output_token_ceiling,
            "schema": kwargs.get("schema"),
            "stop_sequences": list(kwargs.get("stop_sequences") or []),
            "strict_answer_contract": bool(kwargs.get("strict_answer_contract", False)),
            "strict_value_contract": bool(kwargs.get("strict_value_contract", False)),
            "expected_strict_value": str(kwargs.get("expected_strict_value") or ""),
            "proof_evaluation_contract": bool(kwargs.get("proof_evaluation_contract", False)),
            "operator_evidence_contract": bool(kwargs.get("operator_evidence_contract", False)),
            "web_interlocutor_contract": bool(kwargs.get("web_interlocutor_contract", False)),
            "health_probe": bool(kwargs.get("health_probe", False)),
            "warmup_precompile": bool(kwargs.get("warmup_precompile", False)),
            "runtime_fact_status_contract": bool(
                kwargs.get("runtime_fact_status_contract", False)
            ),
            "requires_memory_grounding": bool(
                kwargs.get("requires_memory_grounding", False)
            ),
            "memory_state_contract": bool(
                kwargs.get("memory_state_contract", False)
            ),
            "grounded_recall_contract": bool(
                kwargs.get("grounded_recall_contract", False)
            ),
            "grounded_runtime_status_contract": bool(
                kwargs.get("grounded_runtime_status_contract", False)
            ),
            "clean_user_surface_contract": bool(
                kwargs.get("clean_user_surface_contract", False)
                or kwargs.get("health_probe", False)
            )
            and not bool(kwargs.get("web_interlocutor_contract", False)),
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
            # EXTEND the request's stop list — rebuilding it from the caller
            # kwargs erased every mandatory anti-bleed default appended above.
            for stop in _bridge.extra_stop_sequences:
                if stop not in req["stop_sequences"]:
                    req["stop_sequences"].append(stop)

        if self._active_generations <= 0 and not await self._set_durable_lane_preemptible(
            False
        ):
            logger.info(
                "MLX generation yielded because durable lane ownership is being evicted: %s",
                os.path.basename(self.model_path),
            )
            return None

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
            request_seq=int(req.get("seq", 0)),
        )
        foreground_watchdog = self._start_foreground_first_token_watchdog(
            req_id,
            foreground_request=foreground_request,
            hard_ceiling_s=first_token_hard_ceiling,
        )
        # Ship the caller's production deadline to the worker so its decode
        # loop can stop cooperatively instead of burning GPU past the point
        # anyone is waiting (the worker previously had NO request deadline —
        # only the 360s hard watchdog).
        try:
            _remaining_s = float(deadline.remaining or 0.0)
            if _remaining_s > 0.0:
                req["deadline_unix"] = time.time() + _remaining_s
        except (AttributeError, TypeError, ValueError):
            logger.debug("Request deadline unavailable; worker decodes unbounded.")
        # CP126 a838a49b: this used to read
        # `max(0.5, min(2.0, deadline.remaining or 2.0))`, which turned an
        # ALREADY-EXPIRED budget (remaining == 0.0, falsy) into a 2-second
        # wait and floored every sub-half-second remainder up to 0.5s — so a
        # request could block past its own hard deadline and seed exactly the
        # ownership/event-loop cascades this path exists to prevent. Never
        # enqueue past the deadline; refuse instead.
        _enqueue_remaining = 0.0
        try:
            _enqueue_remaining = max(0.0, float(deadline.remaining or 0.0))
        except (AttributeError, TypeError, ValueError):
            _enqueue_remaining = 2.0
        if _enqueue_remaining <= 0.0:
            await asyncio.shield(
                self._finish_generation_ownership(
                    req_id,
                    fut,
                    foreground_watchdog,
                )
            )
            _record_mlx_degradation(
                TimeoutError("request_deadline_expired_before_enqueue"),
                action="refused to queue work whose deadline had already expired",
                severity="warning",
            )
            return None
        enqueue_timeout = min(2.0, _enqueue_remaining)
        try:
            if self._req_q is None:
                raise BrokenPipeError("MLX request queue is closed")
            await run_io_bound(
                self._req_q.put,
                self._authorize_job(req, principal="mlx_client.generate"),
                True,
                enqueue_timeout,
            )
        except asyncio.CancelledError:
            await asyncio.shield(
                self._finish_generation_ownership(
                    req_id,
                    fut,
                    foreground_watchdog,
                )
            )
            raise
        except (BrokenPipeError, OSError, TimeoutError, queue.Full) as exc:
            await asyncio.shield(
                self._finish_generation_ownership(
                    req_id,
                    fut,
                    foreground_watchdog,
                )
            )
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
                self._record_interoception_from_response(
                    res,
                    foreground_request=foreground_request,
                    owner_label=owner_label,
                )
                raw_text = res.get("text", "")
                if not isinstance(raw_text, str):
                    # A malformed cross-process payload must fail through the
                    # typed empty-response path, not raise AttributeError.
                    _record_mlx_degradation(
                        TypeError(f"worker text payload was {type(raw_text).__name__}"),
                        action="treated non-string worker text as empty response",
                    )
                    raw_text = ""
                text = raw_text.strip()
                self._mark_progress()
                if not text and not res.get("soft_cancelled"):
                    quality_rejection_reasons = _surface_quality_rejection_reasons(
                        self.get_last_surface_control_receipt()
                    )
                    if quality_rejection_reasons:
                        # The worker decoded and deliberately rejected its own
                        # drafts. Treating that as cache corruption used to
                        # repeat the whole request, recycle a healthy 32B
                        # process, and open the Cortex circuit. Preserve the
                        # receipt and leave the lane resident for the caller's
                        # typed recovery policy instead.
                        self._preserve_lane_after_surface_quality_rejection()
                        logger.warning(
                            "🛡️ [MLX] Worker rejected the visible draft for semantic "
                            "quality (%s); preserving the resident lane.",
                            ",".join(quality_rejection_reasons),
                        )
                        return None
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
                        # This is an active recovery transition, not yet a
                        # user-visible failure. Keep the attempt observable
                        # without forwarding a synthetic RuntimeError into
                        # ErrorIntelligence before the retry has a verdict.
                        self._record_degraded_event(
                            "empty_generation_retry",
                            detail=(
                                f"{os.path.basename(self.model_path)}:"
                                f"attempt={empty_count}:cache_reset_retry"
                            ),
                            severity="info",
                            foreground_request=False,
                            classification="non_critical_fallback",
                        )
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
                        self._record_degraded_event(
                            "empty_generation_exhausted",
                            detail=(
                                f"{os.path.basename(self.model_path)}:"
                                f"attempt={empty_count}:no_visible_text"
                            ),
                            severity="error",
                            foreground_request=True,
                        )
                        self._deferred_reboot_reason = "recoverable_empty_generation"
                    else:
                        self._record_degraded_event(
                            "empty_generation",
                            detail=(
                                f"{os.path.basename(self.model_path)}:"
                                f"attempt={empty_count}:background"
                            ),
                            severity="info",
                            foreground_request=False,
                        )
                    if foreground_request and self._is_primary_or_deep_lane() and empty_count >= 3:
                        self._set_lane_state("recovering", "repeated_empty_generation")
                    return None
                self._consecutive_empty = 0
                if res.get("soft_cancelled"):
                    # Deliberate cooperative preemption: return the partial text
                    # without empty-generation telemetry, inline retries, or a
                    # user-facing completion mark — the health machinery must
                    # not treat a requested cancel as a generation failure.
                    logger.info(
                        "✋ [MLX] Generation for %s ended by soft-cancel after partial output (%d chars).",
                        os.path.basename(self.model_path),
                        len(text),
                    )
                    self._set_lane_state("ready")
                    return text or None
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
                # Ask the worker to drop the orphaned generation between
                # tokens; if it acknowledges, the warm lane survives (see
                # _resolve_deferred_reboot). Only an unacknowledged cancel
                # still costs a recycle.
                self.soft_cancel_active_generation("abandoned_generation_deadline")
                self._deferred_reboot_reason = "recoverable_generation_deadline_reached"
                logger.warning(
                    "⏳ [MLX] Deadline reached while worker still looks healthy; "
                    "soft-cancelling the abandoned generation and preserving the warm lane if acknowledged."
                )
            else:
                logger.warning(
                    "⏳ [MLX] Deadline reached but worker still looks healthy; leaving lane warm."
                )
            return None
        finally:
            await asyncio.shield(
                self._finish_generation_ownership(
                    req_id,
                    fut,
                    foreground_watchdog,
                )
            )

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

            tool_call = (
                self._extract_tool_call_payload(
                    response_text,
                    allowed_tools=set(tools.keys()),
                )
                if tools
                else None
            )
            if not tool_call:
                return {
                    "content": response_text,
                    "turns": turn + 1,
                    "tool_calls": tool_calls_made,
                }

            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})
            # CP126 0da5db2e: bind the parsed arguments to the tool's own
            # advertised JSON schema before anything executes. A call whose
            # arguments do not satisfy the schema is a malformed call, not an
            # effect to attempt.
            schema_error = _tool_arguments_schema_error(
                tools.get(tool_name), tool_args
            )
            if schema_error:
                _record_mlx_degradation(
                    ValueError(f"tool_arguments_invalid:{tool_name}:{schema_error}"),
                    action="refused a tool call whose arguments failed its advertised schema",
                    severity="warning",
                )
                messages.append({"role": "assistant", "content": response_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That call to '{tool_name}' was rejected: {schema_error}. "
                            "Correct the arguments or answer directly."
                        ),
                    }
                )
                continue
            # CP126 abd93abf: every call gets a stable id so the assistant
            # turn and its tool result are unambiguously paired in history.
            tool_call_id = f"call_{uuid.uuid4().hex[:12]}"

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

            tool_calls_made.append(
                {
                    "id": tool_call_id,
                    "tool": tool_name,
                    "args": tool_args,
                    "result": tool_result,
                }
            )
            logger.info("[think_and_act] turn=%d tool=%s ok", turn + 1, tool_name)

            # ── Feed result back into history ─────────────────────────
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(
                                    tool_args
                                ),  # [STABILITY v53] Must be a JSON string, not a dict
                            },
                        }
                    ],
                }
            )

            # [STABILITY v53] Protect against massive tool outputs breaking
            # context windows. CP126 abd93abf: a blind character slice cut
            # structured results mid-value, so the model reasoned over
            # syntactically broken JSON. Truncate to a still-parseable form
            # and say so explicitly.
            tool_result = _truncate_tool_result(tool_result, limit=4000)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": tool_result,
                }
            )

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
                        warmup_precompile=True,
                    ),
                    timeout=warmup_timeout + (10.0 * attempt),
                )
                if warmup_text is None and not self.is_alive():
                    raise RuntimeError("warmup_precompile_worker_dead")
                # CP126 cdd743de + b6439433. A nonempty token from a
                # max_tokens=1 "Hello" proves Metal shaders compiled — it does
                # NOT prove this lane can hold a conversation, and skipping the
                # visible probe on that basis is how a lane that cannot answer
                # got marked ready. The probe now ALWAYS runs, and its answer is
                # actually checked against what was asked for (the prompt says
                # "Reply exactly: ready" but any nonblank text used to pass, so
                # hallucinated, garbled, stale, or prompt-echo output proved
                # readiness).
                logger.info(
                    "🔥 [MLX] Verifying conversation readiness for %s with a visible probe.",
                    os.path.basename(self.model_path),
                )
                readiness_text = await asyncio.wait_for(
                    self._generate_inner(
                        _READINESS_PROBE_PROMPT,
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
                        clear_prompt_cache=True,
                    ),
                    timeout=min(max(10.0, warmup_timeout), 60.0),
                )
                if not readiness_text or not str(readiness_text).strip():
                    self._set_lane_state("recovering", "warmup_readiness_no_text")
                    raise RuntimeError("warmup_readiness_no_text")
                if not _readiness_answer_accepted(readiness_text):
                    self._set_lane_state("recovering", "warmup_readiness_answer_mismatch")
                    raise RuntimeError(
                        "warmup_readiness_answer_mismatch:"
                        f"{str(readiness_text).strip()[:60]!r}"
                    )
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
        """Boot the worker and prove the visible conversation path is ready.

        SINGLEFLIGHT (CP126 4d8a7d6b). Concurrent callers used to each set
        ``_warmup_in_flight`` and proceed, so two warmups could load/evict the
        same lane at once; the "stale warmup" recovery then measured
        ``_lane_transition_at`` — a timestamp any other lane transition
        refreshes — and force-cleared the shared flag without proving the prior
        warmup had ended. Callers now JOIN the active warmup, and a genuinely
        stuck one is cancelled and awaited before a replacement starts.
        """
        inflight = self._warmup_inflight
        if inflight is not None and not inflight.done():
            age = max(0.0, time.monotonic() - self._warmup_started_at)
            if age <= _WARMUP_STALE_AFTER_S:
                # Join the in-flight warmup. shield() so that a cancelled
                # joiner cannot kill the warmup the other callers need.
                try:
                    return bool(await asyncio.shield(inflight))
                except asyncio.CancelledError:
                    raise
                except (RuntimeError, TimeoutError, AttributeError, TypeError, ValueError) as exc:
                    _record_mlx_degradation(
                        exc,
                        action="reported warmup failure to a joined singleflight caller",
                        severity="warning",
                    )
                    return False
            logger.warning(
                "🔧 [MLX] Warmup for %s stuck for %.0fs — cancelling the prior "
                "warmup task before starting a replacement.",
                os.path.basename(self.model_path),
                age,
            )
            inflight.cancel()
            # PROVE the prior task ended before starting another one.
            try:
                await asyncio.wait_for(asyncio.shield(inflight), timeout=10.0)
            except (asyncio.CancelledError, TimeoutError):
                pass
            except (RuntimeError, AttributeError, TypeError, ValueError):
                pass
            self._warmup_in_flight = False

        task = asyncio.ensure_future(
            self._warmup_impl(
                foreground_request=foreground_request,
                skip_swap_cooldown=skip_swap_cooldown,
            )
        )
        self._warmup_inflight = task
        self._warmup_started_at = time.monotonic()
        try:
            return bool(await asyncio.shield(task))
        finally:
            if self._warmup_inflight is task and task.done():
                self._warmup_inflight = None

    async def _warmup_impl(
        self,
        *,
        foreground_request: bool | None = None,
        skip_swap_cooldown: bool = False,
    ) -> bool:
        """Boot the worker and prove the visible conversation path is ready."""
        if _shutdown_blocks_model_work(self.model_path, action="warmup"):
            self._warmup_in_flight = False
            if self._lane_state not in {"failed", "cold"}:
                self._set_lane_state("cold", "runtime_shutdown")
            return False
        if foreground_request is None:
            foreground_request = self._is_primary_or_deep_lane()
        else:
            foreground_request = bool(foreground_request)
        request_is_background = not foreground_request
        owner_name = f"warmup:{os.path.basename(self.model_path)}"
        warmup_timeout = self._warmup_timeout()
        self._warmup_attempted = True
        # Stale-warmup recovery lives in warmup()'s singleflight now: it owns
        # the task handle, so it can cancel and PROVE termination instead of
        # force-clearing a shared flag against an unrelated timestamp
        # (CP126 4d8a7d6b). This flag remains the cheap state other lifecycle
        # paths poll.
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

            if _shutdown_blocks_model_work(self.model_path, action="background warmup"):
                return False

            # CP126 811cde6f: this yield check used to run AFTER
            # _ensure_worker_alive, so a background lane could load a 20GB
            # model (or evict the resident one) and only then decide to defer
            # its precompile — defeating the very anti-thrash policy the check
            # exists to enforce. Decide BEFORE touching worker lifecycle.
            #
            # Background lanes (solver promotions, brainstem appraisals) yield
            # to an owned foreground. The PRIMARY lane's own warmup is exempt:
            # the foreground owner is usually a turn WAITING on exactly this
            # warmup, and deferring it deadlocked the lane live (2026-07-10:
            # 206s foreground budget expired every turn while the precompile
            # it needed sat deferred behind it).
            if (
                request_is_background
                and _foreground_owner_active()
                and not self._is_primary_lane()
            ):
                logger.info(
                    "⏸️ [MLX] Background warmup deferred for %s (before worker spawn) while foreground lane is owned by %s.",
                    os.path.basename(self.model_path),
                    _FOREGROUND_OWNER_NAME or "foreground",
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
            if (
                request_is_background
                and _foreground_owner_active()
                and not self._is_primary_lane()
            ):
                # Re-check: a foreground turn can take ownership while the
                # worker was coming up.
                logger.info(
                    "⏸️ [MLX] Background warmup precompile deferred for %s while foreground lane is owned by %s.",
                    os.path.basename(self.model_path),
                    _FOREGROUND_OWNER_NAME or "foreground",
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
        """Forcibly reboots the worker.

        LOCK DISCIPLINE (CP126 ec341dfa). This used to log "forcing reboot
        anyway" and then kill the process, replace queues, cancel futures and
        reset ownership while the actual lock holder was still operating —
        converting a SUSPECTED deadlock into GUARANTEED unsynchronized
        corruption. Contention is now waited out (a real lifecycle op is
        bounded by its own timeouts) and the destructive path is a deliberate,
        receipted last resort after repeated failures to acquire, not the
        first response to 10 seconds of contention.
        """
        self._set_lane_state("recovering", reason)
        acquired = await asyncio.to_thread(self._lock.acquire, True, 10.0)
        if not acquired:
            # Escalate the wait before considering anything unsynchronized.
            acquired = await asyncio.to_thread(
                self._lock.acquire, True, _REBOOT_LOCK_ESCALATED_WAIT_S
            )
        forced_unsynchronized = False
        if not acquired:
            self._reboot_lock_failures += 1
            forced_unsynchronized = self._reboot_lock_failures >= _REBOOT_LOCK_FORCE_AFTER
            if not forced_unsynchronized:
                _record_mlx_degradation(
                    TimeoutError(f"reboot_lock_unavailable:{reason}"),
                    action=(
                        "deferred reboot instead of mutating worker lifecycle state "
                        "without the lifecycle lock"
                    ),
                    severity="error",
                )
                logger.error(
                    "🚨 [MLX] Could not acquire _lock for reboot on %s after %.0fs "
                    "(attempt %d/%d). DEFERRING — another lifecycle operation owns "
                    "this lane.",
                    os.path.basename(self.model_path),
                    10.0 + _REBOOT_LOCK_ESCALATED_WAIT_S,
                    self._reboot_lock_failures,
                    _REBOOT_LOCK_FORCE_AFTER,
                )
                self._set_lane_state("recovering", f"reboot_deferred_lock:{reason}")
                return
            _record_mlx_degradation(
                TimeoutError(f"reboot_lock_wedged:{reason}"),
                action=(
                    "forced an unsynchronized reboot after repeated lock-acquisition "
                    "failures — the lock holder is presumed wedged"
                ),
                severity="critical",
            )
            logger.critical(
                "🚨 [MLX] Lock holder for %s presumed WEDGED after %d failed reboot "
                "acquisitions. Forcing unsynchronized reboot as a last resort.",
                os.path.basename(self.model_path),
                self._reboot_lock_failures,
            )
        else:
            self._reboot_lock_failures = 0
        try:
            if self._process is not None:
                # K4 accounting: the breaker classifies this death by reason
                # (deliberate yields never count; young crashes do).
                _note_lane_worker_death(self, reason)
            if self._process and self._process.is_alive():
                await asyncio.get_running_loop().run_in_executor(
                    None, self._kill_and_join_blocking, self._process
                )
            self._process = None
            self._init_done = False
            self._expert_adapter_path = None  # adapters live in the worker process
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
            self._current_request_seq = 0
            # A reboot orphans any cooperative-cancel request with the worker.
            cancel_seq = getattr(self, "_cancel_seq", None)
            if cancel_seq is not None:
                try:
                    cancel_seq.value = 0
                except (OSError, ValueError):
                    logger.debug("Cancel channel reset skipped during reboot.")
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
        try:
            await self._release_durable_model_lane_owner(reason=reason)
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="worker stopped but durable model-lane owner release failed",
                severity="warning",
            )
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

    async def _set_durable_lane_preemptible(self, preemptible: bool) -> bool:
        fencing_token = int(self._model_lane_fencing_token or 0)
        owner_id = str(self._model_lane_owner_id or "")
        if fencing_token <= 0 or not owner_id:
            return True
        try:
            from core.runtime.model_lane_control import get_model_lane_controller

            return await get_model_lane_controller().update_owner_preemptibility(
                owner_id,
                fencing_token=fencing_token,
                preemptible=preemptible,
            )
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action=(
                    "refused generation before losing active-use protection"
                    if not preemptible
                    else "kept the idle model lane conservatively non-preemptible"
                ),
                severity="critical" if not preemptible else "warning",
            )
            return False

    async def _finish_generation_ownership(
        self,
        request_id: str,
        future: SharedFuture,
        foreground_watchdog: _threading.Timer | None,
        *,
        release_lane: bool = True,
    ) -> None:
        if foreground_watchdog is not None:
            foreground_watchdog.cancel()
        if self._foreground_generation_watchdog is foreground_watchdog:
            self._foreground_generation_watchdog = None
        self._pending_generations.pop(request_id, None)
        if self._current_gen_future is future:
            self._current_gen_future = None
        self._active_generations = max(0, self._active_generations - 1)
        if self._current_request_id == request_id:
            self._clear_active_generation_tracking()
        if release_lane and self._active_generations <= 0:
            await self._set_durable_lane_preemptible(True)

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
        # Programmatic thresholds get the same fail-safe normalization as
        # env values: NaN bypasses every age comparison and a negative value
        # tears down a lane that idled for one tick.
        def _safe_threshold(value: float, default: float) -> float:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return default
            if not math.isfinite(value) or value < 0.0:
                return default
            return value

        pressure_idle_s = _safe_threshold(pressure_idle_s, 90.0)
        hard_idle_s = _safe_threshold(hard_idle_s, 900.0)
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

        # The PRIMARY lane stays resident when there is no memory pressure.
        # The 20260708-final soak started against a cortex the citizenship
        # unload had evicted during a quiet afternoon at 34% system RAM —
        # the first turn then paid a 120-150s cold start (and, pre-fix,
        # seeded the gate-orphan cascade). A resident 20GB cortex on an
        # unpressured 64GB machine is what the machine is FOR; the 90s
        # pressure path still reclaims it the moment RAM actually matters.
        # Small lanes (brainstem/reflex, seconds to reload) keep the
        # citizenship unload. AURA_VRAM_SCAVENGE_PRIMARY_HARD=1 restores
        # the old behavior.
        if not under_pressure and self._is_primary_lane():
            if os.environ.get(
                "AURA_VRAM_SCAVENGE_PRIMARY_HARD", "0"
            ).strip().lower() not in {"1", "true", "yes", "on"}:
                return {
                    "unloaded": False,
                    "reason": "primary_lane_stays_resident_without_pressure",
                    "idle_age_s": round(age, 1),
                }

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
            freed_bytes = _observed_process_rss_bytes(int(process.pid))

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
        # CP126 97aa64fc: close used to give the lifecycle lock ONE second and
        # then destroy the client regardless — cancelling futures, killing the
        # process and closing queues while another lifecycle operation was
        # still using them. close() is terminal so it must always finish, but
        # it now waits long enough for ordinary contention to clear and
        # receipts the case where it genuinely could not.
        acquired = self._lock.acquire(timeout=_CLOSE_LOCK_WAIT_S)
        if not acquired:
            _record_mlx_degradation(
                TimeoutError("close_lock_unavailable"),
                action=(
                    "closed the client without the lifecycle lock after waiting "
                    f"{_CLOSE_LOCK_WAIT_S:.0f}s — shutdown cannot be deferred"
                ),
                severity="error",
            )
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
        try:
            self._release_durable_model_lane_owner_sync(reason="client_close")
        except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            _record_mlx_degradation(
                exc,
                action="client closed but durable model-lane owner release failed",
                severity="warning",
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
        # CP126 84a18b06: swallowing this import failure disabled proof-primary
        # model enforcement EXACTLY when its enforcement infrastructure was
        # unavailable, so a proof run could fall through to an unchecked lane.
        # Fail CLOSED for proof runs; ordinary construction still proceeds.
        _record_mlx_degradation(
            _exc,
            action="could not import proof policy for primary-lane enforcement",
            severity="error",
        )
        if _proof_run_requested(kwargs.get("origin", "mlx_client")):
            raise RuntimeError(
                "proof_policy_unavailable: refusing to build an unenforced model "
                "client for a proof run"
            ) from _exc

    backend = get_local_backend()
    if backend != "mlx" or str(runtime_path).lower().endswith(".gguf"):
        raise RuntimeError(
            "external_cortex_disabled:"
            " live Aura uses the in-process MLX model lane; external Cortex artifacts are retired"
        )

    if client_key not in _CLIENTS:
        _CLIENTS[client_key] = MLXLocalClient(model_path=runtime_path, **kwargs)
    return _CLIENTS[client_key]


_TOOL_ARGS_MAX_KEYS = 64
_TOOL_ARGS_MAX_DEPTH = 6
_TOOL_ARGS_MAX_CHARS = 20_000


def _json_depth(value: Any, *, _depth: int = 0) -> int:
    if _depth > _TOOL_ARGS_MAX_DEPTH:
        return _depth
    if isinstance(value, dict):
        return max(
            (_json_depth(item, _depth=_depth + 1) for item in value.values()),
            default=_depth,
        )
    if isinstance(value, list):
        return max(
            (_json_depth(item, _depth=_depth + 1) for item in value),
            default=_depth,
        )
    return _depth


def _tool_arguments_schema_error(definition: Any, args: Any) -> str:
    """Validate parsed tool arguments against the tool's advertised schema.

    CP126 0da5db2e: parsed arguments went to execution with no binding to the
    schema the tool advertised for this turn — no required-field check, no
    type check, and no size/depth bound. Returns "" when the call is
    acceptable, else a short reason.
    """
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    if len(args) > _TOOL_ARGS_MAX_KEYS:
        return f"too many argument keys ({len(args)} > {_TOOL_ARGS_MAX_KEYS})"
    if _json_depth(args) > _TOOL_ARGS_MAX_DEPTH:
        return "arguments nested too deeply"
    try:
        encoded = json.dumps(args, default=str)
    except (TypeError, ValueError):
        return "arguments are not JSON-serializable"
    if len(encoded) > _TOOL_ARGS_MAX_CHARS:
        return f"arguments too large ({len(encoded)} chars)"

    spec = definition if isinstance(definition, dict) else {}
    # Accept either a bare function spec or an OpenAI-style wrapper.
    if isinstance(spec.get("function"), dict):
        spec = spec["function"]
    parameters = spec.get("parameters")
    if not isinstance(parameters, dict):
        return ""  # Nothing advertised to validate against.
    properties = parameters.get("properties")
    properties = properties if isinstance(properties, dict) else {}
    required = parameters.get("required")
    required = [str(item) for item in required] if isinstance(required, list) else []

    missing = [name for name in required if name not in args]
    if missing:
        return f"missing required argument(s): {', '.join(sorted(missing)[:5])}"
    if properties and parameters.get("additionalProperties") is False:
        unexpected = sorted(set(args) - set(properties))
        if unexpected:
            return f"unexpected argument(s): {', '.join(unexpected[:5])}"

    json_type_map: dict[str, tuple[type, ...]] = {
        "string": (str,),
        "number": (int, float),
        "integer": (int,),
        "boolean": (bool,),
        "object": (dict,),
        "array": (list,),
    }
    for name, value in args.items():
        prop = properties.get(name)
        if not isinstance(prop, dict):
            continue
        expected = prop.get("type")
        if not isinstance(expected, str):
            continue
        allowed = json_type_map.get(expected)
        if allowed is None:
            continue
        if expected in {"number", "integer"} and isinstance(value, bool):
            return f"argument '{name}' must be {expected}"
        if not isinstance(value, allowed):
            return f"argument '{name}' must be {expected}"
        enum = prop.get("enum")
        if isinstance(enum, list) and enum and value not in enum:
            return f"argument '{name}' is not one of its allowed values"
    return ""


def _truncate_tool_result(result: Any, *, limit: int = 4000) -> str:
    """Bound a tool result WITHOUT cutting structured output mid-value.

    CP126 abd93abf: a raw character slice produced syntactically broken JSON
    that the model then reasoned over as if it were the real result.
    """
    text = result if isinstance(result, str) else str(result)
    if len(text) <= limit:
        return text
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            parsed = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        if parsed is not None:
            # Re-emit a VALID, explicitly-marked truncation envelope instead
            # of a broken fragment.
            preview = json.dumps(parsed, default=str)[: max(0, limit - 200)]
            return json.dumps(
                {
                    "truncated": True,
                    "original_chars": len(text),
                    "note": "Result exceeded the context budget; preview only.",
                    "preview": preview,
                },
                default=str,
            )
    return text[:limit] + "\n\n...[OUTPUT TRUNCATED FOR LENGTH]..."


def _proof_run_requested(origin: Any) -> bool:
    """Is a proof run in progress, judged WITHOUT the proof-policy module?

    Used only on the path where importing ``core.runtime.proof_policy`` failed:
    enforcement cannot consult the policy it could not load, so it falls back
    to the environment signals the policy itself is configured from and fails
    closed when either says a proof run is active.
    """
    if str(origin or "").strip().lower().startswith("proof"):
        return True
    for name in ("AURA_PROOF_RUN", "AURA_PROOF_MODEL_TIER", "AURA_PROOF_HEADLESS"):
        value = str(os.environ.get(name, "") or "").strip().lower()
        if value and value not in {"0", "false", "off", "no", "none"}:
            return True
    return False


def _scavenge_env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, "").strip())
    except (TypeError, ValueError):
        return default
    # Infinity would silently disable the citizenship unload forever.
    return value if math.isfinite(value) and value > 0.0 else default


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
    for path, client in list(_CLIENTS.items()):
        lane_label = os.path.basename(str(path or "")) or "unknown"
        unload = getattr(client, "maybe_unload_idle", None)
        if unload is None:
            continue
        try:
            outcome = await unload(pressure_idle_s=pressure_idle_s, hard_idle_s=hard_idle_s)
        except (RuntimeError, OSError, ValueError, AttributeError) as exc:
            logger.debug("Idle VRAM scavenge skipped a lane: %s", exc)
            # Failed lanes must be visible in the report — hiding them made
            # repeated reclaim failures undiagnosable from telemetry.
            results.append(
                {
                    "lane": lane_label,
                    "unloaded": False,
                    "reason": f"scavenge_error:{type(exc).__name__}",
                }
            )
            continue
        entry = dict(outcome) if isinstance(outcome, dict) else {"unloaded": bool(outcome)}
        entry.setdefault("lane", lane_label)
        if entry.get("unloaded"):
            unloaded += 1
        results.append(entry)
    return {"enabled": True, "unloaded": unloaded, "lanes": results}
