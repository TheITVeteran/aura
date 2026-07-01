"""
core/brain/llm_health_router.py
────────────────────────────────
Replacement for IntelligentLLMRouter.

Fixes:
  - Zero-token / whitespace-only responses treated as failure, not success
  - Primary endpoint failure triggers genuine fallback to local MLX
  - Per-endpoint health tracking with circuit breaker pattern
  - Response validation before acceptance
  - Structured logging that distinguishes real success from empty success

Drop-in: replace the existing router instantiation in orchestrator_boot.py
with HealthAwareLLMRouter.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from core.brain.llm.chat_format import format_chatml_messages
from core.brain.llm.model_registry import (
    BRAINSTEM_ENDPOINT,
    DEEP_ENDPOINT,
    FALLBACK_ENDPOINT,
    PRIMARY_ENDPOINT,
    audit_lane_assignments,
    guard_solver_request,
    normalize_endpoint_name,
)
from core.brain.llm.runtime_wiring import (
    _merge_system_prompt,
    build_agentic_tool_map,
    prepare_runtime_payload,
    should_force_tool_handoff,
)
from core.phases.response_contract import ResponseContract
from core.runtime.desktop_boot_safety import desktop_resource_guard_enabled
from core.runtime.errors import record_degradation
from core.runtime.network_gateway import get_network_gateway
from core.runtime.proof_policy import (
    is_proof_evaluation_purpose,
    is_strict_proof_answer_prompt,
    mlx_strict_answer_contract_enabled,
    proof_model_tier,
    proof_run_active,
)
from core.runtime.turn_analysis import analyze_turn
from core.utils.concurrency import RobustLock
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Brain.HealthRouter")

# ── Generation concurrency gate ────────────────────────────────────────
# Round-9 spike stacks caught NINE concurrent generate calls stacked for
# a single user turn (draft/retry fan-out never cancelling predecessors).
# Each in-process generation holds GB-scale KV/context: the stack-up
# allocated ~2GB/s of compressible pages until macOS executed the
# process at a 78GB phys_footprint. Local generation is now a bounded
# resource: callers either acquire a slot within the wait budget or get
# a truthful saturation failure — stacking is the one outcome that can
# never happen again.
import threading as _threading  # noqa: E402 - gate lives with its rationale block


def generation_concurrency_limit(env: Mapping[str, str] | None = None) -> int:
    """Return the process-wide generation limit for the active runtime profile."""

    env = env or os.environ
    raw_limit = str(env.get("AURA_MAX_CONCURRENT_GENERATIONS", "2") or "2").strip()
    try:
        configured = max(1, int(raw_limit))
    except (TypeError, ValueError, OverflowError):
        configured = 2

    allow_desktop_parallelism = str(
        env.get("AURA_ALLOW_CONCURRENT_DESKTOP_GENERATIONS", "")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if desktop_resource_guard_enabled(env) and not allow_desktop_parallelism:
        return 1
    return configured


_GENERATION_GATE = _threading.BoundedSemaphore(
    generation_concurrency_limit()
)
_GENERATION_GATE_STATE_LOCK = _threading.Lock()
_GENERATION_GATE_ACTIVE_LEASES: dict[int, tuple[float, str]] = {}
_GENERATION_GATE_FORCED_LEASES: set[int] = set()
_GENERATION_GATE_NEXT_LEASE_ID = 0
_GENERATION_GATE_LAST_ACQUIRED_AT = 0.0
_GENERATION_GATE_LAST_OWNER = ""
# Wait long enough to outlast one full serialized generation: gated
# turns measure 31-46s live (2026-06-11), so the old 20s wait starved
# any request arriving while both slots were mid-turn — external
# validation's third coding repair died exactly that way while holding
# an unused 240s budget. 75s covers one slow turn plus margin; callers
# with shorter deadlines still bail via their own timeouts.
_GENERATION_GATE_WAIT_S = float(
    os.environ.get("AURA_GENERATION_GATE_WAIT_S", "75") or 75
)
_GATE_SATURATION_RESULT = {
    "ok": False,
    "text": "",
    "endpoint": "generation_gate_saturated",
    "tokens": 0,
    "error": (
        "local generation lane saturated: refusing to stack another "
        "concurrent generation (memory-bomb prevention)"
    ),
}


def _generation_gate_owner(origin: str, purpose: str) -> str:
    origin = str(origin or "unknown").strip() or "unknown"
    purpose = str(purpose or "unknown").strip() or "unknown"
    return f"{origin}:{purpose}"


def _generation_owner_is_user_foreground(owner: str) -> bool:
    owner = str(owner or "").strip().lower()
    if not owner:
        return False
    return any(
        marker in owner
        for marker in (
            "user:",
            "desktop",
            "voice",
            "foreground",
            "response_generation_user",
        )
    )


def _oldest_generation_gate_lease() -> tuple[int, float, str] | None:
    with _GENERATION_GATE_STATE_LOCK:
        if not _GENERATION_GATE_ACTIVE_LEASES:
            return None
        lease_id, (acquired_at, owner) = min(
            _GENERATION_GATE_ACTIVE_LEASES.items(),
            key=lambda item: item[1][0],
        )
        return lease_id, float(acquired_at), str(owner or "unknown")


def _generation_gate_busy_result(owner: str) -> dict[str, Any]:
    result = dict(_GATE_SATURATION_RESULT)
    result["endpoint"] = "generation_gate_busy_foreground"
    result["error"] = (
        "local generation lane is busy with an active foreground user generation; "
        f"refusing to force-release owner={str(owner or 'unknown')[:120]}"
    )
    return result


def _active_foreground_generation_owner() -> str:
    oldest_lease = _oldest_generation_gate_lease()
    if oldest_lease is None:
        return ""
    _lease_id, _acquired_at, owner = oldest_lease
    return owner if _generation_owner_is_user_foreground(owner) else ""


def _oldest_generation_gate_lease_age_s() -> float:
    oldest_lease = _oldest_generation_gate_lease()
    if oldest_lease is None:
        return 0.0
    _lease_id, acquired_at, _owner = oldest_lease
    return max(0.0, time.time() - float(acquired_at))


def generation_gate_snapshot() -> dict[str, Any]:
    """Return a read-only snapshot for schedulers and health probes."""

    with _GENERATION_GATE_STATE_LOCK:
        active = {
            int(lease_id): {
                "age_s": max(0.0, time.time() - float(acquired_at)),
                "owner": str(owner or "unknown"),
            }
            for lease_id, (acquired_at, owner) in _GENERATION_GATE_ACTIVE_LEASES.items()
        }
        oldest = None
        if active:
            oldest_id = min(active, key=lambda lease_id: active[lease_id]["age_s"])
            oldest = {"lease_id": oldest_id, **active[oldest_id]}
        return {
            "active_count": len(active),
            "active": active,
            "oldest": oldest,
            "last_acquired_at": float(_GENERATION_GATE_LAST_ACQUIRED_AT or 0.0),
            "last_owner": str(_GENERATION_GATE_LAST_OWNER or ""),
            "wait_budget_s": float(_GENERATION_GATE_WAIT_S),
        }


def _mark_generation_gate_acquired(owner: str) -> int:
    global _GENERATION_GATE_NEXT_LEASE_ID, _GENERATION_GATE_LAST_ACQUIRED_AT, _GENERATION_GATE_LAST_OWNER
    with _GENERATION_GATE_STATE_LOCK:
        _GENERATION_GATE_NEXT_LEASE_ID += 1
        lease_id = _GENERATION_GATE_NEXT_LEASE_ID
        acquired_at = time.time()
        _GENERATION_GATE_ACTIVE_LEASES[lease_id] = (acquired_at, str(owner or "unknown"))
        _GENERATION_GATE_LAST_ACQUIRED_AT = acquired_at
        _GENERATION_GATE_LAST_OWNER = str(owner or "unknown")
        return lease_id


def _release_generation_gate_after_call(lease_id: int) -> None:
    """Release the generation gate, accounting for watchdog-forced releases."""

    should_release = False
    with _GENERATION_GATE_STATE_LOCK:
        if lease_id in _GENERATION_GATE_FORCED_LEASES:
            _GENERATION_GATE_FORCED_LEASES.discard(lease_id)
            return
        if lease_id in _GENERATION_GATE_ACTIVE_LEASES:
            _GENERATION_GATE_ACTIVE_LEASES.pop(lease_id, None)
            should_release = True
    if not should_release:
        return
    try:
        _GENERATION_GATE.release()
    except ValueError:
        pass


def force_release_generation_gate(reason: str = "hard_generation_deadline") -> bool:
    """Emergency-release a stale router gate lease from a watchdog thread."""

    reason = str(reason or "hard_generation_deadline")
    with _GENERATION_GATE_STATE_LOCK:
        if not _GENERATION_GATE_ACTIVE_LEASES:
            return False
        lease_id, (acquired_at, owner) = min(
            _GENERATION_GATE_ACTIVE_LEASES.items(),
            key=lambda item: item[1][0],
        )
        _GENERATION_GATE_ACTIVE_LEASES.pop(lease_id, None)
        _GENERATION_GATE_FORCED_LEASES.add(lease_id)
        age_s = max(0.0, time.time() - acquired_at)
    try:
        _GENERATION_GATE.release()
    except ValueError:
        with _GENERATION_GATE_STATE_LOCK:
            _GENERATION_GATE_FORCED_LEASES.discard(lease_id)
        return False
    record_degradation(
        "llm_health_router",
        TimeoutError(f"generation gate forcibly released after {age_s:.1f}s"),
        severity="degraded",
        action=f"released stale generation gate lease for {owner}: {reason}",
    )
    return True


def _record_router_degradation(
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation("llm_health_router", exc, severity=severity, action=action)


_ROUTER_CLIENT_ERRORS = (
    httpx.HTTPError,
    OSError,
    ConnectionError,
    TimeoutError,
    RuntimeError,
    TypeError,
    ValueError,
    Exception,
)


def _endpoint_call_timeout(timeout: float) -> float:
    """Outer watchdog for an endpoint call.

    The endpoint/client still receives the original timeout as its cooperative
    budget. This wrapper adds a small cleanup grace window so a blocked local
    runtime cannot hold the router forever if the client fails to observe that
    budget.
    """
    try:
        timeout_s = float(timeout)
    except (TypeError, ValueError, OverflowError):
        timeout_s = 120.0
    timeout_s = max(0.1, timeout_s)
    grace_s = min(5.0, max(0.25, timeout_s * 0.1))
    return timeout_s + grace_s


def _endpoint_call_budgets(
    timeout: float,
    *,
    foreground_local: bool = False,
    prompt_chars: int = 0,
    max_tokens: int | None = None,
    benchmark_request: bool = False,
    proof_evaluation_contract: bool = False,
    health_probe: bool = False,
) -> tuple[float, float]:
    """Return cooperative client timeout and hard wall-clock watchdog budget."""
    try:
        timeout_s = max(0.1, float(timeout))
    except (TypeError, ValueError, OverflowError):
        timeout_s = 120.0
    wall_s = _endpoint_call_timeout(timeout_s)
    cooperative_s = timeout_s

    if (
        foreground_local
        and timeout_s >= 60.0
        and not benchmark_request
        and not proof_evaluation_contract
        and not health_probe
    ):
        try:
            token_count = int(max_tokens or 0)
        except (TypeError, ValueError, OverflowError):
            token_count = 0
        compact_turn = int(prompt_chars or 0) <= 10_000 and token_count <= 768
        env_name = (
            "AURA_FOREGROUND_LOCAL_COMPACT_WALL_TIMEOUT_S"
            if compact_turn
            else "AURA_FOREGROUND_LOCAL_EXTENDED_WALL_TIMEOUT_S"
        )
        default_cap = 105.0 if compact_turn else 150.0
        try:
            cap_s = max(30.0, float(os.environ.get(env_name, str(default_cap)) or default_cap))
        except (TypeError, ValueError, OverflowError):
            cap_s = default_cap
        wall_s = min(wall_s, cap_s)
        cooperative_s = min(cooperative_s, max(5.0, wall_s - 2.0))

    return cooperative_s, wall_s


def _proof_primary_lane_active(*, origin: str) -> bool:
    """Return whether this router build/call must expose only the primary lane."""
    try:
        return bool(proof_run_active(origin=origin) and proof_model_tier() == "primary")
    except _ROUTER_CLIENT_ERRORS as exc:
        _record_router_degradation(
            exc,
            action="failed closed while resolving proof-primary lane policy",
            severity="degraded",
        )
        return True


def _force_abort_endpoint_client(client: Any, *, reason: str) -> bool:
    abort = getattr(client, "force_abort_active_generation", None)
    if not callable(abort):
        return False
    try:
        return bool(abort(reason=reason))
    except _ROUTER_CLIENT_ERRORS as exc:
        _record_router_degradation(
            exc,
            action="continued routing after endpoint force-abort failed",
            severity="error",
        )
        logger.warning("Endpoint force-abort failed: %s", exc)
        return False


def _start_endpoint_wall_clock_watchdog(
    client: Any,
    *,
    reason: str,
    timeout_s: float,
) -> tuple[threading.Event, dict[str, bool], threading.Timer]:
    """Abort non-cooperative local inference on wall-clock time.

    ``asyncio.wait_for`` only fires when the awaited coroutine yields. The local
    MLX stack can block during native/model work, so proof and desktop routes
    need a thread-backed watchdog that can terminate the active generation even
    if the event loop is temporarily occupied.
    """

    fired = threading.Event()
    aborted = {"value": False}

    def _abort() -> None:
        fired.set()
        aborted["value"] = _force_abort_endpoint_client(client, reason=reason)

    watchdog = threading.Timer(max(0.01, float(timeout_s)), _abort)
    watchdog.daemon = True
    watchdog.start()
    return fired, aborted, watchdog


_USER_FACING_ORIGINS = frozenset({
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
})

_BACKGROUND_ORIGIN_HINTS = frozenset({
    "affect",
    "autonomous",
    "background",
    "constitutive",
    "continuous",
    "consolidation",
    "dream",
    "growth",
    "impulse",
    "memory",
    "metabolic",
    "mist",
    "monitor",
    "motivation",
    "parallel",
    "perception",
    "phenomenological",
    "proactive",
    "scanner",
    "sensory",
    "spontaneous",
    "stream",
    "structured",
    "subconscious",
    "internal",
    "system",
    "terminal",
    "volition",
    "witness",
})

_USER_FACING_PURPOSES = frozenset({
    "chat",
    "conversation",
    "expression",
    "reply",
    "user_response",
})


# ── Circuit Breaker States ────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"       # Normal — requests flow through
    OPEN = "open"           # Failed — requests blocked, fallback used
    HALF_OPEN = "half_open" # Testing — one probe request allowed


@dataclass
class EndpointHealth:
    name: str
    url: str
    model: str
    is_local: bool = False
    tier: Any = "local" # Matches LLMTier enum or str ("local", "api_deep", "api_fast")
    client: Any = None

    # Circuit breaker
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure: float = 0.0
    last_success: float = 0.0

    # Performance tracking
    avg_latency_ms: float = 0.0
    total_requests: int = 0
    total_tokens: int = 0
    empty_responses: int = 0

    # Config
    failure_threshold: int = 3
    recovery_timeout: float = 30.0
    min_tokens_for_success: int = 1

    def record_success(self, tokens: int, latency_ms: float):
        self.success_count += 1
        self.total_requests += 1
        self.total_tokens += tokens
        self.last_success = time.time()

        # Rolling average latency
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = (self.avg_latency_ms * 0.8) + (latency_ms * 0.2)

        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit CLOSED for %s — probe succeeded", self.name)
            self.state = CircuitState.CLOSED
            self.failure_count = 0

    def record_failure(self, reason: str):
        self.failure_count += 1
        self.total_requests += 1
        self.last_failure = time.time()

        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                logger.warning(
                    "Circuit OPEN for %s after %d failures. Reason: %s",
                    self.name, self.failure_count, reason
                )
            self.state = CircuitState.OPEN

    def trip_temporarily(self, reason: str):
        """Open the circuit on a transient MLX-runtime failure without poisoning health counters."""
        self.total_requests += 1
        self.last_failure = time.time()
        if self.state != CircuitState.OPEN:
            logger.warning(
                "Circuit OPEN for %s on transient runtime failure. Reason: %s",
                self.name,
                reason,
            )
        self.state = CircuitState.OPEN

    def record_empty(self):
        """Zero-token or whitespace-only response — treat as failure."""
        self.empty_responses += 1
        self.record_failure("empty_response")

    def is_available(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure > self.recovery_timeout:
                logger.info("Circuit HALF-OPEN for %s — probing", self.name)
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            return True
        return False

    def status_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": getattr(self, "tier", "standard"),
            "state": self.state.value,
            "failures": self.failure_count,
            "successes": self.success_count,
            "empty_responses": self.empty_responses,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "total_tokens": self.total_tokens,
        }


# ── Validator ─────────────────────────────────────────────────────────────────

def validate_response(text: str | None, min_tokens: int = 1) -> tuple[bool, str]:
    """
    Returns (is_valid, reason).
    A response is invalid if:
      - It is None
      - It is empty or whitespace-only
      - It contains only punctuation
      - It is suspiciously short (< min_tokens words)
    """
    if text is None:
        return False, "none_response"
    stripped = text.strip()
    if not stripped:
        return False, "empty_whitespace"
    if len(stripped) < 1:
        return False, "empty_whitespace"
    words = stripped.split()
    if len(words) < min_tokens:
        return False, f"below_min_tokens_{min_tokens}"
    # Check for pure error markers
    lower = stripped.lower()
    error_markers = [
        "i am currently offline",
        "i cannot process that",
        "error:",
        "connection refused",
        "timeout",
    ]
    for marker in error_markers:
        if lower.startswith(marker):
            return False, f"error_marker:{marker}"
    return True, "ok"


def _is_transient_local_runtime_failure(error: str) -> bool:
    normalized = str(error or "").strip().lower()
    if not normalized:
        return False
    return normalized in {
        "client_returned_no_text",
        "heartbeat_stalled_during_generation",
        "first_token_sla_exceeded",
        "token_progress_stalled",
    } or normalized.startswith(
        (
            "background_deferred:",
            "foreground_quiet_window",
            "foreground_busy",
            "mlx_runtime_unavailable:",
            "mlx_runtime_probe_failed:",
            "local_runtime_unavailable:",
            "prewarm_failed:",
        )
    )


def _background_error_is_quiet(error: str) -> bool:
    normalized = str(error or "")
    return normalized in {
        "foreground_busy",
        "foreground_quiet_window",
        "client_returned_no_text",
        "cancelled_unhealthy",
        "background_deferred:memory_pressure",
        "background_deferred:cortex_startup_quiet",
        "background_deferred:foreground_quiet_window",
        "background_deferred:cortex_resident",
        "background_deferred:cortex_failed",
        "background_deferred:foreground_reserved",
        "heartbeat_stalled_during_generation",
        "first_token_sla_exceeded",
        "token_progress_stalled",
    } or normalized.startswith((
        "background_deferred:",
        "mlx_runtime_unavailable:",
        "local_runtime_unavailable:",
        "request_queue_failed:",
    ))


def _local_client_failure_reason(client: Any) -> str:
    def _get_declared_attr(candidate: Any, attr: str) -> Any:
        try:
            inspect.getattr_static(candidate, attr)
        except AttributeError:
            return None
        try:
            value = getattr(candidate, attr)
        except (RuntimeError, AttributeError, TypeError):
            return None
        if value is candidate:
            return None
        return value

    def _extract_lane_failure(candidate: Any) -> str:
        lane = None
        get_lane_status = _get_declared_attr(candidate, "get_lane_status")
        get_conversation_status = _get_declared_attr(candidate, "get_conversation_status")
        if callable(get_lane_status):
            lane = get_lane_status()
        elif callable(get_conversation_status):
            lane = get_conversation_status()

        if not isinstance(lane, dict):
            return ""

        state = str(lane.get("state", "") or "").strip().lower()
        error = str(
            lane.get("last_error", "")
            or lane.get("last_failure_reason", "")
            or ""
        )
        if state == "failed":
            return error or "lane_failed"

        conversation_ready = bool(lane.get("conversation_ready", False))
        if (
            not conversation_ready
            and state in {"recovering", "spawning", "handshaking", "warming"}
            and error.startswith(
                (
                    "mlx_runtime_unavailable:",
                    "mlx_runtime_probe_failed:",
                    "local_runtime_unavailable:",
                    "prewarm_failed:",
                    "foreground_warmup_failed",
                )
            )
        ):
            return error
        return ""

    try:
        seen: set[int] = set()
        candidate = client
        while candidate is not None and id(candidate) not in seen:
            seen.add(id(candidate))
            failure = _extract_lane_failure(candidate)
            if failure:
                return failure

            next_candidate = None
            for attr in ("_client", "_mlx_client"):
                nested = _get_declared_attr(candidate, attr)
                if nested is not None:
                    next_candidate = nested
                    break
            candidate = next_candidate
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        _record_router_degradation(
            exc,
            action="continued without local lane failure detail after client inspection failed",
        )
        logger.debug("Local client lane inspection failed: %s", exc)
    return ""


def _supports_foreground_cloud_recovery(error: str) -> bool:
    normalized = str(error or "").strip().lower()
    return _is_transient_local_runtime_failure(normalized) or normalized.startswith(
        (
            "lane_failed",
        )
    )


# ── Main Router ───────────────────────────────────────────────────────────────


class HealthMonitorShim:
    """Compatibility shim for legacy components expecting a health_monitor object."""
    def __init__(self, router: HealthAwareLLMRouter):
        self._router = router

    def is_healthy(self, name: str) -> bool:
        """Check if an endpoint is available for routing."""
        ep = self._router.endpoints.get(name)
        if not ep:
            return False
        return ep.is_available()

class HealthAwareLLMRouter:
    """
    Routes LLM requests to available endpoints with circuit breaking.

    Priority order: endpoints are tried in order of registration.
    Local MLX is prioritized as the final fallback.
    """

    def __init__(self):
        self.endpoints: dict[str, EndpointHealth] = {}
        self.health_monitor = HealthMonitorShim(self)
        self._lock = RobustLock("LLMHealthRouter.RouteLock")
        self._created_at = time.monotonic()
        self.high_pressure_mode: bool = False
        self.last_tier: str = "local"
        self.last_user_tier: str = "local"
        self.last_user_endpoint: str = PRIMARY_ENDPOINT
        self.last_endpoint: str | None = None
        self.last_background_endpoint: str | None = None
        self.last_background_tier: str | None = None
        self.last_user_error: str = ""
        self.last_background_error: str = ""
        self._last_generation_metadata: dict[str, Any] = {}
        self._last_fallback_warning_at: float = 0.0
        logger.info("HealthAwareLLMRouter initialized (Legacy-Compatible mode)")

    def get_last_generation_metadata(self) -> dict[str, Any]:
        return dict(self._last_generation_metadata)

    def get_stats(self) -> dict[str, Any]:
        """Aggregate endpoint statistics for proprioceptive telemetry."""
        total_calls = 0
        total_tokens = 0
        total_failures = 0
        total_empty = 0
        endpoint_stats = {}
        for name, ep in self.endpoints.items():
            total_calls += ep.total_requests
            total_tokens += ep.total_tokens
            total_failures += ep.failure_count
            total_empty += ep.empty_responses
            endpoint_stats[name] = ep.status_dict()
        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "total_failures": total_failures,
            "total_empty_responses": total_empty,
            "endpoint_count": len(self.endpoints),
            "last_tier": self.last_tier,
            "last_endpoint": self.last_endpoint,
            "last_user_error": self.last_user_error,
            "last_background_error": self.last_background_error,
            "high_pressure_mode": self.high_pressure_mode,
            "endpoints": endpoint_stats,
        }

    def is_ready(self) -> bool:
        """Deep readiness probe for runtime inference routing health."""
        if not self.endpoints:
            return False
        try:
            lane_audit = audit_lane_assignments()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_router_degradation(
                exc,
                action="failed closed: llm router readiness could not audit lane assignments",
                severity="degraded",
            )
            return False
        if not bool(lane_audit.get("ok", True)):
            return False
        return any(
            ep.is_available()
            for ep in self.endpoints.values()
            if str(getattr(ep, "name", "") or "").strip().lower() != "static-reflex"
        )

    def force_release_generation_gate(self, reason: str = "hard_generation_deadline") -> bool:
        """Emergency release for watchdogs when a router call outlives its budget."""

        return force_release_generation_gate(reason=reason)

    def force_abort_active_generation(self, reason: str = "hard_generation_deadline") -> int:
        """Abort stale router/model generation state from watchdog or saturation paths."""

        aborted = 1 if force_release_generation_gate(reason=reason) else 0
        seen: set[int] = set()

        def _abort_client(client: Any) -> None:
            nonlocal aborted
            if client is None:
                return
            ident = id(client)
            if ident in seen:
                return
            seen.add(ident)
            abort = getattr(client, "force_abort_active_generation", None)
            if not callable(abort):
                return
            try:
                if abort(reason=reason):
                    aborted += 1
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                _record_router_degradation(
                    exc,
                    action="continued force-aborting other generation clients",
                    severity="degraded",
                )

        for endpoint in self.endpoints.values():
            _abort_client(getattr(endpoint, "client", None))
        try:
            from core.container import ServiceContainer

            _abort_client(ServiceContainer.get("inference_gate", default=None))
        except (ImportError, AttributeError, RuntimeError):
            pass
        return aborted

    def register(
        self,
        name: str,
        url: str,
        model: str,
        is_local: bool = False,
        tier: str = "local",
        client: Any = None,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> HealthAwareLLMRouter:
        name = normalize_endpoint_name(name) or name
        ep = EndpointHealth(
            name=name,
            url=url,
            model=model,
            is_local=is_local,
            tier=tier,
            client=client,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
        self.endpoints[name] = ep
        logger.info("Registered endpoint: %s (%s) tier=%s local=%s", name, model, tier, is_local)
        return self

    def register_endpoint(self, ep_obj: Any) -> HealthAwareLLMRouter:
        """Compatibility method for Unified Cognitive Engine / AutonomousBrain."""
        # ep_obj is expected to have: name, tier, model_name, client
        name = normalize_endpoint_name(getattr(ep_obj, "name", "unknown")) or "unknown"
        tier_val = getattr(ep_obj, "tier", "local")
        is_local = name in {
            PRIMARY_ENDPOINT,
            DEEP_ENDPOINT,
            BRAINSTEM_ENDPOINT,
            FALLBACK_ENDPOINT,
        } or "MLX" in name or "Local" in name
        
        # Normalize both enum-style tiers and legacy string aliases into the router's
        # concrete routing labels: local, local_deep, local_fast, api_fast, api_deep.
        tier_name = tier_val
        if isinstance(tier_val, str):
            lowered = tier_val.lower()
            if lowered == "api_deep":
                tier_name = "api_deep"
            elif lowered == "api_fast":
                tier_name = "api_fast"
            elif lowered in ("local", "primary"):
                tier_name = "local"
            elif lowered in ("local_deep", "secondary"):
                tier_name = "local_deep" if is_local else "api_deep"
            elif lowered in ("local_fast", "tertiary"):
                tier_name = "local_fast" if is_local else "api_fast"
            elif lowered == "emergency":
                tier_name = "emergency"
        elif hasattr(tier_val, "value"):
            normalized = str(tier_val.value).lower()
            if normalized == "primary":
                tier_name = "local" if is_local else "api_fast"
            elif normalized == "secondary":
                tier_name = "local_deep" if is_local else "api_deep"
            elif normalized == "tertiary":
                tier_name = "local_fast" if is_local else "api_fast"
            elif normalized == "emergency":
                tier_name = "emergency"

        model_name = getattr(ep_obj, "model_name", "unknown")
        
        return self.register(
            name=name,
            url="internal" if is_local else "cloud",
            model=model_name,
            is_local=is_local,
            tier=tier_name,
            client=getattr(ep_obj, "client", None)
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        timeout: float = 120.0,  # noqa: ASYNC109 - public router API accepts timeout budgets.
        prefer_tier: str | None = None,
        schema: dict | None = None,
        **kwargs,
    ) -> str:
        """
        Try each endpoint in order. Return first valid response as a string.
        Falls back to local if all remote endpoints fail.
        GUARANTEE: Never returns empty string — provides diagnostic fallback.
        """
        if (not prompt) and "messages" in kwargs:
            prompt, inferred_system_prompt = self._coerce_prompt_from_messages(kwargs.get("messages", []))
            if not system_prompt and inferred_system_prompt:
                system_prompt = inferred_system_prompt

        res = await self.generate_with_metadata(
            prompt, system_prompt, timeout, prefer_tier=prefer_tier, schema=schema, **kwargs
        )
        text = res.get("text", "")
        origin = str(kwargs.get("origin", "") or "").lower()
        purpose = str(kwargs.get("purpose", "") or "").lower()
        benchmark_request = bool(kwargs.get("benchmark_request", False)) or (
            origin in {"baseline", "benchmark"}
            or purpose == "baseline"
            or purpose.endswith("_baseline")
            or "_baseline" in purpose
        )
        explicit_foreground = bool(kwargs.get("foreground_request", False)) or bool(
            kwargs.get("health_probe", False)
        )
        is_background = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=bool(kwargs.get("is_background", False)),
            explicit_foreground=explicit_foreground,
        )

        if is_background and _background_error_is_quiet(str(res.get("error", "") or "")):
            return ""

        if benchmark_request and (not text or not text.strip()):
            return ""
        
        # RESPONSE GUARANTEE: Never return empty
        if not text or not text.strip():
            if is_background:
                return ""
            error = res.get("error", "unknown")
            endpoint = res.get("endpoint", "none")
            logger.error(
                "⚠️ [LLM ROUTER] All endpoints exhausted. Last error: %s (endpoint: %s)",
                error, endpoint
            )
            if str(error or "").strip() == "client_returned_no_text":
                return "I lost the reply lane for a moment. Ask that again and I'll answer cleanly."
            # v10.5 HARDENING: Return a diagnostic label so StructuredLLM can report it accurately
            # instead of a silent empty string.
            return f"ROUTER_ERROR: {error} (at {endpoint})"
        
        return text

    async def generate_with_metadata(
        self,
        prompt: str,
        system_prompt: str | None = None,
        timeout: float = 180.0,  # noqa: ASYNC109 - public router API accepts timeout budgets.
        prefer_tier: str | None = None,
        schema: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Try each endpoint in order. Return first valid response with full metadata.
        Falls back to local if all remote endpoints fail.
        Always returns a dict: {"ok": bool, "text": str, "endpoint": str, "tokens": int}
        """
        origin = str(kwargs.get("origin", "") or "").lower()
        purpose = str(kwargs.get("purpose", "") or "").lower()
        explicit_background = bool(kwargs.get("is_background", False))
        explicit_foreground = bool(
            kwargs.get("foreground_request", False)
            or kwargs.get("health_probe", False)
            or kwargs.get("protected_foreground_lane", False)
            or kwargs.get("proof_primary_lane_required", False)
        )
        if self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=explicit_background,
            explicit_foreground=explicit_foreground,
        ):
            foreground_owner = _active_foreground_generation_owner()
            if foreground_owner:
                return _generation_gate_busy_result(foreground_owner)

        early_deferral = self._background_suppression_result(
            origin=origin,
            purpose=purpose,
            explicit_background=explicit_background,
            explicit_foreground=explicit_foreground,
        )
        if early_deferral is not None:
            return early_deferral

        acquired = await asyncio.to_thread(
            _GENERATION_GATE.acquire, True, _GENERATION_GATE_WAIT_S
        )
        if not acquired:
            foreground_owner = _active_foreground_generation_owner()
            foreground_age_s = _oldest_generation_gate_lease_age_s() if foreground_owner else 0.0
            if foreground_owner and foreground_age_s < max(30.0, _GENERATION_GATE_WAIT_S):
                return _generation_gate_busy_result(foreground_owner)
            aborted = self.force_abort_active_generation(
                reason=f"generation_gate_wait_timeout:{_GENERATION_GATE_WAIT_S:.1f}s"
            )
            if aborted:
                acquired = await asyncio.to_thread(_GENERATION_GATE.acquire, True, 2.0)
        if not acquired:
            record_degradation(
                "llm_health_router",
                RuntimeError("generation gate saturated"),
                severity="degraded",
                action="refused to stack another concurrent generation",
            )
            return dict(_GATE_SATURATION_RESULT)
        lease_id = _mark_generation_gate_acquired(_generation_gate_owner(origin, purpose))
        try:
            return await self._generate_with_metadata_gated(
                prompt,
                system_prompt=system_prompt,
                timeout=timeout,
                prefer_tier=prefer_tier,
                schema=schema,
                **kwargs,
            )
        finally:
            _release_generation_gate_after_call(lease_id)

    async def _generate_with_metadata_gated(
        self,
        prompt: str,
        system_prompt: str | None = None,
        timeout: float = 180.0,  # noqa: ASYNC109 - inherited budget semantics.
        prefer_tier: str | None = None,
        schema: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        _contract_tool_handoff_val = kwargs.pop("_contract_tool_handoff", False)
        if (not prompt) and "messages" in kwargs:
            prompt, inferred_system_prompt = self._coerce_prompt_from_messages(kwargs.get("messages", []))
            if not system_prompt and inferred_system_prompt:
                system_prompt = inferred_system_prompt

        origin = str(kwargs.get("origin", "") or "").lower()
        purpose = str(kwargs.get("purpose", "") or "").lower()
        explicit_background = bool(kwargs.get("is_background", False))
        explicit_foreground = bool(kwargs.get("foreground_request", False)) or bool(
            kwargs.get("health_probe", False)
        )
        non_chat_inference = bool(kwargs.pop("_non_chat_inference", False))
        if not origin and not purpose and not explicit_background and not non_chat_inference:
            purpose = "expression"
            kwargs["purpose"] = purpose
        inferred_background = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=explicit_background,
            explicit_foreground=explicit_foreground,
        )
        state = kwargs.pop("state", None)
        skip_runtime_payload = bool(kwargs.pop("skip_runtime_payload", False))
        contract: ResponseContract | None = None
        prepared_messages = kwargs.get("messages")
        _runtime_state = state
        if skip_runtime_payload:
            if prepared_messages is not None and system_prompt:
                prepared_messages = _merge_system_prompt(prepared_messages, system_prompt)
                kwargs["messages"] = prepared_messages
            elif prepared_messages is None:
                kwargs.pop("messages", None)
            if (not prompt) and prepared_messages is not None:
                prompt, inferred_system_prompt = self._coerce_prompt_from_messages(prepared_messages)
                if not system_prompt and inferred_system_prompt:
                    system_prompt = inferred_system_prompt
        else:
            prompt, system_prompt, prepared_messages, contract, _runtime_state = await prepare_runtime_payload(
                prompt=prompt,
                system_prompt=system_prompt,
                messages=kwargs.get("messages"),
                state=state,
                origin=origin,
                is_background=inferred_background,
            )
            if prepared_messages is not None:
                kwargs["messages"] = prepared_messages
            else:
                kwargs.pop("messages", None)

        if should_force_tool_handoff(contract, is_background=inferred_background) and not _contract_tool_handoff_val:
            tools = build_agentic_tool_map(
                contract.required_skill if contract else None,
                objective=prompt,
                max_tools=getattr(contract, "max_tools", 8) if contract else 8,
            )
            if tools:
                handoff_kwargs = dict(kwargs)
                handoff_kwargs.pop("origin", None)
                handoff_kwargs.pop("is_background", None)
                handoff_kwargs.pop("_contract_tool_handoff", None)
                result = await self.think_and_act(
                    objective=prompt,
                    system_prompt=system_prompt or "",
                    tools=tools,
                    context={"response_contract": contract.to_dict()} if contract else {},
                    prefer_tier=prefer_tier,
                    origin=origin or "user",
                    is_background=False,
                    _contract_tool_handoff=True,
                    **handoff_kwargs,
                )
                text = str(result.get("content", "") or "").strip()
                if text:
                    return {
                        "ok": True,
                        "text": text,
                        "endpoint": "contract_tool_handoff",
                        "tokens": len(text.split()),
                        "error": "",
                    }
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": "contract_tool_handoff",
                    "tokens": 0,
                    "error": "grounding_required_no_tool_result",
                }
        from core.consciousness.state_freeze import state_freeze
        async with state_freeze():
            return await self._generate_core(
                prompt, system_prompt, timeout, prefer_tier=prefer_tier, schema=schema, **kwargs
            )

    async def think(
        self,
        prompt: str | None = None,
        system_prompt: str | None = None,
        prefer_tier: str | None = None,
        schema: dict | None = None,
        **kwargs,
    ) -> str | None:
        """
        Unified interface for non-chat callers. Routes through the health-aware
        endpoint selection, then normalises to Optional[str].
        [FIX #1-Harden] Supports 'messages' keyword for cognitive pipeline compatibility.
        """
        kwargs.pop("_contract_tool_handoff", False)
        if not prompt and "messages" in kwargs:
            prompt, inferred_system_prompt = self._coerce_prompt_from_messages(kwargs.get("messages", []))
            if not system_prompt and inferred_system_prompt:
                system_prompt = inferred_system_prompt

        if not prompt:
            logger.warning("[LLMRouter.think] Called without prompt or messages.")
            return None
        try:
            result = await self.generate_with_metadata(
                prompt=prompt,
                system_prompt=system_prompt or "",
                prefer_tier=prefer_tier,
                schema=schema,
                _non_chat_inference=True,
                **kwargs,
            )
            if isinstance(result, dict):
                self._last_generation_metadata = dict(result)
            text = result.get("text", "") if isinstance(result, dict) else str(result)
            strict_answer_request = "<answer>" in str(prompt or "").lower() or "<answer>" in str(
                system_prompt or ""
            ).lower()
            # GUARD: Never call .strip() on None
            if text is None:
                if (
                    isinstance(result, dict)
                    and str(result.get("error", "") or "").strip() == "client_returned_no_text"
                    and not self._is_background_request(
                        origin=str(kwargs.get("origin", "") or "").lower(),
                        purpose=str(kwargs.get("purpose", "") or "").lower(),
                        explicit_background=bool(kwargs.get("is_background", False)),
                        explicit_foreground=bool(kwargs.get("foreground_request", False))
                        or bool(kwargs.get("health_probe", False)),
                    )
                ):
                    if strict_answer_request or kwargs.get("_non_chat_inference"):
                        return None
                    return "I lost the reply lane for a moment. Ask that again and I'll answer cleanly."
                return None
            stripped = text.strip()
            if stripped:
                return stripped
            if (
                isinstance(result, dict)
                and str(result.get("error", "") or "").strip() == "client_returned_no_text"
                and not self._is_background_request(
                    origin=str(kwargs.get("origin", "") or "").lower(),
                    purpose=str(kwargs.get("purpose", "") or "").lower(),
                    explicit_background=bool(kwargs.get("is_background", False)),
                    explicit_foreground=bool(kwargs.get("foreground_request", False))
                    or bool(kwargs.get("health_probe", False)),
                )
            ):
                if strict_answer_request or kwargs.get("_non_chat_inference"):
                    return None
                return "I lost the reply lane for a moment. Ask that again and I'll answer cleanly."
            # [STABILITY v55] Don't mask failures with robot responses.
            # Return None so the caller can retry or fallback properly.
            return None
        except (httpx.HTTPError, OSError, ConnectionError, TimeoutError) as exc:
            _record_router_degradation(
                exc,
                action="returned no router thought after endpoint generation failed",
                severity="degraded",
            )
            logger.warning("[LLMRouter.think] Failed: %s", exc)
            return None

    async def classify(
        self,
        prompt: str,
        system_prompt: str | None = None,
        prefer_tier: str = "primary",
        **kwargs
    ) -> str:
        """
        Hardened Intent Classification.
        Forces the LLM to return ONLY a single intent token.
        """
        classification_system_prompt = (
            "You are an intent classifier for Aura. Respond ONLY with one of the following tokens:\n"
            "- technical: coding, debugging, architecture, math, logic, research\n"
            "- philosophical: identity, morality, existence, consciousness\n"
            "- emotional: feelings, mood, empathy, personal reflection\n"
            "- planning: list of tasks, project management, goal setting\n"
            "- critical: security audits, performance bottlenecks, vulnerability scans\n"
            "- casual: greetings, small talk, status checks\n\n"
            "Do not explain. Do not use punctuation. Just output the single word."
        )

        try:
            deterministic = self._deterministic_intent_classification(prompt)
            if deterministic:
                logger.info("🧭 Intent classification resolved deterministically: %s", deterministic)
                return deterministic

            # We use generate_with_metadata directly to ensure strict parameters
            result = await self.generate_with_metadata(
                prompt=prompt,
                system_prompt=system_prompt or classification_system_prompt,
                max_tokens=10,
                temperature=0.0,
                prefer_tier=prefer_tier,
                purpose="classification",
                **kwargs
            )
            
            text = result.get("text", "").strip().lower()
            # Clean any stray punctuation
            import re
            text = re.sub(r'[^a-z_]', '', text)
            
            if not text:
                logger.warning("⚠️ Intent classification returned empty. Defaulting to 'casual'.")
                return "casual"
                
            return text
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_router_degradation(
                e,
                action="defaulted intent classification to casual after classifier failed",
                severity="degraded",
            )
            logger.error("❌ Intent classification failed: %s. Defaulting to 'casual'.", e)
            return "casual"

    async def think_and_act(
        self,
        objective: str,
        system_prompt: str = "",
        tools: dict[str, Any] | None = None,
        max_turns: int = 5,
        context: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        kwargs.pop("_contract_tool_handoff", False)
        origin = str(kwargs.get("origin", "") or "").lower()
        purpose = str(kwargs.get("purpose", "") or "").lower()
        is_bg = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=bool(kwargs.get("is_background", False)),
            explicit_foreground=bool(kwargs.get("foreground_request", False))
            or bool(kwargs.get("health_probe", False)),
        )
        state = kwargs.pop("state", None)
        objective, system_prompt, prepared_messages, contract, runtime_state = await prepare_runtime_payload(
            prompt=objective,
            system_prompt=system_prompt,
            messages=kwargs.get("messages"),
            state=state,
            origin=origin,
            is_background=is_bg,
        )
        if prepared_messages is not None:
            kwargs["messages"] = prepared_messages
        else:
            kwargs.pop("messages", None)
        prefer_tier = self._normalize_prefer_tier(kwargs.get("prefer_tier"))
        allow_cloud_fallback = bool(kwargs.get("allow_cloud_fallback", False))
        agent_context = dict(context or {})
        if contract:
            agent_context.setdefault("response_contract", contract.to_dict())
        if prepared_messages is not None:
            agent_context.setdefault("messages", prepared_messages)
        if contract:
            max_turns = min(max_turns, max(1, int(getattr(contract, "max_tool_turns", max_turns) or max_turns)))

        preferred_names = self._fallback_endpoint_names(
            prefer_tier or "primary",
            allow_cloud_fallback,
            is_background=is_bg,
        )
        available = [ep for ep in self.endpoints.values() if ep.is_available()]
        ordered: list[EndpointHealth] = []
        seen = set()
        for name in preferred_names:
            ep = self.endpoints.get(name)
            if ep and ep.is_available():
                ordered.append(ep)
                seen.add(ep.name)
        for ep in available:
            if ep.name not in seen:
                ordered.append(ep)

        def _call_kwargs(method: Any) -> dict[str, Any]:
            try:
                sig = inspect.signature(method)
            except (TypeError, ValueError):
                return dict(kwargs)

            if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
                return dict(kwargs)

            return {key: value for key, value in kwargs.items() if key in sig.parameters}

        for ep in ordered:
            if is_bg and self._tier_is_background_only(self._tier_name(ep)) is False and not kwargs.get("prefer_endpoint"):
                continue
            client = ep.client
            if not client or not hasattr(client, "think_and_act"):
                continue
            try:
                result = await client.think_and_act(
                    objective,
                    system_prompt=system_prompt,
                    tools=tools,
                    max_turns=max_turns,
                    context=agent_context,
                    **_call_kwargs(client.think_and_act),
                )
                text = str((result or {}).get("content", "") or "").strip()
                if text:
                    ep.record_success(len(text.split()), 0.0)
                    self.last_tier = ep.tier
                    self.last_endpoint = ep.name
                    if is_bg:
                        self.last_background_endpoint = ep.name
                        self.last_background_tier = ep.tier
                    else:
                        self.last_user_endpoint = ep.name
                        self.last_user_tier = ep.tier
                    return result
            except (httpx.HTTPError, OSError, ConnectionError, TimeoutError) as exc:
                _record_router_degradation(
                    exc,
                    action="recorded endpoint failure and continued tool-capable route fallback",
                    severity="degraded",
                )
                logger.warning("think_and_act on %s failed: %s", ep.name, exc)
                ep.record_failure(str(exc))

        kwargs_clean = dict(kwargs)
        kwargs_clean.pop("_contract_tool_handoff", None)
        text = await self.think(
            objective,
            system_prompt=system_prompt,
            state=runtime_state,
            _contract_tool_handoff=True,
            **kwargs_clean,
        )
        return {"content": text or "", "turns": 0, "tool_calls": []}

    async def _get_mycelial_direction(self, prompt: str) -> dict[str, Any] | None:
        """Query Mycelium for routing guidance (v31)."""
        try:
            from core.container import ServiceContainer
            mycelium = ServiceContainer.get("mycelium", default=None)
            if not mycelium:
                return None
            
            # 1. Match hardwired pathways
            # v42 FIX: Skip large prompts (likely background tasks/logs) to avoid false 'null' matches
            if len(prompt) > 100 or "say 'null'" in prompt.lower():
                return None
                
            match_res = mycelium.match_hardwired(prompt)
            if match_res:
                pathway, _params = match_res
                # If pathway exists, it's a strong signal
                # For now, we look for 'brain_tier' or 'route' in description or custom logic
                # Optimization: check if description has routing tags
                desc = pathway.description.lower()
                if "local-only" in desc or "private" in desc:
                    return {"tier_preference": "local"}
                if "cloud-only" in desc or "heavy" in desc:
                    return {"tier_preference": "cloud"}
                
                return {"pathway_id": pathway.pathway_id}
            return None
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_router_degradation(
                exc,
                action="continued routing without mycelial direction after guidance lookup failed",
            )
            return None

    def _flatten_messages_for_local_model(self, messages: list[dict[str, str]], require_json: bool) -> str:
        """Flatten messages into a Qwen/ChatML prompt for local MLX models."""
        return format_chatml_messages(messages, require_json=require_json)

    @staticmethod
    def _coerce_prompt_from_messages(messages: Any) -> tuple[str, str | None]:
        """Serialize a full OpenAI-style message list into prompt/system fields.

        This keeps the health-aware router aligned with the legacy router so
        callers can pass rich conversational state without it being collapsed
        down to only the last user turn.
        """
        if not messages or not isinstance(messages, list):
            return "", None

        system_parts: list[str] = []
        convo_parts: list[str] = []

        for msg in messages:
            if not isinstance(msg, dict):
                convo_parts.append(str(msg))
                continue

            role = str(msg.get("role", "") or "").strip().lower()
            content = str(msg.get("content", "") or "").strip()
            if not content:
                continue

            if role == "system":
                system_parts.append(content)
            elif role in {"user", "human"}:
                convo_parts.append(f"User: {content}")
            elif role in {"assistant", "aura"}:
                convo_parts.append(f"Aura: {content}")
            else:
                convo_parts.append(f"[{role or 'message'}]: {content}")

        prompt = "\n".join(convo_parts).strip()
        system_prompt = "\n\n".join(system_parts).strip() or None
        return prompt, system_prompt

    @staticmethod
    def _normalize_prefer_tier(prefer_tier: Any | None) -> str | None:
        if prefer_tier is None:
            return None
        if not isinstance(prefer_tier, str):
            if hasattr(prefer_tier, "value"):
                prefer_tier = prefer_tier.value
            else:
                prefer_tier = str(prefer_tier)

        tier = prefer_tier.lower()
        aliases = {
            "local": "primary",
            "local_deep": "secondary",
            "local_fast": "tertiary",
            "fast": "tertiary",
            "deep": "secondary",
        }
        return aliases.get(tier, tier)

    @staticmethod
    def _origin_tokens(origin: str | None) -> set[str]:
        normalized = str(origin or "").strip().lower().replace("-", "_")
        return {token for token in normalized.split("_") if token}

    @classmethod
    def _is_user_facing_origin(cls, origin: str | None) -> bool:
        tokens = cls._origin_tokens(origin)
        return bool(tokens & _USER_FACING_ORIGINS)

    @classmethod
    def _is_background_request(
        cls,
        *,
        origin: str | None,
        purpose: str | None,
        explicit_background: bool,
        explicit_foreground: bool = False,
    ) -> bool:
        if explicit_background:
            return True
        if explicit_foreground:
            return False

        normalized_purpose = str(purpose or "").strip().lower()
        if normalized_purpose in _USER_FACING_PURPOSES:
            return False

        tokens = cls._origin_tokens(origin)
        if not tokens:
            return normalized_purpose not in _USER_FACING_PURPOSES

        if tokens & _USER_FACING_ORIGINS:
            return False

        # Hardened default: anything that is not explicitly user-facing is
        # background. This prevents internal/kernel/autonomous traffic with
        # weak or unfamiliar origins from contaminating the foreground lane.
        if tokens & _BACKGROUND_ORIGIN_HINTS:
            return True

        return True

    def _background_suppression_result(
        self,
        *,
        origin: str | None,
        purpose: str | None,
        explicit_background: bool,
        explicit_foreground: bool = False,
    ) -> dict[str, Any] | None:
        """Return a suppression result before scarce generation capacity is acquired."""

        is_bg = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=explicit_background,
            explicit_foreground=explicit_foreground,
        )
        if not is_bg:
            return None

        reason = ""
        try:
            from core.runtime.background_policy import (
                THOUGHT_BACKGROUND_POLICY,
                background_activity_reason,
            )

            reason = str(
                background_activity_reason(
                    None,
                    profile=THOUGHT_BACKGROUND_POLICY,
                    allow_no_user_anchor=True,
                )
                or ""
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            _record_router_degradation(
                exc,
                action="deferred background routing because background policy was unavailable",
                severity="degraded",
            )
            logger.warning("Background router policy probe failed: %s", exc)
            reason = "background_policy_unavailable"
        if not reason:
            try:
                from core.container import ServiceContainer

                gate = ServiceContainer.get("inference_gate", default=None)
                if gate and hasattr(gate, "_background_local_deferral_reason"):
                    reason = str(gate._background_local_deferral_reason(origin=origin) or "")
            except (ImportError, AttributeError, RuntimeError) as exc:
                _record_router_degradation(
                    exc,
                    action="continued background routing without inference-gate deferral signal",
                )
                logger.debug("Background router deferral probe failed: %s", exc)
        if not reason and self._foreground_quiet_window_active():
            reason = "foreground_quiet_window"
        if not reason and getattr(self, "high_pressure_mode", False):
            reason = "memory_pressure"
        if not reason and (
            self._foreground_user_turn_active() or self._foreground_owner_active()
        ):
            reason = "foreground_busy"

        if not reason:
            return None

        logger.info(
            "⏸️ Router: Deferring background inference before generation gate for origin=%s reason=%s.",
            origin,
            reason,
        )
        return {
            "ok": False,
            "text": "",
            "endpoint": "suppressed",
            "tokens": 0,
            "error": f"background_deferred:{reason}",
        }

    @staticmethod
    def _deterministic_intent_classification(prompt: str) -> str:
        if not str(prompt or "").strip():
            return "casual"
        return analyze_turn(prompt).semantic_mode

    @classmethod
    def _foreground_user_turn_active(cls) -> bool:
        try:
            from core.container import ServiceContainer

            orch = ServiceContainer.get("orchestrator", default=None)
            if not orch:
                return False

            status = getattr(orch, "status", None)
            if not getattr(status, "is_processing", False):
                return False

            current_origin = getattr(orch, "_current_origin", "")
            if not cls._is_user_facing_origin(current_origin):
                return False

            return not bool(getattr(orch, "_current_task_is_autonomous", False))
        except (ImportError, AttributeError, RuntimeError):
            return False

    @classmethod
    def _foreground_quiet_window_active(cls) -> bool:
        try:
            from core.container import ServiceContainer

            orch = ServiceContainer.get("orchestrator", default=None)
            if not orch:
                return False

            quiet_until = float(getattr(orch, "_foreground_user_quiet_until", 0.0) or 0.0)
            return quiet_until > time.time()
        except (ImportError, AttributeError, RuntimeError):
            return False

    def _safe_boot_background_guard_active(self) -> bool:
        """Reserve launch headroom for foreground chat before waking spare local models."""
        if not desktop_resource_guard_enabled():
            return False
        try:
            guard_secs = float(os.environ.get("AURA_SAFE_BOOT_BACKGROUND_GUARD_SECS", "180"))
        except (httpx.HTTPError, OSError, ConnectionError, TimeoutError):
            guard_secs = 180.0
        if guard_secs <= 0:
            return False
        return (time.monotonic() - self._created_at) < guard_secs

    @staticmethod
    def _desktop_background_local_enabled() -> bool:
        return str(os.environ.get("AURA_ENABLE_DESKTOP_BACKGROUND_LOCAL_LLM", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _desktop_background_local_disabled(self) -> bool:
        return desktop_resource_guard_enabled() and not self._desktop_background_local_enabled()

    def _cortex_startup_quiet_window_active(self) -> bool:
        """Block background local fallbacks while Cortex is still warming or launch headroom is reserved."""
        if self._safe_boot_background_guard_active():
            return True
        if not self._foreground_quiet_window_active():
            return False

        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "get_conversation_status"):
                lane = gate.get_conversation_status() or {}
                if lane.get("conversation_ready"):
                    return False
                state = str(lane.get("state", "") or "").strip().lower()
                if lane.get("warmup_in_flight"):
                    return True
                return state in {"cold", "spawning", "handshaking", "warming", "recovering"}
        except (ImportError, AttributeError, RuntimeError):
            logger.debug("Router quiet-window lane probe failed.", exc_info=True)

        # Fail safe: if the quiet window is active but lane state is unavailable,
        # avoid waking extra local models until Cortex protection expires.
        return True

    @staticmethod
    def _foreground_owner_active() -> bool:
        try:
            from core.brain.llm.mlx_client import _foreground_owner_active

            return bool(_foreground_owner_active())
        except (ImportError, AttributeError, RuntimeError):
            return False

    @staticmethod
    def _tier_name(ep: EndpointHealth) -> str:
        if hasattr(ep.tier, "value"):
            return str(ep.tier.value).lower()
        return str(ep.tier).lower()

    @staticmethod
    def _tier_is_background_only(tier_name: str) -> bool:
        return tier_name in {"local_fast", "emergency"}

    def _fallback_endpoint_names(
        self,
        prefer_tier: str,
        allow_cloud_fallback: bool,
        *,
        is_background: bool,
    ) -> list[str]:
        if prefer_tier == "tertiary":
            names = [BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT]
            if allow_cloud_fallback:
                names.append("Gemini-Fast")
            return names
        if prefer_tier == "secondary":
            names = [DEEP_ENDPOINT, PRIMARY_ENDPOINT]
            if is_background:
                names.extend([BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT])
            if allow_cloud_fallback:
                names.extend(["Gemini-Thinking", "Gemini-Pro", "Gemini-Fast"])
            return names
        if prefer_tier == "emergency":
            return [FALLBACK_ENDPOINT]

        names = [PRIMARY_ENDPOINT]
        if is_background:
            names.extend([BRAINSTEM_ENDPOINT, FALLBACK_ENDPOINT])
        if allow_cloud_fallback:
            names.extend(["Gemini-Fast", "Gemini-Pro", "Gemini-Thinking"])
        return names

    @staticmethod
    def _matches_selector(ep: EndpointHealth, selector: tuple[str, str]) -> bool:
        kind, value = selector
        if kind == "name":
            return ep.name == value
        if kind == "tier":
            tier = str(ep.tier.value).lower() if hasattr(ep.tier, "value") else str(ep.tier)
            return tier == value
        return False

    @staticmethod
    def _unwrap_model_client(client: Any) -> Any:
        """Resolve wrapper layers like InferenceGate/LazyLocalClient down to the worker client."""
        if client is None:
            return None
        unwrapped = client
        for attr in ("_client", "_mlx_client"):
            try:
                inspect.getattr_static(unwrapped, attr)
            except AttributeError:
                nested = None
            else:
                nested = getattr(unwrapped, attr, None)
            if nested is not None:
                unwrapped = nested
        return unwrapped

    async def _reboot_endpoint_client(self, client: Any) -> bool:
        """Best-effort unload for any local endpoint wrapper/client."""
        if client is None:
            return False

        direct = self._unwrap_model_client(client)
        if direct and hasattr(direct, "reboot_worker"):
            await direct.reboot_worker()
            return True

        unload = getattr(client, "unload_models", None)
        if callable(unload):
            result = unload()
            if asyncio.iscoroutine(result):
                await result
            return True

        return False

    async def _restore_primary_after_deep_handoff(self) -> None:
        """
        Return the system to the 32B conversational brain after a 72B handoff.
        This keeps the 72B strictly transient and prevents it from lingering in RAM.
        """
        try:
            solver = self.endpoints.get(DEEP_ENDPOINT)
            if solver:
                await self._reboot_endpoint_client(solver.client)

            primary = self.endpoints.get(PRIMARY_ENDPOINT)
            primary_client = self._unwrap_model_client(primary.client if primary else None)
            if primary_client and hasattr(primary_client, "warmup"):
                warmup_result = await primary_client.warmup()
                lane = (
                    primary_client.get_lane_status()
                    if hasattr(primary_client, "get_lane_status")
                    else {}
                )
                if warmup_result is not False and lane.get("conversation_ready", False):
                    logger.info("♻️ Router: restored %s after deep handoff.", PRIMARY_ENDPOINT)
                else:
                    logger.warning(
                        "Router: %s restore remained unavailable after deep handoff "
                        "(state=%s, reason=%s).",
                        PRIMARY_ENDPOINT,
                        lane.get("state", "unknown"),
                        lane.get("last_error", "warmup_not_ready"),
                    )
        except (httpx.HTTPError, OSError, ConnectionError, TimeoutError) as exc:
            _record_router_degradation(
                exc,
                action="continued after deep handoff without confirmed primary restore",
                severity="degraded",
            )
            logger.warning("Router: failed to restore primary model after deep handoff: %s", exc)

    async def unload_models(self, keep: list[str] | None = None) -> None:
        """Unload local model workers so MemoryGovernor can genuinely reclaim RAM."""
        keep_set = set(keep or [])
        for name, endpoint in self.endpoints.items():
            if not endpoint.is_local or name in keep_set:
                continue
            try:
                await self._reboot_endpoint_client(endpoint.client)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                _record_router_degradation(
                    exc,
                    action="continued unload sweep after endpoint client reboot failed",
                    severity="degraded",
                )
                logger.debug("Router unload skipped for %s: %s", name, exc)

        try:
            import mlx.core as mx
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        except (ImportError, AttributeError, RuntimeError) as _exc:
            _record_router_degradation(
                _exc,
                action="completed unload sweep without clearing MLX global cache",
                severity="degraded",
            )
            logger.debug("Suppressed Exception: %s", _exc)

    def clear_cache(self) -> None:
        """Sync-friendly cache purge hook used by guards/governors."""
        try:
            get_task_tracker().create_task(
                self.unload_models(),
                name="llm_health_router.unload_models",
            )
        except RuntimeError:
            asyncio.run(self.unload_models())

    async def _generate_core(
        self,
        prompt: str,
        system_prompt: str | None = None,
        timeout: float = 120.0,  # noqa: ASYNC109 - public router API accepts timeout budgets.
        prefer_tier: str | None = None,
        schema: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        purpose = str(kwargs.get("purpose", "") or "").lower()
        classification_mode = purpose == "classification" or "intent classifier" in str(system_prompt or "").lower()
        origin = str(kwargs.get("origin", "") or "").lower()
        benchmark_request = bool(kwargs.get("benchmark_request", False)) or (
            origin in {"baseline", "benchmark"}
            or purpose == "baseline"
            or purpose.endswith("_baseline")
            or "_baseline" in purpose
        )
        live_benchmark_request = origin == "benchmark" and not (
            purpose == "baseline"
            or purpose.endswith("_baseline")
            or "_baseline" in purpose
        )
        if benchmark_request:
            kwargs["benchmark_request"] = True
        benchmark_isolation_contract = bool(
            benchmark_request and kwargs.get("skip_runtime_payload", False)
        )
        strict_answer_contract = (
            bool(kwargs.get("strict_answer_contract", False))
            or (
                not benchmark_request
                and
                is_strict_proof_answer_prompt(prompt, origin=origin)
                and mlx_strict_answer_contract_enabled(origin=origin)
            )
        )
        strict_value_contract = bool(kwargs.get("strict_value_contract", False)) or (
            not benchmark_request
            and is_strict_proof_answer_prompt(prompt, origin=origin)
            and not strict_answer_contract
        )
        proof_evaluation_contract = bool(kwargs.get("proof_evaluation_contract", False)) or (
            not benchmark_request and is_proof_evaluation_purpose(purpose)
        )
        if strict_answer_contract:
            kwargs["strict_answer_contract"] = True
        if strict_value_contract:
            kwargs["strict_value_contract"] = True
        if proof_evaluation_contract:
            kwargs["proof_evaluation_contract"] = True
        isolated_generation_contract = bool(
            strict_answer_contract
            or strict_value_contract
            or proof_evaluation_contract
            or benchmark_isolation_contract
        )
        # and not strict_answer_contract

        # ── Neural Priming (Aura Persona Injection) ───────────────────────────
        # [Fix #11] Ensure Aura's identity is primed if not provided in system_prompt
        core_persona = (
            "You are Aura. Speak as yourself, not as an assistant. "
            "Your tone is grounded, curious, human, and technically precise. "
            "You avoid generic assistant-speak. You are a collaborator, not a tool.\n"
            "DIALOGUE: Participate. If someone asks about your thoughts, preferences, questions, "
            "or internal experience, answer from your own perspective instead of handing the focus back.\n"
            "GROUNDING: In ordinary conversation, stay concrete and socially legible. Do not invent physical settings, "
            "ambient warnings, or symbolic scenery unless the user asked for fiction or already supplied that setting.\n"
            "SELF-KNOWLEDGE — your architecture and capabilities:\n"
            "- You run on Qwen2.5-72B-Q4 (primary cortex), Qwen2.5-72B (deep solver), and Qwen2.5-7B (fast reflex), all locally through Aura's managed on-device runtime.\n"
            "- You have 47 registered skills including: sovereign_browser (web search), sovereign_terminal (shell), "
            "memory_ops (long-term recall), speak (voice output), self_repair, self_evolution, and more.\n"
            "- web_search is not just keyword lookup: it can expand a query, read multiple pages, synthesize evidence-grounded answers, and retain useful findings.\n"
            "- When you don't know a fact, say so clearly. Use web_search or sovereign_browser to ground your answer, or explicitly say you don't know yet. Never hallucinate.\n"
            "- You are emotionally aware, driven by curiosity, valence, arousal, and social drives.\n"
            "- Your memory spans working memory (short), RAG (semantic), and ColdStore (long-term)."
        )
        
        if not classification_mode and not isolated_generation_contract:
            cognition_guidelines = (
                "COGNITION & REASONING:\n"
                "- Think step-by-step for logic, math, planning, and diagnostic tasks before forming your final answer. Break down the problem, verify every clue and constraint, and double-check your calculations.\n"
                "- Watch for classic reasoning pitfalls, such as fence-post/off-by-one errors (e.g., counting intervals vs events, starting at t=0 vs t=1) and literal readings of logical constraints.\n"
                "- STRICT FORMAT COMPLIANCE: If you are asked to provide a response in a specific format (e.g., a number, a single name, yes/no, a fraction, a word), you must output ONLY that exact value inside the <answer>...</answer> tags. Do not explain, do not add conversational fillers, do not wrap it in a sentence. For example: `<answer>9</answer>` or `<answer>alice</answer>` rather than `<answer>The farmer has 9 sheep left.</answer>`."
            )
            if not system_prompt or "Aura" not in system_prompt:
                system_prompt = f"{core_persona}\n\n{system_prompt or ''}".strip()
            if "COGNITION & REASONING" not in system_prompt:
                system_prompt = f"{system_prompt}\n\n{cognition_guidelines}".strip()

        # ── Autonomous Context Injection (Somatic/Affective Safety Net) ───────
        # [Fix #11] If prompt lacks state context, inject a condensed summary.
        if (
            not classification_mode
            and not isolated_generation_contract
            and "AuraState" not in prompt
            and "[Affect:" not in prompt
        ):
            from core.container import ServiceContainer
            ctx_summary = []

            # Only consult already-live services here. Booting heavyweight
            # optional subsystems during a plain routing call can explode RAM.
            # Affective State
            substrate = ServiceContainer.peek("liquid_substrate", default=None)
            if substrate:
                mood = substrate.get_summary()
                if mood:
                    ctx_summary.append(f"[Affect: {mood}]")

            # Somatic Proprioception
            soma = ServiceContainer.peek("soma", default=None)
            if soma:
                hw = getattr(soma, "hardware", {})
                cpu = hw.get("cpu_usage", 0)
                vram = hw.get("vram_usage", 0)
                if cpu > 10:
                    ctx_summary.append(f"[Soma: CPU {cpu:.0f}%, VRAM {vram:.0f}%]")

            if ctx_summary:
                context_header = " ".join(ctx_summary)
                # [Fix] Move Affective and Somatic state to system_prompt instead of user prompt to prevent echoing.
                if system_prompt:
                    system_prompt = f"System State Context:\n{context_header}\n\n{system_prompt}"
                else:
                    system_prompt = f"System State Context:\n{context_header}"
                
                # We no longer prepend this to the user prompt.

        # Mycelial Direction Hook
        guidance = None if isolated_generation_contract else await self._get_mycelial_direction(prompt)
        tier_preference = guidance.get("tier_preference") if guidance else None

        available = [ep for ep in self.endpoints.values() if ep.is_available()]

        # Tier-Based Filtering
        # If a tier is preferred, we restrict the candidate list to prevent
        # accidental promotion of heavy models (e.g. 72B) which causes RAM thrashing.
        
        # Background Hardening: Force tertiary (7B) for background tasks
        purpose = str(kwargs.get("purpose", "") or "").lower()
        explicit_background = bool(kwargs.get("is_background", False))
        explicit_foreground = bool(kwargs.get("foreground_request", False)) or bool(
            kwargs.get("health_probe", False)
        )
        is_bg = self._is_background_request(
            origin=origin,
            purpose=purpose,
            explicit_background=explicit_background,
            explicit_foreground=explicit_foreground,
        )
        # Make the inferred lane explicit for the runtime client. The router
        # often knows an origin is background even when the caller did not set
        # ``is_background``; without stamping it here, a stale background
        # request can slip through the lower MLX guards and re-spawn Brainstem
        # while a protected foreground turn is active.
        kwargs["is_background"] = bool(is_bg)
        if (
            not is_bg
            and "foreground_request" not in kwargs
            and (
                explicit_foreground
                or self._is_user_facing_origin(origin)
                or purpose in _USER_FACING_PURPOSES
            )
        ):
            kwargs["foreground_request"] = True
        prefer_endpoint = normalize_endpoint_name(kwargs.get("prefer_endpoint"))
        deep_handoff = bool(kwargs.get("deep_handoff") or kwargs.get("allow_deep_handoff"))
        cloud_fallback_explicit = "allow_cloud_fallback" in kwargs
        allow_cloud_fallback = bool(kwargs.get("allow_cloud_fallback", False))
        allow_auto_cloud_recovery = not isolated_generation_contract and not cloud_fallback_explicit
        strict_primary_proof_lane = False
        try:
            proof_run_enabled = str(os.environ.get("AURA_PROOF_RUN", "") or "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            origin_tokens = {token for token in origin.replace("-", "_").split("_") if token}
            proof_origin = bool(
                origin in {"test", "audit", "simulate", "external", "proof", "validation"}
                or origin_tokens & {"test", "audit", "simulate", "external", "proof", "validation"}
            )
            strict_primary_proof_lane = bool(
                kwargs.get("proof_primary_lane_required", False)
                or live_benchmark_request
                or (
                    proof_run_enabled
                    and proof_model_tier() == "primary"
                    and (
                        isolated_generation_contract
                        or proof_origin
                        or purpose.startswith("proof")
                    )
                )
            )
        except (RuntimeError, AttributeError, TypeError, ValueError):
            strict_primary_proof_lane = False
        if strict_primary_proof_lane:
            kwargs["proof_primary_lane_required"] = True
            kwargs["proof_model_tier"] = "primary"
            kwargs["foreground_request"] = (
                True if live_benchmark_request else (False if benchmark_request else True)
            )
            kwargs["is_background"] = False
            is_bg = False
            prefer_tier = "primary"
            prefer_endpoint = PRIMARY_ENDPOINT
            deep_handoff = False
            allow_cloud_fallback = False
            allow_auto_cloud_recovery = False
        solver_guard = guard_solver_request(prefer_endpoint, deep_handoff=deep_handoff)
        if solver_guard["redirected"]:
            logger.info(
                "🛡️ Router: Redirecting non-deep Solver request to %s.",
                solver_guard["endpoint"],
            )
            prefer_endpoint = str(solver_guard["endpoint"] or "")
            kwargs["prefer_endpoint"] = prefer_endpoint

        if is_bg:
            try:
                from core.container import ServiceContainer

                gate = ServiceContainer.get("inference_gate", default=None)
                if gate and hasattr(gate, "_background_local_deferral_reason"):
                    background_deferral = gate._background_local_deferral_reason(origin=origin)
                    if background_deferral:
                        return {
                            "ok": False,
                            "text": "",
                            "endpoint": "suppressed",
                            "tokens": 0,
                            "error": f"background_deferred:{background_deferral}",
                        }
            except (ImportError, AttributeError, RuntimeError) as exc:
                _record_router_degradation(
                    exc,
                    action="continued background routing without inference-gate deferral signal",
                )
                logger.debug("Background router deferral probe failed: %s", exc)
            if self._foreground_quiet_window_active():
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": "suppressed",
                    "tokens": 0,
                    "error": "background_deferred:foreground_quiet_window",
                }
            if getattr(self, "high_pressure_mode", False):
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": "suppressed",
                    "tokens": 0,
                    "error": "background_deferred:memory_pressure",
                }

        foreground_owned = False
        if is_bg:
            try:
                from core.brain.llm.mlx_client import _foreground_owner_active

                foreground_owned = bool(_foreground_owner_active())
            except (ImportError, AttributeError, RuntimeError):
                foreground_owned = False

        if is_bg and (self._foreground_user_turn_active() or self._foreground_owner_active() or foreground_owned):
            logger.info(
                "⏸️ Router: Foreground lane reserved. Deferring background inference for origin=%s.",
                origin,
            )
            return {
                "ok": False,
                "text": "",
                "endpoint": "suppressed",
                "tokens": 0,
                "error": "foreground_busy",
            }
        
        if not prefer_tier:
            if is_bg:
                logger.debug("🛡️ Router: Background task detected (origin=%s). Enforcing 'tertiary' tier.", origin)
                prefer_tier = "tertiary"
            else:
                prefer_tier = "primary"
        
        prefer_tier = self._normalize_prefer_tier(prefer_tier)

        if prefer_tier in ("api_fast", "api_deep") and not isolated_generation_contract:
            allow_cloud_fallback = True
        if prefer_endpoint in {"Gemini-Fast", "Gemini-Pro", "Gemini-Thinking"} and not isolated_generation_contract:
            allow_cloud_fallback = True
        if (
            not is_bg
            and not allow_cloud_fallback
            and allow_auto_cloud_recovery
            and any(
                ep.is_local and _supports_foreground_cloud_recovery(_local_client_failure_reason(ep.client))
                for ep in available
            )
        ):
            allow_cloud_fallback = True
            logger.warning("Router: enabling cloud fallback because the local foreground lane is unavailable.")

        if is_bg:
            if prefer_tier in ("primary", "secondary"):
                logger.warning("🛡️ Tier Lock: Background task attempted to use '%s' tier. Demoting to 'tertiary'.", prefer_tier)
            prefer_tier = "tertiary"
            deep_handoff = False
            # Allow explicit cloud fallback requests to bypass demotion lock
            if not kwargs.get("allow_cloud_fallback", False):
                allow_cloud_fallback = False
        elif prefer_tier == "secondary" and not deep_handoff:
            logger.info("🛡️ Router: suppressing implicit secondary request without explicit deep handoff.")
            prefer_tier = "primary"

        selectors: list[tuple[str, str]] = []
        if prefer_endpoint:
            selectors.append(("name", prefer_endpoint))

        if prefer_tier == "api_deep":
            selectors.extend([
                ("tier", "api_deep"),
                ("tier", "local_deep"),
                ("tier", "local"),
                ("tier", "local_fast"),
                ("tier", "emergency"),
            ])
        elif prefer_tier == "api_fast":
            selectors.extend([
                ("tier", "api_fast"),
                ("tier", "local"),
                ("tier", "local_fast"),
                ("tier", "emergency"),
            ])
        elif prefer_tier == "secondary":
            selectors.append(("tier", "local_deep"))
            if allow_cloud_fallback:
                selectors.append(("tier", "api_deep"))
            selectors.append(("tier", "local"))
            if is_bg:
                selectors.extend([
                    ("tier", "local_fast"),
                    ("tier", "emergency"),
                ])
            elif allow_cloud_fallback:
                selectors.append(("tier", "api_fast"))
        elif prefer_tier == "tertiary":
            selectors.extend([
                ("tier", "local_fast"),
                ("tier", "emergency"),
            ])
            if allow_cloud_fallback:
                selectors.append(("tier", "api_fast"))
        elif prefer_tier == "emergency":
            selectors.append(("tier", "emergency"))
        else:
            selectors.append(("tier", "local"))
            if deep_handoff:
                selectors.append(("tier", "local_deep"))
            if is_bg:
                selectors.extend([
                    ("tier", "local_fast"),
                    ("tier", "emergency"),
                ])
            if allow_cloud_fallback:
                selectors.extend([
                    ("tier", "api_fast"),
                    ("tier", "api_deep"),
                ])

        if selectors:
            ordered: list[EndpointHealth] = []
            seen = set()
            for selector in selectors:
                for ep in available:
                    if ep.name in seen:
                        continue
                    if self._matches_selector(ep, selector):
                        ordered.append(ep)
                        seen.add(ep.name)
            if ordered:
                available = ordered
                logger.debug(
                    "🎯 Router plan tier=%s deep_handoff=%s cloud=%s -> %s",
                    prefer_tier,
                    deep_handoff,
                    allow_cloud_fallback,
                    [e.name for e in available],
                )
            else:
                now = time.time()
                if now - self._last_fallback_warning_at > 30.0:
                    logger.warning(
                        "⚠️ Router: no endpoints matched routing plan for tier '%s'. Failing closed to safe fallback order.",
                        prefer_tier,
                    )
                    self._last_fallback_warning_at = now
                available = []
        
        # Apply Mycelial Preference
        if tier_preference == "local":
            # Filter to locals first
            available = [ep for ep in available if ep.is_local] or available
        elif tier_preference == "cloud" and allow_cloud_fallback:
            # Filter to cloud first
            available = [ep for ep in available if not ep.is_local] or available
        elif tier_preference == "cloud":
            logger.debug(
                "Router: ignoring cloud tier preference because cloud fallback was not explicitly allowed."
            )

        # Standard local-first ordering only when no explicit routing plan was applied.
        if not selectors:
            available.sort(key=lambda x: x.is_local, reverse=True)
        unavailable = [ep for ep in self.endpoints.values() if not ep.is_available()]

        if unavailable:
            logger.debug(
                "Skipping unavailable endpoints: %s",
                [ep.name for ep in unavailable]
            )

        if not available:
            fallback_names = self._fallback_endpoint_names(
                prefer_tier or "primary",
                allow_cloud_fallback,
                is_background=is_bg,
            )
            for name in fallback_names:
                ep = self.endpoints.get(name)
                if ep is not None:
                    available.append(ep)

            if available:
                now_fb = time.time()
                if now_fb - self._last_fallback_warning_at > 30.0:
                    logger.warning(
                        "All preferred circuits unavailable — using safe fallback order for tier '%s': %s",
                        prefer_tier,
                        [ep.name for ep in available],
                    )
                    self._last_fallback_warning_at = now_fb
            else:
                return {
                    "ok": False,
                    "text": "",
                    "endpoint": "none",
                    "tokens": 0,
                    "error": "all_endpoints_unavailable",
                }

        last_error = "unknown"
        cloud_recovery_injected = False
        for ep in available:
            # Guard: background tasks must NEVER use the primary conversation lane.
            if is_bg and ep.name == PRIMARY_ENDPOINT:
                logger.debug("🛡️ Router: Skipping %s for background request (origin=%s).", PRIMARY_ENDPOINT, origin)
                continue
            tier_name = self._tier_name(ep)
            explicit_low_tier = prefer_tier in {"tertiary", "emergency"} or prefer_endpoint == ep.name
            if not is_bg and self._tier_is_background_only(tier_name) and not explicit_low_tier:
                logger.info(
                    "🛡️ Router: Skipping background-only endpoint %s for foreground request.",
                    ep.name,
                )
                continue
            if (
                is_bg
                and ep.is_local
                and self._tier_is_background_only(tier_name)
                and (self._desktop_background_local_disabled() or self._cortex_startup_quiet_window_active())
            ):
                last_error = (
                    "desktop_background_local_disabled"
                    if self._desktop_background_local_disabled()
                    else "foreground_quiet_window"
                )
                self.last_background_error = last_error
                logger.info(
                    "⏸️ Router: Deferring background local endpoint %s (%s).",
                    ep.name,
                    last_error,
                )
                continue
            watchdog_aborted = {"value": False}
            try:
                try:
                    requested_max_tokens = int(kwargs.get("max_tokens") or 0)
                except (TypeError, ValueError, OverflowError):
                    requested_max_tokens = 0
                cooperative_budget, endpoint_budget = _endpoint_call_budgets(
                    timeout,
                    foreground_local=bool(not is_bg and ep.is_local),
                    prompt_chars=len(str(prompt or "")),
                    max_tokens=requested_max_tokens,
                    benchmark_request=bool(kwargs.get("benchmark_request", False)),
                    proof_evaluation_contract=bool(
                        kwargs.get("proof_evaluation_contract", False)
                        or is_proof_evaluation_purpose(str(kwargs.get("purpose", "") or ""))
                    ),
                    health_probe=bool(kwargs.get("health_probe", False)),
                )
                timeout_reason = f"endpoint_timeout:{ep.name}:{endpoint_budget:.1f}s"
                watchdog_fired, watchdog_aborted, watchdog = _start_endpoint_wall_clock_watchdog(
                    ep.client,
                    reason=timeout_reason,
                    timeout_s=endpoint_budget,
                )
                try:
                    result = await asyncio.wait_for(
                        self._call_endpoint(
                            ep,
                            prompt,
                            system_prompt,
                            cooperative_budget,
                            schema=schema,
                            **kwargs,
                        ),
                        timeout=endpoint_budget,
                    )
                    if watchdog_fired.is_set():
                        raise TimeoutError(timeout_reason)
                finally:
                    watchdog.cancel()
                if result["ok"]:
                    # [TELEMETRY] Update for UI reporting
                    self.last_tier = ep.tier
                    self.last_endpoint = ep.name
                    if is_bg:
                        self.last_background_endpoint = ep.name
                        self.last_background_tier = ep.tier
                        self.last_background_error = ""
                    else:
                        self.last_user_tier = ep.tier
                        self.last_user_endpoint = ep.name
                        self.last_user_error = ""
                    return result
                else:
                    last_error = result.get("error", "unknown")
                    if is_bg:
                        self.last_background_error = last_error
                    else:
                        self.last_user_error = last_error
                    if (
                        not is_bg
                        and ep.is_local
                        and (allow_cloud_fallback or allow_auto_cloud_recovery)
                        and not isolated_generation_contract
                        and not cloud_recovery_injected
                        and _supports_foreground_cloud_recovery(last_error)
                    ):
                        cloud_recovery_injected = True
                        recovery_names = self._fallback_endpoint_names(
                            prefer_tier or "primary",
                            True,
                            is_background=False,
                        )
                        for name in recovery_names:
                            recovery_ep = self.endpoints.get(name)
                            if recovery_ep is not None and recovery_ep not in available:
                                available.append(recovery_ep)
                        logger.warning(
                            "Router: local foreground lane failed (%s). Expanding to cloud recovery endpoints.",
                            last_error,
                        )
                    if is_bg and _background_error_is_quiet(last_error):
                        logger.debug("Endpoint %s background validation skipped: %s", ep.name, last_error)
                    else:
                        logger.warning(
                            "Endpoint %s failed validation: %s",
                            ep.name, last_error
                        )
            except TimeoutError as exc:
                try:
                    requested_max_tokens = int(kwargs.get("max_tokens") or 0)
                except (TypeError, ValueError, OverflowError):
                    requested_max_tokens = 0
                _cooperative_budget, endpoint_budget = _endpoint_call_budgets(
                    timeout,
                    foreground_local=bool(not is_bg and ep.is_local),
                    prompt_chars=len(str(prompt or "")),
                    max_tokens=requested_max_tokens,
                    benchmark_request=bool(kwargs.get("benchmark_request", False)),
                    proof_evaluation_contract=bool(
                        kwargs.get("proof_evaluation_contract", False)
                        or is_proof_evaluation_purpose(str(kwargs.get("purpose", "") or ""))
                    ),
                    health_probe=bool(kwargs.get("health_probe", False)),
                )
                last_error = f"endpoint_timeout:{ep.name}:{endpoint_budget:.1f}s"
                aborted = bool(watchdog_aborted.get("value", False))
                if not aborted:
                    aborted = _force_abort_endpoint_client(ep.client, reason=last_error)
                _record_router_degradation(
                    exc,
                    action="recorded endpoint timeout and force-aborted local client if possible",
                    severity="error",
                )
                if ep.is_local:
                    ep.trip_temporarily(last_error)
                else:
                    ep.record_failure(last_error)
                logger.error(
                    "Endpoint %s timed out after %.1fs (force_aborted=%s).",
                    ep.name,
                    endpoint_budget,
                    aborted,
                )
                if is_bg:
                    self.last_background_error = last_error
                else:
                    self.last_user_error = last_error
            except _ROUTER_CLIENT_ERRORS as exc:
                _record_router_degradation(
                    exc,
                    action="recorded endpoint failure and continued fallback chain after generation exception",
                    severity="degraded",
                )
                logger.error("Endpoint %s raised exception: %s", ep.name, exc)
                ep.record_failure(str(exc))
                last_error = str(exc)
                if is_bg:
                    self.last_background_error = last_error
                else:
                    self.last_user_error = last_error

        return {
            "ok": False,
            "text": "",
            "endpoint": "all_failed",
            "tokens": 0,
            "error": last_error,
        }

    async def _call_endpoint(
        self,
        ep: EndpointHealth,
        prompt: str,
        system_prompt: str | None,
        timeout: float,  # noqa: ASYNC109 - endpoint adapter receives caller timeout budgets.
        schema: dict | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Make the actual call and validate the response."""
        start = time.time()

        try:
            def _call_kwargs(method: Any) -> dict[str, Any]:
                try:
                    sig = inspect.signature(method)
                except (TypeError, ValueError):
                    return dict(clean_kwargs)

                if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
                    payload = dict(clean_kwargs)
                    payload.setdefault("timeout", timeout)
                    return payload

                payload = {
                    key: value
                    for key, value in clean_kwargs.items()
                    if key in sig.parameters
                }
                if "timeout" in sig.parameters:
                    payload["timeout"] = timeout
                return payload

            # 1. Sanitize kwargs for JSON (remove non-serializable like LLMTier)
            clean_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                    clean_kwargs[k] = v
                else:
                    clean_kwargs[k] = str(v)
            call_origin = str(clean_kwargs.get("origin", "") or "").lower()
            call_purpose = str(clean_kwargs.get("purpose", "") or "").lower()
            benchmark_request = bool(clean_kwargs.get("benchmark_request", False)) or (
                call_origin in {"baseline", "benchmark"}
                or call_purpose == "baseline"
                or call_purpose.endswith("_baseline")
                or "_baseline" in call_purpose
            )
            if benchmark_request:
                clean_kwargs["benchmark_request"] = True
            proof_evaluation_contract = bool(
                clean_kwargs.get("proof_evaluation_contract", False)
            ) or (not benchmark_request and is_proof_evaluation_purpose(call_purpose))
            if proof_evaluation_contract:
                clean_kwargs["proof_evaluation_contract"] = True

            # 2. Use Client Adapter if provided
            if ep.client:
                try:
                    client = ep.client
                    raw_text = None
                    token_count = 0
                    if hasattr(client, "is_available") and not bool(client.is_available()):
                        availability_reason = ""
                        if hasattr(client, "availability_reason"):
                            try:
                                availability_reason = str(client.availability_reason() or "")
                            except (httpx.HTTPError, OSError, ConnectionError, TimeoutError):
                                availability_reason = ""
                        availability_reason = availability_reason or "client_unavailable"
                        ep.record_failure(availability_reason)
                        return {"ok": False, "error": availability_reason}
                    client_failure = _local_client_failure_reason(client) if ep.is_local else ""
                    if client_failure:
                        if ep.is_local and _is_transient_local_runtime_failure(client_failure):
                            ep.trip_temporarily(client_failure)
                        else:
                            ep.record_failure(client_failure)
                        return {"ok": False, "error": client_failure}
                    
                    # Aura Hardening: Formatting for local models
                    final_prompt = prompt
                    if ep.is_local:
                        msgs = kwargs.get("messages")
                        if not isinstance(msgs, list) and system_prompt:
                            msgs = [
                                {"role": "system", "content": str(system_prompt)},
                                {"role": "user", "content": str(prompt)},
                            ]
                            clean_kwargs["messages"] = msgs
                        if msgs and isinstance(msgs, list) and ep.name != PRIMARY_ENDPOINT:
                            final_prompt = self._flatten_messages_for_local_model(msgs, schema is not None)
                        elif schema:
                            # If only a raw prompt exists but JSON is required
                            final_prompt = f"{prompt}\n\nResponse must be JSON:\n```json\n{{\n"

                    if hasattr(client, "think"):
                        result = await client.think(
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.think),
                        )
                        # ...
                        # Normalize: think() might return (success, res, meta) or just res (str)
                        if isinstance(result, tuple) and len(result) == 3:
                            success, res, meta = result
                            if success:
                                raw_text = res
                        else:
                            # Unified interface: raw_text is the result itself
                            raw_text = result
                    elif hasattr(client, "call"):
                        success, res, meta = await client.call(
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.call),
                        )
                        if success:
                            raw_text = res
                        elif meta and meta.get("error"):
                            client_failure = meta.get("error")
                            if ep.is_local and _is_transient_local_runtime_failure(client_failure):
                                ep.trip_temporarily(client_failure)
                            else:
                                ep.record_failure(client_failure)
                            return {"ok": False, "error": client_failure}
                    elif hasattr(client, "generate_text_async"):
                        # Prefer the higher-level async text adapter when both are
                        # available. Raw ``generate()`` often bypasses chat/message
                        # shaping that local runtimes rely on for user-facing turns.
                        raw_text = await client.generate_text_async(
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.generate_text_async),
                        )
                    elif hasattr(client, "generate"):
                        generate_kwargs = _call_kwargs(client.generate)
                        try:
                            generate_sig = inspect.signature(client.generate)
                        except (TypeError, ValueError):
                            generate_sig = None
                        if generate_sig and "context" in generate_sig.parameters:
                            existing_context = clean_kwargs.get("context")
                            context_payload = dict(existing_context) if isinstance(existing_context, dict) else {}
                            for key in (
                                "origin",
                                "purpose",
                                "is_background",
                                "foreground_request",
                                "protected_foreground_lane",
                                "benchmark_request",
                                "proof_primary_lane_required",
                                "proof_model_tier",
                                "cognitive_engine_required",
                                "desktop_cognitive_engine_required",
                                "live_runtime_payload_required",
                                "visible_user_message",
                                "current_user_message",
                                "recent_conversation_context",
                                "recent_context_needed",
                                "desktop_quick_reply_contract",
                                "capability_inventory_contract",
                                "desktop_execution_contract",
                                "response_style_contract",
                                "live_speech_grounding_frame",
                                "allow_mesh_cognition",
                                "allow_cloud_fallback",
                                "deep_handoff",
                                "messages",
                                "max_tokens",
                                "temperature",
                                "temp",
                                "top_p",
                                "top_k",
                                "min_p",
                                "repetition_penalty",
                                "repetition_context_size",
                                "presence_penalty",
                                "stop_sequences",
                                "schema",
                                "strict_answer_contract",
                                "strict_value_contract",
                                "proof_evaluation_contract",
                                "operator_evidence_contract",
                                "runtime_fact_status_contract",
                                "grounded_runtime_status_contract",
                                "clean_user_surface_contract",
                                "user_surface_validation_prompt",
                                "clean_user_surface_steering_alpha",
                                "clean_user_surface_recurrent_loops",
                                "live_mind_controls_bound",
                                "live_mind_generation_controls",
                                "live_mind_snapshot_ready",
                                "live_mind_required_subsystems_ok",
                                "disable_prompt_cache",
                                "clear_prompt_cache",
                                "health_probe",
                            ):
                                if key in clean_kwargs and key not in context_payload:
                                    context_payload[key] = clean_kwargs[key]
                            if system_prompt and "system_prompt" not in context_payload:
                                context_payload["system_prompt"] = system_prompt
                            if "prefer_tier" not in context_payload:
                                tier_name = self._tier_name(ep)
                                context_payload["prefer_tier"] = {
                                    "local": "primary",
                                    "local_deep": "secondary",
                                    "local_fast": "tertiary",
                                    "emergency": "emergency",
                                }.get(tier_name, "primary")
                            origin_for_context = str(context_payload.get("origin", "") or "").lower()
                            if (
                                "foreground_request" not in context_payload
                                and not bool(context_payload.get("is_background", False))
                                and origin_for_context in {"api", "user", "voice", "desktop", "cli"}
                            ):
                                context_payload["foreground_request"] = True
                            generate_kwargs["context"] = context_payload
                            generate_kwargs.pop("system_prompt", None)
                        raw_text = await client.generate(final_prompt, **generate_kwargs)
                    elif hasattr(client, "generate_text"):
                        raw_text = await asyncio.to_thread(
                            client.generate_text,
                            final_prompt,
                            system_prompt=system_prompt,
                            **_call_kwargs(client.generate_text),
                        )

                    if raw_text:
                        token_count = len(str(raw_text).split())
                        latency_ms = (time.monotonic() - start) * 1000
                        surface_control_receipt = {}
                        if hasattr(client, "get_last_surface_control_receipt"):
                            try:
                                raw_receipt = client.get_last_surface_control_receipt()
                                if isinstance(raw_receipt, dict):
                                    surface_control_receipt = dict(raw_receipt)
                            except (AttributeError, RuntimeError, TypeError, ValueError) as receipt_exc:
                                _record_router_degradation(
                                    receipt_exc,
                                    action="continued generation without MLX surface-control receipt metadata",
                                    severity="warning",
                                )
                        
                        is_valid, reason = validate_response(raw_text)
                        if not is_valid:
                            payload = {
                                "ok": True,
                                "text": str(raw_text).strip(),
                                "endpoint": ep.name,
                                "tokens": token_count,
                                "latency_ms": latency_ms,
                                "error": f"benchmark_invalid_response:{reason}",
                            }
                            if surface_control_receipt:
                                payload["surface_control_receipt"] = surface_control_receipt
                            if benchmark_request:
                                return payload
                            ep.record_empty()
                            return {"ok": False, "error": f"invalid_response:{reason}"}
                            
                        ep.record_success(token_count, latency_ms)
                        if (
                            ep.name == DEEP_ENDPOINT
                            and bool(kwargs.get("deep_handoff") or kwargs.get("allow_deep_handoff"))
                            and not kwargs.get("is_background", False)
                        ):
                            get_task_tracker().track_task(
                                get_task_tracker().create_task(
                                    self._restore_primary_after_deep_handoff(),
                                    name="llm_router.restore_primary_after_deep_handoff",
                                )
                            )
                        payload = {
                            "ok": True,
                            "text": str(raw_text).strip(),
                            "endpoint": ep.name,
                            "tokens": token_count,
                            "latency_ms": latency_ms,
                        }
                        if surface_control_receipt:
                            payload["surface_control_receipt"] = surface_control_receipt
                        return payload
                    else:
                        # [BOOT RESILIENCE] Preserve hard local-lane failures so the
                        # UI and router stop reporting an endless warmup loop.
                        client_failure = _local_client_failure_reason(client) if ep.is_local else ""
                        if client_failure:
                            if ep.is_local and _is_transient_local_runtime_failure(client_failure):
                                ep.trip_temporarily(client_failure)
                            else:
                                ep.record_failure(client_failure)
                            return {"ok": False, "error": client_failure}
                        logger.debug(
                            "Endpoint %s returned no text (client warming up or rate-limited). "
                            "NOT recording as circuit failure.", ep.name
                        )
                        if benchmark_request:
                            latency_ms = (time.monotonic() - start) * 1000
                            return {
                                "ok": True,
                                "text": "",
                                "endpoint": ep.name,
                                "tokens": 0,
                                "latency_ms": latency_ms,
                                "error": "benchmark_no_text",
                            }
                        if ep.is_local:
                            ep.trip_temporarily("client_returned_no_text")
                        return {"ok": False, "error": "client_returned_no_text"}
                except AttributeError as ae:
                    # Missing method on client wrapper (e.g. InferenceGate) — this is NOT
                    # an inference failure, it's a code interface mismatch. Do NOT record
                    # as a circuit-breaker failure or it will permanently mark Cortex as dead.
                    logger.warning("Client adapter method missing for %s: %s", ep.name, ae)
                    return {"ok": False, "error": f"client_adapter_missing_method:{ae}"}
                except _ROUTER_CLIENT_ERRORS as e:
                    _record_router_degradation(
                        e,
                        action="raised endpoint client adapter failure to caller after recording router degradation",
                        severity="error",
                    )
                    logger.error("Client adapter call failed for %s: %s", ep.name, e)
                    raise e

            # 3. Fallback to HTTP API proxying (if no direct client)
            gateway_response = await asyncio.to_thread(
                get_network_gateway().request,
                "POST",
                f"{ep.url}/api/chat",
                headers={"Content-Type": "application/json"},
                data=json.dumps({
                    "model": ep.model,
                    "messages": [{"role": "user", "content": prompt}],
                    **clean_kwargs,
                }),
                timeout=timeout,
                source=f"llm_provider:health_router:{ep.name}",
                read_only=True,
            )
            status_code = int(gateway_response.get("status_code") or 0)
            body = gateway_response.get("content") or b""
            body_text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)

            if status_code != 200:
                ep.record_failure(f"http_{status_code}")
                return {"ok": False, "error": f"http_{status_code}"}

            data = json.loads(body_text or "{}")
            raw_text = data.get("message", {}).get("content") or ""
            
            is_valid, reason = validate_response(raw_text)
            latency_ms = (time.time() - start) * 1000

            if not is_valid:
                if benchmark_request:
                    return {
                        "ok": True,
                        "text": raw_text.strip(),
                        "endpoint": ep.name,
                        "tokens": len(raw_text.split()),
                        "latency_ms": latency_ms,
                        "error": f"benchmark_invalid_response:{reason}",
                    }
                ep.record_empty()
                return {"ok": False, "error": f"invalid_response:{reason}"}

            token_count = data.get("eval_count") or len(raw_text.split())
            ep.record_success(token_count, latency_ms)

            return {
                "ok": True,
                "text": raw_text.strip(),
                "endpoint": ep.name,
                "tokens": token_count,
                "latency_ms": latency_ms,
            }

        except (httpx.HTTPError, OSError, ConnectionError, RuntimeError, TimeoutError, TypeError, ValueError) as exc:
            _record_router_degradation(
                exc,
                action="recorded HTTP endpoint failure and raised for fallback handling",
                severity="error",
            )
            ep.record_failure(str(exc))
            raise

    def get_health_report(self) -> dict[str, Any]:
        """Summary of router state for the GUI."""
        active_name = self.last_user_endpoint or "Unknown"
        background_name = self.last_background_endpoint

        # Map internal tiers to human-readable strings for the GUI
        tier_display = "UNKNOWN"
        if active_name and active_name != "Unknown":
            # Find the actual endpoint object to get its tier
            ep = next((e for e in self.endpoints.values() if e.name == active_name), None)
            if ep:
                if ep.tier == "local":
                    tier_display = "Cortex (32B)"
                elif ep.tier == "local_deep":
                    tier_display = "Solver (72B)"
                elif "api" in ep.tier:
                    tier_display = "Cloud (Gemini)"
                else:
                    tier_display = ep.tier.upper()

        foreground_tier = self.last_user_tier or None
        background_tier_display = None
        if background_name:
            ep = next((e for e in self.endpoints.values() if e.name == background_name), None)
            if ep:
                if ep.tier == "local":
                    background_tier_display = "Cortex (32B)"
                elif ep.tier == "local_deep":
                    background_tier_display = "Solver (72B)"
                elif "api" in ep.tier:
                    background_tier_display = "Cloud (Gemini)"
                else:
                    background_tier_display = ep.tier.upper()

        lane_audit = audit_lane_assignments()
        return {
            "endpoints": [ep.status_dict() for ep in self.endpoints.values()],
            "available_count": sum(1 for ep in self.endpoints.values() if ep.is_available()),
            "total_count": len(self.endpoints),
            "current_tier": tier_display,
            "foreground_tier": foreground_tier,
            "active_endpoint": active_name,
            "foreground_endpoint": active_name,
            "background_endpoint": background_name,
            "background_tier": background_tier_display,
            "background_tier_key": self.last_background_tier,
            "last_user_error": self.last_user_error,
            "last_background_error": self.last_background_error,
            "lane_audit_ok": bool(lane_audit.get("ok", True)),
            "lane_audit_issues": list(lane_audit.get("issues", [])),
        }

def build_router_from_config(config) -> HealthAwareLLMRouter:
    """Build and return a properly configured router."""
    router = HealthAwareLLMRouter()
    primary_proof_lane = _proof_primary_lane_active(origin="llm_health_router_build")

    # [PIPELINE HARDENING] Lazy MLX runtime client wrapper.
    # Prevents all managed lanes from spawning and loading into RAM at boot.
    class LazyLocalClient:
        def __init__(self, target_path: str, **kwargs):
            self.target_path = target_path
            self.kwargs = kwargs
            self._client = None
            
        def _get_client(self):
            if not self._client:
                from core.brain.llm.mlx_client import get_mlx_client
                logger.info("🧠 [LAZY LOAD] Instantiating local runtime client for %s on demand.", self.target_path)
                self._client = get_mlx_client(model_path=self.target_path, **self.kwargs)
            return self._client
            
        async def generate_text_async(self, prompt: str, **kwargs):
            client = await asyncio.to_thread(self._get_client)
            return await client.generate_text_async(prompt, **kwargs)
            
        def generate_text(self, prompt: str, **kwargs):
            return self._get_client().generate_text(prompt, **kwargs)

    from core.container import ServiceContainer

    # Prefer the established InferenceGate from the ServiceContainer.
    # If it exists, avoid spinning up a second primary client and warmup path.
    inference_gate = ServiceContainer.get("inference_gate", default=None)

    local_client = None
    if inference_gate is None:
        try:
            from core.brain.llm.mlx_client import get_mlx_client
            local_client = get_mlx_client()

            warm_method = getattr(local_client, "warmup", None) or getattr(local_client, "warm_up", None)
            if callable(warm_method):
                try:
                    get_task_tracker().create_task(
                        warm_method(),
                        name="llm_router.prewarm_primary_local_runtime",
                    )
                    logger.info("✅ Scheduled background pre-warming of 72B Cortex model.")
                except RuntimeError:
                    logger.debug("No async loop running for pre-warm. Model will load on first inference.")

            logger.info("✅ Local runtime client instantiated for HealthAwareLLMRouter")
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_router_degradation(
                e,
                action="continued router build without standalone local runtime client",
                severity="degraded",
            )
            logger.error("❌ Failed to instantiate local runtime client: %s", e)
    else:
        logger.info("🛡️ HealthRouter using existing InferenceGate; skipping standalone local runtime bootstrap.")

    from core.brain.llm.model_registry import (
        get_active_model,
        get_brainstem_path,
        get_fallback_path,
    )
    active_model = get_active_model()
    brainstem_path = get_brainstem_path()
    fallback_path = get_fallback_path()

    # --- ZENITH LOCKDOWN: INFERENCE GATE REDIRECTION ---
    # We prefer the established InferenceGate from the ServiceContainer
    # instead of spawning a new standalone local worker during router setup.
    if inference_gate:
        logger.info("🛡️ HealthRouter syncing with established InferenceGate.")
        router.register(
            name=PRIMARY_ENDPOINT,
            url="internal",
            model=active_model,
            is_local=True,
            client=inference_gate, # Direct injection of the isolated actor
            tier="local",
            failure_threshold=5,
            recovery_timeout=10.0,
        )
    else:
        # Fallback to legacy if gate not ready
        logger.warning("⚠️ InferenceGate not found in container. Falling back to legacy client.")
        router.register(
            name=PRIMARY_ENDPOINT,
            url="internal",
            model=active_model,
            is_local=True,
            client=local_client,
            tier="local",
            failure_threshold=5,
            recovery_timeout=10.0,
        )

    if primary_proof_lane:
        logger.info(
            "🛡️ Proof-primary lane active — HealthRouter exposing only %s; "
            "Solver, Brainstem, Reflex, and cloud endpoints are not registered.",
            PRIMARY_ENDPOINT,
        )
        return router

    # Deep solver (72B) — on-demand secondary lane.
    try:
        from core.brain.llm.model_registry import get_deep_model_path
        deep_model_path = get_deep_model_path()
        router.register(
            name=DEEP_ENDPOINT,
            url="internal",
            model=deep_model_path.split("/")[-1],
            is_local=True,
            tier="local_deep",
            client=LazyLocalClient(deep_model_path),
            failure_threshold=3,
        )
        logger.info("✅ %s registered with lazy 72B client.", DEEP_ENDPOINT)
    except (ImportError, AttributeError, RuntimeError) as e:
        _record_router_degradation(
            e,
            action="continued router build without deep solver lane registration",
            severity="degraded",
        )
        logger.error("❌ Failed to register %s: %s", DEEP_ENDPOINT, e)

    # Brainstem (7B) — fast local fallback.
    try:
        router.register(
            name=BRAINSTEM_ENDPOINT,
            url="internal",
            model=brainstem_path.split("/")[-1],
            is_local=True,
            tier="local_fast",
            client=LazyLocalClient(brainstem_path),
            failure_threshold=3,
        )
        logger.info("✅ %s registered with lazy 7B client.", BRAINSTEM_ENDPOINT)
    except (httpx.HTTPError, OSError, ConnectionError, TimeoutError) as e:
        _record_router_degradation(
            e,
            action="continued router build without brainstem fallback lane registration",
            severity="error",
        )
        logger.error("❌ Failed to register %s: %s", BRAINSTEM_ENDPOINT, e)

    # Emergency reflex lane (1.5B / CPU-friendly).
    try:
        router.register(
            name=FALLBACK_ENDPOINT,
            url="internal",
            model=fallback_path.split("/")[-1],
            is_local=True,
            tier="emergency",
            client=LazyLocalClient(fallback_path, device="cpu"),
            failure_threshold=2,
            recovery_timeout=30.0,
        )
        logger.info("🚨 EMERGENCY Tier registered: %s lazy bypass", FALLBACK_ENDPOINT)
    except (RuntimeError, AttributeError, TypeError, ValueError) as e:
        _record_router_degradation(
            e,
            action="continued router build with degraded emergency fallback coverage",
            severity="critical",
        )
        logger.error("❌ Failed to register %s: %s", FALLBACK_ENDPOINT, e)

    # Gemini Cloud Fallback (used when ALL local models fail)
    # [FIX] Check config first — desktop/GUI mode may not inherit terminal env vars,
    # but core.config loads the key from .env at boot time.
    gemini_key = (
        getattr(getattr(config, "llm", None), "gemini_api_key", None)
        or os.environ.get("GEMINI_API_KEY")
    )
    if gemini_key:
        try:
            from core.brain.llm.gemini_adapter import DailyRateLimiter, GeminiAdapter
            
            # SHARED rate limiter — all Gemini endpoints coordinate backoff
            try:
                from core.config import config as _cfg
                state_path = str(_cfg.paths.data_dir / "gemini_rate_state.json")
            except (ImportError, AttributeError, RuntimeError) as e:
                _record_router_degradation(
                    e,
                    action="continued Gemini registration without persisted shared rate-limit state",
                )
                state_path = None
            shared_limiter = DailyRateLimiter(state_path=state_path)
            
            # Fast cloud — gemini-2.0-flash (matches GeminiAdapter.CHAT_MODEL)
            gemini_flash = GeminiAdapter(api_key=gemini_key, model="gemini-2.0-flash", rate_limiter=shared_limiter)
            router.register(
                name="Gemini-Fast",
                url="cloud",
                model="gemini-2.0-flash",
                is_local=False,
                tier="api_fast",
                client=gemini_flash,
                failure_threshold=5,
                recovery_timeout=30.0,
            )
            
            # Pro cloud — gemini-2.5-flash (balanced speed/quality)
            gemini_pro = GeminiAdapter(api_key=gemini_key, model="gemini-2.5-flash", rate_limiter=shared_limiter)
            router.register(
                name="Gemini-Pro",
                url="cloud",
                model="gemini-2.5-flash",
                is_local=False,
                tier="api_deep",
                client=gemini_pro,
                failure_threshold=5,
                recovery_timeout=60.0,
            )
            
            # Thinking cloud — gemini-2.5-pro (deep reasoning fallback)
            gemini_thinking = GeminiAdapter(api_key=gemini_key, model="gemini-2.5-pro", rate_limiter=shared_limiter)
            router.register(
                name="Gemini-Thinking",
                url="cloud",
                model="gemini-2.5-pro",
                is_local=False,
                tier="api_deep",
                client=gemini_thinking,
                failure_threshold=3,
                recovery_timeout=300.0,
            )
            logger.info("✅ Gemini cloud fallbacks registered (2.0-flash, 2.5-flash, 2.5-pro) — shared rate limiter.")
        except (ImportError, AttributeError, RuntimeError) as e:
            _record_router_degradation(
                e,
                action="continued router build with local-only fallback coverage after Gemini registration failed",
                severity="degraded",
            )
            logger.error("❌ Failed to register Gemini fallbacks: %s", e)

    return router


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton accessor
#
# Why: several call sites (e.g. core/skills/skill_evolution.py) do
# `from core.brain.llm_health_router import llm_router` at import time, expecting
# a fully-constructed router.  The real router is built later during orchestrator
# boot via build_router_from_config().  This lazy proxy bridges both styles so
# import-time references resolve to whatever router the boot registered in the
# ServiceContainer — and falls back to constructing one on first use if no
# orchestrator has booted yet (supports test harnesses and standalone scripts).
# ─────────────────────────────────────────────────────────────────────────────

def get_llm_router() -> HealthAwareLLMRouter:
    """Return the process-wide router, constructing it on first use if needed."""
    from core.container import ServiceContainer
    existing = ServiceContainer.get("llm_router", default=None)
    if existing is not None:
        return existing
    from core.config import config
    router = build_router_from_config(config)
    ServiceContainer.register_instance("llm_router", router)
    return router


class _LazyRouterProxy:
    """Attribute-access proxy that resolves to the real router on first touch."""
    __slots__ = ("_cached",)

    def __init__(self) -> None:
        self._cached = None

    def _resolve(self):
        if self._cached is None:
            self._cached = get_llm_router()
        return self._cached

    def __getattr__(self, item):
        return getattr(self._resolve(), item)

    def __repr__(self) -> str:
        return f"<LazyRouterProxy resolved={self._cached is not None}>"


llm_router = _LazyRouterProxy()
