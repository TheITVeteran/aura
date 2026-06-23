from __future__ import annotations
from core.runtime.errors import record_degradation


import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from core.health.degraded_events import get_unified_failure_state

try:  # module-local so tests and diagnostics can patch the exact host probe.
    import psutil
except ImportError:  # pragma: no cover - production hosts should carry psutil.
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.debug("Invalid %s=%r; using %.1f", name, raw, default)
        return float(default)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def foreground_only_runtime() -> bool:
    """Return True when Aura should boot only foreground/user-facing loops."""

    return _env_flag("AURA_FOREGROUND_ONLY", False)


def background_cognition_disabled_reason() -> str:
    """Return why optional background cognition must stay offline.

    Desktop safe boot is foreground-first by default. The live desktop lane has
    repeatedly failed by letting idle autonomy, dreaming, and background model
    work compete with the user's foreground turn. Operators can explicitly
    re-enable these loops after boot with AURA_ENABLE_BACKGROUND_COGNITION=1.
    """

    configured = str(os.getenv("AURA_ENABLE_BACKGROUND_COGNITION", "") or "").strip().lower()
    if configured in {"0", "false", "no", "off"}:
        return "background_cognition_disabled"
    if configured in {"1", "true", "yes", "on"}:
        return ""
    try:
        from core.runtime.desktop_boot_safety import desktop_safe_boot_enabled

        if desktop_safe_boot_enabled():
            return "desktop_background_disabled"
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="blocked background cognition because desktop safe-boot probe failed",
        )
        logger.warning("Background cognition desktop safe-boot probe failed: %s", _exc)
        return "desktop_safe_boot_probe_unavailable"
    return ""


def background_loop_start_reason(origin: Any = None) -> str:
    """Explain why a persistent background loop must not start.

    ``background_activity_reason`` gates individual work items. This helper
    gates loop creation itself for modes where idle autonomy would contaminate
    proof artifacts or compete with the live user lane.
    """

    try:
        from core.runtime.shutdown_coordinator import is_shutdown_requested

        if is_shutdown_requested():
            return "shutdown_requested"
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="blocked background loop start because shutdown probe failed",
        )
        logger.warning("Background loop shutdown probe failed: %s", _exc)
        return "shutdown_probe_unavailable"

    try:
        from core.runtime.proof_policy import proof_run_active

        if proof_run_active(origin=origin):
            return "proof_run_active"
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="blocked background loop start because proof-run signal failed",
        )
        logger.warning("Background loop proof-run probe failed: %s", _exc)
        return "proof_signal_unavailable"

    if foreground_only_runtime():
        return "foreground_only_runtime"

    disabled_reason = background_cognition_disabled_reason()
    if disabled_reason:
        return disabled_reason

    return ""


def background_loop_start_allowed(origin: Any = None) -> bool:
    return not background_loop_start_reason(origin)


_USER_FACING_ORIGIN_TOKENS = frozenset({
    "user",
    "voice",
    "admin",
    "api",
    "gui",
    "ws",
    "websocket",
    "desktop",
    "ui",
    "external",
    "direct",
    "embodied",
    "reflex",
    "motor",
    "test",
})

_BACKGROUND_ORIGIN_HINTS = frozenset({
    "affect",
    "autonomous",
    "background",
    "constitutive",
    "consolidation",
    "context",
    "dream",
    "growth",
    "impulse",
    "internal",
    "memory",
    "metabolic",
    "mist",
    "monitor",
    "motivation",
    "parallel",
    "perception",
    "phenomenological",
    "proactive",
    "pruner",
    "scanner",
    "sensory",
    "spontaneous",
    "stream",
    "structured",
    "subconscious",
    # "system" intentionally omitted: it is too broad and would misclassify
    # user-adjacent routing paths that still use the historical default.
    "terminal",
    "volition",
    "witness",
})


@dataclass(frozen=True)
class BackgroundPolicyProfile:
    min_idle_seconds: float = 10.0
    max_memory_percent: float = 90.0
    max_failure_pressure: float = 0.60
    require_conversation_ready: bool = False


@dataclass(frozen=True)
class ConstitutiveComputeBudget:
    """Runtime budget for always-on embodied/cognitive loops.

    These loops should stay alive, but they must yield hard priority to the live
    foreground conversation and to system memory pressure. The budget is a
    throttle, not a shutdown signal.
    """

    component: str
    base_hz: float
    effective_hz: float
    interval_s: float
    reason: str
    foreground_active: bool
    memory_percent: float | None = None


@dataclass(frozen=True)
class _MemoryPressureSnapshot:
    pressure_pct: float
    reason: str
    refuse_heavy_local_generation: bool = False


def _read_memory_pressure_snapshot() -> _MemoryPressureSnapshot:
    """Read host memory through the local probe and the richer runtime guard.

    The module-local psutil probe is intentionally retained because background
    policy is the fail-closed boundary for constitutive loops. If this direct
    host probe fails, optional/background work must not continue on a stale
    or untestable memory signal.
    """

    host_percent: float | None = None
    if psutil is not None:
        memory = psutil.virtual_memory()
        host_percent = float(getattr(memory, "percent"))

    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        runtime = get_memory_pressure_snapshot()
        runtime_percent = float(getattr(runtime, "pressure_pct", 0.0) or 0.0)
        runtime_reason = str(getattr(runtime, "reason", "") or "")
        runtime_refuse = bool(getattr(runtime, "refuse_heavy_local_generation", False))
        pressure = runtime_percent
        reason = runtime_reason or f"memory_pressure_{runtime_percent:.1f}"
        refuse = runtime_refuse
        if host_percent is not None and host_percent >= runtime_percent:
            pressure = host_percent
            if not runtime_refuse:
                reason = f"memory_pressure_{host_percent:.1f}"
            refuse = runtime_refuse or host_percent >= 92.0
        return _MemoryPressureSnapshot(
            pressure_pct=pressure,
            reason=reason,
            refuse_heavy_local_generation=refuse,
        )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        if host_percent is None:
            raise
        return _MemoryPressureSnapshot(
            pressure_pct=host_percent,
            reason=f"memory_pressure_{host_percent:.1f}",
            refuse_heavy_local_generation=False,
        )


THOUGHT_BACKGROUND_POLICY = BackgroundPolicyProfile(
    min_idle_seconds=30.0,
    max_memory_percent=85.0,
    max_failure_pressure=0.50,
    require_conversation_ready=True,
)

RESEARCH_BACKGROUND_POLICY = BackgroundPolicyProfile(
    min_idle_seconds=900.0,
    max_memory_percent=85.0,
    max_failure_pressure=0.50,
    require_conversation_ready=True,
)

MAINTENANCE_BACKGROUND_POLICY = BackgroundPolicyProfile(
    min_idle_seconds=1800.0,
    max_memory_percent=92.0,
    max_failure_pressure=0.75,
    require_conversation_ready=True,
)


def _component_env_name(component: str) -> str:
    normalized = "".join(ch if ch.isalnum() else "_" for ch in str(component or "loop"))
    return f"AURA_CONSTITUTIVE_{normalized.upper()}_HZ"


def _bounded_hz(value: float, *, lower: float = 0.1, upper: float = 1000.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = lower
    if not value or value != value or value < lower:
        return lower
    return min(float(upper), value)


def constitutive_compute_budget(
    component: str,
    base_hz: float,
    *,
    min_hz: float = 0.5,
    foreground_hz: float = 2.0,
    memory_high_hz: float = 2.0,
    memory_critical_hz: float = 0.5,
    memory_high_percent: float = 85.0,
    memory_critical_percent: float = 92.0,
    failure_pressure_hz: float = 1.0,
    max_failure_pressure: float = 0.75,
) -> ConstitutiveComputeBudget:
    """Return a safe update budget for continuous constitutive loops.

    Unlike ``background_activity_reason``, this does not block the loop. It
    caps its frequency during foreground inference, memory pressure, proof runs,
    or failure pressure so substrate/field/HOT machinery cannot compete with a
    user-facing model turn or drive host RAM spikes.
    """

    base = _bounded_hz(base_hz, lower=0.1, upper=1000.0)
    floor = _bounded_hz(min_hz, lower=0.1, upper=base)
    effective = base
    reason = "nominal"
    foreground_active = False
    memory_percent: float | None = None

    component_override = os.getenv(_component_env_name(component))
    if component_override is not None:
        effective = min(effective, _bounded_hz(component_override, lower=floor, upper=base))
        reason = "component_override"

    global_override = os.getenv("AURA_CONSTITUTIVE_MAX_HZ")
    if global_override is not None:
        effective = min(effective, _bounded_hz(global_override, lower=floor, upper=base))
        reason = "global_override" if reason == "nominal" else f"{reason}+global_override"

    try:
        from core.runtime.proof_policy import proof_run_active

        if proof_run_active():
            effective = min(effective, floor)
            reason = "proof_run_active"
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="throttled constitutive loop because proof-run signal failed",
        )
        effective = min(effective, floor)
        reason = "proof_signal_unavailable"

    if foreground_only_runtime():
        effective = min(effective, floor)
        reason = "foreground_only_runtime"

    disabled_reason = background_cognition_disabled_reason()
    if disabled_reason:
        effective = min(effective, floor)
        reason = disabled_reason

    foreground_reason = _foreground_activity_reason()
    if foreground_reason:
        foreground_active = True
        effective = min(effective, _bounded_hz(foreground_hz, lower=floor, upper=base))
        reason = foreground_reason

    try:
        memory = _read_memory_pressure_snapshot()
        memory_percent = float(memory.pressure_pct)
        memory_reason = str(memory.reason or f"memory_pressure_{memory_percent:.1f}")
        if memory.refuse_heavy_local_generation:
            effective = min(
                effective,
                _bounded_hz(memory_critical_hz, lower=floor, upper=base),
            )
            reason = memory_reason
        elif memory_percent >= float(memory_critical_percent):
            effective = min(
                effective,
                _bounded_hz(memory_critical_hz, lower=floor, upper=base),
            )
            reason = f"memory_critical_{memory_percent:.1f}"
        elif memory_percent >= float(memory_high_percent):
            effective = min(effective, _bounded_hz(memory_high_hz, lower=floor, upper=base))
            reason = f"memory_pressure_{memory_percent:.1f}"
    except (ImportError, OSError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="throttled constitutive loop because memory-pressure probe failed",
        )
        effective = min(effective, floor)
        reason = "memory_probe_unavailable"

    try:
        failure = get_unified_failure_state()
        pressure = float(failure.get("pressure", 0.0) or 0.0)
        if pressure >= float(max_failure_pressure):
            effective = min(effective, _bounded_hz(failure_pressure_hz, lower=floor, upper=base))
            reason = f"failure_pressure_{pressure:.2f}"
    except (OSError, ConnectionError, TimeoutError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="throttled constitutive loop because failure-state probe failed",
        )
        effective = min(effective, floor)
        reason = "failure_state_unavailable"

    effective = max(floor, min(base, float(effective)))
    return ConstitutiveComputeBudget(
        component=str(component or "loop"),
        base_hz=base,
        effective_hz=effective,
        interval_s=1.0 / max(floor, effective),
        reason=reason,
        foreground_active=foreground_active,
        memory_percent=memory_percent,
    )


def normalize_origin(origin: Any) -> str:
    normalized = str(origin or "").strip().lower().replace("-", "_")
    while normalized.startswith("routing_"):
        normalized = normalized[len("routing_"):]
    return normalized


def origin_tokens(origin: Any) -> set[str]:
    normalized = normalize_origin(origin)
    return {token for token in normalized.split("_") if token}


def is_user_facing_origin(origin: Any) -> bool:
    normalized = normalize_origin(origin)
    if not normalized:
        return False
    if normalized in _USER_FACING_ORIGIN_TOKENS:
        return True
    return bool(origin_tokens(normalized) & _USER_FACING_ORIGIN_TOKENS)


def is_background_origin(origin: Any, *, explicit_background: bool = False) -> bool:
    if explicit_background:
        return True
    tokens = origin_tokens(origin)
    if not tokens:
        return False
    if tokens & _USER_FACING_ORIGIN_TOKENS:
        return False
    return bool(tokens & _BACKGROUND_ORIGIN_HINTS)


def _last_user_interaction_time(orchestrator: Any = None) -> float:
    orch = orchestrator
    if orch is None:
        return 0.0

    value = float(getattr(orch, "_last_user_interaction_time", 0.0) or 0.0)
    if value > 0.0:
        return value

    status = getattr(orch, "status", None)
    if status is not None:
        value = float(getattr(status, "last_user_interaction_time", 0.0) or 0.0)
        if value > 0.0:
            return value

    return 0.0


def _runtime_uptime_seconds(orchestrator: Any = None) -> float:
    if orchestrator is None:
        return 0.0

    candidates = [
        getattr(orchestrator, "start_time", None),
        getattr(getattr(orchestrator, "status", None), "start_time", None),
    ]
    for candidate in candidates:
        try:
            start = float(candidate or 0.0)
        except (TypeError, ValueError):
            continue
        if start > 0.0:
            return max(0.0, time.time() - start)
    return 0.0


def _foreground_activity_reason() -> str:
    guard_reason = ""
    try:
        from core.runtime.foreground_guard import foreground_activity_reason

        guard_reason = foreground_activity_reason()
        if guard_reason == "foreground_chat_active":
            return guard_reason
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="blocked background work because foreground guard probe failed",
        )
        logger.warning("Background policy foreground guard probe failed: %s", _exc)
        return "foreground_guard_unavailable"

    try:
        from core.container import ServiceContainer

        gate = ServiceContainer.get("inference_gate", default=None)
        if gate and hasattr(gate, "get_conversation_status"):
            lane = dict(gate.get_conversation_status() or {})
            if bool(lane.get("foreground_owned")) or int(lane.get("active_generations", 0) or 0) > 0:
                return "foreground_generation_active"
            if bool(lane.get("kernel_lock_held")):
                return "foreground_kernel_lock"
            request_age = float(lane.get("request_age_s", 0.0) or 0.0)
            if request_age > 0.0 and str(lane.get("foreground_owner") or "").strip():
                return "foreground_request_active"
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="blocked background work because inference foreground probe failed",
        )
        logger.warning("Background policy inference foreground probe failed: %s", _exc)
        return "foreground_generation_status_unavailable"
    if guard_reason:
        return guard_reason
    return ""


def background_activity_reason(
    orchestrator: Any = None,
    *,
    profile: BackgroundPolicyProfile | None = None,
    min_idle_seconds: float | None = None,
    max_memory_percent: float | None = None,
    max_failure_pressure: float | None = None,
    require_conversation_ready: bool | None = None,
    allow_no_user_anchor: bool = False,
) -> str:
    if profile is not None:
        if min_idle_seconds is None:
            min_idle_seconds = profile.min_idle_seconds
        if max_memory_percent is None:
            max_memory_percent = profile.max_memory_percent
        if max_failure_pressure is None:
            max_failure_pressure = profile.max_failure_pressure
        if require_conversation_ready is None:
            require_conversation_ready = profile.require_conversation_ready

    min_idle_seconds = float(min_idle_seconds if min_idle_seconds is not None else 10.0)
    max_memory_percent = float(max_memory_percent if max_memory_percent is not None else 90.0)
    max_failure_pressure = float(max_failure_pressure if max_failure_pressure is not None else 0.60)
    require_conversation_ready = bool(
        False if require_conversation_ready is None else require_conversation_ready
    )

    now = time.time()

    try:
        from core.runtime.proof_policy import proof_run_active

        if proof_run_active():
            return "proof_run_active"
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="continued background policy evaluation without proof-run signal",
        )
        logger.debug("Proof-run background policy check unavailable: %s", _exc)

    if foreground_only_runtime():
        return "foreground_only_runtime"

    disabled_reason = background_cognition_disabled_reason()
    if disabled_reason:
        return disabled_reason

    orch = orchestrator
    if orch is not None:
        boot_grace_s = _env_float("AURA_BACKGROUND_BOOT_GRACE_S", 300.0)
        uptime_s = _runtime_uptime_seconds(orch)
        if boot_grace_s > 0.0 and 0.0 < uptime_s < boot_grace_s:
            return f"boot_grace_{int(uptime_s)}s"

    if orch is not None:
        if bool(getattr(orch, "is_busy", False)):
            return "orchestrator_busy"

        if float(getattr(orch, "_suppress_unsolicited_proactivity_until", 0.0) or 0.0) > now:
            return "suppressed"

        last_user = _last_user_interaction_time(orch)
        if last_user <= 0.0 and not allow_no_user_anchor:
            return "no_user_anchor"

    foreground_reason = _foreground_activity_reason()
    if foreground_reason:
        return foreground_reason

    if orch is not None:
        quiet_until = float(getattr(orch, "_foreground_user_quiet_until", 0.0) or 0.0)
        if quiet_until > now:
            return "foreground_quiet_window"

        if (now - last_user) < min_idle_seconds:
            return f"recent_user_{int(now - last_user)}"

    try:
        memory = _read_memory_pressure_snapshot()
        memory_pct = float(memory.pressure_pct)
        if bool(memory.refuse_heavy_local_generation):
            return str(memory.reason or f"memory_pressure_guard_{memory_pct:.1f}")
        if memory_pct >= max_memory_percent:
            return f"memory_pressure_{memory_pct:.1f}"
    except (ImportError, OSError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="blocked background work because memory-pressure probe failed",
        )
        logger.warning("Background policy memory-pressure probe failed: %s", _exc)
        return "memory_probe_unavailable"

    try:
        failure = get_unified_failure_state()
        pressure = float(failure.get("pressure", 0.0) or 0.0)
        if pressure >= max_failure_pressure:
            return f"failure_lockdown_{pressure:.2f}"
    except (OSError, ConnectionError, TimeoutError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="blocked background work because failure-state probe failed",
        )
        logger.warning("Background policy failure-state probe failed: %s", _exc)
        return "failure_state_unavailable"

    try:
        from core.organism.welfare import welfare_block_reason

        welfare_reason = welfare_block_reason()
        if welfare_reason:
            # The organism's vital interests (memory integrity, repair
            # capacity) gate optional work: welfare is causal machinery
            # here, not narrative.
            return welfare_reason
    except (ImportError, AttributeError, RuntimeError) as _exc:
        record_degradation(
            "background_policy",
            _exc,
            action="continued background gating without welfare model",
        )

    if require_conversation_ready:
        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "get_conversation_status"):
                lane = gate.get_conversation_status() or {}
                if not bool(lane.get("conversation_ready", False)):
                    return f"conversation_lane_{str(lane.get('state', 'unready') or 'unready').lower()}"
        except (ImportError, AttributeError, RuntimeError) as _exc:
            record_degradation(
                "background_policy",
                _exc,
                action="blocked background work because conversation readiness probe failed",
            )
            logger.warning("Background policy conversation readiness probe failed: %s", _exc)
            return "conversation_lane_probe_unavailable"

    return ""


def background_activity_allowed(
    orchestrator: Any = None,
    *,
    profile: BackgroundPolicyProfile | None = None,
    min_idle_seconds: float | None = None,
    max_memory_percent: float | None = None,
    max_failure_pressure: float | None = None,
    require_conversation_ready: bool | None = None,
    allow_no_user_anchor: bool = False,
) -> bool:
    return not background_activity_reason(
        orchestrator,
        profile=profile,
        min_idle_seconds=min_idle_seconds,
        max_memory_percent=max_memory_percent,
        max_failure_pressure=max_failure_pressure,
        require_conversation_ready=require_conversation_ready,
        allow_no_user_anchor=allow_no_user_anchor,
    )
