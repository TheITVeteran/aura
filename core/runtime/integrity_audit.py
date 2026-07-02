"""System integrity audit — make silent subsystem failures speak.

The critique's third gap: "the silence of subsystem failures … the degradation
receipt system is comprehensive but requires active reading." Failures are recorded
(``record_degradation``), the CRSM loop can quietly stop closing, and CAA steering can
quietly run below capacity — but nothing pulls these together and says so out loud.

This audit consolidates the three signals — degradation receipts, CRSM→LoRA loop
closure, and CAA steering readiness — into one report, logs a single loud summary when
anything is wrong, and is throttled so it can be called from a hot path (health
heartbeat) without spamming. Under ``AURA_STRICT_RUNTIME=1`` it always emits the
report even when clean, so production runs surface the activation state every interval
instead of requiring someone to go read receipts.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger("Aura.IntegrityAudit")

_last_run = 0.0
_last_report: dict[str, Any] | None = None
_lock = threading.Lock()

# Degradations above this for a single subsystem are flagged as a concern.
_DEGRADATION_CONCERN = 10
# Concern verdicts look at a trailing window so the runtime can recover after
# a degradation storm instead of staying "unhealthy" for its whole lifetime.
_DEGRADATION_CONCERN_WINDOW_S = 1800.0


def strict_mode() -> bool:
    return os.environ.get("AURA_STRICT_RUNTIME") == "1"


def run_integrity_audit(*, log: bool = True) -> dict[str, Any]:
    """Aggregate degradations + CRSM loop + CAA readiness; log loudly if degraded."""
    # RUNTIME-HEALTH concerns (can the process actually serve?) vs ADVISORY concerns
    # (operational facts like "training hasn't run yet"). Only the former may gate
    # health — an open CRSM loop or runtime-derived CAA vectors are real and worth
    # surfacing, but they do NOT mean Aura can't converse, so they must never make the
    # runtime report "degraded"/not-ready.
    concerns: list[str] = []
    advisory: list[str] = []

    degradations: dict[str, Any] = {}
    try:
        from core.runtime.errors import get_degradation_tracker

        tracker = get_degradation_tracker()
        degradations = tracker.status()
        # Health verdicts use a trailing window, not lifetime counters — a
        # long-lived runtime must be able to RECOVER once a storm passes.
        recent_counts = tracker.recent_counts_by_subsystem(_DEGRADATION_CONCERN_WINDOW_S)
        degradations["recent_window_s"] = _DEGRADATION_CONCERN_WINDOW_S
        degradations["recent_counts_by_subsystem"] = recent_counts
        for sub, sevs in recent_counts.items():
            total = sum(sevs.values())
            if total >= _DEGRADATION_CONCERN:
                concerns.append(
                    f"{sub}: {total} degradations in the last "
                    f"{int(_DEGRADATION_CONCERN_WINDOW_S // 60)}m"
                )
    except (ImportError, AttributeError, RuntimeError, TypeError):
        degradations = {}

    crsm_loop: dict[str, Any] = {}
    try:
        from core.consciousness.crsm_loop_monitor import get_crsm_loop_monitor

        crsm_loop = get_crsm_loop_monitor().loop_state()
        if crsm_loop.get("state") == "open":
            advisory.append(f"CRSM→LoRA loop OPEN ({crsm_loop.get('unconsumed')} captures untrained)")
    except (ImportError, AttributeError, RuntimeError, TypeError):
        crsm_loop = {}

    caa_readiness: dict[str, Any] = {}
    try:
        from core.consciousness.caa.readiness_report import verify_readiness

        caa_readiness = verify_readiness()
        if caa_readiness.get("below_design_capacity"):
            advisory.append(
                f"CAA steering at {caa_readiness.get('steering_capacity_pct')}% "
                f"({caa_readiness.get('level')})"
            )
    except (ImportError, AttributeError, RuntimeError, TypeError):
        caa_readiness = {}

    # Failure pressure with its top contributors: when background policy
    # reports failure_lockdown_X, this names the feeder without log archaeology.
    failure_state: dict[str, Any] = {}
    try:
        from core.health.degraded_events import get_unified_failure_state

        unified = get_unified_failure_state()
        failure_state = {
            "pressure": unified.get("pressure", 0.0),
            "count": unified.get("count", 0),
            "critical": unified.get("critical", 0),
            "top_subsystems": unified.get("top_subsystems", []),
        }
        if float(failure_state.get("pressure") or 0.0) >= 0.5:
            advisory.append(
                f"failure pressure {failure_state['pressure']:.2f} "
                f"(top: {', '.join(str(t) for t in failure_state['top_subsystems'][:3]) or 'n/a'})"
            )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        failure_state = {}

    report = {
        # 'healthy' reflects RUNTIME health only — advisory operational facts never make
        # the runtime unhealthy. 'concerns' (the health-blocking list) holds runtime
        # concerns; 'advisory' holds the surfaced-but-non-blocking operational notes.
        "healthy": not concerns,
        "concerns": concerns,
        "advisory": advisory,
        "strict_mode": strict_mode(),
        "degradations": degradations,
        "crsm_loop": crsm_loop,
        "caa_readiness": caa_readiness,
        "failure_state": failure_state,
        "at": time.time(),
    }

    global _last_report
    _last_report = report

    if log:
        if concerns:
            logger.warning("🩺 [Integrity] %d runtime concern(s): %s", len(concerns), " | ".join(concerns))
            try:
                from core.observability.metrics import get_metrics

                get_metrics().increment_counter("integrity_concern_total")
            except (ImportError, AttributeError, RuntimeError, TypeError):
                pass
        if advisory:
            logger.info("🩺 [Integrity] advisory (non-blocking): %s", " | ".join(advisory))
        if not concerns and not advisory and strict_mode():
            logger.info("🩺 [Integrity] all subsystems nominal (strict mode).")
    return report


def maybe_run(*, interval_s: float = 300.0) -> dict[str, Any] | None:
    """Throttled audit — safe to call from a hot path (e.g. the health heartbeat)."""
    global _last_run
    now = time.time()
    with _lock:
        if now - _last_run < interval_s:
            return _last_report
        _last_run = now
    return run_integrity_audit()


def last_report() -> dict[str, Any] | None:
    return _last_report
