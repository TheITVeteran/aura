"""System integrity audit — make silent subsystem failures speak.

The critique's third gap: "the silence of subsystem failures … the degradation
receipt system is comprehensive but requires active reading." Failures are recorded
(``record_degradation``), the CRSM loop can quietly stop closing, and CAA steering can
quietly run below capacity — but nothing pulls these together and says so out loud.

This audit consolidates the three signals — degradation receipts, CRSM→LoRA loop
closure, and CAA steering readiness — into one report and logs a single loud summary
when anything is wrong. Health callers consume a bounded stale-while-revalidate read
model; filesystem hashing, dataset parsing, and CAA readiness checks never run on the
HTTP event loop. Under ``AURA_STRICT_RUNTIME=1`` the collector emits the report even
when clean, so production runs surface the activation state without requiring someone
to go read receipts.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from core.health.read_model import HealthReadModelConfig, HealthSnapshotReadModel

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


def _env_positive_float(name: str, default: float) -> float:
    try:
        return max(0.05, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return float(default)


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


def _integrity_snapshot_fallback() -> dict[str, Any]:
    return {
        "healthy": False,
        "concerns": ["integrity_snapshot_initializing"],
        "advisory": [],
        "strict_mode": strict_mode(),
        "degradations": {},
        "crsm_loop": {},
        "caa_readiness": {},
        "failure_state": {},
        "at": None,
    }


def _new_integrity_read_model() -> HealthSnapshotReadModel:
    refresh_s = _env_positive_float("AURA_INTEGRITY_REFRESH_S", 15.0)
    return HealthSnapshotReadModel(
        run_integrity_audit,
        _integrity_snapshot_fallback,
        config=HealthReadModelConfig(
            refresh_interval_s=refresh_s,
            max_stale_s=max(
                refresh_s,
                _env_positive_float("AURA_INTEGRITY_MAX_STALE_S", 90.0),
            ),
            collection_timeout_s=_env_positive_float(
                "AURA_INTEGRITY_COLLECTION_TIMEOUT_S", 8.0
            ),
            retry_base_s=_env_positive_float("AURA_INTEGRITY_RETRY_BASE_S", 2.0),
            retry_max_s=_env_positive_float("AURA_INTEGRITY_RETRY_MAX_S", 30.0),
            schema_version="aura.integrity.snapshot.v1",
            metadata_key="integrity_read_model",
            worker_name_prefix="AuraIntegritySnapshot",
            incident_prefix="integrity-refresh",
            log_label="Integrity snapshot",
        ),
    )


_INTEGRITY_READ_MODEL = _new_integrity_read_model()


def start_integrity_read_model() -> bool:
    """Prewarm integrity evidence without joining the collector."""

    return _INTEGRITY_READ_MODEL.start()


def stop_integrity_read_model() -> None:
    _INTEGRITY_READ_MODEL.close()


def reset_integrity_read_model_for_test() -> None:
    _INTEGRITY_READ_MODEL.reset_for_test()


def read_integrity_audit() -> dict[str, Any]:
    """Return immediately with current or explicitly stale integrity evidence."""

    return _INTEGRITY_READ_MODEL.read()


def maybe_run(*, interval_s: float = 300.0) -> dict[str, Any] | None:
    """Legacy synchronous throttle for CLI/background callers.

    Event-loop and request paths must use :func:`read_integrity_audit`.
    """
    global _last_run
    now = time.time()
    with _lock:
        if now - _last_run < interval_s:
            return _last_report
        _last_run = now
    return run_integrity_audit()


def last_report() -> dict[str, Any] | None:
    return _last_report
