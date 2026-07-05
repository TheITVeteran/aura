"""Unified runtime pressure — the pull-based provider for a real contract.

The health contract requires ``unified_runtime_pressure`` ("Aura must not
claim healthy when scheduling lag or substrate survival pressure is high"),
but no provider ever existed: the requirement was a phantom and the runtime
could sit DEGRADED forever against a service nobody could start or heal
(observed live 2026-07-05, 84 minutes of DEGRADED with this entry dead).

This provider is deliberately pull-based: no thread, no loop, no task —
nothing that can die and pin the contract. Every snapshot is computed on
demand from organs that already exist:

* event-loop lag   — the registered EventLoopMonitor's status
* memory pressure  — psutil RSS / system percent
* thermal pressure — core/runtime/thermal (NSProcessInfo on macOS)

``is_alive`` is True exactly when a fresh snapshot can be produced and no
pressure dimension is in its red zone — so the contract entry now measures
real pressure instead of the liveness of a nonexistent loop.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("Aura.Runtime.Pressure")

# Red-zone thresholds: past these, the runtime should not claim healthy.
_LOOP_LAG_RED_S = 5.0
_MEMORY_RED_PCT = 92.0
_THERMAL_RED_LEVEL = 3  # critical


class UnifiedRuntimePressure:
    """On-demand pressure snapshot over existing organs. Nothing to crash."""

    def __init__(self) -> None:
        self._last_snapshot: dict[str, Any] = {}
        self._last_snapshot_at = 0.0

    def runtime_pressure_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"at_unix": time.time()}

        loop_lag_s = 0.0
        monitor_alive = None
        try:
            from core.runtime.service_registry import get_runtime_service

            monitor = get_runtime_service("event_loop_monitor", default=None)
            status = monitor.get_status() if monitor is not None else {}
            if isinstance(status, dict):
                loop_lag_s = float(status.get("last_lag_s", 0.0) or 0.0)
                monitor_alive = bool(status.get("alive", False))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Pressure snapshot: loop-lag source unavailable: %s", exc)
        snapshot["loop_lag_s"] = round(loop_lag_s, 4)
        snapshot["loop_monitor_alive"] = monitor_alive

        memory_pct = 0.0
        try:
            import psutil

            memory_pct = float(psutil.virtual_memory().percent)
        except (ImportError, RuntimeError, OSError) as exc:
            logger.debug("Pressure snapshot: memory source unavailable: %s", exc)
        snapshot["memory_pct"] = memory_pct

        thermal_level = 0
        try:
            from core.runtime.thermal import thermal_state

            thermal_level = int(thermal_state().level)
        except (ImportError, RuntimeError, AttributeError, ValueError) as exc:
            logger.debug("Pressure snapshot: thermal source unavailable: %s", exc)
        snapshot["thermal_level"] = thermal_level

        red_zones = []
        if loop_lag_s >= _LOOP_LAG_RED_S:
            red_zones.append(f"loop_lag_{loop_lag_s:.1f}s")
        if memory_pct >= _MEMORY_RED_PCT:
            red_zones.append(f"memory_{memory_pct:.0f}pct")
        if thermal_level >= _THERMAL_RED_LEVEL:
            red_zones.append(f"thermal_level_{thermal_level}")
        snapshot["red_zones"] = red_zones
        snapshot["pressure_ok"] = not red_zones

        self._last_snapshot = snapshot
        self._last_snapshot_at = time.monotonic()
        return snapshot

    def is_alive(self) -> bool:
        """Fresh snapshot succeeds and no pressure dimension is red."""
        try:
            return bool(self.runtime_pressure_snapshot().get("pressure_ok", False))
        except Exception as exc:  # noqa: BLE001 — liveness must never raise
            logger.warning("Runtime pressure snapshot failed: %s", exc)
            return False

    def get_status(self) -> dict[str, Any]:
        return dict(self._last_snapshot)


_instance: UnifiedRuntimePressure | None = None


def get_unified_runtime_pressure() -> UnifiedRuntimePressure:
    global _instance
    if _instance is None:
        _instance = UnifiedRuntimePressure()
    return _instance


__all__ = ["UnifiedRuntimePressure", "get_unified_runtime_pressure"]
