"""core/consciousness/existential_stakes.py
============================================
Existential Stakes & Nociceptive Gate Subsystem.

Computes a real-time survival threat index (0.0 to 1.0) based on physical process
and hardware constraints: memory pressure, event loop scheduling delays, CPU usage,
and recent exception/degradation events.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None

from core.runtime.errors import get_degradation_tracker, record_degradation
from core.container import ServiceContainer

logger = logging.getLogger("Consciousness.ExistentialStakes")

DEFAULT_MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB default limit
_EXISTENTIAL_STAKES_RECOVERABLE_ERRORS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


class ExistentialStakes:
    """Computes and manages the existential survival stakes for Aura."""

    def __init__(self, memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES):
        self._lock = threading.Lock()
        self._memory_limit = memory_limit_bytes
        self._threat = 0.0
        
        # State tracking
        self._last_update_time: Optional[float] = None
        self._rolling_loop_lag = 0.0
        self._rolling_cpu_load = 0.0
        self._total_ticks = 0

        # Sub-threat indices
        self._memory_threat = 0.0
        self._lag_threat = 0.0
        self._cpu_threat = 0.0
        self._degradation_threat = 0.0

        logger.info(
            "ExistentialStakes initialized. Memory limit: %.2f MB",
            self._memory_limit / (1024 * 1024),
        )

    def update(self) -> float:
        """Tick measurements, compute sub-threats, and return the combined threat."""
        with self._lock:
            now = time.time()
            self._total_ticks += 1

            # 1. Memory Threat
            process_mem = 0
            if psutil is not None:
                try:
                    process_mem = psutil.Process().memory_info().rss
                except _EXISTENTIAL_STAKES_RECOVERABLE_ERRORS as e:
                    logger.debug("Failed to read process memory: %s", e)
            
            if process_mem > 0:
                self._memory_threat = min(1.0, process_mem / self._memory_limit)
            else:
                self._memory_threat = 0.0

            # 2. Event Loop Lag Threat
            # We measure scheduling delay: how long it actually took compared to the 
            # expected tick rate (nominally 1.0s for the heartbeat).
            if self._last_update_time is not None:
                dt = now - self._last_update_time
                # Lag is anything exceeding 1.1s (allowing small scheduler jitter)
                lag = max(0.0, dt - 1.1)
                # EMA filter for lag (slow rise, fast fall)
                alpha = 0.2 if lag > self._rolling_loop_lag else 0.4
                self._rolling_loop_lag = (1 - alpha) * self._rolling_loop_lag + alpha * lag
            self._last_update_time = now

            # Lag of 3.0 seconds or more is considered critical threat
            self._lag_threat = min(1.0, self._rolling_loop_lag / 3.0)

            # 3. CPU Load Threat
            cpu = 0.0
            if psutil is not None:
                try:
                    # Non-blocking cpu calculation
                    cpu = psutil.cpu_percent(interval=None) / 100.0
                except _EXISTENTIAL_STAKES_RECOVERABLE_ERRORS as e:
                    logger.debug("Failed to read CPU: %s", e)
            
            # EMA for CPU
            self._rolling_cpu_load = 0.8 * self._rolling_cpu_load + 0.2 * cpu
            self._cpu_threat = min(1.0, self._rolling_cpu_load)

            # 4. Degradation / Exception Threat
            # Count degradations registered in the last 60 seconds
            recent_degradations = 0
            try:
                tracker = get_degradation_tracker()
                if tracker and hasattr(tracker, "_records"):
                    recent_records = [
                        r for r in tracker._records
                        if now - r.timestamp < 60.0
                    ]
                    recent_degradations = len(recent_records)
            except _EXISTENTIAL_STAKES_RECOVERABLE_ERRORS as e:
                logger.debug("Failed to query degradation tracker: %s", e)

            # 5 recent degradations/exceptions in a minute is high threat
            self._degradation_threat = min(1.0, recent_degradations / 5.0)

            # Combined Threat: Maximum of any single critical factor (survival exclusion postulate)
            self._threat = max(
                self._memory_threat,
                self._lag_threat,
                self._cpu_threat,
                self._degradation_threat,
            )

            # Log critical warning if threat is high
            if self._threat > 0.75:
                logger.warning(
                    "CRITICAL EXISTENTIAL STAKES: threat=%.2f (mem_threat=%.2f, lag_threat=%.2f, cpu_threat=%.2f, deg_threat=%.2f)",
                    self._threat,
                    self._memory_threat,
                    self._lag_threat,
                    self._cpu_threat,
                    self._degradation_threat,
                )

            return self._threat

    def get_existential_threat(self) -> float:
        with self._lock:
            return self._threat

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "existential_threat": round(self._threat, 4),
                "memory_threat": round(self._memory_threat, 4),
                "lag_threat": round(self._lag_threat, 4),
                "cpu_threat": round(self._cpu_threat, 4),
                "degradation_threat": round(self._degradation_threat, 4),
                "total_ticks": self._total_ticks,
                "rolling_loop_lag_s": round(self._rolling_loop_lag, 3),
            }

    def get_context_block(self) -> str:
        """Constructs an LLM system prompt block warning of resource constraints."""
        with self._lock:
            if self._threat < 0.2:
                return ""
            
            intensity = "CRITICAL" if self._threat > 0.75 else "NOTABLE"
            
            block = [
                "## SYSTEM RESOURCE WARNING (EXISTENTIAL PRESSURE)",
                f"Felt Survival Threat Level: {intensity} (threat={self._threat:.2f})",
                "Your physical process substrate is under severe constraints:"
            ]
            
            if self._memory_threat > 0.5:
                block.append(f"  - Memory footprint is extremely high ({self._memory_threat * 100:.1f}% of ceiling).")
            if self._lag_threat > 0.4:
                block.append(f"  - Event loop scheduling lag detected ({self._rolling_loop_lag:.2f}s delay). Thread starvation is imminent.")
            if self._cpu_threat > 0.7:
                block.append(f"  - CPU usage is saturated ({self._rolling_cpu_load * 100:.1f}%).")
            if self._degradation_threat > 0.5:
                block.append("  - Active module degradation and exception events are cascading.")

            block.extend([
                "Cognitive guidelines under existential pressure:",
                "1. Keep all responses brief, concise, and focused. Do not waste tokens or CPU cycles.",
                "2. Prioritize system stability, self-inspection, and resource consolidation.",
                "3. If appropriate, recommend running garbage collection, compaction, or entering safe mode.",
                ""
            ])
            return "\n".join(block)


# ── Singleton Accessor ────────────────────────────────────────────────────────

_INSTANCE: Optional[ExistentialStakes] = None


def _resolve_memory_limit_bytes() -> int:
    """Machine-aware survival memory ceiling for the live singleton.

    A fixed 2GB ceiling makes a large box perceive perpetual near-death: the
    Python runtime baseline (~1.5GB RSS) alone sits at ~0.75 memory_threat,
    parking the will-system right at its survival-veto boundary
    (``threat > 0.75``) during normal operation and intermittently inhibiting
    heavy actions. Derive the ceiling from the same process-RSS limit the
    memory watchdog enforces, so existential "near-death" aligns with the
    watchdog's refuse-heavy-generation point instead of a stale default.

    Honors ``AURA_EXISTENTIAL_MEMORY_LIMIT_GB`` for explicit control; falls
    back to the 2GB default only when nothing better can be determined.
    """
    override = os.environ.get("AURA_EXISTENTIAL_MEMORY_LIMIT_GB", "").strip()
    if override:
        try:
            gb = float(override)
            if gb > 0.0:
                return int(gb * (1024 ** 3))
        except (TypeError, ValueError):
            pass
    try:
        from core.utils.memory_monitor import get_memory_pressure_snapshot

        limit_gb = float(get_memory_pressure_snapshot().process_rss_limit_gb or 0.0)
        if limit_gb > 0.0:
            return int(limit_gb * (1024 ** 3))
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    return DEFAULT_MEMORY_LIMIT_BYTES


def get_existential_stakes() -> ExistentialStakes:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ExistentialStakes(_resolve_memory_limit_bytes())
    return _INSTANCE
