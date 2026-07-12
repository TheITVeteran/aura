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
    from core.runtime import resource_psutil as psutil
except ImportError:
    psutil = None

from core.runtime.errors import get_degradation_tracker
from core.runtime.resource_observation import get_resource_observer

logger = logging.getLogger("Consciousness.ExistentialStakes")

DEFAULT_MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB default limit

# Ceiling on how much OPERATIONAL pressure (high CPU, event-loop lag) can
# contribute to the combined survival threat. High CPU and lag are normal
# during heavy local generation — a busy machine is not a dying one — so they
# must never alone reach the will-system's survival-inhibition veto threshold
# (will.py: threat > 0.75), which would block Aura's own actions whenever it is
# working hard. Genuine death risk (memory exhaustion → OOM, degradation
# cascades) is uncapped and can still reach 1.0. Kept just below the veto line.
OPERATIONAL_THREAT_CAP = 0.70
CRITICAL_THREAT_THRESHOLD = 0.75
DEGRADATION_THREAT_WINDOW_S = 60.0
DEGRADATION_THREAT_DENOMINATOR = 5.0
DEGRADATION_THREAT_SATURATION_EPSILON = 1e-4
CRITICAL_LOG_COOLDOWN_S = 30.0
DEGRADATION_SEVERITY_WEIGHTS = {
    "critical": 2.0,
    "degraded": 1.0,
    "warning": 0.20,
    "debug": 0.0,
}
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
        self._recent_degradation_weight = 0.0
        self._last_critical_log_time = 0.0
        self._last_critical_log_bucket = -1
        self._was_critical = False

        logger.info(
            "ExistentialStakes initialized. Memory limit: %.2f MB",
            self._memory_limit / (1024 * 1024),
        )

    @staticmethod
    def _degradation_record_weight(record: Any, *, now: float) -> float:
        severity = str(getattr(record, "severity", "") or "").lower()
        base = DEGRADATION_SEVERITY_WEIGHTS.get(severity, 0.0)
        if base <= 0.0:
            return 0.0
        age_s = max(0.0, now - float(getattr(record, "timestamp", now) or now))
        if age_s >= DEGRADATION_THREAT_WINDOW_S:
            return 0.0
        # A resolved transient should fade quickly instead of holding the live
        # Will in an existential veto for a full minute. Cascades still rise
        # because fresh records keep adding full weight.
        decay = 1.0 - (age_s / DEGRADATION_THREAT_WINDOW_S)
        return base * decay

    def _should_log_critical(self, now: float) -> bool:
        bucket = int(self._threat * 10.0)
        if not self._was_critical:
            self._last_critical_log_time = now
            self._last_critical_log_bucket = bucket
            self._was_critical = True
            return True
        if bucket != self._last_critical_log_bucket:
            self._last_critical_log_time = now
            self._last_critical_log_bucket = bucket
            return True
        if now - self._last_critical_log_time >= CRITICAL_LOG_COOLDOWN_S:
            self._last_critical_log_time = now
            return True
        return False

    def update(self) -> float:
        """Tick measurements, compute sub-threats, and return the combined threat."""
        with self._lock:
            now = time.time()
            self._total_ticks += 1

            # 1. Memory Threat
            process_mem = 0
            try:
                process = get_resource_observer().process(os.getpid())
                process_mem = int(process.rss_bytes) if process is not None else 0
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
            # Use severity-weighted, decaying degradation pressure. Counting
            # every recent warning as full existential danger made one repaired
            # foreground failure keep flooding the neural stream and blocking
            # actions. Critical/degraded cascades remain existential pressure;
            # warnings are weak signal; debug/lifecycle noise is ignored.
            recent_degradation_weight = 0.0
            try:
                tracker = get_degradation_tracker()
                if tracker and hasattr(tracker, "_records"):
                    recent_degradation_weight = sum(
                        self._degradation_record_weight(r, now=now)
                        for r in tracker._records
                    )
            except _EXISTENTIAL_STAKES_RECOVERABLE_ERRORS as e:
                logger.debug("Failed to query degradation tracker: %s", e)

            self._recent_degradation_weight = recent_degradation_weight
            # 5 fresh degraded-equivalent events in a minute is high threat.
            if recent_degradation_weight >= (
                DEGRADATION_THREAT_DENOMINATOR - DEGRADATION_THREAT_SATURATION_EPSILON
            ):
                self._degradation_threat = 1.0
            else:
                self._degradation_threat = min(
                    1.0,
                    recent_degradation_weight / DEGRADATION_THREAT_DENOMINATOR,
                )

            # Combined Threat. Distinguish SURVIVAL pressure (genuine death
            # risk) from OPERATIONAL pressure (busy/laggy but not dying):
            #   - memory exhaustion → OOM kill (the 110GB incident) and
            #     degradation cascades are real survival threats; uncapped, they
            #     may reach 1.0 and trigger the will-system's survival veto.
            #   - high CPU and event-loop lag are NORMAL during heavy 32B
            #     generation. Treating them as maximal survival threat made the
            #     will-veto block Aura's own actions whenever it worked hard
            #     (observed: continual-learning battery blocked at threat=1.00
            #     under load). Operational pressure still raises the felt threat
            #     so survival perception isn't blind, but it is capped BELOW the
            #     veto threshold so load alone can never inhibit action. Loop
            #     wedges are owned by the StallWatchdog, not this veto.
            survival_pressure = max(self._memory_threat, self._degradation_threat)
            operational_pressure = min(
                OPERATIONAL_THREAT_CAP,
                max(self._lag_threat, self._cpu_threat),
            )
            self._threat = max(survival_pressure, operational_pressure)

            # Log critical warning if threat is high, but coalesce repeated
            # ticks. The neural stream should show a state transition, not a
            # log storm.
            if self._threat > CRITICAL_THREAT_THRESHOLD and self._should_log_critical(now):
                logger.warning(
                    "CRITICAL EXISTENTIAL STAKES: threat=%.2f (mem_threat=%.2f, lag_threat=%.2f, cpu_threat=%.2f, deg_threat=%.2f)",
                    self._threat,
                    self._memory_threat,
                    self._lag_threat,
                    self._cpu_threat,
                    self._degradation_threat,
                )
            elif self._was_critical and self._threat <= CRITICAL_THREAT_THRESHOLD:
                self._was_critical = False
                self._last_critical_log_bucket = -1
                logger.info(
                    "Existential stakes recovered below critical threshold: threat=%.2f "
                    "(mem_threat=%.2f, lag_threat=%.2f, cpu_threat=%.2f, deg_threat=%.2f)",
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
                "recent_degradation_weight": round(self._recent_degradation_weight, 4),
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
