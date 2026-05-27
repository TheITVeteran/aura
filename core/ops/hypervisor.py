"""
core/ops/hypervisor.py
Enterprise Sentinel: Watchdog Hypervisor for Aura.
Monitors event loop health, memory leaks, and severe freezes.
"""

import asyncio
import logging
import time

from core.observability.metrics import get_metrics
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Hypervisor")
metrics = get_metrics()


class Hypervisor:
    def __init__(self, lag_threshold_s: float = 0.5):
        self._lag_threshold = lag_threshold_s
        self._active_lag_threshold = max(lag_threshold_s, 5.0)
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_tick = time.time()

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = get_task_tracker().create_task(self._watchdog_loop())
        logger.info("👁️ Hypervisor Watchdog active (Threshold: %.2fs)", self._lag_threshold)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _e:
                logger.debug("Ignored asyncio.CancelledError in hypervisor.py: %s", _e)
        logger.info("👁️ Hypervisor Watchdog shutdown.")

    def is_alive(self) -> bool:
        """Return True only when the watchdog loop is actively supervised."""
        return bool(self._running and self._task is not None and not self._task.done())

    def _active_runtime_reason(self) -> str:
        try:
            from core.runtime.proof_policy import proof_run_active

            if proof_run_active():
                return "proof_run_active"
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Hypervisor proof-run probe unavailable: %s", exc)

        try:
            from core.runtime.foreground_guard import foreground_activity_reason

            reason = foreground_activity_reason()
            if reason:
                return reason
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug("Hypervisor foreground probe unavailable: %s", exc)

        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            if gate and hasattr(gate, "get_conversation_status"):
                status = dict(gate.get_conversation_status() or {})
                if bool(status.get("foreground_owned")) or int(
                    status.get("active_generations", 0) or 0
                ) > 0:
                    return "foreground_generation_active"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Hypervisor inference-gate probe unavailable: %s", exc)
        return ""

    def _lag_threshold_for_context(self) -> tuple[float, str]:
        reason = self._active_runtime_reason()
        if reason:
            return self._active_lag_threshold, reason
        return self._lag_threshold, "idle"

    async def _watchdog_loop(self):
        while self._running:
            start = time.time()
            # Simple async sleep to measure lag
            await asyncio.sleep(1.0)
            actual_sleep = time.time() - start
            lag = actual_sleep - 1.0

            self._last_tick = time.time()
            metrics.gauge("hypervisor.loop_lag_s", lag)

            lag_threshold, lag_context = self._lag_threshold_for_context()
            if lag > lag_threshold:
                logger.warning(
                    "🚨 HIGH EVENT LOOP LAG detected: %.3fs (context=%s threshold=%.3fs)",
                    lag,
                    lag_context,
                    lag_threshold,
                )
                metrics.increment("hypervisor.lag_spikes_total")

                if lag > 5.0:
                    logger.critical(
                        "🚨 SEVERE FREEZE: Loop lag > 5s. System stability compromised."
                    )
                    # In a real enterprise system, we might trigger a graceful restart here.

            # Memory Check
            import psutil

            mem = psutil.Process().memory_info().rss / (1024 * 1024)
            metrics.gauge("system.memory_rss_mb", mem)
            if mem > 8192:  # 8GB threshold for M5 Pro 64GB (Aura's base limit)
                logger.warning("🚨 HIGH MEMORY USAGE: %.1f MB", mem)


_hypervisor: Hypervisor | None = None


def get_hypervisor() -> Hypervisor:
    global _hypervisor
    if _hypervisor is None:
        _hypervisor = Hypervisor()
    return _hypervisor
