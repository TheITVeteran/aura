"""
core/ops/hypervisor.py
Enterprise Sentinel: Watchdog Hypervisor for Aura.
Monitors event loop health, memory leaks, and severe freezes.
"""

import asyncio
import logging
import time

from core.observability.metrics import get_metrics
from core.runtime.errors import record_degradation
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
        self._last_lag = 0.0
        self._last_severe_lag_at = 0.0
        self._last_failure_reason = ""
        self._healthy_lag_samples_after_failure = 0
        self._required_recovery_samples = 3

    async def start(self):
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
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
        if not bool(self._running and self._task is not None and not self._task.done()):
            return False
        if self._last_severe_lag_at and (
            self._healthy_lag_samples_after_failure < self._required_recovery_samples
        ):
            return False
        return True

    def get_status(self) -> dict[str, float | bool | str]:
        return {
            "alive": self.is_alive(),
            "last_lag_s": self._last_lag,
            "last_severe_lag_at": self._last_severe_lag_at,
            "last_failure_reason": self._last_failure_reason,
            "healthy_recovery_samples": self._healthy_lag_samples_after_failure,
            "required_recovery_samples": self._required_recovery_samples,
        }

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
            self._last_lag = lag

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
                    uptime = time.time() - getattr(self, "_start_time", time.time())
                    if uptime < 180.0:
                        logger.warning(
                            "🚨 Loop lag > 5s during boot/warmup grace period (uptime: %.1fs). "
                            "Skipping severe freeze failure recording to allow model load to complete.",
                            uptime,
                        )
                    else:
                        self._last_severe_lag_at = time.time()
                        self._last_failure_reason = f"severe event-loop lag {lag:.3f}s"
                        self._healthy_lag_samples_after_failure = 0
                        record_degradation(
                            "hypervisor",
                            RuntimeError(self._last_failure_reason),
                            severity="critical",
                            action="marked hypervisor unhealthy until healthy lag samples confirm recovery",
                            enforce_failure_policy=False,
                        )
                        logger.critical(
                            "🚨 SEVERE FREEZE: Loop lag > 5s. System stability compromised."
                        )
            elif self._last_severe_lag_at:
                self._healthy_lag_samples_after_failure += 1
                if self._healthy_lag_samples_after_failure >= self._required_recovery_samples:
                    logger.info(
                        "Hypervisor recovered after %d healthy event-loop samples.",
                        self._healthy_lag_samples_after_failure,
                    )
                    self._last_severe_lag_at = 0.0
                    self._last_failure_reason = ""

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
