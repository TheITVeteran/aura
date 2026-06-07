import asyncio
import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

# Use the centralized enhanced logger
try:
    from core.utils.aura_logging import core_logger as logger
except ImportError:
    logger = logging.getLogger("Aura.Core")

# Sentinels
LOCK_SENTINEL = "LOCK_ACQUIRED"


async def run_io_bound(func, *args, **kwargs):
    """
    Runs a blocking I/O bound function in a separate thread to avoid blocking the event loop.
    """
    import functools

    if hasattr(asyncio, "to_thread"):
        # Python 3.9+
        return await asyncio.to_thread(func, *args, **kwargs)
    else:
        # Fallback for older versions
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


class RobustLock:
    """
    A loop-agnostic version of a lock that implements the 'Timeout & Retry' protocol.
    Uses an internal threading.Lock to allow sharing across multiple asyncio event loops.
    ZENITH LOCKDOWN: Adaptive timeouts and GPU load scaling.
    """

    def __init__(
        self,
        name: str = "UnnamedLock",
        *,
        watchdog_threshold_s: float | None = None,
        force_release_on_stall: bool = False,
        timeout_s: float | None = None,
    ):
        import uuid

        self.name = name
        full_id = str(uuid.uuid4())
        self.id = full_id[:8]
        self._lock = threading.Lock()
        self.timeout = float(timeout_s) if timeout_s is not None else 30.0  # Base timeout (Zenith default)
        self.adaptive = True
        self.last_acquire_start = 0.0
        self.watchdog_threshold_s = watchdog_threshold_s
        self.force_release_on_stall = force_release_on_stall

    @staticmethod
    def _watchdog_report_acquire_start(
        watchdog: Any, lock_id: str, name: str, callback: Any, threshold_s: float | None = None
    ) -> None:
        try:
            watchdog.report_acquire_start(
                lock_id,
                name,
                on_stall=callback,
                threshold_s=threshold_s,
            )
        except TypeError:
            watchdog.report_acquire_start(lock_id, name)

    async def acquire_robust(  # noqa: ASYNC109
        self,
        timeout: float | None = None,  # noqa: ASYNC109
        max_retries: int = 3,
    ) -> bool:
        """
        Attempts to acquire the lock with a timeout and retries.
        """
        wait_time = timeout or self.timeout

        # 0. Register with Watchdog
        from core.resilience.lock_watchdog import get_lock_watchdog

        watchdog = get_lock_watchdog()
        stall_callback = self.force_release if self.force_release_on_stall else None
        self._watchdog_report_acquire_start(
            watchdog,
            self.id,
            self.name,
            stall_callback,
            self.watchdog_threshold_s,
        )

        # Adaptive Scaling: if GPU is saturated, extend timeout
        if self.adaptive:
            # We check the metrics collector for GPU load if possible
            try:
                from core.observability.metrics import get_metrics

                m = get_metrics()._custom_gauges.get("gpu_utilization", 0)
                if m > 0.8:
                    wait_time = max(wait_time, 180.0)
                    logger.debug(
                        "🛡️ [ADAPTIVE] GPU Saturated (%s). Extending '%s' timeout to %ss",
                        f"{m:.2f}",
                        self.name,
                        wait_time,
                    )
            except (ImportError, AttributeError, RuntimeError) as _exc:
                record_degradation("concurrency", _exc)
                logger.debug("Suppressed Exception: %s", _exc)

        async def _await_threaded_acquire(acquire_timeout: float) -> bool:
            acquire_task = get_task_tracker().create_task(
                asyncio.to_thread(self._lock.acquire, timeout=acquire_timeout),
                name=f"lock_acquire:{self.name}",
            )
            try:
                # Pulse the watchdog while waiting so it doesn't trigger a stall on long adaptive timeouts
                start_wait = time.monotonic()
                while not acquire_task.done():
                    elapsed = time.monotonic() - start_wait
                    if elapsed > 0:
                        watchdog.report_wait_progress(
                            self.id
                        )  # Notify watchdog we are actively waiting
                    done, pending = await asyncio.wait([acquire_task], timeout=1.0)
                    if done:
                        break
                return acquire_task.result()
            except asyncio.CancelledError:
                # asyncio.to_thread cannot be cancelled safely; wait for the worker to
                # finish and immediately release any lock it may have acquired so we
                # don't strand the mutex or its watchdog entry forever.
                try:
                    acquired = await asyncio.wait_for(acquire_task, timeout=1.0)
                except (RuntimeError, asyncio.CancelledError, TimeoutError, AttributeError):
                    acquired = False
                if acquired:
                    try:
                        self._lock.release()
                    except (RuntimeError, AttributeError, TypeError, ValueError) as _exc:
                        record_degradation("concurrency", _exc)
                        logger.debug("Suppressed Exception: %s", _exc)
                watchdog.report_release(self.id)
                raise

        for attempt in range(max_retries):
            self.last_acquire_start = time.monotonic()
            self._watchdog_report_acquire_start(
                watchdog,
                self.id,
                self.name,
                stall_callback,
                self.watchdog_threshold_s,
            )
            try:
                success = await _await_threaded_acquire(wait_time)
            except asyncio.CancelledError:
                raise
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation("concurrency", e)
                watchdog.report_release(self.id)
                logger.error("Unexpected error acquiring lock '%s': %s", self.name, e)
                break

            if success:
                watchdog.report_acquire_success(self.id)
                logger.debug("Successfully locked: '%s'", self.name)
                return True

            watchdog.report_release(self.id)
            logger.warning(
                "Attempt %s/%s: Timeout waiting for '%s'.", attempt + 1, max_retries, self.name
            )
            await asyncio.sleep(random.uniform(0.1, 0.5))

        # Safety valve: force-release the EXISTING lock only for locks that
        # explicitly permit corruption-risk recovery. Boot and promotion locks
        # prefer fail-closed acquisition over replacing a live mutex.
        if not self.force_release_on_stall:
            watchdog.report_release(self.id)
            return False

        if not self.force_release():
            watchdog.report_release(self.id)
            return False
        watchdog.report_release(self.id)

        self._watchdog_report_acquire_start(
            watchdog,
            self.id,
            self.name,
            stall_callback,
            self.watchdog_threshold_s,
        )
        try:
            success = await _await_threaded_acquire(10.0)
        except asyncio.CancelledError:
            raise
        except (RuntimeError, AttributeError, TypeError, ValueError):
            watchdog.report_release(self.id)
            raise
        if success:
            watchdog.report_acquire_success(self.id)
            return True
        watchdog.report_release(self.id)
        return False

    async def acquire(self) -> bool:
        """Standard async acquire with adaptive timeout."""
        return await self.acquire_robust()

    def release(self):
        """Release the lock and reset timing."""
        try:
            if self._lock.locked():
                self._lock.release()
                from core.resilience.lock_watchdog import get_lock_watchdog

                get_lock_watchdog().report_release(self.id)
                logger.debug("Released lock: '%s'", self.name)
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation("concurrency", e)
            logger.debug("RobustLock.release() error for '%s': %s", self.name, e)

    def force_release(self) -> bool:
        """CRITICAL: Force release the lock to break a detected deadlock.

        Replaces the lock entirely so that the blocked thread can proceed.
        The thread holding the old lock will release the old lock safely.
        """
        if not self.force_release_on_stall:
            exc = RuntimeError(f"force release disabled for lock '{self.name}'")
            record_degradation(
                "concurrency",
                exc,
                severity="critical",
                action="blocked_lock_force_release_without_explicit_opt_in",
                extra={"lock_id": self.id, "lock_name": self.name},
            )
            logger.error(
                "Force release blocked for lock '%s'; lock was configured fail-closed.",
                self.name,
            )
            return False
        logger.critical("⚠️ FORCE RELEASING LOCK '%s' due to deadlock watchdog!", self.name)
        try:
            self._lock = threading.Lock()
        except RuntimeError:
            # release() on an unlocked lock — harmless
            pass  # no-op: intentional
        except (AttributeError, TypeError, ValueError) as _exc:
            record_degradation("concurrency", _exc)
            logger.debug("Suppressed Exception: %s", _exc)
            return False
        return True

    def locked(self) -> bool:
        return self._lock.locked()

    @property
    def is_locked(self) -> bool:
        """Compatibility property for Watchdog."""
        return self.locked()

    @property
    def held_duration(self) -> float:
        if not self.locked():
            return 0.0
        return time.monotonic() - self.last_acquire_start

    async def __aenter__(self):
        acquired = await self.acquire()
        if not acquired:
            raise TimeoutError(f"failed to acquire robust lock '{self.name}'")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


@dataclass(frozen=True, order=True)
class LockableResource:
    """A resource identifier for strict ordering."""

    name: str
    lock: RobustLock


class DeadlockPrevention:
    """
    Implements the 'Strict Lock Ordering + Timeout Fallback' protocol.
    """

    @staticmethod
    async def acquire_multiple(  # noqa: ASYNC109
        resources: list[LockableResource],
        timeout: float = 2.0,  # noqa: ASYNC109
        max_retries: int = 5,
    ) -> bool:
        """
        Acquires multiple locks in a strict alphabetical order with timeout/retry logic.
        """
        # Step 1: Strict Ordering (Sort the requested resources by name)
        sorted_resources = sorted(resources, key=lambda r: r.name)
        resource_names = [r.name for r in sorted_resources]
        logger.debug("Starting multi-lock acquisition for: %s", resource_names)

        for attempt in range(max_retries):
            acquired_locks: list[LockableResource] = []
            success = True

            for res in sorted_resources:
                # Inner acquisition uses no retries here because the outer loop handles it
                # But we use the robust method for the timeout logic
                if await res.lock.acquire_robust(timeout=timeout, max_retries=1):
                    acquired_locks.append(res)
                else:
                    success = False
                    break

            if success:
                logger.info("All locks acquired for: %s", resource_names)
                return True
            else:
                # Phase 4a: Release what we managed to grab
                for res in reversed(acquired_locks):
                    res.lock.release()

                logger.debug(
                    "Backing off after failed attempt %s for %s", attempt + 1, resource_names
                )
                # Phase 4b: Randomized backoff before next attempt
                await asyncio.sleep(random.uniform(0.1, 0.5))

        # CRITICAL: Failed to resolve contention
        logger.error(
            f"MULTI-LOCK TRANSACTION FAILED: Max retries ({max_retries}) reached "
            f"for resources {resource_names}. Possible systemic deadlock."
        )
        return False

    @staticmethod
    def release_multiple(resources: list[LockableResource]):
        """Releases multiple locks in reverse order."""
        # Sorting is not strictly necessary for release but good for consistency
        sorted_resources = sorted(resources, key=lambda r: r.name, reverse=True)
        for res in sorted_resources:
            if res.lock.locked():
                res.lock.release()


# Global Lock Registry to enforce unique names
_LOCK_REGISTRY: dict[str, RobustLock] = {}


def get_robust_lock(name: str) -> RobustLock:
    """Returns a named RobustLock instance, creating it if necessary."""
    if name not in _LOCK_REGISTRY:
        _LOCK_REGISTRY[name] = RobustLock(name)
    return _LOCK_REGISTRY[name]


class EventLoopMonitor:
    """
    Monitors the asyncio event loop for blocking operations.
    If the loop is delayed by more than 'threshold' seconds beyond its
    intended sleep interval, it logs a warning.
    """

    def __init__(
        self, threshold: float = 0.75, interval: float = 1.0, startup_grace: float = 300.0
    ):
        try:
            self.threshold = float(os.getenv("AURA_EVENT_LOOP_MONITOR_THRESHOLD_S", str(threshold)))
        except (TypeError, ValueError):
            self.threshold = float(threshold)
        try:
            self.active_threshold = max(
                self.threshold,
                float(os.getenv("AURA_EVENT_LOOP_MONITOR_ACTIVE_THRESHOLD_S", "5.0")),
            )
        except (TypeError, ValueError):
            self.active_threshold = max(self.threshold, 5.0)
        self.interval = interval
        try:
            self.startup_grace = float(
                os.getenv("AURA_EVENT_LOOP_MONITOR_STARTUP_GRACE_S", str(startup_grace))
            )
        except (TypeError, ValueError):
            self.startup_grace = float(startup_grace)
        self.log_transient_lag = os.getenv(
            "AURA_EVENT_LOOP_LOG_TRANSIENTS", ""
        ).strip().lower() in {"1", "true", "yes"}
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._last_lag: float = 0.0
        self._consecutive_breaches: int = 0
        self._started_at: float = 0.0
        self._last_failure_at: float = 0.0
        self._last_failure_reason: str = ""
        self._healthy_lag_samples_after_failure: int = 0
        try:
            self.hard_failure_threshold = float(
                os.getenv("AURA_EVENT_LOOP_MONITOR_HARD_FAILURE_S", "5.0")
            )
        except (TypeError, ValueError):
            self.hard_failure_threshold = 5.0
        try:
            self.failure_recovery_window_s = float(
                os.getenv("AURA_EVENT_LOOP_MONITOR_FAILURE_RECOVERY_S", "300.0")
            )
        except (TypeError, ValueError):
            self.failure_recovery_window_s = 300.0
        try:
            self.failure_recovery_samples = max(
                1,
                int(os.getenv("AURA_EVENT_LOOP_MONITOR_RECOVERY_SAMPLES", "3")),
            )
        except (TypeError, ValueError):
            self.failure_recovery_samples = 3

    def _active_runtime_reason(self) -> str | None:
        try:
            from core.runtime.proof_policy import proof_run_active

            if proof_run_active():
                return "proof_run_active"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
            logger.debug("Suppressed %s in core.utils.concurrency: %s", type(_exc).__name__, _exc)

        try:
            from core.runtime import foreground_guard

            reason = foreground_guard.foreground_activity_reason()
            if reason:
                return str(reason)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
            logger.debug("Suppressed %s in core.utils.concurrency: %s", type(_exc).__name__, _exc)

        try:
            from core.container import ServiceContainer

            gate = ServiceContainer.get("inference_gate", default=None)
            status_getter = getattr(gate, "get_conversation_status", None)
            if callable(status_getter):
                status = status_getter()
                if bool(getattr(status, "active", False)):
                    return "foreground_generation"
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as _exc:
            logger.debug("Suppressed %s in core.utils.concurrency: %s", type(_exc).__name__, _exc)

        return None

    def _lag_threshold_for_context(self) -> tuple[float, str]:
        reason = self._active_runtime_reason()
        if reason:
            return self.active_threshold, reason
        return self.threshold, "idle"

    def start(self):
        """Starts the monitor in a background task."""
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._started_at = time.perf_counter()
        self._task = get_task_tracker().create_task(self._run())
        logger.info(
            "🕒 EventLoopMonitor started (threshold=%.2fs, interval=%.1fs)",
            self.threshold,
            self.interval,
        )

    async def stop(self):
        """Stops the monitor gracefully."""
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError as _exc:
                logger.debug("Suppressed asyncio.CancelledError: %s", _exc)
        logger.info("🕒 EventLoopMonitor stopped.")

    def is_alive(self) -> bool:
        """Return True when the monitor task is running and accepting ticks."""
        if self._task is None or self._task.done() or self._stop_event.is_set():
            return False
        if self._last_failure_at and (
            self._healthy_lag_samples_after_failure < self.failure_recovery_samples
        ):
            return False
        return True

    def get_status(self) -> dict[str, Any]:
        return {
            "alive": self.is_alive(),
            "last_lag_s": self._last_lag,
            "consecutive_breaches": self._consecutive_breaches,
            "last_failure_at": self._last_failure_at,
            "last_failure_reason": self._last_failure_reason,
            "healthy_recovery_samples": self._healthy_lag_samples_after_failure,
            "required_recovery_samples": self.failure_recovery_samples,
        }

    async def _run(self):
        while not self._stop_event.is_set():
            start_time = time.perf_counter()
            try:
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break

            end_time = time.perf_counter()
            actual_elapsed = end_time - start_time
            lag = actual_elapsed - self.interval
            threshold, context = self._lag_threshold_for_context()
            in_startup_grace = (
                self.startup_grace > 0
                and self._started_at > 0
                and (end_time - self._started_at) < self.startup_grace
            )

            if lag > threshold and not in_startup_grace:
                self._last_lag = lag
                self._consecutive_breaches += 1
                if lag >= self.hard_failure_threshold:
                    self._last_failure_at = time.time()
                    self._last_failure_reason = (
                        f"hard event-loop lag {lag:.4f}s exceeded {self.hard_failure_threshold:.2f}s"
                    )
                    self._healthy_lag_samples_after_failure = 0
                    record_degradation(
                        "event_loop_monitor",
                        RuntimeError(self._last_failure_reason),
                        severity="critical",
                        action="marked event loop monitor unhealthy until healthy lag samples confirm recovery",
                        enforce_failure_policy=False,
                    )
                severe = lag >= max(threshold * 3.0, 0.50)
                if severe or self._consecutive_breaches >= 5:
                    logger.warning(
                        "🚨 EVENT LOOP LAG DETECTED: %.4fs (context=%s threshold=%.2fs, streak=%d). "
                        "Something is blocking the event loop!",
                        lag,
                        context,
                        threshold,
                        self._consecutive_breaches,
                    )
                    try:
                        from core.resilience.omni_tracer import write_trace

                        write_trace(
                            "event_loop_monitor",
                            "EventLoopLag",
                            (
                                f"lag={lag:.4f}s context={context} threshold={threshold:.2f}s "
                                f"streak={self._consecutive_breaches}"
                            ),
                        )
                    except (ImportError, AttributeError, RuntimeError) as _exc:
                        logger.debug(
                            "Suppressed %s in core.utils.concurrency: %s", type(_exc).__name__, _exc
                        )
                elif self.log_transient_lag:
                    logger.debug(
                        "EventLoopMonitor: transient lag %.4fs observed (context=%s threshold=%.2fs).",
                        lag,
                        context,
                        threshold,
                    )
            else:
                self._consecutive_breaches = 0
                if self._last_failure_at:
                    self._healthy_lag_samples_after_failure += 1
                    if self._healthy_lag_samples_after_failure >= self.failure_recovery_samples:
                        logger.info(
                            "EventLoopMonitor recovered after %d healthy lag samples.",
                            self._healthy_lag_samples_after_failure,
                        )
                        self._last_failure_at = 0.0
                        self._last_failure_reason = ""


class RobustSemaphore:
    """
    A loop-agnostic version of a semaphore that uses threading.Semaphore
    to bridge across multiple event loops.
    """

    def __init__(self, value: int = 1, name: str = "UnnamedSemaphore"):
        self.name = name
        self._sem = threading.BoundedSemaphore(value)

    async def acquire(self, timeout: float | None = None) -> bool:  # noqa: ASYNC109
        """Acquire the semaphore asynchronously using to_thread.

        When a timeout is provided, the timeout is enforced inside the backing
        thread primitive itself. That avoids orphaned background threads that
        can still acquire the permit after asyncio-side cancellation/timeout.
        """
        logger.debug("Attempting to acquire semaphore: '%s'", self.name)
        if timeout is None:
            acquired = await asyncio.to_thread(self._sem.acquire)
        else:
            acquired = await asyncio.to_thread(self._sem.acquire, True, max(0.0, float(timeout)))
        if acquired:
            logger.debug("Acquired semaphore: '%s'", self.name)
        else:
            logger.debug("Semaphore acquire timed out: '%s'", self.name)
        return bool(acquired)

    def release(self):
        """Release the underlying threading.Semaphore."""
        self._sem.release()
        logger.debug("Released semaphore: '%s'", self.name)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()


_SEM_REGISTRY: dict[str, RobustSemaphore] = {}


def get_robust_semaphore(name: str, value: int = 1) -> RobustSemaphore:
    """Returns a named RobustSemaphore instance, creating it if necessary."""
    if name not in _SEM_REGISTRY:
        _SEM_REGISTRY[name] = RobustSemaphore(value, name)
    return _SEM_REGISTRY[name]
