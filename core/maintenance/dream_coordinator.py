"""core/maintenance/dream_coordinator.py
Single exclusive gate for all dream/sleep/consolidation subsystems.

Prevents concurrent memory writes from DreamProcessor, DreamerV2,
maintenance dream_cycle, and resilience DreamCycle — which previously
had no coordination and could write to the same episodic store simultaneously.

Priority order (highest → lowest):
  1. resilience/DLQ re-ingestion          (every 5 min if DLQ non-empty)
  2. maintenance/WAL checkpoint + pruning  (every hour)
  3. DreamerV2 full biological sleep cycle (when idle > 10 min, every 2h)
  4. DreamProcessor                        (DEPRECATED — do not re-enable)
"""
import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, Optional

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.DreamCoordinator")

_coordinator: Optional["DreamCoordinator"] = None


def get_dream_coordinator() -> "DreamCoordinator":
    global _coordinator
    if _coordinator is None:
        _coordinator = DreamCoordinator()
    return _coordinator


class DreamCoordinator:
    """Single exclusive async lock for all memory consolidation subsystems."""

    def __init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        self._last_run: Dict[str, float] = {}
        self._running: Dict[str, bool] = {}
        self._run_count: Dict[str, int] = {}

    async def run_if_due(
        self,
        name: str,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
        interval_s: float,
        *,
        priority: int = 0,
    ) -> bool:
        """Run the named dream subsystem if its interval has elapsed.

        Returns True if the coroutine ran, False if skipped.
        Thread-safe: the internal asyncio.Lock prevents concurrent runs.
        """
        now = time.monotonic()
        last = self._last_run.get(name, 0.0)
        if now - last < interval_s:
            return False

        if self._lock.locked():
            logger.debug(
                "DreamCoordinator: '%s' skipped — another subsystem is running.", name
            )
            return False

        async with self._lock:
            # Double-check after acquiring lock
            now2 = time.monotonic()
            if now2 - self._last_run.get(name, 0.0) < interval_s:
                return False

            self._running[name] = True
            self._run_count[name] = self._run_count.get(name, 0) + 1
            logger.info(
                "🌙 DreamCoordinator: running '%s' (run #%d, priority=%d)",
                name, self._run_count[name], priority,
            )
            started = time.monotonic()
            try:
                await coro_factory()
                elapsed = time.monotonic() - started
                self._last_run[name] = time.monotonic()
                logger.info("✅ DreamCoordinator: '%s' complete in %.1fs", name, elapsed)
                return True
            except Exception as exc:
                elapsed = time.monotonic() - started
                record_degradation(
                    "dream_coordinator",
                    exc,
                    action=f"Subsystem '{name}' failed after {elapsed:.1f}s; retrying next interval",
                )
                logger.error("DreamCoordinator: '%s' failed: %s", name, exc)
                return False
            finally:
                self._running[name] = False

    def status(self) -> Dict[str, Any]:
        return {
            "last_run_monotonic": {k: round(v, 1) for k, v in self._last_run.items()},
            "currently_running": {k: v for k, v in self._running.items() if v},
            "run_counts": dict(self._run_count),
        }
