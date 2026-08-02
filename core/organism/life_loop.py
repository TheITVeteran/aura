"""Compatibility loop for isolated organism simulations and tests.

The live Aura runtime uses ``core.mind_tick.MindTick`` as its single cognitive
and organism rhythm. This module remains available for boxed simulations that
operate on the standalone ``LifeState`` schema; it must not be started beside
MindTick in production.
"""
import asyncio
import logging
import os
import time
from typing import Optional

from core.organism.life_state import LifeState
from core.organism.life_tick import LifeTickProcessor
from core.organism.cycle_clock import CycleClock
from core.event_bus import get_event_bus, EventPriority
from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Organism.LifeLoop")

_LIFE_LOOP_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class LifeLoop:
    """Standalone organism simulation loop, separate from the live runtime."""

    def __init__(self, tick_rate_hz: float = 0.5):
        self.state = LifeState()
        self.processor = LifeTickProcessor()
        self.clock = CycleClock(tick_rate_hz=tick_rate_hz)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.event_bus = get_event_bus()
        self._consecutive_failures = 0
        self._last_error = ""
        self._last_success_at = 0.0

    #: Env override for the one legitimate case: deliberately running the
    #: boxed simulation on a host that also has a live runtime registered.
    ALLOW_BESIDE_MIND_TICK_ENV = "AURA_ALLOW_LIFE_LOOP_BESIDE_MIND_TICK"

    @staticmethod
    def _mind_tick_is_live() -> bool:
        """Is the production cognitive rhythm already running?"""
        try:
            from core.container import ServiceContainer

            return ServiceContainer.get("mind_tick", default=None) is not None
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            # Cannot tell. Say no rather than blocking a boxed simulation on
            # a container that is not up — this guard exists to prevent a
            # double rhythm, not to gate the simulator on import health.
            return False

    async def start(self) -> None:
        """Start the standalone simulation loop in the background.

        This module's own docstring has always said the loop "must not be
        started beside MindTick in production". That was a comment, and
        ``start()`` enforced nothing — a caller who had not read it got two
        organism rhythms driving one state, which is exactly the failure the
        sentence was written to prevent.

        The rule is now the code. Refuses rather than warns, because a second
        life loop is not a degraded mode: both rhythms would write the same
        organism state and neither would be wrong from where it stood.
        """
        if self._running:
            return

        if self._mind_tick_is_live() and not os.environ.get(
            self.ALLOW_BESIDE_MIND_TICK_ENV, ""
        ).strip():
            message = (
                "refusing to start the standalone organism simulation: MindTick "
                "is live and is the runtime's single organism rhythm. Set "
                f"{self.ALLOW_BESIDE_MIND_TICK_ENV}=1 to override deliberately."
            )
            logger.error("%s", message)
            record_degradation(
                "organism_life_loop",
                RuntimeError(message),
                severity="warning",
                action="refused to start a second organism rhythm beside MindTick",
                enforce_failure_policy=False,
            )
            return

        self._running = True
        self._task = get_task_tracker().create_task(
            self._loop_run(),
            name="organism.life_loop",
        )
        logger.info("Standalone organism simulation loop started.")

    async def stop(self) -> None:
        """Stop the standalone simulation loop gracefully."""
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("Life loop task acknowledged cancellation.")
        logger.info("Standalone organism simulation loop stopped.")

    async def _loop_run(self) -> None:
        """Core asynchronous daemon executing continuous life ticks."""
        while self._running:
            start_time = asyncio.get_event_loop().time()
            try:
                # Execute unified tick
                await self.processor.execute_tick(self.state)

                # Broadcast LifeState snapshot to the event bus
                await self.event_bus.publish(
                    "organism.state_tick",
                    self.state.to_dict(),
                    priority=EventPriority.AUTONOMIC
                )
                self._consecutive_failures = 0
                self._last_error = ""
                self._last_success_at = time.time()
            except asyncio.CancelledError:
                break
            except _LIFE_LOOP_RECOVERABLE_ERRORS as e:
                self._consecutive_failures += 1
                self._last_error = f"{type(e).__name__}: {e}"
                if self._consecutive_failures >= 3:
                    self.processor = LifeTickProcessor()
                record_degradation(
                    "organism.life_loop",
                    e,
                    severity="degraded",
                    action=(
                        "kept the supervised life loop alive with bounded backoff"
                        + (
                            " and rebuilt the tick processor after repeated failures"
                            if self._consecutive_failures >= 3
                            else ""
                        )
                    ),
                )
                logger.error("Error in life tick execution: %s", e, exc_info=True)

            # Sleep calculated interval
            tick_duration = asyncio.get_event_loop().time() - start_time
            sleep_interval = max(0.1, self.clock.tick_sleep(tick_duration) - tick_duration)
            if self._consecutive_failures:
                sleep_interval = max(
                    sleep_interval,
                    min(30.0, float(2 ** min(self._consecutive_failures, 5))),
                )
            await asyncio.sleep(sleep_interval)

    def get_health_status(self) -> dict[str, object]:
        """Expose whether the standalone simulation loop is making forward progress."""
        return {
            "running": self._running,
            "task_alive": bool(self._task and not self._task.done()),
            "consecutive_failures": self._consecutive_failures,
            "last_error": self._last_error,
            "last_success_at": self._last_success_at,
            "healthy": bool(
                self._running
                and self._task
                and not self._task.done()
                and self._consecutive_failures == 0
                and self._last_success_at > 0.0
            ),
        }


# Singleton lifecycle access
_life_loop: Optional[LifeLoop] = None


def get_life_loop() -> LifeLoop:
    """Canonical singleton provider for the organism life loop."""
    global _life_loop
    if _life_loop is None:
        _life_loop = LifeLoop()
    return _life_loop
