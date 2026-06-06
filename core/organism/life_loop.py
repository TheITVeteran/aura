"""core/organism/life_loop.py
Unified canonical life loop executor for Aura.
Maintains continuous non-blocking background ticks.
"""
import asyncio
import logging
from typing import Optional

from core.organism.life_state import LifeState
from core.organism.life_tick import LifeTickProcessor
from core.organism.cycle_clock import CycleClock
from core.event_bus import get_event_bus, EventPriority
from core.runtime.errors import record_degradation

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
    """Master organism control daemon running the continuous life cycles."""

    def __init__(self, tick_rate_hz: float = 0.5):
        self.state = LifeState()
        self.processor = LifeTickProcessor()
        self.clock = CycleClock(tick_rate_hz=tick_rate_hz)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.event_bus = get_event_bus()

    async def start(self) -> None:
        """Starts the canonical life loop in the background."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop_run())
        logger.info("Aura Canonical Life Loop started.")

    async def stop(self) -> None:
        """Stops the canonical life loop gracefully."""
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Aura Canonical Life Loop stopped.")

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
            except asyncio.CancelledError:
                break
            except _LIFE_LOOP_RECOVERABLE_ERRORS as e:
                record_degradation("organism.life_loop", e)
                logger.error("Error in life tick execution: %s", e, exc_info=True)

            # Sleep calculated interval
            tick_duration = asyncio.get_event_loop().time() - start_time
            sleep_interval = max(0.1, self.clock.tick_sleep(tick_duration) - tick_duration)
            await asyncio.sleep(sleep_interval)


# Singleton lifecycle access
_life_loop: Optional[LifeLoop] = None


def get_life_loop() -> LifeLoop:
    """Canonical singleton provider for the organism life loop."""
    global _life_loop
    if _life_loop is None:
        _life_loop = LifeLoop()
    return _life_loop
