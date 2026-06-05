"""core/world/perception_scheduler.py — Periodic Ingestion Perception Scheduler.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from core.world.perception_hub import get_perception_hub

logger = logging.getLogger("Aura.PerceptionScheduler")


class PerceptionScheduler:
    """Schedules and runs background perception sweeps."""

    def __init__(self, interval_seconds: float = 3600.0) -> None:
        self.interval = interval_seconds
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("⏱️  Perception Scheduler started with interval %.1fs", self.interval)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("⏱️  Perception Scheduler stopped.")

    async def _loop(self) -> None:
        hub = get_perception_hub()
        while self._running:
            try:
                # Periodic general intelligence sweep
                await hub.perceive(query="AI agents, sovereign runtime, model councils")
            except Exception as e:
                logger.error("Error during scheduled perception sweep: %s", e)
            await asyncio.sleep(self.interval)
