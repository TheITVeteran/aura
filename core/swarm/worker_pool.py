"""core/swarm/worker_pool.py — Swarm Local Worker Pool.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List

logger = logging.getLogger("Aura.SwarmPool")


class LocalWorkerPool:
    """Orchestrates parallel execution pools locally on the host machine."""

    def __init__(self, pool_size: int = 8) -> None:
        self.pool_size = pool_size
        self.executor = ThreadPoolExecutor(max_workers=pool_size)

    async def execute_all(self, tasks: List[Callable[[], Any]]) -> List[Any]:
        """Dispatches tasks in parallel using the ThreadPoolExecutor."""
        logger.info("🧵 Swarm LocalPool: dispatching %d tasks locally...", len(tasks))
        loop = asyncio.get_running_loop()
        futures = [loop.run_in_executor(self.executor, t) for t in tasks]
        return await asyncio.gather(*futures)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)
