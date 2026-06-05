"""core/swarm/swarm_runtime.py — Swarm Runtime Orchestrator.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List

from core.swarm.ray_backend import RayBackend
from core.swarm.worker_pool import LocalWorkerPool

logger = logging.getLogger("Aura.SwarmRuntime")


class SwarmRuntime:
    """Entry point for managing computing resource pools and routing parallel subtasks."""

    def __init__(self) -> None:
        self.ray_backend = RayBackend()
        self.local_pool = LocalWorkerPool()

    async def run_subtasks(self, tasks: List[Callable[[], Any]]) -> List[Any]:
        """Runs subtasks in parallel across Ray if available, else local process pool."""
        logger.info("🐝 Swarm Runtime routing %d subtasks...", len(tasks))
        if self.ray_backend.is_available():
            logger.info("🐝 Swarm: Routing to Ray cluster backend.")
            return await self.ray_backend.execute_parallel(tasks)
        else:
            logger.info("🐝 Swarm: Routing to Local Worker Pool.")
            return await self.local_pool.execute_all(tasks)

    def shutdown(self) -> None:
        self.local_pool.shutdown()


# Singleton
_swarm_instance: SwarmRuntime | None = None


def get_swarm_runtime() -> SwarmRuntime:
    global _swarm_instance
    if _swarm_instance is None:
        _swarm_instance = SwarmRuntime()
    return _swarm_instance
