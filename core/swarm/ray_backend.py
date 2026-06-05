"""core/swarm/ray_backend.py — Distributed Swarm Ray Integration.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, List

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SwarmRay")
_RAY_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    ray = None
    _RAY_AVAILABLE = False


class RayBackend:
    """Interfaces with a Ray distributed cluster to dispatch tasks across worker nodes."""

    def __init__(self) -> None:
        self.active = False
        if _RAY_AVAILABLE:
            try:
                # Eagerly initialize ray if not already initialized
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                self.active = True
                logger.info("⚡ Ray distributed backend connected successfully.")
            except _RAY_RECOVERABLE_ERRORS as e:
                record_degradation(
                    "ray_backend",
                    e,
                    action="used local thread execution after optional Ray backend initialization failed",
                )
                logger.warning("Failed to initialize Ray cluster: %s. Falling back to local.", e)

    def is_available(self) -> bool:
        return self.active

    async def execute_parallel(self, tasks: List[Callable[[], Any]]) -> List[Any]:
        """Dispatches tasks in parallel across Ray actors."""
        if not self.active:
            # Fallback local execute
            import asyncio
            futures = [asyncio.to_thread(t) for t in tasks]
            return await asyncio.gather(*futures)

        # Ray remote task execution
        @ray.remote
        def ray_task_runner(fn: Callable[[], Any]) -> Any:
            return fn()

        logger.info("⚡ Swarm Ray: dispatching %d tasks to cluster...", len(tasks))
        ray_refs = [ray_task_runner.remote(t) for t in tasks]
        return ray.get(ray_refs)
