"""Parallel execution — run many fluid loops at once (the CI 'fork').

The existing swarm (``swarm_delegation``) forks *cognitive* sub-agents (parallel LLM
reasoning). This is the complementary half: forking parallel **action** execution, so
Aura can actually *do* several things in the world simultaneously — open Notes, research
three sources, and set a wallpaper at the same time — each as its own governed,
verified :class:`FluidExecutor` loop.

Each task runs in its own executor (no shared mutable state), under a concurrency bound
(a semaphore — Aura's hands are finite even if her attention forks) with a per-task
timeout. Failures and timeouts are isolated: one worker stalling never aborts the
others. The result is a :class:`SwarmReceipt` aggregating every task's outcome — the
provenance the autonomy layer uses to decide what to do next.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from core.runtime.errors import record_degradation
from core.skills.fluid_executor import ExecutionReceipt, FluidExecutor, Step

logger = logging.getLogger("Aura.ParallelExecutor")


@dataclass
class ParallelTask:
    goal: str
    steps: list[Step]


@dataclass
class SwarmReceipt:
    tasks: list[ExecutionReceipt] = field(default_factory=list)
    elapsed_s: float = 0.0
    peak_concurrency: int = 0

    @property
    def completed_count(self) -> int:
        return sum(1 for t in self.tasks if t.completed)

    @property
    def stalled_count(self) -> int:
        return sum(1 for t in self.tasks if t.stalled)

    @property
    def failed_count(self) -> int:
        return sum(1 for t in self.tasks if not t.completed and not t.stalled)

    @property
    def all_completed(self) -> bool:
        return bool(self.tasks) and all(t.completed for t in self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed": self.completed_count,
            "stalled": self.stalled_count,
            "failed": self.failed_count,
            "all_completed": self.all_completed,
            "peak_concurrency": self.peak_concurrency,
            "elapsed_s": round(self.elapsed_s, 3),
            "tasks": [t.to_dict() for t in self.tasks],
        }


class ParallelExecutor:
    """Fan out independent goals as concurrent, bounded, isolated fluid loops."""

    def __init__(
        self,
        *,
        max_concurrency: int = 4,
        per_task_timeout_s: float = 120.0,
        executor_factory: Callable[[], FluidExecutor] | None = None,
    ) -> None:
        self.max_concurrency = max(1, int(max_concurrency))
        self.per_task_timeout_s = float(per_task_timeout_s)
        self._executor_factory = executor_factory or (lambda: FluidExecutor())
        self._active = 0
        self._peak = 0
        self._active_lock = asyncio.Lock()

    async def _run_one(self, task: ParallelTask, sem: asyncio.Semaphore) -> ExecutionReceipt:
        async with sem:
            async with self._active_lock:
                self._active += 1
                self._peak = max(self._peak, self._active)
            try:
                executor = self._executor_factory()
                try:
                    return await asyncio.wait_for(
                        executor.run(task.goal, task.steps),
                        timeout=self.per_task_timeout_s,
                    )
                except asyncio.TimeoutError:
                    logger.warning("⏱️ [Parallel] task '%s' timed out after %.0fs", task.goal, self.per_task_timeout_s)
                    return ExecutionReceipt(goal=task.goal, completed=False, stalled=True, elapsed_s=self.per_task_timeout_s)
                except (RuntimeError, AttributeError, TypeError, ValueError, OSError) as exc:
                    record_degradation("parallel_executor", exc)
                    return ExecutionReceipt(goal=task.goal, completed=False, stalled=False)
            finally:
                async with self._active_lock:
                    self._active -= 1

    async def run(self, tasks: list[ParallelTask]) -> SwarmReceipt:
        """Run all tasks concurrently (bounded), isolating failures. Returns a receipt."""
        receipt = SwarmReceipt()
        if not tasks:
            return receipt
        started = time.monotonic()
        self._active = 0
        self._peak = 0
        sem = asyncio.Semaphore(self.max_concurrency)
        results = await asyncio.gather(
            *[self._run_one(t, sem) for t in tasks],
            return_exceptions=True,
        )
        for task, res in zip(tasks, results, strict=True):
            if isinstance(res, ExecutionReceipt):
                receipt.tasks.append(res)
            else:  # an unexpected exception escaped — isolate it as a failed task
                if isinstance(res, BaseException):
                    record_degradation("parallel_executor", res)
                receipt.tasks.append(ExecutionReceipt(goal=task.goal, completed=False, stalled=False))
        receipt.peak_concurrency = self._peak
        receipt.elapsed_s = time.monotonic() - started
        logger.info(
            "🍴 [Parallel] %d tasks → %d completed, %d stalled, %d failed (peak concurrency %d)",
            len(tasks), receipt.completed_count, receipt.stalled_count, receipt.failed_count, receipt.peak_concurrency,
        )
        return receipt


_instance: ParallelExecutor | None = None


def get_parallel_executor() -> ParallelExecutor:
    global _instance
    if _instance is None:
        _instance = ParallelExecutor()
    return _instance
