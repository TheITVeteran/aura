"""Continuum adapter for Aura main-15.

Aura already has BeingRuntime, ContinuousSelfField, SemanticStream, WelfareState,
and organism life loops. This scheduler does not spin the LLM in the dark. It
schedules cheap continuity work and yields immediately to external user I/O.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable
import asyncio
import inspect
import time
import uuid


JobFunc = Callable[[], Any | Awaitable[Any]]
_CONTINUITY_JOB_ERRORS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


@dataclass
class ContinuityJob:
    name: str
    cadence_s: float
    budget_cost: float
    priority: int
    func: JobFunc
    requires_idle: bool = True
    permission_level: str = "maintenance"
    last_run: float = 0.0
    failure_count: int = 0
    job_id: str = field(default_factory=lambda: f"continuity-job-{uuid.uuid4()}")

    def due(self, now: float) -> bool:
        return (now - self.last_run) >= self.cadence_s


class ContinuumAdapter:
    def __init__(self, *, production_mode: bool = True, max_budget_per_tick: float = 1.0) -> None:
        self.production_mode = production_mode
        self.max_budget_per_tick = float(max_budget_per_tick)
        self.external_io_active = False
        self.jobs: list[ContinuityJob] = []
        self.event_log: list[dict[str, Any]] = []

    def add_job(self, job: ContinuityJob) -> None:
        self.jobs.append(job)
        self.jobs.sort(key=lambda j: (-j.priority, j.name))
        self.event_log.append({"event": "job_added", "job": self._job_payload(job), "timestamp": time.time()})

    @staticmethod
    def _job_payload(job: ContinuityJob) -> dict[str, Any]:
        payload = asdict(job)
        payload.pop("func", None)
        return payload

    def set_external_io(self, active: bool) -> None:
        self.external_io_active = bool(active)
        self.event_log.append({"event": "external_io", "active": self.external_io_active, "timestamp": time.time()})

    async def tick(self, *, available_budget: float | None = None, now: float | None = None) -> list[str]:
        now = time.time() if now is None else now
        budget = self.max_budget_per_tick if available_budget is None else min(self.max_budget_per_tick, float(available_budget))
        ran: list[str] = []
        for job in list(self.jobs):
            if not job.due(now):
                continue
            if job.requires_idle and self.external_io_active:
                self.event_log.append({"event": "job_deferred", "job": job.name, "reason": "external_io_active", "timestamp": now})
                continue
            if job.budget_cost > budget:
                self.event_log.append({"event": "job_deferred", "job": job.name, "reason": "budget_exceeded", "timestamp": now})
                continue
            if self.production_mode and job.permission_level not in {"maintenance", "consolidation", "audit"}:
                self.event_log.append({"event": "job_blocked", "job": job.name, "reason": "permission_not_allowed", "timestamp": now})
                continue
            try:
                result = job.func()
                if inspect.isawaitable(result):
                    result = await result
                job.last_run = now
                budget -= job.budget_cost
                ran.append(job.name)
                self.event_log.append({"event": "job_ran", "job": job.name, "result": str(result)[:300], "timestamp": now})
            except _CONTINUITY_JOB_ERRORS as exc:
                job.failure_count += 1
                self.event_log.append({"event": "job_failed", "job": job.name, "error": f"{type(exc).__name__}: {exc}", "timestamp": now})
        return ran


def install_default_continuity_jobs(adapter: ContinuumAdapter, *, being_runtime: Any) -> None:
    """Register cheap continuity jobs against existing BeingRuntime organs."""

    def evolve_semantic_stream() -> str:
        stream = getattr(being_runtime, "semantic_stream", None)
        if stream is not None and hasattr(stream, "evolve"):
            stream.evolve()
            return "semantic_stream.evolve"
        return "semantic_stream_unavailable"

    def sample_idle_state() -> str:
        if hasattr(being_runtime, "sample"):
            being_runtime.sample(None, objective="")
            return "being_runtime.sample_idle"
        return "being_runtime_unavailable"

    adapter.add_job(ContinuityJob("semantic_stream_evolve", 5.0, 0.10, 10, evolve_semantic_stream, requires_idle=True, permission_level="consolidation"))
    adapter.add_job(ContinuityJob("idle_being_sample", 15.0, 0.20, 5, sample_idle_state, requires_idle=True, permission_level="maintenance"))
