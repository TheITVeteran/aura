"""DurableWorkflowEngine — resumable multi-step task runtime.

Each long task runs as a `Workflow` with an immutable plan (a list of
`WorkflowStep`s). After every committed step, the engine writes a
durable checkpoint via the canonical AtomicWriter so the workflow can
be resumed on the next process from the exact step it failed/paused at,
without re-running already-committed side effects.

Key contracts:

* Each step has a unique ``step_id`` and an ``apply`` callable.
* ``apply`` receives the prior outputs and must be idempotent — the
  engine guarantees it is invoked at-most-once per ``step_id`` across
  resumes by checking the checkpoint first.
* A step may flag ``human_approval=True``; the engine pauses there and
  records a checkpoint with status PAUSED until ``resume()`` is called.
* If a step fails and ``rollback`` is provided, the engine runs rollback
  and marks the workflow FAILED.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_json, read_json_envelope
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.DurableWorkflow")

_WORKFLOW_STEP_ERRORS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED_FOR_APPROVAL = "paused_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class WorkflowStep:
    step_id: str
    name: str
    apply: Callable[[dict[str, Any]], Any | Awaitable[Any]]
    rollback: Callable[[dict[str, Any]], None | Awaitable[None]] | None = None
    human_approval: bool = False
    receipt_id: str | None = None


@dataclass
class WorkflowCheckpoint:
    workflow_id: str
    objective: str
    status: WorkflowStatus
    completed_steps: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    failed_step: str | None = None
    failure_reason: str | None = None
    paused_at_step: str | None = None
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


#: Statuses that describe work still owed. PAUSED_FOR_APPROVAL counts: the
#: workflow is waiting on a human, not finished, and a restart must not lose it.
_RESUMABLE_STATUSES = frozenset({
    WorkflowStatus.PENDING,
    WorkflowStatus.RUNNING,
    WorkflowStatus.PAUSED_FOR_APPROVAL,
})


class WorkflowStore:
    """Atomic-writer-backed checkpoint store."""

    SCHEMA_VERSION = 1

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else (state_root() / "workflows")
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: WorkflowCheckpoint) -> None:
        path = self.root / f"{checkpoint.workflow_id}.json"
        payload = asdict(checkpoint)
        payload["status"] = checkpoint.status.value
        atomic_write_json(
            path,
            payload,
            schema_version=self.SCHEMA_VERSION,
            schema_name="workflow_checkpoint",
        )

    def unfinished(self) -> list[WorkflowCheckpoint]:
        """Every workflow that was interrupted rather than completed.

        This is what made resume() unreachable in practice: the store could
        save and load BY ID, but nothing could discover which workflows were
        still owed work — and knowing the id is exactly what a crash destroys.
        Recovery has to start from "what was I doing?", not from a caller
        remembering.

        Corrupt or unreadable checkpoints are skipped rather than aborting the
        scan: one bad file must not hide every other resumable workflow.
        """
        out: list[WorkflowCheckpoint] = []
        try:
            paths = sorted(self.root.glob("*.json"))
        except OSError:
            return out
        for path in paths:
            try:
                env = read_json_envelope(path)
                payload = env.get("payload") or {}
                payload["status"] = WorkflowStatus(payload.get("status", "pending"))
                checkpoint = WorkflowCheckpoint(**payload)
            except (OSError, ValueError, TypeError, KeyError):
                continue
            if checkpoint.status in _RESUMABLE_STATUSES:
                out.append(checkpoint)
        return out

    def load(self, workflow_id: str) -> WorkflowCheckpoint | None:
        path = self.root / f"{workflow_id}.json"
        if not path.exists():
            return None
        env = read_json_envelope(path)
        payload = env.get("payload") or {}
        payload["status"] = WorkflowStatus(payload.get("status", "pending"))
        return WorkflowCheckpoint(**payload)


class DurableWorkflowEngine:
    def __init__(self, *, store: WorkflowStore | None = None):
        self.store = store or WorkflowStore()

    async def run(
        self,
        objective: str,
        steps: list[WorkflowStep],
        *,
        workflow_id: str | None = None,
    ) -> WorkflowCheckpoint:
        workflow_id = workflow_id or f"wf-{uuid.uuid4()}"
        checkpoint = self.store.load(workflow_id)
        if checkpoint is None:
            checkpoint = WorkflowCheckpoint(
                workflow_id=workflow_id,
                objective=objective,
                status=WorkflowStatus.RUNNING,
            )
            self.store.save(checkpoint)
        else:
            checkpoint.status = WorkflowStatus.RUNNING
            self.store.save(checkpoint)

        for step in steps:
            if step.step_id in checkpoint.completed_steps:
                continue  # Idempotent: skip already-committed
            if step.human_approval:
                checkpoint.status = WorkflowStatus.PAUSED_FOR_APPROVAL
                checkpoint.paused_at_step = step.step_id
                checkpoint.updated_at = time.time()
                self.store.save(checkpoint)
                return checkpoint
            try:
                result = step.apply(checkpoint.outputs)
                if asyncio.iscoroutine(result):
                    result = await result
                checkpoint.outputs[step.step_id] = result
                checkpoint.completed_steps.append(step.step_id)
                checkpoint.updated_at = time.time()
                self.store.save(checkpoint)
            except _WORKFLOW_STEP_ERRORS as exc:
                checkpoint.failed_step = step.step_id
                checkpoint.failure_reason = repr(exc)
                checkpoint.status = WorkflowStatus.FAILED
                checkpoint.updated_at = time.time()
                self.store.save(checkpoint)
                if step.rollback is not None:
                    try:
                        rb = step.rollback(checkpoint.outputs)
                        if asyncio.iscoroutine(rb):
                            await rb
                    except _WORKFLOW_STEP_ERRORS as rb_exc:
                        logger.error(
                            "Workflow %s rollback for %s failed: %s",
                            workflow_id, step.step_id, rb_exc,
                        )
                return checkpoint

        checkpoint.status = WorkflowStatus.COMPLETED
        checkpoint.updated_at = time.time()
        self.store.save(checkpoint)
        return checkpoint

    async def resume(
        self,
        workflow_id: str,
        steps: list[WorkflowStep],
    ) -> WorkflowCheckpoint:
        return await self.run(objective=workflow_id, steps=steps, workflow_id=workflow_id)
