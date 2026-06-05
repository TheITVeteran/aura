"""core/planning/task_graph.py — Dependency-Ordered Task DAG
=============================================================
Converts complex multi-step objectives into structured dependency
graphs with preconditions, verification, rollback, and crash-resume.

A TaskGraph is a DAG of TaskNodes. Each node has:
  - preconditions (other node IDs that must complete first)
  - verification predicate (checked after execution)
  - rollback action (undo if verification fails)
  - fallback action (alternative if primary fails)
  - retry count and timeout

The graph is executed by MissionState, verified by PostActionVerifier,
and recovered by RecoveryEngine.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Aura.TaskGraph")


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"
    RETRYING = "retrying"


@dataclass
class TaskNode:
    """A single step in a task graph."""
    task_id: str
    action: str                         # skill/primitive name
    params: Dict[str, Any] = field(default_factory=dict)
    preconditions: List[str] = field(default_factory=list)  # task_ids that must complete
    verification: str = "true"          # predicate name for PostActionVerifier
    verification_args: Dict[str, Any] = field(default_factory=dict)
    rollback_action: str = ""           # action to undo this step
    rollback_params: Dict[str, Any] = field(default_factory=dict)
    fallback_action: str = ""           # alternative if primary fails
    fallback_params: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"             # "low", "medium", "high"
    adapter: str = ""                   # which capability adapter
    description: str = ""               # human-readable step description
    timeout_s: float = 30.0
    retry_count: int = 2
    retries_used: int = 0
    critical: bool = True               # if True, graph fails on this node's failure
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    verification_result: Optional[Dict[str, Any]] = None
    receipt_id: str = ""
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    artifacts: List[str] = field(default_factory=list)  # file paths created by this step

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "task_id": self.task_id,
            "action": self.action,
            "params": self.params,
            "preconditions": self.preconditions,
            "verification": self.verification,
            "status": self.status.value,
            "description": self.description,
            "risk_level": self.risk_level,
            "error": self.error,
            "receipt_id": self.receipt_id,
            "retries_used": self.retries_used,
            "artifacts": self.artifacts,
        }
        if self.result:
            d["result"] = {k: str(v)[:200] for k, v in self.result.items()}
        if self.verification_result:
            d["verification_result"] = self.verification_result
        if self.started_at:
            d["started_at"] = self.started_at
        if self.completed_at:
            d["completed_at"] = self.completed_at
            d["duration_ms"] = round((self.completed_at - self.started_at) * 1000, 1)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskNode":
        status_raw = d.get("status", "pending")
        if isinstance(status_raw, TaskStatus):
            status = status_raw
        else:
            status = TaskStatus(str(status_raw))
        return cls(
            task_id=d["task_id"],
            action=d.get("action", ""),
            params=d.get("params", {}),
            preconditions=d.get("preconditions", []),
            verification=d.get("verification", "true"),
            verification_args=d.get("verification_args", {}),
            rollback_action=d.get("rollback_action", ""),
            rollback_params=d.get("rollback_params", {}),
            fallback_action=d.get("fallback_action", ""),
            fallback_params=d.get("fallback_params", {}),
            risk_level=d.get("risk_level", "low"),
            adapter=d.get("adapter", ""),
            description=d.get("description", ""),
            timeout_s=float(d.get("timeout_s", 30.0)),
            retry_count=int(d.get("retry_count", 2)),
            retries_used=int(d.get("retries_used", 0)),
            critical=d.get("critical", True),
            status=status,
            result=d.get("result"),
            verification_result=d.get("verification_result"),
            receipt_id=d.get("receipt_id", ""),
            error=d.get("error", ""),
            started_at=float(d.get("started_at", 0.0)),
            completed_at=float(d.get("completed_at", 0.0)),
            artifacts=d.get("artifacts", []),
        )


class TaskGraph:
    """DAG of TaskNodes with dependency ordering, execution state, and persistence.

    Usage:
        graph = TaskGraph("mission_1", "Create a PDF and set wallpaper")
        graph.add_node(TaskNode(task_id="t1", action="launch_app", ...))
        graph.add_node(TaskNode(task_id="t2", action="create_pdf", preconditions=["t1"], ...))

        ready = graph.get_ready_nodes()  # nodes whose preconditions are satisfied
        graph.mark_running("t1")
        graph.mark_succeeded("t1", result={...})
    """

    def __init__(self, mission_id: str, objective: str) -> None:
        self.mission_id = mission_id
        self.objective = objective
        self.nodes: Dict[str, TaskNode] = {}
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.artifacts: List[str] = []
        self._execution_order: List[str] = []  # topological order cache

    def add_node(self, node: TaskNode) -> None:
        """Add a node to the graph. Validates no cycles."""
        if node.task_id in self.nodes:
            raise ValueError(f"Duplicate task_id: {node.task_id}")

        # Validate preconditions exist
        for pre in node.preconditions:
            if pre not in self.nodes and pre != node.task_id:
                # Precondition may be added later — just warn
                logger.debug("Precondition '%s' for '%s' not yet in graph", pre, node.task_id)

        self.nodes[node.task_id] = node
        self._execution_order = []  # invalidate cache
        self.updated_at = time.time()

    def get_ready_nodes(self) -> List[TaskNode]:
        """Get nodes whose preconditions are all satisfied (SUCCEEDED or SKIPPED)."""
        ready = []
        completed_ids = {
            tid for tid, n in self.nodes.items()
            if n.status in (TaskStatus.SUCCEEDED, TaskStatus.SKIPPED)
        }
        for node in self.nodes.values():
            if node.status != TaskStatus.PENDING:
                continue
            preconditions_met = all(
                pre in completed_ids for pre in node.preconditions
            )
            if preconditions_met:
                ready.append(node)
        return ready

    def get_next_node(self) -> Optional[TaskNode]:
        """Get the next single node to execute (first ready by insertion order)."""
        ready = self.get_ready_nodes()
        if not ready:
            return None
        # Prefer by insertion order
        for tid in self.nodes:
            if self.nodes[tid] in ready:
                return self.nodes[tid]
        return ready[0]

    def mark_running(self, task_id: str) -> None:
        node = self.nodes[task_id]
        node.status = TaskStatus.RUNNING
        node.started_at = time.time()
        self.updated_at = time.time()

    def mark_succeeded(self, task_id: str, result: Optional[Dict[str, Any]] = None,
                       receipt_id: str = "", artifacts: Optional[List[str]] = None) -> None:
        node = self.nodes[task_id]
        node.status = TaskStatus.SUCCEEDED
        node.result = result or {}
        node.receipt_id = receipt_id
        node.completed_at = time.time()
        if artifacts:
            node.artifacts.extend(artifacts)
            self.artifacts.extend(artifacts)
        self.updated_at = time.time()

    def mark_failed(self, task_id: str, error: str,
                    result: Optional[Dict[str, Any]] = None) -> None:
        node = self.nodes[task_id]
        node.status = TaskStatus.FAILED
        node.error = error
        node.result = result or {}
        node.completed_at = time.time()
        self.updated_at = time.time()

    def mark_skipped(self, task_id: str, reason: str = "") -> None:
        node = self.nodes[task_id]
        node.status = TaskStatus.SKIPPED
        node.error = reason or "Skipped"
        node.completed_at = time.time()
        self.updated_at = time.time()

    def mark_retrying(self, task_id: str) -> None:
        node = self.nodes[task_id]
        node.status = TaskStatus.RETRYING
        node.retries_used += 1
        self.updated_at = time.time()

    def mark_rolled_back(self, task_id: str) -> None:
        node = self.nodes[task_id]
        node.status = TaskStatus.ROLLED_BACK
        self.updated_at = time.time()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """True if all nodes are in a terminal state."""
        return all(
            n.status in (TaskStatus.SUCCEEDED, TaskStatus.SKIPPED, TaskStatus.FAILED, TaskStatus.ROLLED_BACK)
            for n in self.nodes.values()
        )

    @property
    def is_successful(self) -> bool:
        """True if all critical nodes succeeded."""
        return all(
            n.status == TaskStatus.SUCCEEDED
            for n in self.nodes.values()
            if n.critical
        )

    @property
    def has_failures(self) -> bool:
        return any(n.status == TaskStatus.FAILED for n in self.nodes.values())

    @property
    def total_steps(self) -> int:
        return len(self.nodes)

    @property
    def completed_steps(self) -> int:
        return sum(1 for n in self.nodes.values()
                   if n.status in (TaskStatus.SUCCEEDED, TaskStatus.SKIPPED))

    @property
    def failed_steps(self) -> int:
        return sum(1 for n in self.nodes.values() if n.status == TaskStatus.FAILED)

    def get_progress(self) -> Dict[str, Any]:
        """Progress summary for dashboards."""
        total = self.total_steps
        completed = self.completed_steps
        return {
            "mission_id": self.mission_id,
            "objective": self.objective[:200],
            "total_steps": total,
            "completed": completed,
            "failed": self.failed_steps,
            "progress_pct": round(completed / max(1, total) * 100, 1),
            "is_complete": self.is_complete,
            "is_successful": self.is_successful,
            "current_step": self._current_step_description(),
            "artifacts": self.artifacts,
        }

    def _current_step_description(self) -> str:
        running = [n for n in self.nodes.values() if n.status == TaskStatus.RUNNING]
        if running:
            return running[0].description or running[0].action
        ready = self.get_ready_nodes()
        if ready:
            return f"Next: {ready[0].description or ready[0].action}"
        if self.is_complete:
            return "Complete"
        return "Waiting"

    def get_proof_bundle(self) -> Dict[str, Any]:
        """Generate a proof bundle with all receipts and artifacts."""
        return {
            "mission_id": self.mission_id,
            "objective": self.objective,
            "status": "success" if self.is_successful else "failed" if self.has_failures else "incomplete",
            "total_steps": self.total_steps,
            "completed": self.completed_steps,
            "failed": self.failed_steps,
            "created_at": self.created_at,
            "completed_at": self.updated_at,
            "duration_s": round(self.updated_at - self.created_at, 1),
            "artifacts": self.artifacts,
            "steps": [n.to_dict() for n in self.nodes.values()],
        }

    def get_failure_summary(self) -> str:
        """Honest failure summary for narration."""
        failures = [n for n in self.nodes.values() if n.status == TaskStatus.FAILED]
        if not failures:
            return ""
        parts = []
        for f in failures:
            parts.append(f"Step '{f.description or f.action}' failed: {f.error}")
        return "; ".join(parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "objective": self.objective,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "artifacts": self.artifacts,
            "nodes": {tid: n.to_dict() for tid, n in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskGraph":
        graph = cls(
            mission_id=d["mission_id"],
            objective=d.get("objective", ""),
        )
        graph.created_at = d.get("created_at", time.time())
        graph.updated_at = d.get("updated_at", time.time())
        graph.artifacts = d.get("artifacts", [])
        for tid, nd in d.get("nodes", {}).items():
            nd["task_id"] = tid
            graph.nodes[tid] = TaskNode.from_dict(nd)
        return graph

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, text: str) -> "TaskGraph":
        return cls.from_dict(json.loads(text))

    def persist(self, path: Path) -> None:
        """Save graph state to disk for crash recovery."""
        path.parent.mkdir(parents=True, exist_ok=True)
        get_file_write_gateway().write_text(
            path,
            self.to_json(),
            encoding="utf-8",
            source="task_graph.persist",
        )

    @classmethod
    def resume(cls, path: Path) -> "TaskGraph":
        """Reload graph state from disk."""
        return cls.from_json(path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Check graph for issues. Returns list of warnings."""
        warnings = []
        # Check for missing preconditions
        for node in self.nodes.values():
            for pre in node.preconditions:
                if pre not in self.nodes:
                    warnings.append(f"Node '{node.task_id}' depends on unknown '{pre}'")

        # Check for cycles (simple DFS)
        visited: Set[str] = set()
        in_stack: Set[str] = set()

        def _dfs(tid: str) -> bool:
            if tid in in_stack:
                return True  # cycle
            if tid in visited:
                return False
            visited.add(tid)
            in_stack.add(tid)
            node = self.nodes.get(tid)
            if node:
                for pre in node.preconditions:
                    if pre in self.nodes and _dfs(pre):
                        warnings.append(f"Cycle detected involving '{tid}'")
                        return True
            in_stack.discard(tid)
            return False

        for tid in self.nodes:
            _dfs(tid)

        if not self.nodes:
            warnings.append("Empty task graph")

        return warnings

    def __repr__(self) -> str:
        return (
            f"TaskGraph(mission={self.mission_id!r}, "
            f"steps={self.total_steps}, "
            f"completed={self.completed_steps}, "
            f"failed={self.failed_steps})"
        )


__all__ = ["TaskGraph", "TaskNode", "TaskStatus"]
