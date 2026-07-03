"""Self-repair backlog — route detected test defects to governed repair goals.

Closes the mandate's "code catches and fixes regressions before users do":
the chunk runner detects order-dependence and real failures and (with
``--defect-register``) writes a machine-readable register. This module turns
that register into concrete, approval-gated repair goals for the autonomous
task engine — so Aura can burn down her own defect backlog behind the same
human-approval gate that guards every consequential self-modification.

Design guarantees:
- **Never auto-executes.** Repair goals are created ``requires_approval=True``
  and shadow-planned; a human calls ``approve_plan`` to let one run.
- **Idempotent.** Each defect has a stable id; re-ingesting the same register
  never double-creates a goal.
- **Read-only ingestion.** Parsing a register mutates nothing until a plan is
  explicitly created.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.SelfRepairBacklog")


@dataclass(frozen=True)
class RepairItem:
    defect_id: str
    kind: str          # "order_dependence" | "real_failure"
    target: str        # the failing test id
    goal: str          # the repair goal text for the task engine

    def to_dict(self) -> dict[str, str]:
        return {
            "defect_id": self.defect_id,
            "kind": self.kind,
            "target": self.target,
            "goal": self.goal,
        }


def _defect_id(kind: str, target: str) -> str:
    return kind[:4] + "-" + hashlib.sha256(f"{kind}|{target}".encode()).hexdigest()[:12]


_ORDER_DEP_GOAL = (
    "Diagnose and fix the ORDER-DEPENDENCE defect in test '{target}': it "
    "fails inside a chunk but passes in isolation, which means shared process "
    "state (module globals, singletons, the ServiceContainer, monkeypatched "
    "module attributes, or _PROCESS_STARTED_AT-style clocks) leaks across "
    "tests. Find the leaking state, make the test hermetic (fixture-scoped "
    "reset), and prove it by running the test both alone and after the "
    "polluting sibling. Do not weaken the assertion."
)

_REAL_FAILURE_GOAL = (
    "Diagnose and fix the FAILING test '{target}' (fails both in-chunk and in "
    "isolation). Root-cause it, fix the underlying defect (not the test unless "
    "the test is provably wrong), and verify green. Do not silence it."
)


@dataclass
class SelfRepairBacklog:
    """Parses defect registers into approval-gated repair goals."""

    seen_path: Path = field(
        default_factory=lambda: Path.home() / ".aura" / "data" / "self_repair_seen.json"
    )
    _seen: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        try:
            if self.seen_path.exists():
                self._seen = set(json.loads(self.seen_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            record_degradation("self_repair_backlog", exc, severity="debug")
            self._seen = set()

    def parse_register(self, register_path: str | Path) -> list[RepairItem]:
        """Read a defect register into RepairItems (pure — mutates nothing)."""
        try:
            data = json.loads(Path(register_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            record_degradation("self_repair_backlog", exc, severity="warning")
            return []
        if not isinstance(data, dict):
            return []
        items: list[RepairItem] = []
        for target in data.get("order_dependent", []) or []:
            target = str(target)
            items.append(
                RepairItem(
                    _defect_id("order_dependence", target),
                    "order_dependence",
                    target,
                    _ORDER_DEP_GOAL.format(target=target),
                )
            )
        for target in data.get("real_failures", []) or []:
            target = str(target)
            items.append(
                RepairItem(
                    _defect_id("real_failure", target),
                    "real_failure",
                    target,
                    _REAL_FAILURE_GOAL.format(target=target),
                )
            )
        return items

    def new_items(self, register_path: str | Path) -> list[RepairItem]:
        """RepairItems not already ingested (idempotent by defect_id)."""
        return [it for it in self.parse_register(register_path) if it.defect_id not in self._seen]

    def _mark_seen(self, items: list[RepairItem]) -> None:
        self._seen.update(it.defect_id for it in items)
        try:
            self.seen_path.parent.mkdir(parents=True, exist_ok=True)
            self.seen_path.write_text(json.dumps(sorted(self._seen)) + "\n", encoding="utf-8")
        except OSError as exc:
            record_degradation("self_repair_backlog", exc, severity="debug")

    async def enqueue_repairs(
        self,
        register_path: str | Path,
        *,
        task_engine: Any = None,
        max_items: int = 5,
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        """Create approval-gated, shadow-planned repair goals for new defects.

        Returns a summary per item. ``dry_run`` (or no task engine) parses and
        marks-seen without creating plans. Real plans are created with
        ``requires_approval=True`` + ``is_shadow=True`` — a human must approve
        each before it executes.
        """
        items = self.new_items(register_path)[: max(0, int(max_items))]
        if not items:
            return []

        if dry_run or task_engine is None:
            if task_engine is None and not dry_run:
                try:
                    from core.agency.autonomous_task_engine import get_task_engine

                    task_engine = get_task_engine()
                except (ImportError, RuntimeError, AttributeError) as exc:
                    record_degradation("self_repair_backlog", exc, severity="warning")
                    task_engine = None
            if task_engine is None:
                self._mark_seen(items)
                return [{**it.to_dict(), "created": False, "reason": "no_task_engine"} for it in items]

        results: list[dict[str, Any]] = []
        for item in items:
            created = False
            reason = "dry_run"
            if not dry_run:
                try:
                    result = await task_engine.execute_goal(
                        item.goal,
                        context={
                            "origin": "self_repair_backlog",
                            "defect_id": item.defect_id,
                            "defect_kind": item.kind,
                            "target": item.target,
                            "requires_approval": True,
                        },
                        is_shadow=True,
                    )
                    created = True
                    reason = getattr(result, "status", "planned")
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation("self_repair_backlog", exc, severity="warning")
                    reason = f"error:{type(exc).__name__}"
            results.append({**item.to_dict(), "created": created, "reason": reason})

        self._mark_seen(items)
        logger.info(
            "🔧 [SELF-REPAIR] Ingested %d defect(s) into approval-gated repair goals.",
            len(results),
        )
        return results


_BACKLOG: SelfRepairBacklog | None = None


def get_self_repair_backlog() -> SelfRepairBacklog:
    global _BACKLOG
    if _BACKLOG is None:
        _BACKLOG = SelfRepairBacklog()
    return _BACKLOG
