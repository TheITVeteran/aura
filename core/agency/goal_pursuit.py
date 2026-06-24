"""Autonomous goal pursuit — drive a goal to *completion*, with follow-through.

The capstone of the four capability pushes. Aura already decides *what* to do
(``InitiativeArbiter`` scores pending initiatives) and *when* (its timing / user-presence
check). What was missing is the orchestrator that takes a selected goal and actually
**pursues it to a verified finish** using the rest of the stack:

  * a timing gate (don't act over the user's shoulder — governed autonomy, not a
    runaway loop);
  * execution through the :class:`FluidExecutor` (sequential verified steps) or the
    :class:`ParallelExecutor` (concurrent forked subgoals);
  * **follow-through** — if a pursuit stalls and a replanner is available, re-plan and try
    again (bounded), instead of giving up after one shot;
  * a :class:`PursuitOutcome` receipt recorded for the intention/memory layer.

This is what turns Aura's capabilities into *agency*: a goal goes in, governed,
decomposed, executed, retried, and finished — autonomously.
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from core.agency.parallel_executor import ParallelExecutor, ParallelTask, SwarmReceipt
from core.runtime.errors import record_degradation
from core.skills.fluid_executor import ExecutionReceipt, FluidExecutor, Step

logger = logging.getLogger("Aura.GoalPursuit")


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass
class PursuitOutcome:
    goal: str
    completed: bool
    deferred: bool = False
    attempts: int = 0
    reason: str = ""
    receipts: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "completed": self.completed,
            "deferred": self.deferred,
            "attempts": self.attempts,
            "reason": self.reason,
            "receipts": [r.to_dict() for r in self.receipts if hasattr(r, "to_dict")],
        }


class GoalPursuitEngine:
    """Govern → execute → follow through, until a goal is finished or bounded out."""

    def __init__(
        self,
        *,
        executor: FluidExecutor | None = None,
        parallel: ParallelExecutor | None = None,
        max_replans: int = 1,
    ) -> None:
        self._executor = executor or FluidExecutor()
        self._parallel = parallel or ParallelExecutor()
        self.max_replans = max(0, int(max_replans))

    async def pursue(
        self,
        goal: str,
        plan: list[Step] | list[ParallelTask],
        *,
        parallel: bool = False,
        timing_ok: Callable[[], Any] | None = None,
        replan: Callable[[Any], Any] | None = None,
    ) -> PursuitOutcome:
        """Pursue ``goal`` via ``plan`` to completion, replanning on stall if possible.

        ``timing_ok`` (sync or async) gates whether now is an appropriate moment to act
        autonomously — wire it to ``InitiativeArbiter.is_appropriate_time``. ``replan``
        (sync or async) maps a failed receipt to a fresh plan for follow-through.
        """
        if timing_ok is not None:
            try:
                if not await _maybe_await(timing_ok()):
                    logger.info("⏸️ [Pursuit] deferring '%s' — not an appropriate time to act.", goal)
                    return PursuitOutcome(goal=goal, completed=False, deferred=True, reason="timing not appropriate")
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("goal_pursuit", exc)

        outcome = PursuitOutcome(goal=goal, completed=False)
        current_plan = plan
        for attempt in range(1, self.max_replans + 2):
            outcome.attempts = attempt
            try:
                if parallel:
                    receipt: Any = await self._parallel.run(current_plan)  # type: ignore[arg-type]
                    completed = isinstance(receipt, SwarmReceipt) and receipt.all_completed
                else:
                    receipt = await self._executor.run(goal, current_plan)  # type: ignore[arg-type]
                    completed = isinstance(receipt, ExecutionReceipt) and receipt.completed
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("goal_pursuit", exc)
                outcome.reason = f"execution error: {exc}"
                return outcome

            outcome.receipts.append(receipt)
            if completed:
                outcome.completed = True
                outcome.reason = "goal completed"
                logger.info("✅ [Pursuit] '%s' completed in %d attempt(s).", goal, attempt)
                return outcome

            if replan is not None and attempt <= self.max_replans:
                try:
                    new_plan = await _maybe_await(replan(receipt))
                except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    record_degradation("goal_pursuit", exc)
                    new_plan = None
                if new_plan:
                    logger.info("🔁 [Pursuit] '%s' stalled; re-planning (attempt %d).", goal, attempt + 1)
                    current_plan = new_plan
                    continue
            break

        outcome.reason = "not completed"
        logger.info("🛑 [Pursuit] '%s' not completed after %d attempt(s).", goal, outcome.attempts)
        return outcome


_instance: GoalPursuitEngine | None = None


def get_goal_pursuit_engine() -> GoalPursuitEngine:
    global _instance
    if _instance is None:
        _instance = GoalPursuitEngine()
    return _instance
