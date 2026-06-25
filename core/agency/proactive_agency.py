"""Proactive agency — the bridge from 'a goal was selected' to 'pursue it to completion'.

The autonomous-initiative loop already *selects* goals (InitiativeArbiter) and advances
missions, but advancing only marks progress — it never drove a goal through the
capability stack. This is the connective tissue: given a goal, it builds a plan (an
injected planner) and pursues it to a verified finish via :class:`GoalPursuitEngine`
(fluid + parallel execution), strictly gated so autonomous action only happens when it
is *allowed* (background policy) and *appropriate* (timing / user-presence).

Safe by construction: no planner ⇒ no autonomous execution (returns ``None``), and every
pursuit passes the background-allowed and timing gates first. The planner is injected, so
this is testable and pluggable — a computational/reasoning goal can plan to a deliberation
step, a desktop goal to verified UI steps, etc.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.ProactiveAgency")

Planner = Callable[[str], Awaitable[Any]]   # goal -> list[Step] | list[ParallelTask]


class ProactiveAgency:
    def __init__(
        self,
        *,
        pursuit: Any | None = None,
        planner: Planner | None = None,
        background_allowed: Callable[[], bool] | None = None,
        timing_ok: Callable[[], Any] | None = None,
        default_planner_enabled: bool | None = None,
    ) -> None:
        self._pursuit = pursuit
        self._planner = planner
        self._background_allowed = background_allowed
        self._timing_ok = timing_ok
        # Proactive autonomy is ON by default — Aura is always self-directed. It is made
        # safe not by disabling it but by being non-blocking (fire-and-forget),
        # single-flight (one pursuit at a time), and running on the cheap BACKGROUND lane,
        # so it never stalls the event loop or competes with the foreground conversation.
        # AURA_PROACTIVE_AUTONOMY=0 is a kill-switch.
        self._default_planner_enabled = (
            default_planner_enabled
            if default_planner_enabled is not None
            else os.getenv("AURA_PROACTIVE_AUTONOMY", "1") != "0"
        )
        self._pursued = 0
        self._completed = 0
        self._running = False   # single-flight guard

    @property
    def enabled(self) -> bool:
        """True if proactive pursuit can run (an explicit planner, or env opt-in)."""
        return self._planner is not None or self._default_planner_enabled

    def register_planner(self, planner: Planner) -> None:
        self._planner = planner

    def _get_planner(self) -> Planner | None:
        """Resolve a planner — default to the GoalPlanner so open-ended goals are plannable."""
        if self._planner is not None:
            return self._planner
        if self._default_planner_enabled:
            try:
                from core.agency.goal_planner import GoalPlanner

                # Cheap background planner: a single deliberation sample (not 3-5) so
                # proactive autonomous thinking is light on the background lane.
                self._planner = GoalPlanner(deliberate_samples=1)
                return self._planner
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation("proactive_agency", exc)
        return None

    def _engine(self) -> Any | None:
        if self._pursuit is not None:
            return self._pursuit
        try:
            from core.agency.goal_pursuit import get_goal_pursuit_engine

            self._pursuit = get_goal_pursuit_engine()
            return self._pursuit
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("proactive_agency", exc)
            return None

    async def pursue_goal(self, goal: str, *, parallel: bool = False) -> Any | None:
        """Plan and pursue ``goal`` to completion — or ``None`` if not allowed/plannable.

        Single-flight: only one proactive pursuit runs at a time, so background
        deliberation can never pile up and saturate the model lane.
        """
        if not goal or not str(goal).strip():
            return None
        if self._running:
            return None   # a pursuit is already in flight — don't stack another
        if self._background_allowed is not None:
            try:
                if not self._background_allowed():
                    logger.debug("⏸️ [Proactive] background action not allowed; skipping '%s'.", goal[:50])
                    return None
            except (RuntimeError, AttributeError, TypeError) as exc:
                record_degradation("proactive_agency", exc)
        planner = self._get_planner()
        if planner is None:
            return None
        self._running = True
        try:
            try:
                plan = await planner(goal)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation("proactive_agency", exc)
                return None
            if not plan:
                return None
            engine = self._engine()
            if engine is None:
                return None
            self._pursued += 1
            outcome = await engine.pursue(goal, plan, parallel=parallel, timing_ok=self._timing_ok)
            if getattr(outcome, "completed", False):
                self._completed += 1
                logger.info("✅ [Proactive] autonomously completed goal: %s", goal[:60])
            elif getattr(outcome, "deferred", False):
                logger.debug("⏸️ [Proactive] goal deferred (timing): %s", goal[:50])
            return outcome
        finally:
            self._running = False

    def status(self) -> dict[str, Any]:
        return {"pursued": self._pursued, "completed": self._completed, "has_planner": self._planner is not None}


_instance: ProactiveAgency | None = None


def get_proactive_agency() -> ProactiveAgency:
    global _instance
    if _instance is None:
        _instance = ProactiveAgency()
    return _instance
