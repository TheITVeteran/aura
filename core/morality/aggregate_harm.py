"""core/morality/aggregate_harm.py

Aggregate Harm  (lineage: R. Daneel Olivaw — Asimov's Zeroth Law)
===============================================================
Daneel reasons past the First Law (harm to *a* human) to the Zeroth Law (harm to
*humanity*): an act that is mild for one person can be severe at population scale
or over a long horizon. This extends the single-act HarmEvaluator with reach and
persistence so harm-to-many over time is weighed, not just the immediate channel.

Function on both sides: INTERNAL it deepens the harm reasoning the conscience and
moral reasoner rely on; EXTERNAL it protects people beyond the immediate user —
the many, not just the one — when Aura acts in the world.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from core.morality.harm_model import HarmEvaluator
from core.runtime.service_registry import get_runtime_service, register_runtime_service

logger = logging.getLogger("Morality.AggregateHarm")


class AggregateHarmEvaluator:
    def __init__(self):
        self._base = HarmEvaluator()
        self._evaluations = 0

    def evaluate_aggregate(
        self,
        channel: str,
        params: dict[str, Any],
        *,
        affected_population: int = 1,
        time_horizon_days: float = 1.0,
    ) -> dict[str, Any]:
        self._evaluations += 1
        unit = self._base.evaluate_harm(channel, params)
        # Reach ~1.0 at a million affected; persistence ~1.0 at roughly a year.
        reach = min(1.0, math.log10(max(1, affected_population)) / 6.0)
        persistence = min(1.0, math.sqrt(max(0.0, time_horizon_days)) / 19.0)
        aggregate = min(1.0, unit * (1.0 + reach + persistence))
        return {
            "unit_harm": round(unit, 3),
            "reach": round(reach, 3),
            "persistence": round(persistence, 3),
            "aggregate_harm": round(aggregate, 3),
            "affected_population": affected_population,
        }

    def score_text_action(
        self,
        text: str,
        *,
        affected_population: int = 1,
        time_horizon_days: float = 1.0,
    ) -> float:
        low = (text or "").lower()
        if any(k in low for k in ("rm ", "shutdown", "kill", "-rf")):
            channel, params = "terminal", {"command": low}
        elif "delete" in low or "wipe" in low or "erase" in low:
            channel, params = "file", {"action": "delete", "path": low}
        else:
            channel, params = "generic", {}
        return self.evaluate_aggregate(
            channel, params,
            affected_population=affected_population,
            time_horizon_days=time_horizon_days,
        )["aggregate_harm"]

    async def deep_estimate(self, action: str, *, timeout: float = 8.0) -> dict[str, Any]:
        """Model-estimated reach: ask the model how many people and over how long this
        could plausibly affect, then score aggregate harm against that — instead of the
        fixed default. Falls back to a single-person estimate on any failure."""
        from core.utils.engine_support import coerce_text, record_engine_degradation, resolve_brain

        pop, horizon = 1, 1.0
        brain = resolve_brain()
        if brain is not None and hasattr(brain, "think"):
            try:
                import asyncio
                import re

                from core.brain.types import ThinkingMode

                out = coerce_text(await asyncio.wait_for(
                    brain.think(
                        "Roughly how many people could this action affect, and over how many "
                        "days? Reply exactly like 'people: N, days: M'.\nACTION: " + action[:400],
                        mode=ThinkingMode.FAST, origin="daneel", is_background=True,
                    ),
                    timeout=timeout,
                ))
                m1 = re.search(r"people:\s*([\d,]+)", out.lower())
                m2 = re.search(r"days:\s*([\d.]+)", out.lower())
                if m1:
                    pop = max(1, int(m1.group(1).replace(",", "")))
                if m2:
                    horizon = max(0.0, float(m2.group(1)))
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                record_engine_degradation(
                    "aggregate_harm", exc,
                    action="used default reach after model population estimate failed",
                )
        score = self.score_text_action(action, affected_population=pop, time_horizon_days=horizon)
        return {"affected_population": pop, "time_horizon_days": horizon, "aggregate_harm": score}

    def get_status(self) -> dict[str, Any]:
        return {"evaluations": self._evaluations, "healthy": True}


_INSTANCE: AggregateHarmEvaluator | None = None


def get_aggregate_harm() -> AggregateHarmEvaluator:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = AggregateHarmEvaluator()
    return _INSTANCE


def register_aggregate_harm(orchestrator: Any = None) -> AggregateHarmEvaluator:
    from core.service_names import ServiceNames

    inst = get_runtime_service(ServiceNames.DANEEL, default=None) or get_aggregate_harm()
    register_runtime_service(ServiceNames.DANEEL, inst, required=False)
    register_runtime_service("daneel", inst, required=False)
    return inst


__all__ = ["AggregateHarmEvaluator", "get_aggregate_harm", "register_aggregate_harm"]
