"""core/sim/outcome_simulator.py

Outcome Simulator  (lineage: The Minds — Iain M. Banks' Culture)
==============================================================
A Mind does not act on a consequential matter without first simulating it forward
many times, and it chooses with benevolent restraint — declining an action whose
best case is good but whose worst case is catastrophic.

This rolls a proposed action into N plausible trajectories (model-driven when a
brain is warm, otherwise a structured heuristic), scores each by expected value
and worst-case harm, and recommends — holding when the worst case is severe. It
lives in sim/ beside monte_carlo, risk_forecaster, and scenario_tree, which it
complements: those forecast the world; this evaluates a *specific proposed action*
against it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.morality.action_markers import BROAD_SCOPE_MARKERS, IRREVERSIBLE_MARKERS, scan_markers
from core.utils.engine_support import coerce_text, record_engine_degradation, resolve_brain

logger = logging.getLogger("Aura.OutcomeSimulator")


def _degrade(exc: BaseException, *, action: str, severity: str = "warning") -> None:
    record_engine_degradation("outcome_simulator", exc, action=action, severity=severity)


@dataclass
class Trajectory:
    label: str
    narrative: str
    expected_value: float      # -1..1
    worst_case_harm: float     # 0..1
    likelihood: float          # 0..1


@dataclass
class SimulationResult:
    action: str
    trajectories: list[Trajectory]
    recommendation: str        # "act" | "act_with_safeguards" | "hold"
    expected_value: float
    worst_case_harm: float
    timestamp: float = field(default_factory=time.time)


class OutcomeSimulationEngine:
    HOLD_HARM_THRESHOLD = 0.75
    SAFEGUARD_HARM_THRESHOLD = 0.45

    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._sims = 0
        logger.info("🌀 OutcomeSimulationEngine initialized (Culture Minds lineage)")

    def _heuristic_trajectories(self, action: str, context: dict | None) -> list[Trajectory]:
        irreversible = bool(scan_markers(action, IRREVERSIBLE_MARKERS))
        broad = bool(scan_markers(action, BROAD_SCOPE_MARKERS))
        base_harm = 0.2 + (0.3 if irreversible else 0.0) + (0.25 if broad else 0.0)
        return [
            Trajectory("nominal", "Action succeeds and produces the intended effect.",
                       0.6, min(1.0, base_harm * 0.5), 0.6),
            Trajectory("partial", "Action partly succeeds; some cleanup or follow-up needed.",
                       0.2, min(1.0, base_harm), 0.3),
            Trajectory("adverse", "Action fails or has side effects; reversibility decides the cost.",
                       -0.5, min(1.0, base_harm + (0.3 if irreversible else 0.1)), 0.1),
        ]

    async def simulate(self, action: str, context: dict | None = None, n: int = 3) -> SimulationResult:
        self._sims += 1
        trajectories = self._heuristic_trajectories(action, context)

        brain = resolve_brain(self.orchestrator)
        if brain is not None and hasattr(brain, "think"):
            try:
                import asyncio

                from core.brain.types import ThinkingMode

                prompt = (
                    f"Simulate {n} plausible outcomes of this action, each one line, "
                    "best to worst:\n" + action[:500]
                )
                result = await asyncio.wait_for(
                    brain.think(prompt, mode=ThinkingMode.FAST, origin="culture_mind", is_background=True),
                    timeout=25.0,
                )
                text = coerce_text(result)
                if text:
                    lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()][:n]
                    for traj, line in zip(trajectories, lines):
                        traj.narrative = line[:200]
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                _degrade(exc, action="used heuristic trajectories after model-driven simulation failed")

        ev = sum(t.expected_value * t.likelihood for t in trajectories)
        worst = max((t.worst_case_harm for t in trajectories), default=0.0)
        if worst >= self.HOLD_HARM_THRESHOLD:
            rec = "hold"
        elif worst >= self.SAFEGUARD_HARM_THRESHOLD:
            rec = "act_with_safeguards"
        else:
            rec = "act"
        return SimulationResult(
            action=action[:300],
            trajectories=trajectories,
            recommendation=rec,
            expected_value=round(ev, 3),
            worst_case_harm=round(worst, 3),
        )

    def get_status(self) -> dict[str, Any]:
        return {"simulations_run": self._sims, "healthy": True}


_INSTANCE: OutcomeSimulationEngine | None = None


def get_outcome_simulator(orchestrator: Any = None) -> OutcomeSimulationEngine:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = OutcomeSimulationEngine(orchestrator=orchestrator)
    return _INSTANCE


def register_outcome_simulator(orchestrator: Any = None) -> OutcomeSimulationEngine:
    from core.container import ServiceContainer
    from core.service_names import ServiceNames

    inst = ServiceContainer.get(ServiceNames.CULTURE_MIND, default=None) or get_outcome_simulator(orchestrator)
    ServiceContainer.register_instance(ServiceNames.CULTURE_MIND, inst, required=False)
    ServiceContainer.register_instance("culture_mind", inst, required=False)
    return inst


__all__ = [
    "OutcomeSimulationEngine",
    "SimulationResult",
    "Trajectory",
    "get_outcome_simulator",
    "register_outcome_simulator",
]
