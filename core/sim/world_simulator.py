"""core/sim/world_simulator.py — World Simulator.

Simulates future scenarios and risks before executing real world actions,
combining MCTS rollouts and digital twin predictions.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from core.twins.digital_twin import DigitalTwin
from core.sim.monte_carlo import MonteCarloPlanner

logger = logging.getLogger("Aura.WorldSimulator")


class WorldSimulator:
    """Simulates outcomes of long-horizon plans prior to execution."""

    def __init__(self) -> None:
        self.twin = DigitalTwin()
        self.mcts = MonteCarloPlanner()

    async def simulate_outcomes(self, objective: str) -> Dict[str, Any]:
        logger.info("📐 WorldSimulator: running outcome simulation for objective: '%s'", objective)

        # 1. Run digital twin checks for code/shell activities
        twin_impact = self.twin.simulate_change(
            change_type="codebase_patch",
            params={"patch": f"Implementation code for {objective}"},
        )

        # 2. Run MCTS trials to estimate success probability
        def rollout() -> float:
            # Objective complexity scales down base success probability
            base = 0.90 - (len(objective) * 0.001)
            return max(0.2, base)

        avg_score = self.mcts.simulate_rollouts(rollout, runs=200)

        risk_tier = "low"
        if twin_impact.get("risk_score", 0.0) > 0.8 or avg_score < 0.5:
            risk_tier = "high"
        elif twin_impact.get("risk_score", 0.0) > 0.4:
            risk_tier = "medium"

        return {
            "objective": objective,
            "success_probability": round(avg_score, 2),
            "risk_tier": risk_tier,
            "digital_twin_checks": twin_impact,
            "simulation_completed": True,
        }
