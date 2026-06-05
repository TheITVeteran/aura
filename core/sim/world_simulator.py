"""core/sim/world_simulator.py — World Simulation Orchestrator.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from core.sim.causal_graph import CausalInterventionGraph, InterventionNode
from core.sim.monte_carlo import MonteCarloPlanner
from core.sim.scenario_tree import ScenarioTreeBuilder
from core.sim.risk_forecaster import RiskForecaster

logger = logging.getLogger("Aura.WorldSimulator")


class WorldSimulator:
    """Orchestrates Monte Carlo tree searches, decision scenario trees, and risk profiles."""

    def __init__(self) -> None:
        self.causal_graph = CausalInterventionGraph()
        # Initialize standard node
        self.causal_graph.register_node(InterventionNode("optimize_indices", 0.85, 0.90))
        self.causal_graph.register_node(InterventionNode("regression_fault", -0.90, 0.05))
        self.causal_graph.add_link("optimize_indices", "regression_fault")

    async def simulate_outcomes(self, objective: str) -> Dict[str, Any]:
        """Runs MCTS planning rollouts and risk forecasts for a proposed objective."""
        logger.info("🌲 WorldSimulator simulating outcomes for objective: '%s'", objective)

        # 1. Build scenario decision tree
        action_options = [f"plan_a_{objective[:10]}", f"plan_b_{objective[:10]}"]
        root_scenario = ScenarioTreeBuilder.build_tree(action_options)

        # 2. Run Monte Carlo rollouts
        def rollout_scorer() -> float:
            # Simulate a baseline score
            return 0.785

        mc_score = MonteCarloPlanner.simulate_rollouts(rollout_scorer, runs=100)

        # 3. Forecast risks
        risk_profile = RiskForecaster.forecast_risk(action_options)

        # 4. Evaluate causal intervention impact
        causal_impact = self.causal_graph.simulate_intervention("optimize_indices")

        return {
            "ok": True,
            "objective": objective,
            "monte_carlo_score": mc_score,
            "risk_profile": risk_profile,
            "causal_impact": causal_impact,
            "optimal_path": action_options[0] if mc_score > 0.50 else "abort_mission",
        }


# Singleton
_simulator_instance: WorldSimulator | None = None


def get_world_simulator() -> WorldSimulator:
    global _simulator_instance
    if _simulator_instance is None:
        _simulator_instance = WorldSimulator()
    return _simulator_instance
