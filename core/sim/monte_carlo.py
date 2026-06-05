"""core/sim/monte_carlo.py — Monte Carlo Tree Search Planning Simulator.
"""
from __future__ import annotations

import logging
import random
from typing import Callable, List

logger = logging.getLogger("Aura.MonteCarloSim")


class MonteCarloPlanner:
    """Runs randomized rollouts over state variables to optimize plan paths."""

    @staticmethod
    def simulate_rollouts(
        rollout_fn: Callable[[], float],
        runs: int = 500
    ) -> float:
        """Runs the rollout evaluation function and returns the average score."""
        logger.info("🎲 Starting %d Monte Carlo rollout runs...", runs)
        total_score = 0.0
        for _ in range(runs):
            # Evaluate random seed fluctuations to model uncertainty
            noise = random.uniform(0.90, 1.10)
            score = rollout_fn() * noise
            total_score += score
        average = total_score / runs
        logger.info("🎲 Rollout finished. Average outcome score: %.3f", average)
        return average
