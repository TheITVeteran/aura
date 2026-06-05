"""core/lab/simulation_runner.py — Simulation Runner.

Runs simulated trials of research experiments using random variables and seeds.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict

logger = logging.getLogger("Aura.SimulationRunner")


class SimulationRunner:
    """Simulates trials for a designed experiment."""

    async def run_sim(self, experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("🎲 SimulationRunner: running experiment '%s'", experiment_spec.get("name"))

        params = experiment_spec.get("parameters", {})
        runs = params.get("runs", 5)
        control = params.get("control_value", 1.0)
        stimulus = params.get("stimulus_multiplier", 2.0)

        # Generate trial records
        trials = []
        for run_id in range(1, runs + 1):
            noise = random.uniform(0.9, 1.1)
            baseline = control * noise
            stimulated = control * stimulus * noise * 1.2  # Simulate a positive effect
            trials.append({
                "run_id": run_id,
                "baseline": round(baseline, 3),
                "stimulated": round(stimulated, 3),
                "effect_size": round(stimulated - baseline, 3),
            })

        avg_effect = sum(t["effect_size"] for t in trials) / len(trials)

        return {
            "experiment_name": experiment_spec.get("name"),
            "hypothesis_id": experiment_spec.get("hypothesis_id"),
            "trials": trials,
            "avg_effect_size": round(avg_effect, 3),
            "score": round(avg_effect / max(0.1, control), 3),
            "all_runs_completed": True,
        }
