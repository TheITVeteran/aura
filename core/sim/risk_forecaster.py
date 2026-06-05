"""core/sim/risk_forecaster.py — Consequence and Plan Risk Forecaster.
"""
from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger("Aura.RiskForecaster")


class RiskForecaster:
    """Calculates risk ratings, cost forecasts, and failure likelihoods."""

    @staticmethod
    def forecast_risk(plan_steps: List[str]) -> Dict[str, Any]:
        logger.info("🔮 Forecaster evaluating risk profiles for %d plan steps...", len(plan_steps))
        
        failure_likelihood = 0.05
        expected_welfare_cost = 10.0
        consequences = []

        for step in plan_steps:
            lowered = step.lower()
            if "delete" in lowered or "destroy" in lowered:
                failure_likelihood += 0.20
                expected_welfare_cost += 30.0
                consequences.append(f"Risk: possible permanent loss of data in step: {step}")
            elif "network" in lowered or "download" in lowered or "api" in lowered:
                failure_likelihood += 0.10
                expected_welfare_cost += 15.0
                consequences.append(f"Risk: external connectivity side-effects in step: {step}")

        # Limit likelihood
        failure_likelihood = min(0.95, failure_likelihood)

        logger.info("🔮 Risk Forecast: likelihood=%.2f, cost=%.1f", failure_likelihood, expected_welfare_cost)
        return {
            "failure_likelihood": failure_likelihood,
            "expected_welfare_cost": expected_welfare_cost,
            "risk_consequences": consequences,
            "risk_tier": "critical" if failure_likelihood > 0.40 else "medium" if failure_likelihood > 0.15 else "low",
        }
