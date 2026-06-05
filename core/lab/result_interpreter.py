"""core/lab/result_interpreter.py — Research Result Interpreter.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger("Aura.ResultInterpreter")


class ResultInterpreter:
    """Interprets experimental outputs to prove/disprove hypotheses."""

    @staticmethod
    def interpret(result_data: Dict[str, Any], verification_metric: str) -> Dict[str, Any]:
        logger.info("🔬 ResultInterpreter analyzing outcomes against: '%s'", verification_metric)
        
        # Simple parser for metrics e.g. "latency_delta_ratio >= 0.30"
        passed = False
        details = "Metric check failed."

        if "latency_delta_ratio" in verification_metric:
            ratio_str = verification_metric.split(">=")[-1].strip()
            threshold = float(ratio_str)
            actual_ratio = result_data.get("latency_delta_ratio", 0.0)
            passed = actual_ratio >= threshold
            details = f"Actual ratio: {actual_ratio:.2f} vs Threshold: {threshold:.2f}"

        logger.info("🔬 Hypothesis validation outcome: %s. Details: %s", "VALIDATED" if passed else "REJECTED", details)
        return {
            "hypothesis_validated": passed,
            "interpretation_details": details,
            "metric_checked": verification_metric,
        }
