"""core/lab/result_interpreter.py — Research Result Interpreter.

Analyzes experimental trials to determine whether hypotheses are validated.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from core.lab.hypothesis_engine import Hypothesis

logger = logging.getLogger("Aura.ResultInterpreter")


class ResultInterpreter:
    """Interprets simulation outcomes against hypotheses statements."""

    def interpret(
        self,
        hypothesis: Hypothesis,
        simulation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.info("🎯 ResultInterpreter: analyzing hypothesis '%s'", hypothesis.hypothesis_id)

        avg_effect = simulation_result.get("avg_effect_size", 0.0)

        # Determine if independent variable positively impacted the dependent variable
        validated = avg_effect > 0.05

        confidence_change = 0.2 if validated else -0.2
        new_confidence = min(1.0, max(0.0, hypothesis.confidence + confidence_change))

        return {
            "hypothesis_id": hypothesis.hypothesis_id,
            "statement": hypothesis.statement,
            "validated": validated,
            "avg_effect_size": avg_effect,
            "old_confidence": hypothesis.confidence,
            "new_confidence": round(new_confidence, 2),
            "conclusion": (
                f"Hypothesis validated with average effect size of {avg_effect}. "
                f"Stimulating {hypothesis.variables.get('independent')} increases {hypothesis.variables.get('dependent')}."
                if validated else
                f"Hypothesis rejected. No significant effect observed."
            )
        }
