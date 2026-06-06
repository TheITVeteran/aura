"""core/learning/regression_guard.py
Regression Guard rolling back active model adapters if performance drops.
"""
from typing import Dict, Any
import logging

logger = logging.getLogger("Learning.RegressionGuard")


class RegressionGuard:
    """Monitors live performance signals to trigger rollback of degraded weights."""

    def evaluate_live_regression(self, error_rate: float, baseline_error_rate: float) -> bool:
        """Determines if the active model shows significant regressions."""
        if error_rate > baseline_error_rate * 1.5:
            logger.warning("Regression detected! Live error rate (%.2f) exceeds baseline (%.2f).",
                           error_rate, baseline_error_rate)
            return True
        return False
