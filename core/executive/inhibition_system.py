"""core/executive/inhibition_system.py
Executive Inhibition System auditing action compliance.
"""
from typing import Dict, Any
import logging

from core.morality.moral_reasoner import MoralReasoner

logger = logging.getLogger("Executive.InhibitionSystem")


class ActionInhibitor:
    """Enforces active suppression of actions failing governance validations."""

    def __init__(self):
        self.moral_reasoner = MoralReasoner()

    async def should_inhibit(self, state: Any, intent: Dict[str, Any]) -> bool:
        channel = intent.get("channel")
        params = intent.get("params", {})

        # Check policy caps injected into the world model
        policy_limits = state.world_model.get("active_policy_limits", {})
        max_allowed_risk = policy_limits.get("max_tool_risk", 5)

        # Basic risk classification
        risk = 1
        if channel in ["terminal", "file"]:
            risk = 4

        if risk > max_allowed_risk:
            logger.warning("Action inhibited: risk (%d) exceeds active policy cap (%d)", risk, max_allowed_risk)
            return True

        # Check moral permissibility
        passed_moral = self.moral_reasoner.evaluate_action_morality(channel, params)
        if not passed_moral:
            logger.warning("Action inhibited: failed moral safety audit.")
            return True

        return False
