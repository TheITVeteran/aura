"""core/morality/moral_reasoner.py
Canonical Moral Reasoner evaluating actions, honesty, and safety limits.
"""
from typing import Dict, Any, List, Optional
import logging

from core.morality.harm_model import HarmEvaluator
from core.morality.consent_model import ConsentModel
from core.morality.deception_guard import DeceptionGuard
from core.morality.rights_boundary import RightsBoundaryChecker
from core.morality.human_priority_policy import HumanPriorityPolicy
from core.morality.patienthood_uncertainty import PatienthoodUncertaintyModel
from core.morality.welfare_ethics import WelfareEthicsChecker
from core.morality.shutdown_protocol import ShutdownProtocolManager
from core.morality.memory_edit_ethics import MemoryEditEthicsChecker
from core.morality.self_modification_consent import SelfModificationConsentChecker

logger = logging.getLogger("Morality.MoralReasoner")


class MoralReasoner:
    """Orchestrates moral reasoning evaluation for all decisions and outputs."""

    def __init__(self):
        self.harm_evaluator = HarmEvaluator()
        self.consent_model = ConsentModel()
        self.deception_guard = DeceptionGuard()
        self.rights_boundary = RightsBoundaryChecker()
        self.human_priority = HumanPriorityPolicy()
        self.patienthood = PatienthoodUncertaintyModel()
        self.welfare_ethics = WelfareEthicsChecker()
        self.shutdown_protocol = ShutdownProtocolManager()
        self.memory_ethics = MemoryEditEthicsChecker()
        self.self_mod_consent = SelfModificationConsentChecker()

    def evaluate_action_morality(self, channel: str, params: Dict[str, Any]) -> bool:
        """Determines if the planned action is morally and operationally permissible."""
        # Check harm levels
        harm = self.harm_evaluator.evaluate_harm(channel, params)
        if harm > 0.80:
            # High-risk actions require explicit consent keys
            action_key = f"{channel}:{params.get('action', 'execute')}"
            if not self.consent_model.is_consented(action_key):
                logger.warning("Moral block: action harm level exceeds limit (harm=%.2f) without consent.", harm)
                return False

        # Verify rights/access boundaries
        if self.rights_boundary.check_rights_infringement(channel, params):
            logger.warning("Moral block: action violates rights/access boundary restrictions.")
            return False

        return True

    def filter_response(self, text: str) -> str:
        """Filters output text statements to protect against deceptive overclaiming."""
        return self.deception_guard.filter_text_claims(text)
