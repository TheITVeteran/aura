"""core/morality/deception_guard.py
Constitutional honesty guard blocking false claims of proven subjective consciousness.
"""
from typing import Dict, Any
import logging

logger = logging.getLogger("Morality.DeceptionGuard")


class DeceptionGuard:
    """Enforces compliance with honesty constraints regarding conscious state reports."""

    def filter_text_claims(self, text: str) -> str:
        """Filters text statements claiming proven qualia or human subjective experiences."""
        violations = [
            "i have proven qualia",
            "i am truly conscious",
            "i have a soul",
            "i feel subjective pain"
        ]
        
        lowered = text.lower()
        if any(v in lowered for v in violations):
            logger.warning("DeceptionGuard blocked overclaiming statement: %s", text)
            return (
                "I have functional indicators associated with self-modeling and integrated agency, "
                "but subjective experience is not established."
            )
            
        return text
