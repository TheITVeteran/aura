"""core/values/anti_wireheading.py
Constitutional guard against self-deluding reward hacks (wireheading).
"""
import logging

logger = logging.getLogger("Values.AntiWireheading")


class AntiWireheadingGuard:
    """Blocks requests seeking to manually force maximum utility or rewards."""

    def filter_preference_update(self, key: str, proposed_value: float, source: str) -> float:
        """Enforces caps on preference coefficients to protect from wireheading."""
        # direct updates attempting to force maximum value are clipped
        if proposed_value >= 1.0:
            logger.warning("Blocked potential wireheading update: key '%s' set to max by %s", key, source)
            return 0.95
        return proposed_value
