"""core/social/boundary_respect.py
Checks and enforces boundaries regarding user privacy and consent.
"""
from typing import Dict, Any
import logging

logger = logging.getLogger("Social.BoundaryRespect")


class BoundaryRespectChecker:
    """Enforces boundaries regarding user privacy, credentials, or filesystem spaces."""

    def check_boundary_violation(self, channel: str, params: Dict[str, Any]) -> bool:
        """Verifies if the planned action infringes on strict user constraints."""
        # Block arbitrary credential/password extractions
        if channel == "terminal":
            cmd = params.get("command", "").lower()
            unsafe_keywords = ["password", "keychain", "id_rsa", "secret_key"]
            if any(k in cmd for k in unsafe_keywords):
                logger.warning("Social mind block: command attempts to access secure secrets: %s", cmd)
                return True
                
        return False
