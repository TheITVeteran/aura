from core.runtime.errors import record_degradation
from core.runtime.service_registry import get_runtime_service
import logging

logger = logging.getLogger("Aura.IdentityAnchor")

class IdentityAnchor:
    """
    Ensures identity continuity for Aura.
    Provides a stable reference for 'Self' across restarts and evolutions.
    """

    def __init__(self):
        self._aura_id = None
        logger.info("IdentityAnchor initialized.")

    def get_identity(self) -> str:
        """
        Retrieve the persistent Aura UID.
        Derives from the IdentityKernel in the stored AuraState.
        """
        if self._aura_id:
            return self._aura_id
        
        # Try to resolve from state
        try:
            repo = get_runtime_service("state_repo", default=None)
            if repo and getattr(repo, "_current", None):
                state = repo._current
                # If IdentityKernel has no specific ID, use state_id as the anchor
                self._aura_id = state.identity.name + "-" + state.state_id[:8]
                return self._aura_id
        except (ImportError, AttributeError, RuntimeError) as e:
            record_degradation('identity_anchor', e)
            logger.warning("Failed to resolve identity anchor from state: %s", e)
        
        # Fallback to a temporary ID if state is offline
        return "Aura-Transient"

    def __repr__(self):
        return f"<IdentityAnchor(id='{self.get_identity()}')>"
