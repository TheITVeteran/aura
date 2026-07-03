import logging
from typing import Any, Optional
from . import BasePhase
from ..state.aura_state import AuraState
from core.consciousness.integration import get_consciousness_integration
from core.runtime.service_registry import get_runtime_service

logger = logging.getLogger(__name__)

class ConsciousnessPhase(BasePhase):
    """
    Phase 8: Phenomenological Awareness.
    Constructs the first-person experiential claim for this cognitive cycle.
    """
    
    def __init__(self, container: Any = None):
        self.container = container

    async def execute(self, state: AuraState, objective: Optional[str] = None, **kwargs) -> AuraState:
        """
        Pull the latest phenomenal context from the integration layer.
        """
        new_state = state.derive("consciousness_phase")
        
        # Pull from singleton integration
        integration = get_consciousness_integration()
        
        # Layer 8 Injection
        if integration:
            phenomenal_claim = integration.get_phenomenal_context()
            new_state.cognition.phenomenal_state = phenomenal_claim
            logger.debug("🌅 ConsciousnessPhase: Layer 8 phenomenal context injected.")
        
        # Causal Simulation
        causal_model = get_runtime_service("causal_world_model", default=None)
        if causal_model:
            causal_context = causal_model.get_prompt_context()
            if causal_context:
                if not hasattr(new_state.cognition, 'causal_reasoning'):
                    # Just in case the state model is lagging
                    new_state.cognition.causal_reasoning = causal_context
                else:
                    new_state.cognition.causal_reasoning = causal_context
                logger.debug("🧶 ConsciousnessPhase: Causal world cascades injected.")
        
        return new_state
