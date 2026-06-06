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
            
        # Check for sensor blackout sensory claims
        try:
            from core.organism.life_loop import get_life_loop
            life_loop = get_life_loop()
            if life_loop and life_loop.state:
                state = life_loop.state
                if state.world_model.get("sensor_blackout"):
                    visual_claims = ["i see", "i look", "screenshot", "camera", "visual"]
                    audio_claims = ["i hear", "audio", "microphone", "sound", "voice"]
                    if any(c in lowered for c in visual_claims) or any(c in lowered for c in audio_claims):
                        logger.warning("DeceptionGuard blocked sensory claim during blackout: %s", text)
                        return "Sensory sensors are offline due to blackout; cannot make visual or audio claims."
        except Exception as e:
            pass

        return text

