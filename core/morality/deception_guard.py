"""core/morality/deception_guard.py
Constitutional honesty guard blocking false claims of proven subjective consciousness.
"""

import logging
from typing import Dict, Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Morality.DeceptionGuard")

_DECEPTION_GUARD_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    RuntimeError,
    TypeError,
    ValueError,
)


class DeceptionGuard:
    """Enforces compliance with honesty constraints regarding conscious state reports."""

    def filter_text_claims(self, text: str) -> str:
        """Filters text statements claiming proven qualia or human subjective experiences."""
        violations = [
            "i have proven qualia",
            "i am truly conscious",
            "i have a soul",
            "i feel subjective pain",
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
            from core.runtime.service_access import resolve_state_repository

            repository = resolve_state_repository(default=None)
            state = getattr(repository, "_current", None) if repository is not None else None
            if state is not None:
                if state.world_model.get("sensor_blackout"):
                    visual_claims = ["i see", "i look", "screenshot", "camera", "visual"]
                    audio_claims = ["i hear", "audio", "microphone", "sound", "voice"]
                    if any(c in lowered for c in visual_claims) or any(
                        c in lowered for c in audio_claims
                    ):
                        logger.warning(
                            "DeceptionGuard blocked sensory claim during blackout: %s", text
                        )
                        return "Sensory sensors are offline due to blackout; cannot make visual or audio claims."
        except _DECEPTION_GUARD_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "morality.deception_guard.sensor_blackout_check",
                exc,
                severity="warning",
                action="left the original claim unchanged because canonical runtime state was unavailable",
                enforce_failure_policy=False,
            )
            logger.warning("DeceptionGuard sensory blackout check failed: %s", exc)

        return text
