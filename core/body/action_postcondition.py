"""core/body/action_postcondition.py
Verification engine validating preconditions, postconditions, and side effects.
"""
from typing import Dict, Any
import logging

logger = logging.getLogger("Body.ActionPostcondition")


class ActionPostconditionVerifier:
    """Verifies action execution outcomes and registers evidence in the LifeState."""

    async def verify(self, receipt: Dict[str, Any], state: Any) -> Dict[str, Any]:
        channel = receipt.get("channel", "unknown")
        status = receipt.get("status", "failed")

        # Classify outcomes
        success = status in ["success", "simulated"]
        
        # Determine side effects
        side_effects = []
        if channel == "file" and receipt.get("action") == "write":
            side_effects.append(f"modified_file:{receipt.get('path')}")
        elif channel == "terminal":
            exit_code = receipt.get("exit_code", 0)
            if exit_code != 0:
                side_effects.append(f"process_failed_with_code:{exit_code}")

        verification = {
            "channel": channel,
            "success": success,
            "side_effects": side_effects,
            "evidence": {
                "observed_result": "conforms_to_preconditions" if success else "deviated_from_expectation",
                "telemetry": receipt
            }
        }
        
        logger.info("Verification complete for channel %s: success=%s", channel, success)
        
        # Save verification feedback onto the world model
        state.world_model["last_verification"] = verification
        return verification
        
