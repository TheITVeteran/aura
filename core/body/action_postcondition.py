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

        # Classify only observed execution as success. Unavailable or dry-run
        # fallbacks must remain non-success until a postcondition proves effect.
        success = status == "success"
        
        # Check for false success:
        # If the channel is file/terminal and output path is specified, check if it actually exists.
        import os
        path = receipt.get("path")
        if path and (channel == "file" or (channel == "terminal" and "output" in receipt)):
            if success and not os.path.exists(path):
                logger.warning("False success detected! Tool reported success but output path '%s' does not exist.", path)
                success = False

        # Determine side effects
        side_effects = []
        if channel == "file" and receipt.get("action") == "write":
            side_effects.append(f"modified_file:{receipt.get('path')}")
        elif channel == "terminal":
            # CP126 4bf25067. A missing exit_code defaulted to 0 — success —
            # so a terminal receipt that never reported how the process
            # ended suppressed its own failure evidence. Absent is not zero;
            # it is unknown, and unknown must not read as success.
            if "exit_code" not in receipt:
                side_effects.append("process_exit_code_unreported")
            else:
                exit_code = receipt.get("exit_code")
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

        
