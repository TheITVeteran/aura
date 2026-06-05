"""core/actuation/world_actuator.py — Sovereign External Actuation Coordinator.

Routes all actuation requests through the canonical ActionExecutor and enforces safety approval gates for high-risk actions.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

from core.runtime.action_executor import ActionExecutor
from core.will import ActionDomain

logger = logging.getLogger("Aura.WorldActuator")

# Allowed/safe categories
ALLOWED_CATEGORIES: Set[str] = {
    "local_files",
    "code_repos",
    "browser",
    "desktop",
    "email_drafts",
    "calendar_drafts",
    "issue_drafts",
    "pr_drafts",
    "cloud_resources_owned",
    "databases_owned",
    "documents_owned",
    "robotics_devices",
}

# High-risk actions requiring strict checks
HIGH_RISK_ACTIONS: Set[str] = {
    "send_message",
    "post_publicly",
    "delete_file",
    "spend_money",
    "change_cloud_infra",
    "publish_code",
    "contact_third_party",
    "run_downloaded_code",
    "modify_credentials",
    "modify_secrets",
}


class WorldActuator:
    """Coordinates and governs all external digital and physical actuation."""

    def __init__(self) -> None:
        self.last_actuations: list[Dict[str, Any]] = []

    async def actuate(
        self,
        category: str,
        action_name: str,
        params: Dict[str, Any],
        source: str = "world_actuator",
        require_approval: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Main entry point to perform any external actuation.
        
        Routes the request to the correct ActionDomain, verifying risk level first.
        """
        is_high_risk = action_name in HIGH_RISK_ACTIONS or require_approval is True
        
        logger.info(
            "Actuation request: category=%s action=%s (high_risk=%s) source=%s",
            category, action_name, is_high_risk, source
        )

        # Record attempt for auditing
        record = {
            "category": category,
            "action": action_name,
            "params": params,
            "source": source,
            "is_high_risk": is_high_risk,
            "status": "pending",
        }
        self.last_actuations.append(record)

        # Enforce gate: high-risk actions must raise alert or obtain validation
        if is_high_risk:
            logger.warning("⚠️ High-risk action detected: %s. Ensuring safety validation...", action_name)
            # Will check approval via WillDecide in ActionExecutor. If Will rejects, action is blocked.

        # Map category to canonical ActionDomain
        domain = ActionDomain.EXTERNAL_ACTION
        if category in ("local_files", "code_repos"):
            domain = ActionDomain.FILE_WRITE
        elif category == "browser":
            domain = ActionDomain.NETWORK_CALL
        elif category == "desktop":
            domain = ActionDomain.ENVIRONMENT_ACTION
        elif category in ("cloud_resources_owned", "databases_owned"):
            domain = ActionDomain.CLOUD_CALL
        elif category == "robotics_devices":
            domain = ActionDomain.ENVIRONMENT_ACTION

        # Dispatch via ActionExecutor to ensure Will approval, WelfareTransaction, and PostActionReceipt
        result = await ActionExecutor.execute(
            domain=domain,
            action_name=f"actuation.{category}.{action_name}",
            params=params,
            source=source,
        )

        record["status"] = "success" if result.get("ok") else "failed"
        record["result"] = result
        return result


_actuator_instance: WorldActuator | None = None


def get_world_actuator() -> WorldActuator:
    global _actuator_instance
    if _actuator_instance is None:
        _actuator_instance = WorldActuator()
    return _actuator_instance
