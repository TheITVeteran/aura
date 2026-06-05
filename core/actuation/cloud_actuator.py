"""core/actuation/cloud_actuator.py — Cloud Resources and Databases Actuator."""
from __future__ import annotations

from typing import Any, Dict
from core.actuation.world_actuator import get_world_actuator


class CloudActuator:
    """Wrapper for managing cloud infrastructure and DB connections."""

    @classmethod
    async def query_db(cls, db_name: str, query: str, source: str = "cloud_actuator") -> Dict[str, Any]:
        return await get_world_actuator().actuate(
            category="databases_owned",
            action_name="query_database",
            params={"db_name": db_name, "query": query},
            source=source,
        )

    @classmethod
    async def modify_infra(cls, service: str, state: str, source: str = "cloud_actuator") -> Dict[str, Any]:
        # High risk action: changing cloud configuration
        return await get_world_actuator().actuate(
            category="cloud_resources_owned",
            action_name="change_cloud_infra",
            params={"service": service, "desired_state": state},
            source=source,
            require_approval=True,
        )
