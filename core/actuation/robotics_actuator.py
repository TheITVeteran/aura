"""core/actuation/robotics_actuator.py — Robotics Devices Actuator."""
from __future__ import annotations

from typing import Any, Dict
from core.actuation.world_actuator import get_world_actuator


class RoboticsActuator:
    """Wrapper for external physical/robotics device interactions."""

    @classmethod
    async def command_device(cls, device_id: str, command: str, params: Dict[str, Any], source: str = "robotics_actuator") -> Dict[str, Any]:
        return await get_world_actuator().actuate(
            category="robotics_devices",
            action_name="command_device",
            params={"device_id": device_id, "command": command, **params},
            source=source,
        )
