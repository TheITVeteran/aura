"""core/actuation/desktop_actuator.py — OS Desktop and AppleScript Actuator."""
from __future__ import annotations

from typing import Any, Dict
from core.actuation.world_actuator import get_world_actuator


class DesktopActuator:
    """Wrapper for AppleScript desktop GUI/system commands."""

    @classmethod
    async def run_gui_action(cls, script: str, source: str = "desktop_actuator") -> Dict[str, Any]:
        return await get_world_actuator().actuate(
            category="desktop",
            action_name="run_applescript",
            params={"script": script},
            source=source,
        )
