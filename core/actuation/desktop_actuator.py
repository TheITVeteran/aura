"""core/actuation/desktop_actuator.py — OS Desktop and AppleScript Actuator."""
from __future__ import annotations

from typing import Any

from core.actuation.world_actuator import get_world_actuator


def _applescript_string(value: str) -> str:
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class DesktopActuator:
    """Wrapper for AppleScript desktop GUI/system commands."""

    @classmethod
    async def run_gui_action(cls, script: str, source: str = "desktop_actuator") -> dict[str, Any]:
        return await get_world_actuator().actuate(
            category="desktop",
            action_name="run_applescript",
            params={"script": script},
            source=source,
        )

    @classmethod
    async def change_wallpaper(cls, image_path: str, source: str = "desktop_actuator") -> dict[str, Any]:
        """Change the macOS desktop wallpaper through the desktop actuator path."""
        picture_path = _applescript_string(image_path)
        applescript = f'''
        tell application "System Events"
            tell every desktop
                set picture to {picture_path}
            end tell
        end tell
        '''
        return await cls.run_gui_action(applescript, source=source)
