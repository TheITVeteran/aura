"""core/body/app_focus_sensor.py
App focus sensor querying active windows using macOS AppleScript.
"""
import logging
import os
import time
from subprocess import SubprocessError
from typing import Any, Dict

from core.body.sensor_registry import BaseSensor
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Body.AppFocusSensor")

_APP_FOCUS_SENSOR_ERRORS = (OSError, RuntimeError, SubprocessError, TimeoutError, TypeError, ValueError)


class AppFocusSensor(BaseSensor):
    """Tracks active frontmost window process name."""

    @property
    def name(self) -> str:
        return "app_focus"

    async def read(self) -> Dict[str, Any]:
        """Queries the active application on macOS via AppleScript."""
        try:
            if os.path.exists("/usr/bin/osascript"):
                cmd = ["/usr/bin/osascript", "-e", 'tell application "System Events" to get name of first process whose frontmost is true']
                res = await get_subprocess_gateway().run_async(
                    cmd,
                    read_only=True,
                    check=True,
                    timeout=2.0,
                    source="body.app_focus_sensor",
                )
                app_name = res.stdout.strip()
                return {
                    "active_app": app_name,
                    "is_browser": app_name in ["Google Chrome", "Safari", "Firefox"],
                    "timestamp": time.time(),
                }
        except _APP_FOCUS_SENSOR_ERRORS as e:
            record_degradation("body.app_focus_sensor", e)
            logger.debug("Failed to query active app via AppleScript: %s", e)

        return {
            "active_app": None,
            "is_browser": False,
            "status": "unavailable",
            "error": "Failed to query macOS front window",
        }
