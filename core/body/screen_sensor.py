"""core/body/screen_sensor.py
Perceptual sensor capturing desktop screenshot frames and layout status.
"""
import logging
import os
import tempfile
from subprocess import SubprocessError
from typing import Any, Dict

from core.body.sensor_registry import BaseSensor
from core.runtime.errors import record_degradation
from core.runtime.permission_gates import screen_allowed
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Body.ScreenSensor")

_SCREEN_SENSOR_ERRORS = (OSError, RuntimeError, SubprocessError, TimeoutError, TypeError, ValueError)


class ScreenSensor(BaseSensor):
    """Captures visual layout and dimensions of the screen."""

    @property
    def name(self) -> str:
        return "screen"

    async def read(self) -> Dict[str, Any]:
        """Capture screen layout with macOS screencapture when available."""
        if not screen_allowed():
            return {
                "available": False,
                "error": "Screen perception disabled by user setting (permissions.screen)",
                "resolution": "1920x1080",
                "ocr_status": "not_available",
            }
        try:
            from core.security.permission_guard import PermissionType, get_permission_guard
            guard = get_permission_guard()
            perm = await guard.check_permission(PermissionType.SCREEN)
            if not perm.get("granted", False):
                return {
                    "available": False,
                    "error": "Screen recording permission not granted",
                    "resolution": "1920x1080",
                    "ocr_status": "not_available",
                }
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("body.screen_sensor.permission_check", exc)

        screenshot_path = os.path.join(tempfile.gettempdir(), "aura_perception_screen.png")
        try:
            if os.path.exists("/usr/sbin/screencapture"):
                await get_subprocess_gateway().run_async(
                    ["/usr/sbin/screencapture", "-x", screenshot_path],
                    read_only=True,
                    check=True,
                    timeout=3.0,
                    source="body.screen_sensor",
                )
                file_size = os.path.getsize(screenshot_path)
                return {
                    "available": True,
                    "file_path": screenshot_path,
                    "file_size": file_size,
                    "resolution": "unknown",
                    "ocr_status": "not_performed",
                }
        except _SCREEN_SENSOR_ERRORS as e:
            record_degradation("body.screen_sensor", e)
            logger.debug("screencapture utility failed: %s", e)

        return {
            "available": False,
            "error": "Hardware capture disabled or not supported",
            "resolution": "1920x1080",
            "ocr_status": "not_available",
        }
