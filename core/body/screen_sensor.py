"""core/body/screen_sensor.py
Perceptual sensor capturing desktop screenshot frames and layout status.
"""
import asyncio
import logging
import os
import tempfile
from pathlib import Path
from subprocess import SubprocessError
from typing import Any

from core.body.sensor_registry import BaseSensor
from core.runtime.errors import record_degradation
from core.runtime.permission_gates import screen_allowed
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.security.screen_capture_policy import evaluate_screen_capture_admission_async

logger = logging.getLogger("Body.ScreenSensor")

_SCREEN_SENSOR_ERRORS = (OSError, RuntimeError, SubprocessError, TimeoutError, TypeError, ValueError)


class ScreenSensor(BaseSensor):
    """Captures visual layout and dimensions of the screen."""

    @property
    def name(self) -> str:
        return "screen"

    async def read(self) -> dict[str, Any]:
        """Capture screen layout with macOS screencapture when available."""
        if not screen_allowed():
            return {
                "available": False,
                "error": "Screen perception disabled by user setting (permissions.screen)",
                "resolution": "1920x1080",
                "ocr_status": "not_available",
            }
        admission = await evaluate_screen_capture_admission_async()
        if not admission.allowed:
            return {
                "available": False,
                "error": admission.public_error,
                "resolution": "unknown",
                "ocr_status": "not_available",
                "capture_admission": admission.to_receipt(),
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

        capture_binary = Path("/usr/sbin/screencapture")
        screenshot_path = os.path.join(tempfile.gettempdir(), "aura_perception_screen.png")
        try:
            if await asyncio.to_thread(capture_binary.exists):
                await get_subprocess_gateway().run_async(
                    [str(capture_binary), "-x", screenshot_path],
                    read_only=True,
                    check=True,
                    timeout=3.0,
                    source="body.screen_sensor",
                    accelerator_capability="none",
                )
                file_size = await asyncio.to_thread(
                    lambda: Path(screenshot_path).stat().st_size
                )
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
