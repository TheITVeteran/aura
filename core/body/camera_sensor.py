"""core/body/camera_sensor.py
Camera frame and visual environment sensor.
"""
from typing import Dict, Any
from core.body.sensor_registry import BaseSensor


class CameraSensor(BaseSensor):
    """Monitors optical inputs and camera availability."""

    @property
    def name(self) -> str:
        return "camera"

    async def read(self) -> Dict[str, Any]:
        return {
            "status": "disabled_by_policy",  # Defaults to off in config for macOS AVFoundation safety
            "has_optical_feed": False,
            "last_frame_timestamp": None
        }
