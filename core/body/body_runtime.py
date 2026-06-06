"""core/body/body_runtime.py
Central controller for Aura's perceptual body, coordinating all sensors.
"""
from typing import Dict, Any, Optional
import logging

from core.body.sensor_registry import get_sensor_registry
from core.body.screen_sensor import ScreenSensor
from core.body.microphone_sensor import MicrophoneSensor
from core.body.camera_sensor import CameraSensor
from core.body.keyboard_mouse_state import KeyboardMouseSensor
from core.body.app_focus_sensor import AppFocusSensor
from core.body.clipboard_sensor import ClipboardSensor
from core.body.filesystem_sensor import FilesystemSensor
from core.body.browser_state_sensor import BrowserStateSensor
from core.body.ui_accessibility_sensor import UiAccessibilitySensor
from core.body.environment_snapshot import EnvironmentSnapshotSensor

logger = logging.getLogger("Body.BodyRuntime")


class BodyRuntime:
    """Manages structural sensory perception routines."""

    def __init__(self):
        self.registry = get_sensor_registry()
        self._initialized = False

    def initialize_sensors(self) -> None:
        """Register default sensor plugins."""
        if self._initialized:
            return
        
        self.registry.register(ScreenSensor())
        self.registry.register(MicrophoneSensor())
        self.registry.register(CameraSensor())
        self.registry.register(KeyboardMouseSensor())
        self.registry.register(AppFocusSensor())
        self.registry.register(ClipboardSensor())
        self.registry.register(FilesystemSensor())
        self.registry.register(BrowserStateSensor())
        self.registry.register(UiAccessibilitySensor())
        self.registry.register(EnvironmentSnapshotSensor())
        
        self._initialized = True
        logger.info("Perceptual body sensors initialized successfully.")

    async def perceive_all(self) -> Dict[str, Any]:
        """Poll all sensors and consolidate results."""
        self.initialize_sensors()
        return await self.registry.read_all()

    async def get_system_status(self) -> Dict[str, Any]:
        """Utility extracting vital body stats to feed LifeState directly."""
        self.initialize_sensors()
        
        env_sensor = self.registry.get_sensor("environment_snapshot")
        focus_sensor = self.registry.get_sensor("app_focus")
        clip_sensor = self.registry.get_sensor("clipboard")

        status = {}
        if env_sensor:
            env_data = await env_sensor.read()
            status["cpu"] = env_data.get("cpu_percent", 10.0)
            status["memory"] = env_data.get("memory_percent", 50.0)
            status["battery"] = env_data.get("battery_percent", 100.0)
        
        if focus_sensor:
            focus_data = await focus_sensor.read()
            status["focus_app"] = focus_data.get("active_app", "Terminal")
        
        if clip_sensor:
            clip_data = await clip_sensor.read()
            status["clipboard"] = clip_data.get("content", "")

        return status


# Singleton Access
_body_runtime: Optional[BodyRuntime] = None


def get_body_runtime() -> BodyRuntime:
    global _body_runtime
    if _body_runtime is None:
        _body_runtime = BodyRuntime()
    return _body_runtime
