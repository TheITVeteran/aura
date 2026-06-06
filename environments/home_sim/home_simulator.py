"""environments/home_sim/home_simulator.py
Simulates a virtual home environment with smart device triggers and status feeds.
"""
from typing import Dict, Any


class VirtualHomeWorld:
    """Simulates smart home appliance states (IoT switches, lights, thermostat)."""

    def __init__(self):
        self._states = {
            "living_room_light": "off",
            "thermostat_temperature": 21.0
        }

    def get_state(self, key: str) -> Any:
        return self._states.get(key)

    def set_state(self, key: str, value: Any) -> None:
        self._states[key] = value
