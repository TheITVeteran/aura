"""environments/robotic_world/robotic_simulator.py
Simulates robotic mobile arm joints and physical spatial dimensions.
"""
from typing import Dict, List, Any


class VirtualRoboticWorld:
    """Simulates robotic joints and actuator statuses."""

    def __init__(self):
        self._arm_joints = {"shoulder": 0.0, "elbow": 0.0, "wrist": 0.0}

    def get_joints(self) -> Dict[str, float]:
        return self._arm_joints

    def set_joints(self, joints: Dict[str, float]) -> None:
        self._arm_joints.update(joints)
