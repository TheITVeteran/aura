import json
import logging

import websockets

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Embodiment")


class UnityEmbodiment:
    def __init__(self):
        self.ws = None
        self.avatar_state = {
            "position": [0, 0, 0],
            "rotation": [0, 0, 0, 1],
            "gaze": [0, 0],
            "expression": "neutral",
            "breathing": 0.5,
            "energy": 100.0,
            "heat": 37.0,
            "integrity": 100.0,
        }

    async def connect_unity(self):
        """Connect to Unity WebRTC server"""
        uri = "ws://localhost:8765/avatar"
        try:
            self.ws = await websockets.connect(uri)
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation("unity_bridge", e)
            logger.warning("Unity connection failed: %s", e)

    async def update_affect(self, affect_wheel: dict):
        """Drive avatar from emotional state"""
        if not self.ws:
            return

        primary = affect_wheel.get("primary", {})
        dimensions = affect_wheel.get("dimensions", {})
        somatic = affect_wheel.get("somatic_indices", {})
        valence = float(dimensions.get("valence", 0.0) or 0.0)

        # Map emotions → FACS Action Units
        expression_map = {"joy": "smile_au12", "fear": "eyes_wide_au5", "anger": "furrow_brow_au4"}

        # Send to Unity
        try:
            msg = {
                "type": "affect_update",
                "valence": valence,
                "expression": max(expression_map, key=lambda k: primary.get(k, 0))
                if primary
                else "neutral",
                "activation_index": float(somatic.get("activation", 0.0) or 0.0),
                "somatic_classification": "simulated_functional_index",
            }
            await self.ws.send(json.dumps(msg))
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation("unity_bridge", e)
            logger.error("Failed to send affect update to Unity: %s", e)
            self.ws = None

    async def get_sensor_data(self) -> dict:
        """Read Unity sensors"""
        if not self.ws:
            return {}
        try:
            msg = await self.ws.recv()
            data = json.loads(msg)
            return {
                "proprioception": data.get("joint_angles", []),
                "tactile": data.get("touch_sensors", []),
                "vestibular": data.get("acceleration", [0, 0, 0]),
            }
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation("unity_bridge", e)
            logger.error("Failed to receive sensor data from Unity: %s", e)
            return {}

    # Adapter for Heartbeat
    def update(self) -> dict:
        """Called by heartbeat to get body state."""
        return self.avatar_state
