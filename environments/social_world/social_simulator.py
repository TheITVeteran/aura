"""environments/social_world/social_simulator.py
Simulates social interactions, messages, and operator directives.
"""
from typing import List, Dict, Any
import time


class VirtualSocialWorld:
    """Simulates communication incoming channels from the operator."""

    def __init__(self):
        self._queue: List[Dict[str, Any]] = [
            {"sender": "Bryan", "message": "Help me refactor the tests folder.", "timestamp": time.time()}
        ]

    def poll_messages(self) -> List[Dict[str, Any]]:
        messages = list(self._queue)
        self._queue.clear()
        return messages

    def send_agent_message(self, text: str) -> None:
        pass
