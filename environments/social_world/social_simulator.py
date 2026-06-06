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
        self._outbox: List[Dict[str, Any]] = []

    def poll_messages(self) -> List[Dict[str, Any]]:
        messages = list(self._queue)
        self._queue.clear()
        return messages

    def send_agent_message(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("agent message text must be non-empty")
        message = {"sender": "Aura", "message": text, "timestamp": time.time()}
        self._outbox.append(message)
        return message

    def sent_messages(self) -> List[Dict[str, Any]]:
        return list(self._outbox)
