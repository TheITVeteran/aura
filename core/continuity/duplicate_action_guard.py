"""core/continuity/duplicate_action_guard.py
Duplicate action guard preventing duplicate execution of side-effect commands.
"""
from typing import Dict, Any, List
import logging

logger = logging.getLogger("Continuity.DuplicateActionGuard")


class DuplicateActionGuard:
    """Tracks recently executed commands to prevent duplicate side effects."""

    def __init__(self):
        # Maps action_hash -> timestamp
        self._execution_registry: Dict[str, float] = {}

    def is_duplicate(self, action_hash: str) -> bool:
        return action_hash in self._execution_registry

    def register_execution(self, action_hash: str, timestamp: float) -> None:
        self._execution_registry[action_hash] = timestamp
        logger.info("Registered action execution hash to guard registry: %s", action_hash)
        
    def clear_old_entries(self, cutoff_s: float = 3600.0) -> None:
        import time
        limit = time.time() - cutoff_s
        self._execution_registry = {
            h: t for h, t in self._execution_registry.items() if t > limit
        }
