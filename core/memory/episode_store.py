"""core/memory/episode_store.py
Episode store driving autobiographical disk persistence.
"""
import json
import os
import logging
from typing import List, Dict, Any
from core.config import get_config
from core.memory.life_event import LifeEvent

logger = logging.getLogger("Memory.EpisodeStore")


class EpisodeStore:
    """Handles disk writing and querying of structured life events."""

    def __init__(self):
        cfg = get_config()
        self.db_path = os.path.join(cfg.paths.memory_dir, "autobiography.jsonl")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    async def save_event(self, event: LifeEvent) -> None:
        """Append event transaction to autobiographical log file."""
        try:
            with open(self.db_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.error("Failed to persist life event: %s", e)

    async def load_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Loads the most recent N events from disk storage."""
        if not os.path.exists(self.db_path):
            return []
        
        events = []
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line.strip()))
        except Exception as e:
            logger.error("Failed to read autobiographical logs: %s", e)

        return events[-limit:]
