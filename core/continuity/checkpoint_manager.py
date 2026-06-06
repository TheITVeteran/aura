"""core/continuity/checkpoint_manager.py
Checkpoint manager writing and reading LifeState snapshots to disk.
"""
import json
import os
import logging
from typing import Dict, Any, Optional
from core.config import get_config
from core.continuity.state_snapshot import StateSnapshotSerializer
from core.organism.life_state import LifeState

logger = logging.getLogger("Continuity.CheckpointManager")


class CheckpointManager:
    """Manages LifeState file checkpointers for runtime survival."""

    def __init__(self):
        self.config = get_config()
        self.serializer = StateSnapshotSerializer()
        self.checkpoint_path = os.path.join(self.config.paths.data_dir, "life_checkpoint.json")

    def save_checkpoint(self, state: LifeState) -> None:
        try:
            snapshot = self.serializer.serialize(state)
            with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=4)
            logger.info("Saved runtime checkpoint snapshot to: %s", self.checkpoint_path)
        except Exception as e:
            logger.error("Failed to save checkpoint: %s", e)

    def load_checkpoint(self, state: LifeState) -> bool:
        if not os.path.exists(self.checkpoint_path):
            return False
            
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            self.serializer.deserialize(snapshot, state)
            logger.info("Restored runtime state from checkpoint: %s", self.checkpoint_path)
            return True
        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e)
            return False
