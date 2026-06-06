import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from core.config import get_config
from core.continuity.state_snapshot import StateSnapshotSerializer
from core.organism.life_state import LifeState
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Continuity.CheckpointManager")

_CHECKPOINT_IO_ERRORS = (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError)


class CheckpointManager:
    """Manages LifeState file checkpointers for runtime survival."""

    def __init__(self):
        self.config = get_config()
        self.serializer = StateSnapshotSerializer()
        self.checkpoint_path = Path(self.config.paths.data_dir) / "life_checkpoint.json"

    def save_checkpoint(self, state: LifeState) -> None:
        try:
            snapshot = self.serializer.serialize(state)
            get_file_write_gateway().write_text(
                self.checkpoint_path,
                json.dumps(snapshot, indent=4, sort_keys=True),
                source="continuity.checkpoint_manager",
            )
            logger.info("Saved runtime checkpoint snapshot to: %s", self.checkpoint_path)
        except _CHECKPOINT_IO_ERRORS as e:
            record_degradation("continuity.checkpoint.save", e)
            logger.error("Failed to save checkpoint: %s", e)

    def load_checkpoint(self, state: LifeState) -> bool:
        if not self.checkpoint_path.exists():
            return False
            
        try:
            snapshot = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            self.serializer.deserialize(snapshot, state)
            logger.info("Restored runtime state from checkpoint: %s", self.checkpoint_path)
            return True
        except _CHECKPOINT_IO_ERRORS as e:
            record_degradation("continuity.checkpoint.load", e)
            logger.error("Failed to load checkpoint: %s", e)
            return False
