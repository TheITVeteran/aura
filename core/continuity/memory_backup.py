"""core/continuity/memory_backup.py
Memory backup manager making periodic copies of memory.
"""
import shutil
import os
import logging
from core.config import get_config

logger = logging.getLogger("Continuity.MemoryBackup")


class MemoryBackupManager:
    """Creates file copies of sqlite/jsonl files for recovery safety."""

    def __init__(self):
        self.config = get_config()

    def backup_database(self) -> bool:
        src = os.path.join(self.config.paths.memory_dir, "autobiography.jsonl")
        dest = os.path.join(self.config.paths.memory_dir, "autobiography_backup.jsonl")

        if not os.path.exists(src):
            return False
            
        try:
            shutil.copyfile(src, dest)
            logger.info("Created memory database backup.")
            return True
        except Exception as e:
            logger.error("Failed to copy database: %s", e)
            return False
