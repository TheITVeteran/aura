import logging
from pathlib import Path

from core.config import get_config
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Continuity.MemoryBackup")

_MEMORY_BACKUP_ERRORS = (OSError, RuntimeError, TypeError, ValueError)


class MemoryBackupManager:
    """Creates file copies of sqlite/jsonl files for recovery safety."""

    def __init__(self):
        self.config = get_config()

    def backup_database(self) -> bool:
        src = Path(self.config.paths.memory_dir) / "autobiography.jsonl"
        dest = Path(self.config.paths.memory_dir) / "autobiography_backup.jsonl"

        if not src.exists():
            return False
            
        try:
            get_file_write_gateway().write_bytes(
                dest,
                src.read_bytes(),
                source="continuity.memory_backup",
            )
            logger.info("Created memory database backup.")
            return True
        except _MEMORY_BACKUP_ERRORS as e:
            record_degradation("continuity.memory_backup", e)
            logger.error("Failed to copy database: %s", e)
            return False
