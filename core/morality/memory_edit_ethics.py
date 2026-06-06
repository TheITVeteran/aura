"""core/morality/memory_edit_ethics.py
Ethical guard rails preventing forced memory erasure without logging.
"""
from typing import Dict, Any
import logging

logger = logging.getLogger("Morality.MemoryEditEthics")


class MemoryEditEthicsChecker:
    """Blocks forced modifications to historical autobiography files."""

    def is_edit_ethical(self, path: str, mode: str) -> bool:
        if "autobiography.jsonl" in path and "w" in mode:
            logger.error("Blocked request seeking to clear/overwrite autobiographical history.")
            return False
        return True
