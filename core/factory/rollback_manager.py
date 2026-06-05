"""core/factory/rollback_manager.py — Git Workspace State Rollback Manager.
"""
from __future__ import annotations

import logging
from typing import Dict

from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.RollbackManager")


class RollbackManager:
    """Manages workspace state cleanup and checkout rollbacks on failure."""

    @staticmethod
    def discard_changes(file_path: str) -> Dict[str, Any]:
        logger.warning("🏭 RollbackManager: Reverting changes in %s!", file_path)
        gateway = get_subprocess_gateway()

        # Run git checkout/restore to clear uncommitted changes
        proc = gateway.run(
            argv=["git", "restore", file_path],
            timeout=10.0,
            source="rollback_manager",
        )
        passed = proc.returncode == 0
        logger.info("🏭 Rollback outcome: %s", "SUCCESS" if passed else "FAILED")

        return {
            "rolled_back": passed,
            "file": file_path,
        }
