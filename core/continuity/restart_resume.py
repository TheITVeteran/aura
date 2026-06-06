"""core/continuity/restart_resume.py
Restoration boot script resuming states after abrupt exits.
"""
from core.continuity.checkpoint_manager import CheckpointManager
from core.organism.life_state import LifeState
import logging

logger = logging.getLogger("Continuity.RestartResume")


class RestartResumeEngine:
    """Manages boot sequence state reload triggers."""

    def __init__(self):
        self.manager = CheckpointManager()

    def resume_from_abrupt_exit(self, state: LifeState) -> bool:
        logger.info("RestartResumeEngine attempting state restoration on boot...")
        success = self.manager.load_checkpoint(state)
        if success:
            logger.info("Successfully recovered LifeState from last checkpoint.")
            return True
        logger.info("No valid checkpoint found. Starting fresh lifecycle.")
        return False
