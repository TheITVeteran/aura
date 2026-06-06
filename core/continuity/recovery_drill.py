"""core/continuity/recovery_drill.py
Simulates abrupt process termination to test checkpoint resumption.
"""
from core.continuity.checkpoint_manager import CheckpointManager
from core.organism.life_state import LifeState
import logging

logger = logging.getLogger("Continuity.RecoveryDrill")


class RecoveryDrillSuite:
    """Runs test drills simulating a crash by dumping state and loading it into a clean object."""

    def __init__(self):
        self.manager = CheckpointManager()

    def run_drill(self, active_state: LifeState) -> bool:
        logger.warning("Initiating simulated system crash and recovery drill...")
        
        # 1. Save checkpoint
        self.manager.save_checkpoint(active_state)
        
        # 2. Spawn clean state target
        clean_state = LifeState()
        
        # 3. Load checkpoint
        recovered = self.manager.load_checkpoint(clean_state)
        
        # Verify tick count matches
        passed = recovered and clean_state.tick_count == active_state.tick_count
        logger.info("Recovery drill result: %s", "PASSED" if passed else "FAILED")
        return passed
