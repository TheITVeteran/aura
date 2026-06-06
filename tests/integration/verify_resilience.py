################################################################################


import asyncio
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.orchestrator import RobustOrchestrator

async def test_resilience():
    print("--- Testing Resilience (State Snapshot) ---")
    original_restore_history = os.environ.get("AURA_RESTORE_HISTORY")
    try:
        # 1. Set up a real orchestrator instance.
        print("Initializing Orchestrator...")
        orchestrator = RobustOrchestrator()
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_dir = Path(tmp) / "snapshots"
            orchestrator.state_manager.snapshot_dir = snapshot_dir

            orchestrator.status.cycle_count = 42
            orchestrator.conversation_history = [
                {"role": "user", "content": "Hello Aura"},
                {"role": "assistant", "content": "Hello User"}
            ]
            orchestrator.boredom = 5

            # 2. Save Snapshot
            print("Saving Snapshot...")
            orchestrator._save_state("test_verification")

            snapshot_path = snapshot_dir / "latest_snapshot.json"
            assert snapshot_path.exists(), "latest snapshot file was not created"
            print(f"✓ Snapshot file created at {snapshot_path}")

            # 3. Exercise restart restoration with a new instance.
            print("Running restart restoration with a new instance...")
            os.environ["AURA_RESTORE_HISTORY"] = "1"
            new_orchestrator = RobustOrchestrator()
            new_orchestrator.state_manager.snapshot_dir = snapshot_dir
            new_orchestrator._load_state()

            assert new_orchestrator.status.cycle_count == 42
            assert len(new_orchestrator.conversation_history) == 2
            assert new_orchestrator.boredom == 5
            print("✓ Cycle Count Restored (42)")
            print("✓ Conversation History Restored")
            print("✓ Boredom Level Restored")
    finally:
        if original_restore_history is None:
            os.environ.pop("AURA_RESTORE_HISTORY", None)
        else:
            os.environ["AURA_RESTORE_HISTORY"] = original_restore_history

    print("--- Resilience Test Complete ---")

if __name__ == "__main__":
    asyncio.run(test_resilience())


##
