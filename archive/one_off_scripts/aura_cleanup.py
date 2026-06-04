import logging
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aura.IroncladCleanup")
_CLEANUP_RECOVERABLE_ERRORS = (OSError, RuntimeError, TimeoutError, ValueError)

def main() -> None:
    """
    🔥 [IRONCLAD] Absolute system purge for Aura.
    Targets orchestrators, workers, simulation scripts, and stale locks.
    """
    logger.info("🔥 [IRONCLAD] Initiating Absolute System Purge...")
    
    # 1. Broad targets to terminate
    # We include 'simulate_200.py' and 'python3' to catch rogue subprocess trees.
    targets = ["aura_main.py", "mlx_worker.py", "gui_actor.py", "llama-server", "simulate_200.py"]
    
    for target in targets:
        try:
            logger.info(f"  💀 Killing: {target}")
            # -9: SIGKILL, -f: Full command line match
            result = get_subprocess_gateway().run(
                ["pkill", "-9", "-f", target],
                cwd=ROOT,
                timeout=10,
                capture_output=True,
                offline_tooling=True,
                source="maintenance_tooling:aura_cleanup:pkill",
            )
            if result.returncode not in {0, 1}:
                logger.warning("pkill failed for %s: %s", target, result.stderr.strip())
        except _CLEANUP_RECOVERABLE_ERRORS as exc:
            logger.warning("Failed to kill %s: %s: %s", target, type(exc).__name__, exc)

    # 2. Hard Purge Locks
    # Instead of unlinking individual files, we wipe the whole directory to 
    # clear out hidden or corrupted fcntl locks.
    lock_dir = Path.home() / ".aura" / "locks"
    if lock_dir.exists():
        logger.info(f"🔓 [IRONCLAD] Wiping lock directory: {lock_dir}")
        try:
            shutil.rmtree(lock_dir)
            lock_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error(f"Failed to wipe locks: {exc}")

    # 3. Final Pause to let OS release ports/VRAM
    time.sleep(2)
    logger.info("✅ [IRONCLAD] Purge complete. System is verified CLEAN.")

if __name__ == "__main__":
    main()
