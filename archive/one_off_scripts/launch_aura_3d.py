#!/usr/bin/env python3
"""
Aura 3D Launcher for macOS
==========================

This script launches Aura with the 3D VirtualBody viewer enabled.
On macOS, MuJoCo's passive viewer requires the 'mjpython' wrapper to 
handle the Cocoa event loop correctly.

Usage:
    mjpython launch_aura_3d.py
"""

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Aura.3DLauncher")
_LAUNCHER_RECOVERABLE_ERRORS = (OSError, RuntimeError, TimeoutError, ValueError)


def _orchestrator_lock_path() -> Path:
    return Path.home() / ".aura" / "locks" / "orchestrator.lock"


def _primary_runtime_is_active() -> bool:
    """Use the canonical runtime lock instead of stale state timestamps."""
    lock_path = _orchestrator_lock_path()
    if not lock_path.exists():
        return False
    try:
        pid = int(lock_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True

def check_mjpython():
    """Try to find mjpython in PATH or local .venv."""
    # 1. Check PATH
    try:
        result = get_subprocess_gateway().run(
            ["mjpython", "--version"],
            cwd=ROOT,
            timeout=15,
            read_only=True,
            source="maintenance_tooling:launch_aura_3d:mjpython_check",
        )
        if result.returncode == 0:
            return "mjpython"
    except _LAUNCHER_RECOVERABLE_ERRORS as exc:
        logger.debug("mjpython PATH check failed: %s: %s", type(exc).__name__, exc)
        
    # 2. Check local .venv
    venv_mjpy = ROOT / ".venv" / "bin" / "mjpython"
    if venv_mjpy.exists():
        return str(venv_mjpy)
        
    return None

def main():
    if sys.platform == "darwin":
        mjpython_bin = check_mjpython()
        
        # Check if we are already running under mjpython
        if "mjpython" not in sys.executable and not mjpython_bin:
            logger.error("❌ 'mjpython' not found. Please install it with: pip install mujoco")
            logger.info("Once installed, run this script with: mjpython launch_aura_3d.py")
            sys.exit(1)
            
        if "mjpython" not in sys.executable:
            logger.info(f"🚀 Re-launching Aura with '{mjpython_bin}' for 3D support...")
            cmd = [mjpython_bin, __file__] + sys.argv[1:]
            os.execv(mjpython_bin, cmd)

    logger.info("✨ Starting Aura 3D Interface...")
    
    # Set environment variables
    os.environ["AURA_SHOW_3D"] = "1"
    
    try:
        import asyncio

        from core.soma.virtual_body import VirtualBody

        from core.state.state_repository import StateRepository

        async def sync_mode(repo, body, stop_event: asyncio.Event):
            """Viewer-only mode: follow the primary Aura process."""
            logger.info("🔗 [Sync Mode] Connected to live Aura process. Tracking movements...")
            body.show_viewer = True
            body._start_viewer()
            
            while not stop_event.is_set():
                state = await repo._load_latest()
                if state and state.soma and body.data:
                    # Sync joint positions
                    body.data.qpos[:] = state.soma.qpos
                    # Sync sensors for consistency if needed
                    body.sensors.update(state.soma.sensors)
                await asyncio.sleep(0.05) # 20fps sync

        async def run_launcher():
            stop_event = asyncio.Event()
            # 1. Initialize State Repository to check for life
            repo = StateRepository()
            await repo.initialize()
            
            # 2. Setup VirtualBody (Viewer only if syncing)
            body = VirtualBody(show_viewer=False) 
            
            if _primary_runtime_is_active():
                # 3a. Sync Mode (Viewer only)
                await sync_mode(repo, body, stop_event)
            else:
                # 3b. Dual Mode (Start Engine + Viewer)
                logger.info("🌱 [Dual Mode] No active Aura found. Starting full engine...")
                from core.agency_core import AgencyCore
                aura = AgencyCore()
                await aura.start()
                # VirtualBody in Dual mode handles its own viewer if started via AgencyCore
                await stop_event.wait()
                
        asyncio.run(run_launcher())
    except ImportError as e:
        logger.error(f"❌ Could not import Aura core: {e}")
        logger.info("Ensure you are running from the project root and PYTHONPATH is set.")
    except _LAUNCHER_RECOVERABLE_ERRORS as exc:
        logger.error(f"💥 Runtime error: {type(exc).__name__}: {exc}")

if __name__ == "__main__":
    main()
