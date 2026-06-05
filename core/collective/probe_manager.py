"""core/collective/probe_manager.py
Phase 16.4: Ghost Deployment - Resource Monitoring Probes.
Spawns and manages lightweight monitoring scripts.
"""
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.Collective.ProbeManager")

class ProbeManager:
    """Manages external 'Ghost Probes' for long-term monitoring."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.probes: dict[str, asyncio.subprocess.Process] = {}
        self.probe_metadata: dict[str, dict[str, Any]] = {}
        self._running = True

    async def deploy_probe(self, probe_id: str, target: str, type: str = "file", duration: int = 3600) -> bool:
        """Spawn a ghost probe process."""
        if probe_id in self.probes:
            logger.warning("Probe %s already active.", probe_id)
            return False

        try:
            duration_seconds = max(1, min(int(duration), 86_400))
        except (TypeError, ValueError):
            duration_seconds = 3_600

        # Keep probe scripts standalone and data-only: no Aura imports and no
        # raw interpolation of target/type into executable Python syntax.
        probe_script = f"""
import os
import sys
import time

target = {json.dumps(str(target))}
probe_type = {json.dumps(str(type))}
duration = {duration_seconds}
print(f"ghost_probe_start:{{probe_type}}:{{target}}")
try:
    start_time = time.time()
    while time.time() - start_time < duration:
        if probe_type == "file":
            if os.path.exists(target):
                mtime = os.path.getmtime(target)
                print(f"ghost_update:file_exists:{{mtime}}")
        elif probe_type == "ping":
             # Simple ping simulation
             print(f"ghost_update:ping_ok")
        
        sys.stdout.flush()
        remaining = duration - (time.time() - start_time)
        if remaining <= 0:
            break
        time.sleep(min(1.0, remaining))
except (OSError, IOError) as e:
    print(f"ghost_error:{{e}}")
    sys.stdout.flush()
"""
        probe_path = Path(tempfile.gettempdir()) / f"aura_probe_{probe_id}.py"
        atomic_write_text(probe_path, probe_script)
        
        try:
            # Spawn in background with asyncio
            process = await get_subprocess_gateway().spawn_async(
                [sys.executable, str(probe_path)],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                source="tool_execution:probe_manager.ghost_probe",
            )
            
            self.probes[probe_id] = process
            self.probe_metadata[probe_id] = {
                "target": target,
                "type": type,
                "start_time": time.time(),
                "expiry": time.time() + duration_seconds,
                "path": str(probe_path)
            }
            
            # Start a background listener for this probe
            get_task_tracker().create_task(self._listen_to_probe(probe_id))
            
            logger.info("👻 Ghost Probe '%s' deployed to watch %s.", probe_id, target)
            return True
        except (subprocess.SubprocessError, OSError) as e:
            record_degradation('probe_manager', e)
            logger.error("Failed to deploy probe %s: %s", probe_id, e)
            return False

    async def _listen_to_probe(self, probe_id: str):
        """Listen for telemetry from a specific probe."""
        process = self.probes.get(probe_id)
        if not process:
            return

        while process.returncode is None:
            line_bytes = await process.stdout.readline()
            if not line_bytes:
                break
            
            line = line_bytes.decode().strip()
            if line.startswith("ghost_update:"):
                update = line.split(":", 2)[1:]
                enqueue = getattr(self.orchestrator, "enqueue_message", None)
                if callable(enqueue):
                    enqueue(f"Impulse [GHOST:{probe_id}]: {update}")
                else:
                    logger.debug("Ghost Probe %s update dropped; orchestrator has no enqueue_message.", probe_id)
            elif line.startswith("ghost_error:"):
                err = line.split(":", 1)[1]
                logger.error("Ghost Probe %s error: %s", probe_id, err)

        # Cleanup
        await self.cleanup_probe(probe_id)

    async def cleanup_probe(self, probe_id: str) -> bool:
        """Terminate and cleanup a probe's resources."""
        ok = True
        if probe_id in self.probes:
            proc = self.probes.pop(probe_id)
            try:
                if proc.returncode is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
                    except TimeoutError:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        await asyncio.wait_for(proc.wait(), timeout=5.0)
            except ProcessLookupError:
                pass
            except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as e:
                ok = False
                record_degradation('probe_manager', e)
                logger.debug("Failed to kill probe process group %d: %s", proc.pid, e)
            
        meta = self.probe_metadata.pop(probe_id, {})
        path = meta.get("path")
        if path:
            try:
                await asyncio.to_thread(Path(path).unlink, missing_ok=True)
            except OSError as e:
                ok = False
                record_degradation('probe_manager', e)
                logger.debug("Failed to remove probe script %s: %s", path, e)

        if ok:
            logger.info("👻 Ghost Probe '%s' cleaned up.", probe_id)
        return ok

    async def auto_cleanup_loop(self):
        """Periodically remove expired probes."""
        while self._running:
            await asyncio.sleep(60)
            now = time.time()
            to_remove = [pid for pid, meta in self.probe_metadata.items() if now > meta["expiry"]]
            for pid in to_remove:
                await self.cleanup_probe(pid)
