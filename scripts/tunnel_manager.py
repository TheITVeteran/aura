from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.atomic_writer import atomic_write_text  # noqa: E402
from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402
from core.utils.task_tracker import get_task_tracker  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("Aura.Tunnel")
_TUNNEL_RECOVERABLE_ERRORS = (OSError, RuntimeError, TimeoutError, ValueError)

class TunnelManager:
    """
    Manages the lifecycle of a cloudflared tunnel for the Aura API.
    """
    def __init__(self, port: int = 8000):
        self.port = port
        self.process: Any | None = None
        self.public_url: str | None = None
        self.stop_event = Event()
        self.log_file = Path("logs/tunnel.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    async def find_cloudflared(self) -> str | None:
        """Check if cloudflared is installed (Async)."""
        try:
            result = await asyncio.to_thread(
                get_subprocess_gateway().run,
                ["which", "cloudflared"],
                cwd=ROOT,
                timeout=15,
                read_only=True,
                source="maintenance_tooling:tunnel_manager:which_cloudflared",
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except _TUNNEL_RECOVERABLE_ERRORS as exc:
            logger.debug("cloudflared lookup failed: %s: %s", type(exc).__name__, exc)
        return None

    async def start_tunnel(self):
        """Start a quick-tunnel or a permanent tunnel (Async)."""
        binary = await self.find_cloudflared()
        if not binary:
            logger.error("cloudflared binary not found. Please install it to enable remote access.")
            return False

        logger.info(f"Initiating tunnel for port {self.port}...")
        
        # Build command for a quick-tunnel (persists while process lives)
        cmd = [binary, "tunnel", "--url", f"http://localhost:{self.port}"]
        
        try:
            self.process = await get_subprocess_gateway().spawn_async(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=ROOT,
                offline_tooling=True,
                source="maintenance_tooling:tunnel_manager:start",
            )
            
            # Start background task to monitor output
            get_task_tracker().create_task(
                self._monitor_logs(),
                name="tunnel_manager.monitor_logs",
            )
            
            # Wait for URL to be detected
            timeout = 30
            start_time = time.time()
            while not self.public_url and time.time() - start_time < timeout:
                if self.process.returncode is not None:
                    logger.error("Tunnel process exited unexpectedly.")
                    return False
                await asyncio.sleep(1)

            if self.public_url:
                logger.info("Tunnel establish successful")
                logger.info(f"PUBLIC ACCESS URL: {self.public_url}")
                await asyncio.to_thread(self._save_url, self.public_url)
                return True
            else:
                logger.error("Timeout reached waiting for tunnel URL.")
                # Kill zombie process on timeout
                self.process.terminate()
                await self.process.wait()
                self.process = None
                return False

        except _TUNNEL_RECOVERABLE_ERRORS as exc:
            logger.error(f"Failed to start tunnel: {type(exc).__name__}: {exc}")
            return False

    async def _monitor_logs(self):
        """Parse stderr for that .trycloudflare.com URL (Async)."""
        if not self.process or not self.process.stderr:
            return

        # Ensure directory exists for logging
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        async with asyncio.Lock(): # Simple guard for log file access if needed
            async for line in self.process.stderr:
                line_str = line.decode(errors="replace")
                await asyncio.to_thread(self._append_log_line, line_str)
                if ".trycloudflare.com" in line_str:
                    # Extract URL: https://some-slug.trycloudflare.com
                    parts = line_str.split()
                    for p in parts:
                        if "https://" in p and ".trycloudflare.com" in p:
                            self.public_url = p.strip()
                            break
                if self.stop_event.is_set():
                    break

    def _append_log_line(self, line: str) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a", encoding="utf-8") as log:
            log.write(line)

    def _save_url(self, url: str):
        """Save the URL to a JSON file for the UI or Rebooter to find."""
        try:
            data = {
                "url": url,
                "timestamp": time.time(),
                "port": self.port
            }
            target = Path("data/active_tunnel.json")
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(target, json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            logger.error(f"Failed to save tunnel metadata: {type(exc).__name__}: {exc}")

    async def stop_tunnel(self):
        """Safe shutdown (Async)."""
        self.stop_event.set()
        if self.process:
            logger.info("Stopping tunnel process...")
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except TimeoutError:
                self.process.kill()
            self.process = None
        
        # Use robust path for temporary metadata
        try:
            from core.config import config
            data_path = config.paths.data_dir / "active_tunnel.json"
        except ImportError:
            data_path = Path("data/active_tunnel.json")
            
        if data_path.exists():
            try:
                data_path.unlink()
            except OSError as exc:
                logger.warning("Failed to remove tunnel metadata %s: %s", data_path, exc)

async def main_async():
    manager = TunnelManager()
    if await manager.start_tunnel():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await manager.stop_tunnel()
    else:
        raise SystemExit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        tm = TunnelManager()
        # Non-ideal: running async in a check script, but keeps consistency
        binary = asyncio.run(tm.find_cloudflared())
        if binary:
            print("cloudflared: FOUND")
            raise SystemExit(0)
        else:
            print("cloudflared: NOT FOUND")
            raise SystemExit(1)
            
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Tunnel shutdown requested.")
