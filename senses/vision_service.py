"""
Vision Service (The Eyes)
Runs in the Background (Sandbox).
Captures screen content and updates 'sensory_memory.json'.
"""
import asyncio
import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from core.runtime.errors import record_degradation
except ImportError:
    def record_degradation(_component: str, _error: BaseException) -> None:
        return None

# Try to import mss for screenshots
try:
    import mss
    mss_available = True
except ImportError:
    mss_available = False

_VISION_CAPTURE_ERRORS = (OSError, RuntimeError, ValueError)
_VISION_SERVICE_ERRORS = (IndexError, OSError, RuntimeError, ValueError)


async def run_vision_loop(
    *,
    stop_event: asyncio.Event | None = None,
    interval_s: float = 5.0,
    output_dir: str | os.PathLike[str] | None = None,
    monitor_index: int = 1,
    max_frames: int | None = None,
) -> None:
    print("Vision Service Starting (Async)...")

    if not mss_available:
        print("Error: 'mss' not installed. Please run 'install_package mss'.")
        return

    sys.stdout.flush()
    target_dir = await asyncio.to_thread(lambda: Path(output_dir or ".").resolve())
    await asyncio.to_thread(target_dir.mkdir, parents=True, exist_ok=True)
    interval_s = max(0.1, float(interval_s))
    frames_attempted = 0

    # mss.mss() is a context manager, we wrap the whole loop to reuse the connection
    def _capture_and_save(sct, monitor):
        # 1. Capture Screen
        screenshot = sct.grab(monitor)
        # 2. Convert to PNG bytes
        png = mss.tools.to_png(screenshot.rgb, screenshot.size)
        # 3. Base64 Encode
        b64_data = base64.b64encode(png).decode('utf-8')
        # 4. Write to Shared Memory
        memory = {
            "timestamp": datetime.now().isoformat(),
            "type": "visual",
            "status": "active",
            "image_data": b64_data,
            "description": "Screen capture active. Vision analysis (LLaVA/Ollama) is initialized and waiting for integration."
        }
        tmp_path = target_dir / "vision_memory.tmp"
        final_path = target_dir / "sensory_vision.json"
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(memory, f)
        os.replace(tmp_path, final_path)
        return True

    try:
        with mss.mss() as sct:
            monitor = sct.monitors[monitor_index] # Primary monitor
            while stop_event is None or not stop_event.is_set():
                try:
                    await asyncio.to_thread(_capture_and_save, sct, monitor)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Frame captured.")
                    sys.stdout.flush()
                except _VISION_CAPTURE_ERRORS as e:
                    record_degradation("vision_service.capture", e)
                    print(f"Vision Error: {e}")
                    sys.stdout.flush()

                frames_attempted += 1
                if max_frames is not None and frames_attempted >= max_frames:
                    break

                await asyncio.sleep(interval_s) # Non-blocking sleep
    except _VISION_SERVICE_ERRORS as e:
        record_degradation("vision_service", e)
        print(f"Fatal Vision Error: {e}")

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(run_vision_loop())
    except KeyboardInterrupt:
        print("Vision Service Stopping.")
