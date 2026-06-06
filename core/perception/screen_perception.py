"""core/perception/screen_perception.py — Visual Desktop Perception
=====================================================================
Gives Aura eyes good enough to recover when the UI changes.

Can answer: "What app is open? Did the file appear? Did the Google Doc
load? Is the note written? Did the wallpaper change?"

Uses screencapture + OCR (pytesseract or macOS Vision) for text,
and AppleScript for structured queries (app name, window title).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.ScreenPerception")


@dataclass
class ScreenSnapshot:
    """A structured snapshot of the current screen state."""
    active_app: str = ""
    window_title: str = ""
    screen_text: str = ""           # OCR text
    text_hash: str = ""             # for change detection
    screenshot_path: str = ""       # saved screenshot file
    has_modal: bool = False         # dialog/alert detected
    modal_text: str = ""
    has_loading: bool = False       # spinner/progress bar detected
    timestamp: float = field(default_factory=time.time)


class ScreenPerception:
    """Visual perception of the desktop.

    Usage:
        perc = get_screen_perception()
        snap = await perc.capture()
        print(snap.active_app, snap.window_title)

        found = await perc.find_text_on_screen("Save")
        changed = await perc.detect_change(previous_hash)
    """

    def __init__(self) -> None:
        self._last_hash: str = ""
        self._capture_count: int = 0
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("screen_perception", self, required=False)
        self._started = True
        logger.info("ScreenPerception ONLINE")

    async def capture(self, save_screenshot: bool = False) -> ScreenSnapshot:
        """Capture a full snapshot of the current screen state."""
        snap = ScreenSnapshot()
        self._capture_count += 1

        # Get active app and window title (fast, via AppleScript)
        try:
            proc = await get_subprocess_gateway().spawn_async(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to get name of first application process whose frontmost is true',
                ],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=True,
                source="screen_perception.active_app",
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            if proc.returncode == 0 and stdout:
                snap.active_app = stdout.decode("utf-8", errors="replace").strip()
        except (OSError, RuntimeError, asyncio.TimeoutError) as exc:
            record_degradation("screen_perception.active_app", exc)

        if snap.active_app:
            try:
                app_name = snap.active_app.replace("\\", "\\\\").replace('"', '\\"')
                proc = await get_subprocess_gateway().spawn_async(
                    [
                        "osascript",
                        "-e",
                        f'tell application "System Events" to get name of front window of process "{app_name}"',
                    ],
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    read_only=True,
                    source="screen_perception.window_title",
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
                if proc.returncode == 0 and stdout:
                    snap.window_title = stdout.decode("utf-8", errors="replace").strip()[:200]
            except (OSError, RuntimeError, asyncio.TimeoutError) as exc:
                record_degradation("screen_perception.window_title", exc)

        # Take screenshot
        if save_screenshot:
            snap.screenshot_path = await self._take_screenshot()

        # OCR (if screenshot was taken or we need text)
        if snap.screenshot_path:
            snap.screen_text = await self._ocr_screenshot(snap.screenshot_path)
        if snap.screen_text:
            snap.text_hash = hashlib.sha256(snap.screen_text.encode()).hexdigest()[:16]

        # Modal/loading detection from window title heuristics
        title_lower = snap.window_title.lower()
        snap.has_modal = any(
            kw in title_lower for kw in ("alert", "error", "warning", "permission", "allow")
        )
        snap.has_loading = any(
            kw in title_lower for kw in ("loading", "saving", "progress", "processing")
        )

        self._last_hash = snap.text_hash
        return snap

    async def get_active_window(self) -> Dict[str, str]:
        """Get just the active window info (fast, no screenshot)."""
        result = {"app": "", "title": "", "bounds": ""}
        try:
            proc = await get_subprocess_gateway().spawn_async(
                [
                    "osascript",
                    "-e",
                    '''tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    set winTitle to name of front window of process frontApp
                    return frontApp & "|" & winTitle
                end tell''',
                ],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=True,
                source="screen_perception.active_window",
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            if proc.returncode == 0 and stdout:
                parts = stdout.decode("utf-8", errors="replace").strip().split("|", 1)
                result["app"] = parts[0] if parts else ""
                result["title"] = parts[1] if len(parts) > 1 else ""
        except (OSError, RuntimeError, asyncio.TimeoutError) as exc:
            record_degradation("screen_perception.active_window", exc)
        return result

    async def find_text_on_screen(self, target: str) -> Dict[str, Any]:
        """Check if specific text appears on screen (via OCR)."""
        screenshot = await self._take_screenshot()
        if not screenshot:
            return {"found": False, "error": "Screenshot failed"}

        text = await self._ocr_screenshot(screenshot)
        found = target.lower() in text.lower() if target and text else False
        return {
            "found": found,
            "text_length": len(text),
            "screenshot": screenshot,
        }

    async def detect_change(self, previous_hash: str = "") -> Dict[str, Any]:
        """Detect if the screen content changed since a previous capture."""
        if not previous_hash:
            previous_hash = self._last_hash

        screenshot = await self._take_screenshot()
        text = await self._ocr_screenshot(screenshot) if screenshot else ""
        current_hash = hashlib.sha256(text.encode()).hexdigest()[:16] if text else ""

        changed = current_hash != previous_hash if previous_hash and current_hash else True
        self._last_hash = current_hash

        return {
            "changed": changed,
            "previous_hash": previous_hash,
            "current_hash": current_hash,
        }

    async def compare_screenshots(
        self, before_path: str, after_path: str
    ) -> Dict[str, Any]:
        """Compare two screenshots for differences."""
        before = Path(before_path)
        after = Path(after_path)

        if not before.exists() or not after.exists():
            return {"error": "Screenshot not found", "change_magnitude": 1.0}

        # Simple byte comparison
        before_hash = hashlib.sha256(before.read_bytes()).hexdigest()
        after_hash = hashlib.sha256(after.read_bytes()).hexdigest()

        if before_hash == after_hash:
            return {"change_magnitude": 0.0, "identical": True}

        # Size-based heuristic for change magnitude
        before_size = before.stat().st_size
        after_size = after.stat().st_size
        size_diff = abs(before_size - after_size)
        magnitude = min(1.0, size_diff / max(1, max(before_size, after_size)))

        return {
            "change_magnitude": magnitude,
            "identical": False,
            "before_size": before_size,
            "after_size": after_size,
        }

    async def _take_screenshot(self) -> str:
        """Take a screenshot and return the file path."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_dir = Path.home() / ".aura" / "data" / "screenshots"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(save_dir / f"screen_{ts}_{self._capture_count}.png")

        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["screencapture", "-x", save_path],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="screen_perception.take_screenshot",
            )
            await asyncio.wait_for(proc.communicate(), timeout=5.0)
            if proc.returncode == 0 and Path(save_path).exists():
                return save_path
        except (OSError, RuntimeError, asyncio.TimeoutError) as exc:
            record_degradation("screen_perception.take_screenshot", exc)
        return ""

    async def _ocr_screenshot(self, screenshot_path: str) -> str:
        """Extract text from a screenshot via OCR."""
        if not screenshot_path or not Path(screenshot_path).exists():
            return ""

        # Try pytesseract
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(screenshot_path)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except ImportError:
            return "[OCR not available — install pytesseract for screen text extraction]"
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("screen_perception.ocr", e)
            logger.debug("pytesseract OCR failed: %s", e)

        # Fallback: try macOS shortcuts or textutil
        # (limited but better than nothing)
        return "[OCR not available — install pytesseract for screen text extraction]"

    def get_status(self) -> Dict[str, Any]:
        return {
            "captures": self._capture_count,
            "last_hash": self._last_hash,
        }


_instance: Optional[ScreenPerception] = None


def get_screen_perception() -> ScreenPerception:
    global _instance
    if _instance is None:
        _instance = ScreenPerception()
    return _instance


__all__ = ["ScreenPerception", "ScreenSnapshot", "get_screen_perception"]
