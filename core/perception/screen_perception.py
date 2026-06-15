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
from typing import Any

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

    @staticmethod
    def _prepare_screenshot_path(capture_count: int) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_dir = Path.home() / ".aura" / "data" / "screenshots"
        save_dir.mkdir(parents=True, exist_ok=True)
        return str(save_dir / f"screen_{ts}_{capture_count}.png")

    @staticmethod
    def _path_exists(path: str) -> bool:
        return Path(path).exists()

    @staticmethod
    def _compare_screenshot_files(before_path: str, after_path: str) -> dict[str, Any]:
        before = Path(before_path)
        after = Path(after_path)

        if not before.exists() or not after.exists():
            return {"error": "Screenshot not found", "change_magnitude": 1.0}

        before_bytes = before.read_bytes()
        after_bytes = after.read_bytes()
        before_hash = hashlib.sha256(before_bytes).hexdigest()
        after_hash = hashlib.sha256(after_bytes).hexdigest()

        if before_hash == after_hash:
            return {"change_magnitude": 0.0, "identical": True}

        before_size = len(before_bytes)
        after_size = len(after_bytes)
        size_diff = abs(before_size - after_size)
        magnitude = min(1.0, size_diff / max(1, max(before_size, after_size)))

        return {
            "change_magnitude": magnitude,
            "identical": False,
            "before_size": before_size,
            "after_size": after_size,
        }

    @staticmethod
    def _ocr_screenshot_sync(screenshot_path: str) -> str:
        if not screenshot_path or not Path(screenshot_path).exists():
            return ""

        try:
            import pytesseract
            from PIL import Image

            img = Image.open(screenshot_path)
            text = pytesseract.image_to_string(img)
            return text.strip()
        except ImportError:
            return "[OCR not available — install pytesseract for screen text extraction]"

    async def _run_osascript(
        self,
        script: str,
        *,
        source: str,
        timeout_s: float = 3.0,
    ) -> str:
        """Run a bounded, read-only AppleScript probe and always reap it."""

        proc = None
        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["osascript", "-e", script],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=True,
                source=source,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            if proc.returncode == 0 and stdout:
                return stdout.decode("utf-8", errors="replace").strip()
            return ""
        except TimeoutError as exc:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except (TimeoutError, OSError, RuntimeError) as kill_exc:
                    reaper_registered = False
                    try:
                        from core.reaper import register_reaper_pid

                        pid = int(getattr(proc, "pid", 0) or 0)
                        if pid > 0:
                            register_reaper_pid(pid)
                            reaper_registered = True
                    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as reaper_exc:
                        logger.error(
                            "Failed to register timed-out AppleScript child for reaping: %s",
                            reaper_exc,
                        )
                    reap_action = (
                        "registered the timed-out child PID with Aura's process reaper "
                        "for supervised cleanup"
                        if reaper_registered
                        else "reported the unreaped child explicitly after kill and reaper registration failed"
                    )
                    record_degradation(
                        f"{source}.reap",
                        kill_exc,
                        severity="degraded",
                        action=reap_action,
                    )
            record_degradation(source, exc)
            return ""
        except (OSError, RuntimeError) as exc:
            record_degradation(source, exc)
            return ""

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
        snap.active_app = await self._run_osascript(
            'tell application "System Events" to get name of first application process whose frontmost is true',
            source="screen_perception.active_app",
        )

        if snap.active_app:
            app_name = snap.active_app.replace("\\", "\\\\").replace('"', '\\"')
            snap.window_title = (
                await self._run_osascript(
                    f'tell application "System Events" to get name of front window of process "{app_name}"',
                    source="screen_perception.window_title",
                )
            )[:200]

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

    async def get_active_window(self) -> dict[str, str]:
        """Get just the active window info (fast, no screenshot)."""
        result = {"app": "", "title": "", "bounds": ""}
        output = await self._run_osascript(
            '''tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    set winTitle to name of front window of process frontApp
                    return frontApp & "|" & winTitle
                end tell''',
            source="screen_perception.active_window",
        )
        if output:
            parts = output.split("|", 1)
            result["app"] = parts[0] if parts else ""
            result["title"] = parts[1] if len(parts) > 1 else ""
        return result

    async def find_text_on_screen(self, target: str) -> dict[str, Any]:
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

    async def detect_change(self, previous_hash: str = "") -> dict[str, Any]:
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
    ) -> dict[str, Any]:
        """Compare two screenshots for differences."""
        return await asyncio.to_thread(
            self._compare_screenshot_files,
            before_path,
            after_path,
        )

    async def _take_screenshot(self) -> str:
        """Take a screenshot and return the file path."""
        save_path = await asyncio.to_thread(
            self._prepare_screenshot_path,
            self._capture_count,
        )

        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["screencapture", "-x", save_path],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                source="screen_perception.take_screenshot",
            )
            await asyncio.wait_for(proc.communicate(), timeout=5.0)
            if proc.returncode == 0 and await asyncio.to_thread(self._path_exists, save_path):
                return save_path
        except (TimeoutError, OSError, RuntimeError) as exc:
            record_degradation("screen_perception.take_screenshot", exc)
        return ""

    async def _ocr_screenshot(self, screenshot_path: str) -> str:
        """Extract text from a screenshot via OCR."""
        try:
            return await asyncio.to_thread(self._ocr_screenshot_sync, screenshot_path)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as e:
            record_degradation("screen_perception.ocr", e)
            logger.debug("pytesseract OCR failed: %s", e)

        # Fallback: try macOS shortcuts or textutil
        # (limited but better than nothing)
        return "[OCR not available — install pytesseract for screen text extraction]"

    def get_status(self) -> dict[str, Any]:
        return {
            "captures": self._capture_count,
            "last_hash": self._last_hash,
        }


_instance: ScreenPerception | None = None


def get_screen_perception() -> ScreenPerception:
    global _instance
    if _instance is None:
        _instance = ScreenPerception()
    return _instance


__all__ = ["ScreenPerception", "ScreenSnapshot", "get_screen_perception"]
