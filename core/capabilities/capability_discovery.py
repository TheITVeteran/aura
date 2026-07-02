"""core/capabilities/capability_discovery.py — Machine Capability Scan
========================================================================
Discovers what this machine can do BEFORE attempting actions.

Produces a CapabilityReport that the TaskDecomposer uses to plan
realistic task graphs. Before attempting any workflow, Aura can say:
"I have access to microphone, screen, Chrome, Finder, file writes,
 and wallpaper settings. Google Docs may require login."
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.subprocess_gateway import get_subprocess_gateway
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.CapabilityDiscovery")


@dataclass
class CapabilityReport:
    """Structured report of machine capabilities."""
    # Apps
    installed_apps: list[str] = field(default_factory=list)
    has_browser: bool = False
    preferred_browser: str = ""
    has_text_editor: bool = False
    has_terminal: bool = True

    # Permissions
    has_accessibility: bool = False
    has_screen_recording: bool = False
    has_microphone: bool = False
    has_camera: bool = False
    has_full_disk_access: bool = False

    # System
    has_network: bool = False
    has_python_packages: dict[str, bool] = field(default_factory=dict)
    writable_directories: list[str] = field(default_factory=list)
    available_models: list[str] = field(default_factory=list)

    # Tools
    has_screencapture: bool = True
    has_osascript: bool = True
    has_pbcopy: bool = True
    has_say: bool = True

    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        """Human-readable capability summary."""
        parts = ["I have access to:"]
        if self.has_microphone:
            parts.append("microphone")
        if self.has_screen_recording:
            parts.append("screen recording")
        if self.has_accessibility:
            parts.append("accessibility")
        if self.has_browser:
            parts.append(f"{self.preferred_browser} browser")
        if self.has_text_editor:
            parts.append("text editor")
        if self.writable_directories:
            parts.append("file writes")
        if self.has_network:
            parts.append("network")
        capabilities = ", ".join(parts[1:]) if len(parts) > 1 else "limited capabilities"
        result = f"I have access to: {capabilities}."

        # Warnings
        if not self.has_accessibility:
            result += " Accessibility permission not detected — UI automation may be limited."
        if not self.has_network:
            result += " Network appears unavailable."
        return result


class CapabilityDiscovery:
    """Discovers what this machine can do."""

    def __init__(self) -> None:
        self._report: CapabilityReport | None = None
        self._started = False
        self._scan_task: asyncio.Task[Any] | None = None

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("capability_discovery", self, required=False)
        self._started = True
        self._report = self._report or CapabilityReport()
        try:
            self._scan_task = get_task_tracker().create_task(
                self._run_initial_scan(),
                name="capability_discovery.initial_scan",
            )
        except (RuntimeError, AttributeError, TypeError):
            self._scan_task = asyncio.create_task(
                self._run_initial_scan(),
                name="capability_discovery.initial_scan",
            )
        logger.info("CapabilityDiscovery ONLINE — initial scan scheduled")

    async def _run_initial_scan(self) -> None:
        try:
            report = await self.discover()
        except (ImportError, AttributeError, RuntimeError, OSError, TimeoutError) as exc:
            record_degradation("capability_discovery.initial_scan", exc)
            logger.warning("CapabilityDiscovery initial scan failed: %s", exc)
            return
        self._report = report
        logger.info("CapabilityDiscovery scan complete — %s", report.summary()[:120])

    async def discover(self) -> CapabilityReport:
        """Run full capability scan."""
        report = CapabilityReport()

        # Run all checks concurrently
        await asyncio.gather(
            self._discover_apps(report),
            self._discover_permissions(report),
            self._discover_network(report),
            self._discover_tools(report),
            self._discover_python_packages(report),
            self._discover_writable_dirs(report),
            self._discover_models(report),
            return_exceptions=True,
        )

        self._report = report
        return report

    async def _discover_apps(self, report: CapabilityReport) -> None:
        """Discover installed applications."""
        try:
            registry = ServiceContainer.get("app_registry", default=None)
            if registry:
                apps = registry.all_apps()
                report.installed_apps = [a.name for a in apps]
                pref_browser = registry.get_preferred_browser()
                if pref_browser:
                    report.has_browser = True
                    report.preferred_browser = pref_browser.name
                pref_editor = registry.get_preferred_text_editor()
                if pref_editor:
                    report.has_text_editor = True
                return
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("capability_discovery.app_registry", exc)

        # Fallback: scan /Applications directly (directory listing is
        # blocking I/O — keep it off the event loop).
        try:
            apps = await asyncio.to_thread(self._scan_applications_dir_sync)
            report.installed_apps = apps
            for browser in ["Google Chrome", "Safari", "Firefox"]:
                if browser in apps:
                    report.has_browser = True
                    report.preferred_browser = browser
                    break
            for editor in ["TextEdit", "Notes", "Visual Studio Code"]:
                if editor in apps:
                    report.has_text_editor = True
                    break
        except (OSError, PermissionError) as exc:
            record_degradation("capability_discovery.app_scan", exc)

    def _scan_applications_dir_sync(self) -> list[str]:
        app_dir = Path("/Applications")
        if not app_dir.exists():
            return []
        return sorted(e.stem for e in app_dir.iterdir() if e.suffix == ".app")

    async def _discover_permissions(self, report: CapabilityReport) -> None:
        """Check system permissions."""
        # Accessibility
        try:
            proc = await get_subprocess_gateway().spawn_async(
                [
                    "osascript", "-e",
                    'tell application "System Events" to get name of first application process whose frontmost is true',
                ],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=True,
                source="capability_discovery.accessibility_probe",
            )
            await asyncio.wait_for(proc.communicate(), timeout=5.0)
            report.has_accessibility = proc.returncode == 0
        except (OSError, TimeoutError, RuntimeError) as exc:
            record_degradation("capability_discovery.accessibility_probe", exc)
            report.has_accessibility = False

        # Screen recording
        try:
            from core.security.permission_guard import PermissionType, get_permission_guard

            guard = get_permission_guard()
            res = await guard.check_permission(PermissionType.SCREEN)
            report.has_screen_recording = res.get("granted", False)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("capability_discovery.screen_recording_probe", exc)
            report.has_screen_recording = False

        # Microphone — check TCC database heuristic
        tcc_db = Path.home() / "Library" / "Application Support" / "com.apple.TCC" / "TCC.db"
        report.has_microphone = tcc_db.exists()  # Can't read it directly, but presence suggests TCC active

        # Camera — similar heuristic
        report.has_camera = report.has_microphone  # Same TCC mechanism

    async def _discover_network(self, report: CapabilityReport) -> None:
        """Check network connectivity."""
        try:
            proc = await get_subprocess_gateway().spawn_async(
                ["ping", "-c", "1", "-W", "2", "8.8.8.8"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                read_only=True,
                source="capability_discovery.network_probe",
            )
            await asyncio.wait_for(proc.communicate(), timeout=5.0)
            report.has_network = proc.returncode == 0
        except (OSError, TimeoutError, RuntimeError) as exc:
            record_degradation("capability_discovery.network_probe", exc)
            report.has_network = False

    async def _discover_tools(self, report: CapabilityReport) -> None:
        """Check for required CLI tools (PATH stats are blocking I/O)."""
        tools = {
            "screencapture": "has_screencapture",
            "osascript": "has_osascript",
            "pbcopy": "has_pbcopy",
            "say": "has_say",
        }
        found = await asyncio.to_thread(
            lambda: {tool: shutil.which(tool) is not None for tool in tools}
        )
        for tool, attr in tools.items():
            setattr(report, attr, found.get(tool, False))

    async def _discover_python_packages(self, report: CapabilityReport) -> None:
        """Check for useful Python packages.

        find_spec answers "is it installed?" without executing module init
        code — importing numpy/PIL on the event loop at boot costs seconds
        of loop stall and loads megabytes nothing asked for yet.
        """
        packages = [
            "fpdf", "reportlab", "pytesseract", "PIL", "pyautogui",
            "psutil", "httpx", "numpy",
        ]

        def _availability() -> dict[str, bool]:
            import importlib.util

            status: dict[str, bool] = {}
            for pkg in packages:
                try:
                    status[pkg] = importlib.util.find_spec(pkg) is not None
                except (ImportError, ValueError, AttributeError):
                    status[pkg] = False
            return status

        report.has_python_packages = await asyncio.to_thread(_availability)

    @staticmethod
    def _probe_writable_dir(d: Path) -> None:
        """Blocking write probe — must run OFF the event loop.

        Every one of the 12 recorded live loop-wedge crashes (20-minute
        event-loop freezes ending in liveness-sentinel SIGKILL) had this
        probe's mkdir/write/unlink syscalls on the loop while the disk was
        thrashing. Probe files are worthless after a crash, so the write is
        atomic but non-durable (no fsync).
        """
        d.mkdir(parents=True, exist_ok=True)
        test_file = d / ".aura_write_test"
        with local_internal_governed_scope(
            "capability_discovery.writable_dir_probe",
            receipt_prefix="capability-write-probe",
        ):
            get_file_write_gateway().write_text(
                test_file,
                "test",
                source="capability_discovery.writable_dir_probe",
                durable=False,
            )
        test_file.unlink()

    async def _discover_writable_dirs(self, report: CapabilityReport) -> None:
        """Check which directories are writable."""
        candidates = [
            Path.home() / "Documents" / "Aura",
            Path.home() / "Desktop" / "Aura",
            Path.home() / "Downloads",
            Path.home() / ".aura" / "data",
            Path(tempfile.gettempdir()),
        ]
        for d in candidates:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self._probe_writable_dir, d),
                    timeout=30.0,
                )
                report.writable_directories.append(str(d))
            except (OSError, PermissionError, RuntimeError, TimeoutError) as exc:
                record_degradation("capability_discovery.writable_dir_probe", exc)

    async def _discover_models(self, report: CapabilityReport) -> None:
        """Check for available LLM models."""
        try:
            router = ServiceContainer.get("llm_router", default=None)
            if router and hasattr(router, "list_models"):
                models = router.list_models()
                report.available_models = [str(m) for m in models[:10]]
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("capability_discovery.models", exc)

    def get_report(self) -> CapabilityReport:
        """Get the latest capability report."""
        if self._report is None:
            self._report = CapabilityReport()
        return self._report

    def get_status(self) -> dict[str, Any]:
        r = self._report or CapabilityReport()
        return {
            "discovered": self._report is not None,
            "apps": len(r.installed_apps),
            "browser": r.preferred_browser,
            "accessibility": r.has_accessibility,
            "screen_recording": r.has_screen_recording,
            "network": r.has_network,
            "writable_dirs": len(r.writable_directories),
        }


_instance: CapabilityDiscovery | None = None


def get_capability_discovery() -> CapabilityDiscovery:
    global _instance
    if _instance is None:
        _instance = CapabilityDiscovery()
    return _instance


__all__ = ["CapabilityDiscovery", "CapabilityReport", "get_capability_discovery"]
