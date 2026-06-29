"""Bounded ambient developer-environment sensory stream.

This service gives Aura a continuous, low-cost stream of local runtime context:
repository changes, active watched directories, and recent log warnings. It is
not a prompt wrapper. It publishes compact frames into WorldState and the
TimescaleBridge so foreground cognition can be grounded in verified background
evidence without inventing idle-time events.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.container import ServiceContainer
from core.runtime.background_policy import (
    background_loop_start_reason,
    constitutive_compute_budget,
)
from core.runtime.errors import record_degradation
from core.runtime.task_ownership import create_tracked_task

logger = logging.getLogger("Aura.AmbientDeveloperStream")

_RUNTIME_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    OSError,
    ImportError,
    TimeoutError,
    asyncio.TimeoutError,
)
_DEFAULT_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "htmlcov",
    "llm_data",
    "models",
    "node_modules",
    "venv",
}
_CODE_SUFFIXES = {
    ".cfg",
    ".css",
    ".env",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
_LOG_PATTERN = re.compile(
    r"\b(ERROR|CRITICAL|Traceback|Exception|DEGRADATION|memory_pressure|OOM|unhealthy|blocked)\b",
    re.IGNORECASE,
)


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, value))


def _bounded_text(value: Any, limit: int = 240) -> str:
    text = str(value or "").replace("\x00", "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].strip() or text[:limit]


def _project_root() -> Path:
    try:
        from core.config import config

        return config.paths.project_root.resolve()
    except _RUNTIME_ERRORS:
        return Path(__file__).resolve().parents[2]


def _log_roots(project_root: Path) -> tuple[Path, ...]:
    roots = [project_root / "logs"]
    try:
        from core.config import config

        roots.append(config.paths.log_dir)
    except _RUNTIME_ERRORS:
        pass
    return tuple(dict.fromkeys(path.resolve() for path in roots))


def _default_watch_roots(project_root: Path) -> tuple[Path, ...]:
    configured = os.environ.get("AURA_AMBIENT_WATCH_DIRS", "").strip()
    if configured:
        raw = [part.strip() for part in configured.split(os.pathsep) if part.strip()]
        return tuple((Path(part).expanduser() if Path(part).is_absolute() else project_root / part).resolve() for part in raw)
    names = ("core", "interface", "tools", "training", "tests", "config", "docs")
    return tuple((project_root / name).resolve() for name in names if (project_root / name).exists())


@dataclass(frozen=True)
class AmbientFileEvent:
    path: str
    kind: str
    mtime: float
    size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AmbientLogEvent:
    path: str
    line: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AmbientDeveloperFrame:
    frame_id: int
    timestamp: float = field(default_factory=time.time)
    repo_root: str = ""
    git_dirty_count: int = 0
    git_status: tuple[str, ...] = ()
    recent_files: tuple[AmbientFileEvent, ...] = ()
    log_events: tuple[AmbientLogEvent, ...] = ()
    repair_candidates: tuple[str, ...] = ()
    throttled_reason: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["git_status"] = list(self.git_status)
        data["recent_files"] = [event.to_dict() for event in self.recent_files]
        data["log_events"] = [event.to_dict() for event in self.log_events]
        data["repair_candidates"] = list(self.repair_candidates)
        return data

    @property
    def event_count(self) -> int:
        return self.git_dirty_count + len(self.recent_files) + len(self.log_events)


class AmbientDeveloperStream:
    """Continuously sample local developer/runtime context with hard bounds."""

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        watch_roots: tuple[Path, ...] | None = None,
        log_roots: tuple[Path, ...] | None = None,
        sample_interval_s: float | None = None,
        max_scan_files: int | None = None,
        recent_window_s: float | None = None,
    ) -> None:
        self.project_root = (project_root or _project_root()).resolve()
        self.watch_roots = watch_roots or _default_watch_roots(self.project_root)
        self.log_roots = log_roots or _log_roots(self.project_root)
        self.sample_interval_s = (
            sample_interval_s
            if sample_interval_s is not None
            else _env_float("AURA_AMBIENT_STREAM_INTERVAL_S", 30.0, minimum=5.0, maximum=900.0)
        )
        self.max_scan_files = max_scan_files or _env_int(
            "AURA_AMBIENT_STREAM_MAX_SCAN_FILES",
            3500,
            minimum=100,
            maximum=20000,
        )
        self.recent_window_s = (
            recent_window_s
            if recent_window_s is not None
            else _env_float("AURA_AMBIENT_STREAM_RECENT_WINDOW_S", 180.0, minimum=15.0, maximum=3600.0)
        )
        self.running = False
        self._task: asyncio.Task | None = None
        self._frame_id = 0
        self._errors = 0
        self._started_at = 0.0
        self._latest_frame: AmbientDeveloperFrame | None = None
        self._frames: deque[AmbientDeveloperFrame] = deque(maxlen=120)

    async def start(self) -> None:
        if self.running:
            return
        reason = background_loop_start_reason("ambient_developer_stream")
        if reason:
            ServiceContainer.register_instance("ambient_developer_stream", self, required=False)
            logger.info("AmbientDeveloperStream not started: %s", reason)
            return
        self.running = True
        self._started_at = time.time()
        ServiceContainer.register_instance("ambient_developer_stream", self, required=False)
        self._task = create_tracked_task(
            self._run_loop(),
            name="Aura.AmbientDeveloperStream",
        )
        logger.info("AmbientDeveloperStream ONLINE — %ss interval", self.sample_interval_s)

    async def stop(self) -> None:
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @property
    def latest_frame(self) -> AmbientDeveloperFrame | None:
        return self._latest_frame

    async def _run_loop(self) -> None:
        while self.running:
            try:
                budget = constitutive_compute_budget(
                    "ambient_developer_stream",
                    base_hz=0.1,
                    foreground_hz=0.1,
                    memory_high_hz=0.1,
                    memory_critical_hz=0.1,
                )
                await self.sample_once(
                    throttled_reason=budget.reason if self._budget_is_throttled(budget) else ""
                )
                await asyncio.sleep(max(self.sample_interval_s, budget.interval_s))
            except asyncio.CancelledError:
                raise
            except _RUNTIME_ERRORS as exc:
                self._errors += 1
                record_degradation("ambient_developer_stream", exc)
                logger.debug("AmbientDeveloperStream tick failed: %s", exc)
                await asyncio.sleep(self.sample_interval_s)

    @staticmethod
    def _budget_is_throttled(budget: Any) -> bool:
        reason = str(getattr(budget, "reason", "") or "")
        return reason not in {"", "nominal", "component_override", "global_override"}

    async def sample_once(self, *, throttled_reason: str = "") -> AmbientDeveloperFrame:
        frame = await asyncio.to_thread(self._collect_frame, throttled_reason=throttled_reason)
        self._publish_frame(frame)
        return frame

    def _collect_frame(self, *, throttled_reason: str = "") -> AmbientDeveloperFrame:
        self._frame_id += 1
        if throttled_reason.startswith("memory") or throttled_reason == "foreground_generation_active":
            frame = AmbientDeveloperFrame(
                frame_id=self._frame_id,
                repo_root=str(self.project_root),
                throttled_reason=throttled_reason,
                summary=f"ambient stream throttled: {throttled_reason}",
            )
            return frame

        git_status = self._collect_git_status()
        recent_files = self._collect_recent_files()
        log_events = self._collect_log_events()
        repair_candidates = self._build_repair_candidates(git_status, recent_files, log_events)
        summary = self._summarize(git_status, recent_files, log_events, repair_candidates)
        return AmbientDeveloperFrame(
            frame_id=self._frame_id,
            repo_root=str(self.project_root),
            git_dirty_count=len(git_status),
            git_status=tuple(git_status),
            recent_files=tuple(recent_files),
            log_events=tuple(log_events),
            repair_candidates=tuple(repair_candidates),
            summary=summary,
        )

    def _collect_git_status(self) -> list[str]:
        try:
            from core.runtime.subprocess_gateway import get_subprocess_gateway

            result = get_subprocess_gateway().run(
                ["git", "status", "--porcelain=v1", "-uno"],
                cwd=str(self.project_root),
                capture_output=True,
                timeout=2.0,
                read_only=True,
                source="ambient_developer_stream.git_status",
            )
            if result.returncode != 0:
                return []
            lines = [
                _bounded_text(line, 180)
                for line in str(result.stdout or "").splitlines()
                if line.strip()
            ]
            return lines[:40]
        except _RUNTIME_ERRORS as exc:
            record_degradation("ambient_developer_stream.git_status", exc)
            return []

    def _collect_recent_files(self) -> list[AmbientFileEvent]:
        cutoff = time.time() - self.recent_window_s
        events: list[AmbientFileEvent] = []
        scanned = 0
        stack = [root for root in self.watch_roots if root.exists()]
        while stack and scanned < self.max_scan_files:
            current = stack.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        if scanned >= self.max_scan_files:
                            break
                        name = entry.name
                        if name in _DEFAULT_SKIP_DIRS or name.startswith(".#"):
                            continue
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                                continue
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            scanned += 1
                            path = Path(entry.path)
                            if path.suffix.lower() not in _CODE_SUFFIXES:
                                continue
                            stat = entry.stat(follow_symlinks=False)
                            if stat.st_mtime < cutoff:
                                continue
                            events.append(
                                AmbientFileEvent(
                                    path=self._relative(path),
                                    kind="modified",
                                    mtime=round(float(stat.st_mtime), 3),
                                    size=int(stat.st_size),
                                )
                            )
                        except _RUNTIME_ERRORS:
                            continue
            except _RUNTIME_ERRORS:
                continue
        events.sort(key=lambda event: event.mtime, reverse=True)
        return events[:25]

    def _collect_log_events(self) -> list[AmbientLogEvent]:
        candidates: list[Path] = []
        for root in self.log_roots:
            if not root.exists():
                continue
            try:
                files = [path for path in root.glob("*.log") if path.is_file()]
            except _RUNTIME_ERRORS:
                continue
            candidates.extend(files)
        candidates.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True)
        events: list[AmbientLogEvent] = []
        for path in candidates[:4]:
            try:
                with path.open("rb") as handle:
                    handle.seek(0, os.SEEK_END)
                    size = handle.tell()
                    handle.seek(max(0, size - 16000))
                    text = handle.read().decode("utf-8", errors="replace")
                for line in text.splitlines()[-120:]:
                    if _LOG_PATTERN.search(line):
                        events.append(
                            AmbientLogEvent(
                                path=self._relative(path),
                                line=_bounded_text(line, 260),
                            )
                        )
                        if len(events) >= 12:
                            return events
            except _RUNTIME_ERRORS:
                continue
        return events

    def _build_repair_candidates(
        self,
        git_status: list[str],
        recent_files: list[AmbientFileEvent],
        log_events: list[AmbientLogEvent],
    ) -> list[str]:
        candidates: list[str] = []
        if log_events:
            candidates.append("review_recent_log_errors")
        if any("DEGRADATION" in event.line.upper() for event in log_events):
            candidates.append("triage_degradation_events")
        if any("MEMORY_PRESSURE" in event.line.upper() or "OOM" in event.line.upper() for event in log_events):
            candidates.append("check_memory_pressure_guard")
        if git_status or recent_files:
            candidates.append("run_targeted_tests_for_recent_changes")
        return candidates[:6]

    def _summarize(
        self,
        git_status: list[str],
        recent_files: list[AmbientFileEvent],
        log_events: list[AmbientLogEvent],
        repair_candidates: list[str],
    ) -> str:
        parts = []
        if git_status:
            parts.append(f"{len(git_status)} tracked repo change(s)")
        if recent_files:
            parts.append(f"{len(recent_files)} recent watched file event(s)")
        if log_events:
            parts.append(f"{len(log_events)} recent warning/error log line(s)")
        if repair_candidates:
            parts.append("repair candidates: " + ", ".join(repair_candidates[:3]))
        return "; ".join(parts) if parts else "ambient developer stream observed no material changes"

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_root))
        except _RUNTIME_ERRORS:
            return str(path)

    def _publish_frame(self, frame: AmbientDeveloperFrame) -> None:
        self._latest_frame = frame
        self._frames.append(frame)
        try:
            ws = ServiceContainer.get("world_state", default=None)
            if ws is not None and hasattr(ws, "record_event") and frame.event_count:
                ws.record_event(
                    f"Ambient developer stream: {frame.summary}",
                    source="ambient_developer_stream",
                    salience=0.45 if frame.log_events else 0.25,
                    ttl=300,
                )
        except _RUNTIME_ERRORS as exc:
            record_degradation("ambient_developer_stream.world_state", exc)
        try:
            from core.runtime.timescale_bridge import get_timescale_bridge

            get_timescale_bridge().ingest_ambient_developer_frame(frame)
        except _RUNTIME_ERRORS as exc:
            record_degradation("ambient_developer_stream.timescale_bridge", exc)

    def get_status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "schema": "aura.ambient_developer_stream.status.v1",
            "sample_interval_s": self.sample_interval_s,
            "frames": len(self._frames),
            "errors": self._errors,
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0.0,
            "latest_frame": self._latest_frame.to_dict() if self._latest_frame else None,
            "watch_roots": [str(path) for path in self.watch_roots],
        }

    status = get_status


_AMBIENT_STREAM: AmbientDeveloperStream | None = None


def get_ambient_developer_stream() -> AmbientDeveloperStream:
    global _AMBIENT_STREAM
    existing = ServiceContainer.get("ambient_developer_stream", default=None)
    if isinstance(existing, AmbientDeveloperStream):
        _AMBIENT_STREAM = existing
        return existing
    if _AMBIENT_STREAM is None:
        _AMBIENT_STREAM = AmbientDeveloperStream()
    ServiceContainer.register_instance("ambient_developer_stream", _AMBIENT_STREAM, required=False)
    return _AMBIENT_STREAM


def render_ambient_developer_prompt_block(frame: dict[str, Any] | AmbientDeveloperFrame | None) -> str:
    if frame is None:
        return ""
    data = frame.to_dict() if isinstance(frame, AmbientDeveloperFrame) else dict(frame or {})
    summary = str(data.get("summary") or "").strip()
    if not summary:
        return ""
    lines = ["## AMBIENT DEVELOPER STREAM"]
    lines.append(f"- Summary: {summary}")
    candidates = data.get("repair_candidates") if isinstance(data.get("repair_candidates"), list) else []
    if candidates:
        lines.append("- Repair candidates: " + ", ".join(str(item) for item in candidates[:4]))
    lines.append("Use as verified background evidence; do not invent file/log events beyond this frame.")
    return "\n".join(lines)


__all__ = [
    "AmbientDeveloperFrame",
    "AmbientDeveloperStream",
    "AmbientFileEvent",
    "AmbientLogEvent",
    "get_ambient_developer_stream",
    "render_ambient_developer_prompt_block",
]
