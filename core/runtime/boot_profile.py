"""Boot-phase wall-clock profiler.

A live desktop boot has taken 13 minutes with nothing in the logs naming the
slow phase. This module gives boot a flight recorder: each phase's duration
is recorded as it completes, slow phases are called out in real time, and a
summary line plus a JSON artifact land when the runtime reaches ready.

Two APIs:
- ``mark(name)`` — attribute everything since the previous mark to ``name``.
  One-line insertions between existing boot steps; no re-indentation.
- ``phase(name)`` — context manager for isolated timed blocks.

Thread-safe; import-light; never raises into the boot path.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("Aura.BootProfile")

SLOW_PHASE_WARN_S = 20.0


class BootProfiler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phases: list[dict[str, Any]] = []
        self._started_monotonic = time.perf_counter()
        self._started_at = time.time()
        self._last_mark_monotonic = self._started_monotonic

    def _record(self, name: str, duration_s: float, offset_s: float) -> None:
        with self._lock:
            self._phases.append(
                {
                    "name": str(name),
                    "duration_s": round(max(0.0, duration_s), 3),
                    "offset_s": round(max(0.0, offset_s), 3),
                }
            )
        if duration_s >= SLOW_PHASE_WARN_S:
            logger.warning(
                "🐢 [BOOT] phase '%s' took %.1fs (offset +%.1fs)",
                name,
                duration_s,
                offset_s,
            )

    def mark(self, name: str) -> float:
        """Attribute the time since the previous mark to ``name``."""
        now = time.perf_counter()
        with self._lock:
            since = self._last_mark_monotonic
            self._last_mark_monotonic = now
        duration = now - since
        self._record(name, duration, since - self._started_monotonic)
        return duration

    @contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._record(name, time.perf_counter() - t0, t0 - self._started_monotonic)

    def total_s(self) -> float:
        return time.perf_counter() - self._started_monotonic

    def phases(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(p) for p in self._phases]

    def summary(self, top: int = 6) -> str:
        phases = self.phases()
        if not phases:
            return f"boot {self.total_s():.1f}s (no phases recorded)"
        slowest = sorted(phases, key=lambda p: p["duration_s"], reverse=True)[:top]
        parts = ", ".join(f"{p['name']}={p['duration_s']:.1f}s" for p in slowest)
        return (
            f"boot {self.total_s():.1f}s across {len(phases)} phases; "
            f"slowest: {parts}"
        )

    def to_report(self) -> dict[str, Any]:
        return {
            "schema": "aura.boot_profile.v1",
            "started_at_unix": self._started_at,
            "total_s": round(self.total_s(), 3),
            "slow_phase_warn_s": SLOW_PHASE_WARN_S,
            "phases": self.phases(),
        }

    def write_artifact(self, path: Optional[Path] = None) -> Optional[Path]:
        """Persist the profile for post-mortems. Never raises."""
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            target = Path(path) if path is not None else (
                Path("artifacts") / "current" / "boot_profile.json"
            )
            with local_internal_governed_scope(
                "boot_profile.write_artifact",
                receipt_prefix="boot-profile-artifact",
            ):
                get_file_write_gateway().write_text(
                    target,
                    json.dumps(self.to_report(), indent=2),
                    source="boot_profile.write_artifact",
                    durable=False,
                )
            return target
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("boot profile artifact write skipped: %s", exc)
            return None


_profiler: BootProfiler | None = None
_profiler_lock = threading.Lock()


def get_boot_profiler() -> BootProfiler:
    global _profiler
    with _profiler_lock:
        if _profiler is None:
            _profiler = BootProfiler()
        return _profiler


def reset_boot_profiler() -> BootProfiler:
    """Fresh profiler (tests and in-process reboots)."""
    global _profiler
    with _profiler_lock:
        _profiler = BootProfiler()
        return _profiler


__all__ = [
    "BootProfiler",
    "SLOW_PHASE_WARN_S",
    "get_boot_profiler",
    "reset_boot_profiler",
]
