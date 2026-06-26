"""core/utils/engine_support.py — shared helpers for derived cognitive engines.

Small, dependency-light utilities shared by several organs (ethics, sim, brain,
knowledge): best-effort brain resolution, defensive text extraction from a
brain.think() result, a data-dir resolver, and a degradation recorder. These live
here — not inside any single organ — because more than one organ needs them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.EngineSupport")


def record_engine_degradation(
    subsystem: str,
    exc: BaseException,
    *,
    action: str,
    severity: str = "warning",
) -> None:
    record_degradation(subsystem, exc, severity=severity, action=action)


def resolve_brain(orchestrator: Any = None) -> Any:
    """Best-effort brain lookup. None is a valid answer — every caller degrades to
    a heuristic when no model is available, so nothing blocks on this."""
    if orchestrator is not None and getattr(orchestrator, "brain", None) is not None:
        return orchestrator.brain
    try:
        from core.container import ServiceContainer
        from core.service_names import ServiceNames

        orch = ServiceContainer.get("orchestrator", default=None)
        if orch is not None and getattr(orch, "brain", None) is not None:
            return orch.brain
        return ServiceContainer.get(ServiceNames.BRAIN, default=None)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_engine_degradation(
            "engine_support",
            exc,
            action="resolved brain as unavailable and fell back to heuristic reasoning",
        )
        return None


def coerce_text(result: Any) -> str:
    """brain.think() return shapes vary by path; pull a string out defensively."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    for attr in ("text", "content", "response", "answer", "output"):
        val = getattr(result, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    if isinstance(result, dict):
        for key in ("text", "content", "response", "answer", "output"):
            val = result.get(key)
            if isinstance(val, str) and val.strip():
                return val
    return ""


def data_root(subdir: str) -> Path:
    from core.config import config

    path = Path(config.paths.data_dir) / subdir
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    return path
