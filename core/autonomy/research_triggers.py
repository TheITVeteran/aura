"""core/autonomy/research_triggers.py
─────────────────────────────────────
Lightweight queue for "things the system noticed it should research".

Producers (e.g. executive_core when deferring a contested belief-update):
   emit_research_trigger(topic=..., source_intent_id=..., ...)

Consumers (curiosity_scheduler):
   for trigger in drain_research_triggers(): ...

Persisted to disk so triggers survive restarts. Bounded ring (last N entries)
so a runaway producer can't blow up storage.

This module is intentionally tiny so the executive can import it without
pulling in the inference/memory stacks.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("research_triggers")

DEFAULT_TRIGGER_PATH = state_root() / "data" / "autonomy" / "research-triggers.jsonl"
RING_LIMIT = 500
_TRIGGER_PERSISTENCE_ERRORS = (OSError, TypeError, ValueError)
_BLOCKED_PAYLOAD_SURFACES = (
    "i lost the reply lane",
    "failed the reply-quality gate",
    "not starting a second foreground generation",
    "as an ai language model",
    "i don't have personal feelings",
    "i do not have personal feelings",
    "i don't have personal beliefs",
    "i do not have personal beliefs",
)


def _default_trigger_path() -> Path:
    override = os.environ.get("AURA_RESEARCH_TRIGGER_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return DEFAULT_TRIGGER_PATH


def _resolve_trigger_path(path: Path | None) -> Path:
    return Path(path).expanduser() if path is not None else _default_trigger_path()


@dataclass(frozen=True)
class ResearchTrigger:
    topic: str
    source_intent_id: str
    contested_count: int
    payload_hint: dict[str, Any]
    emitted_at: float
    consumed_at: float | None = None


def emit_research_trigger(
    topic: str,
    source_intent_id: str = "",
    contested_count: int = 0,
    payload_hint: dict[str, Any] | None = None,
    path: Path | None = None,
) -> None:
    """Append a trigger to the persistent queue.

    Persistence failures are recorded and returned because curiosity research is
    supportive, not critical-path. Programmer/invariant errors still surface.
    """
    resolved_path = _resolve_trigger_path(path)
    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        safe_payload = _json_safe_payload(payload_hint or {})
        if _contains_blocked_payload_surface(topic) or _contains_blocked_payload_surface(safe_payload):
            logger.warning(
                "Blocked contaminated research trigger from persistent queue (topic=%s, source_intent_id=%s).",
                topic,
                source_intent_id,
            )
            return
        record = {
            "topic": topic,
            "source_intent_id": source_intent_id,
            "contested_count": int(contested_count),
            "payload_hint": safe_payload,
            "emitted_at": time.time(),
            "consumed_at": None,
        }
        with resolved_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        _maybe_truncate_ring(resolved_path)
    except _TRIGGER_PERSISTENCE_ERRORS as exc:
        record_degradation("research_triggers", exc)
        logger.warning("Failed to persist research trigger at %s: %s", resolved_path, exc)


def drain_pending_triggers(
    path: Path | None = None,
    max_age_seconds: float = 86400.0 * 7,
) -> list[ResearchTrigger]:
    """Return all unconsumed, non-expired triggers. Does not mark them
    consumed — caller calls ``mark_consumed`` once the trigger has actually
    been picked up by the curiosity scheduler.
    """
    resolved_path = _resolve_trigger_path(path)
    if not resolved_path.exists():
        return []
    out: list[ResearchTrigger] = []
    now = time.time()
    try:
        for line in resolved_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("consumed_at") is not None:
                continue
            if _contains_blocked_payload_surface(rec.get("topic", "")) or _contains_blocked_payload_surface(
                rec.get("payload_hint", {})
            ):
                continue
            if (now - float(rec.get("emitted_at", 0.0))) > max_age_seconds:
                continue
            out.append(
                ResearchTrigger(
                    topic=str(rec.get("topic", "")),
                    source_intent_id=str(rec.get("source_intent_id", "")),
                    contested_count=int(rec.get("contested_count", 0)),
                    payload_hint=dict(rec.get("payload_hint", {})),
                    emitted_at=float(rec.get("emitted_at", 0.0)),
                    consumed_at=None,
                )
            )
    except (OSError, ConnectionError, TimeoutError) as exc:
        record_degradation("research_triggers", exc)
        return []
    return out


def mark_consumed(
    source_intent_id: str,
    path: Path | None = None,
) -> None:
    """Mark a trigger as consumed by rewriting the file with the consumed_at
    timestamp set. Best-effort; concurrent producers may race but the worst
    case is a duplicate trigger, which the scheduler should be idempotent on.
    """
    resolved_path = _resolve_trigger_path(path)
    if not resolved_path.exists():
        return
    try:
        lines = resolved_path.read_text(encoding="utf-8").splitlines()
        now = time.time()
        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("source_intent_id") == source_intent_id and rec.get("consumed_at") is None:
                rec["consumed_at"] = now
            new_lines.append(json.dumps(rec))
        tmp = resolved_path.with_suffix(resolved_path.suffix + ".tmp")
        atomic_write_text(tmp, "\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        os.replace(tmp, resolved_path)
    except (OSError, ConnectionError, TimeoutError) as exc:
        record_degradation("research_triggers", exc)
        logger.debug("Failed to mark research trigger consumed at %s: %s", resolved_path, exc)


def _maybe_truncate_ring(path: Path) -> None:
    """If the file exceeds RING_LIMIT lines, keep the most recent RING_LIMIT."""
    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > RING_LIMIT:
            keep = lines[-RING_LIMIT:]
            tmp = path.with_suffix(path.suffix + ".tmp")
            atomic_write_text(tmp, "".join(keep), encoding="utf-8")
            os.replace(tmp, path)
    except OSError as exc:
        record_degradation(
            "research_triggers",
            exc,
            severity="warning",
            action="left the append-only queue intact so the next emission can retry bounded compaction",
        )
        logger.debug("Failed to truncate research trigger ring at %s: %s", path, exc)


def _json_safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe payload without leaking unserializable runtime objects."""
    safe: dict[str, Any] = {}
    for key, value in dict(payload).items():
        safe[str(key)] = _json_safe_value(value)
    return safe


def _contains_blocked_payload_surface(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        lowered = value.lower()
        return any(surface in lowered for surface in _BLOCKED_PAYLOAD_SURFACES)
    if isinstance(value, dict):
        return any(
            _contains_blocked_payload_surface(key) or _contains_blocked_payload_surface(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple | set):
        return any(_contains_blocked_payload_surface(item) for item in value)
    return _contains_blocked_payload_surface(repr(value))


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list | tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    return repr(value)
