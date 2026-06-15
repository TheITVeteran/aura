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

logger = logging.getLogger("research_triggers")

DEFAULT_TRIGGER_PATH = Path.home() / ".aura/live-source/aura/knowledge/research-triggers.jsonl"
RING_LIMIT = 500
_TRIGGER_PERSISTENCE_ERRORS = (OSError, TypeError, ValueError)


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
    path: Path = DEFAULT_TRIGGER_PATH,
) -> None:
    """Append a trigger to the persistent queue.

    Persistence failures are recorded and returned because curiosity research is
    supportive, not critical-path. Programmer/invariant errors still surface.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "topic": topic,
            "source_intent_id": source_intent_id,
            "contested_count": int(contested_count),
            "payload_hint": _json_safe_payload(payload_hint or {}),
            "emitted_at": time.time(),
            "consumed_at": None,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        _maybe_truncate_ring(path)
    except _TRIGGER_PERSISTENCE_ERRORS as exc:
        record_degradation("research_triggers", exc)
        logger.warning("Failed to persist research trigger at %s: %s", path, exc)


def drain_pending_triggers(
    path: Path = DEFAULT_TRIGGER_PATH,
    max_age_seconds: float = 86400.0 * 7,
) -> list[ResearchTrigger]:
    """Return all unconsumed, non-expired triggers. Does not mark them
    consumed — caller calls ``mark_consumed`` once the trigger has actually
    been picked up by the curiosity scheduler.
    """
    if not path.exists():
        return []
    out: list[ResearchTrigger] = []
    now = time.time()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("consumed_at") is not None:
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
    path: Path = DEFAULT_TRIGGER_PATH,
) -> None:
    """Mark a trigger as consumed by rewriting the file with the consumed_at
    timestamp set. Best-effort; concurrent producers may race but the worst
    case is a duplicate trigger, which the scheduler should be idempotent on.
    """
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
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
        tmp = path.with_suffix(path.suffix + ".tmp")
        atomic_write_text(tmp, "\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
        os.replace(tmp, path)
    except (OSError, ConnectionError, TimeoutError) as exc:
        record_degradation("research_triggers", exc)
        logger.debug("Failed to mark research trigger consumed at %s: %s", path, exc)


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


def _json_safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list | tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    return repr(value)
