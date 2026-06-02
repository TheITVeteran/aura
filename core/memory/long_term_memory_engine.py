"""
Grok-Level Long-Term Memory with Forgetting Curve + Emotional Tagging for Aura.

The runtime contract here is deliberately practical: durable memory must either
load, save, or fail visibly without destroying existing evidence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from core.config import config
from core.container import ServiceContainer
from core.event_bus import get_event_bus
from core.memory.atomic_storage import atomic_write
from core.memory.retention_policy import MemoryRetentionPolicy, long_term_retention_policy
from core.runtime.errors import FallbackClassification, PersistenceCorruption, record_degradation
from core.utils.task_tracker import task_tracker

logger = logging.getLogger("Aura.LongTermMemory")

_LTM_RECOVERABLE_ERRORS = (
    AttributeError,
    FileNotFoundError,
    ImportError,
    json.JSONDecodeError,
    OSError,
    PersistenceCorruption,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _record_ltm_degradation(
    error: BaseException,
    *,
    action: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
):
    return record_degradation(
        "long_term_memory_engine",
        error,
        severity=severity,
        action=action,
        classification=FallbackClassification.SAFE_FALLBACK,
        receipt_required=True,
        extra=extra,
    )


def _bounded_float(value: Any, default: float, *, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError as exc:
        _record_ltm_degradation(
            exc,
            action=f"used default {name} after invalid environment value",
            extra={"name": name, "value": raw, "default": default},
        )
        return default


def _tokenize(text: str) -> set[str]:
    return {
        token.strip(".,!?;:()[]{}\"'").lower()
        for token in str(text or "").split()
        if len(token.strip(".,!?;:()[]{}\"'")) >= 2
    }


@dataclass
class TaggedMemory:
    id: str
    content: str
    timestamp: float
    emotional_valence: float
    importance: float
    decay_rate: float
    last_rehearsed: float
    tags: list[str] = field(default_factory=list)


class LongTermMemoryEngine:
    name = "long_term_memory_engine"

    def __init__(self):
        self.memories: list[TaggedMemory] = []
        self._retention_policy = long_term_retention_policy()
        self._max_memories = self._retention_policy.max_items
        self.memory_facade = None
        self.drive_engine = None
        self.cel = None
        self.running = False
        self._consolidation_task: asyncio.Task[None] | None = None
        self._storage_healthy = True
        self.db_path = config.paths.data_dir / "long_term_memories.json"
        self.consolidation_interval_s = _env_float(
            "AURA_LTM_CONSOLIDATION_INTERVAL_S",
            86400.0,
            minimum=300.0,
        )
        self.rehearsal_min_age_s = _env_float(
            "AURA_LTM_REHEARSAL_MIN_AGE_S",
            3600.0,
            minimum=60.0,
        )

    async def start(self) -> bool:
        self.memory_facade = self._resolve_service("memory_facade")
        self.drive_engine = self._resolve_service("drive_engine")
        self.cel = self._resolve_service("constitutive_expression_layer")

        loaded = self._load_memories()
        self.running = True
        if self._consolidation_task is None or self._consolidation_task.done():
            self._consolidation_task = task_tracker.create_task(
                self._nightly_consolidation(),
                name="LongTermMemory",
            )

        logger.info(
            "Long-Term Memory with emotional tagging online; durable=%s count=%s",
            self._storage_healthy,
            len(self.memories),
        )

        try:
            await get_event_bus().publish(
                "mycelium.register",
                {
                    "component": "long_term_memory_engine",
                    "hooks_into": ["memory_facade", "drive_engine", "cel", "dream_processor"],
                },
            )
        except (ImportError, AttributeError, RuntimeError, TimeoutError) as exc:
            _record_ltm_degradation(
                exc,
                action="kept long-term memory online and skipped optional mycelium registration",
                extra={"event": "mycelium.register"},
            )
            logger.debug("Event bus publish missed for Mycelium hook: %s", exc)
        return loaded

    async def stop(self) -> bool:
        self.running = False
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await asyncio.wait_for(self._consolidation_task, timeout=2.0)
            except (asyncio.CancelledError, TimeoutError):
                pass
            finally:
                self._consolidation_task = None
        return self._save_memories()

    def _load_memories(self) -> bool:
        if not self.db_path.exists():
            self.memories = []
            self._storage_healthy = True
            return True

        try:
            data = json.loads(self.db_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise PersistenceCorruption("long-term memory store root must be a list")

            memories = []
            rejected = 0
            for item in data:
                try:
                    memories.append(self._memory_from_payload(item))
                except (TypeError, ValueError) as exc:
                    rejected += 1
                    logger.debug("Rejected malformed long-term memory entry: %s", exc)

            if rejected:
                _record_ltm_degradation(
                    ValueError("malformed long-term memory entries skipped"),
                    action="loaded valid long-term memories and skipped malformed rows",
                    extra={"rejected": rejected, "loaded": len(memories)},
                )

            self.memories = memories
            self._storage_healthy = True
            logger.info("Loaded %s emotionally tagged memories", len(self.memories))
            return True
        except _LTM_RECOVERABLE_ERRORS as exc:
            self._storage_healthy = False
            self.memories = []
            self._quarantine_corrupt_store(exc)
            _record_ltm_degradation(
                exc,
                action="started with empty in-memory long-term memory after preserving corrupt store",
                severity="degraded",
                extra={"path": str(self.db_path)},
            )
            return False

    def _save_memories(self) -> bool:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            data = [self._memory_to_payload(memory) for memory in self.memories]
            atomic_write(str(self.db_path), json.dumps(data, indent=2, sort_keys=True))
            self._storage_healthy = True
            return True
        except _LTM_RECOVERABLE_ERRORS as exc:
            self._storage_healthy = False
            _record_ltm_degradation(
                exc,
                action="kept long-term memories in memory after durable save failed",
                severity="degraded",
                extra={"path": str(self.db_path), "count": len(self.memories)},
            )
            logger.error("Memory save failed: %s", exc)
            return False

    async def store(
        self,
        content: str,
        valence: float = 0.0,
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> TaggedMemory | None:
        content = str(content or "").strip()
        if not content:
            return None

        normalized_tags = self._normalize_tags(tags)
        importance = _bounded_float(importance, 0.5, minimum=0.0, maximum=1.0)
        valence = _bounded_float(valence, 0.0, minimum=-1.0, maximum=1.0)

        try:
            from core.constitution import get_constitutional_core

            approved, reason = await get_constitutional_core().approve_memory_write(
                memory_type="long_term_memory",
                content=content,
                source="long_term_memory",
                importance=importance,
                metadata={"tags": normalized_tags, "valence": valence},
            )
            if not approved:
                logger.warning("LongTermMemory write blocked: %s", reason)
                try:
                    from core.health.degraded_events import record_degraded_event

                    record_degraded_event(
                        "long_term_memory",
                        "memory_write_blocked",
                        detail=str(reason),
                        severity="warning",
                        classification="background_degraded",
                        context={"importance": importance, "valence": valence},
                    )
                except (ImportError, AttributeError, RuntimeError, TimeoutError) as exc:
                    _record_ltm_degradation(
                        exc,
                        action="blocked memory write and skipped degraded-event telemetry",
                        extra={"reason": str(reason)[:200]},
                    )
                    logger.debug("LongTermMemory degraded-event logging skipped: %s", exc)
                return None
        except (ImportError, AttributeError, RuntimeError, TimeoutError) as exc:
            runtime_live = self._runtime_is_live()
            _record_ltm_degradation(
                exc,
                action="blocked runtime long-term memory write when constitutional gate was unavailable",
                extra={"runtime_live": runtime_live},
            )
            logger.debug("LongTermMemory constitutional gate skipped: %s", exc)
            if runtime_live:
                logger.warning("LongTermMemory write blocked: constitutional gate unavailable")
                return None

        decay = 0.001 if importance > 0.7 or abs(valence) > 0.7 else 0.02
        now = time.time()
        memory = TaggedMemory(
            id=f"mem_{time.time_ns()}_{uuid.uuid4().hex[:8]}",
            content=content[:800],
            timestamp=now,
            emotional_valence=valence,
            importance=importance,
            decay_rate=decay,
            last_rehearsed=now,
            tags=normalized_tags,
        )
        self.memories.append(memory)

        self._enforce_retention_cap()
        
        self._save_memories()

        if self.cel is not None:
            try:
                await self.cel.emit(
                    {
                        "first_person": f"I just etched this moment into my long-term memory... {content[:100]}",
                        "phi": 0.78,
                        "origin": "long_term_memory",
                    }
                )
            except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
                _record_ltm_degradation(
                    exc,
                    action="stored memory but skipped immediate first-person reflection",
                    extra={"memory_id": memory.id},
                )
                logger.debug("LongTermMemory CEL reflection skipped: %s", exc)
        return memory

    def _policy(self) -> MemoryRetentionPolicy:
        policy = getattr(self, "_retention_policy", None)
        if isinstance(policy, MemoryRetentionPolicy):
            return policy
        max_memories = int(getattr(self, "_max_memories", 10_000) or 10_000)
        return MemoryRetentionPolicy(max_items=max_memories, prune_keep_fraction=0.92, basis="legacy_fallback")

    def _enforce_retention_cap(self) -> None:
        policy = self._policy()
        self._max_memories = policy.max_items
        if len(self.memories) <= policy.max_items:
            return
        self.memories.sort(key=lambda memory: (memory.importance, abs(memory.emotional_valence), memory.timestamp))
        keep_count = policy.keep_count(len(self.memories))
        self.memories = self.memories[-keep_count:]
        logger.info(
            "LongTermMemory retention cap: kept %d memories using %s policy.",
            len(self.memories),
            policy.basis,
        )

    async def recall_relevant(self, query: str, limit: int = 5) -> list[TaggedMemory]:
        now = time.time()
        query_tokens = _tokenize(query)
        scored = []
        for memory in self.memories:
            age = now - memory.timestamp
            strength = (
                memory.importance
                * max(0.01, 1.0 - memory.decay_rate * age)
                * (1.0 + abs(memory.emotional_valence))
            )
            memory_tokens = _tokenize(f"{memory.content} {' '.join(memory.tags)}")
            overlap = len(query_tokens & memory_tokens)
            relevance = overlap / max(1, len(query_tokens)) if query_tokens else 0.0
            scored.append((strength + relevance * 2.0, memory.timestamp, memory))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [memory for _, _, memory in scored[: max(0, int(limit or 0))]]

    async def _nightly_consolidation(self):
        while self.running:
            try:
                await asyncio.sleep(self.consolidation_interval_s)
                now = time.time()
                for memory in self.memories:
                    age = now - memory.last_rehearsed
                    if age > self.rehearsal_min_age_s and memory.importance > 0.6:
                        memory.last_rehearsed = now
                        if self.cel:
                            try:
                                await self.cel.emit(
                                    {
                                        "first_person": f"During my dream cycle I revisited: {memory.content[:80]}...",
                                        "phi": 0.65,
                                        "origin": "long_term_memory",
                                    }
                                )
                            except (RuntimeError, AttributeError, TypeError, ValueError, TimeoutError) as exc:
                                _record_ltm_degradation(
                                    exc,
                                    action="continued consolidation after dream-cycle reflection failed",
                                    extra={"memory_id": memory.id},
                                )
                                logger.debug("CEL emission failed in nightly consolidation: %s", exc)
                self._save_memories()
            except asyncio.CancelledError:
                raise
            except _LTM_RECOVERABLE_ERRORS as exc:
                _record_ltm_degradation(
                    exc,
                    action="kept long-term memory consolidation loop alive after recoverable failure",
                    severity="degraded",
                )

    def _resolve_service(self, name: str):
        try:
            return ServiceContainer.get(name, default=None)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_ltm_degradation(
                exc,
                action="started long-term memory without optional service dependency",
                extra={"service": name},
            )
            return None

    def _runtime_is_live(self) -> bool:
        try:
            return bool(
                getattr(ServiceContainer, "_registration_locked", False)
                or ServiceContainer.has("executive_core")
                or ServiceContainer.has("aura_kernel")
                or ServiceContainer.has("kernel_interface")
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_ltm_degradation(
                exc,
                action="treated runtime as live after runtime-status probe failed",
            )
            return True

    def _memory_from_payload(self, payload: Any) -> TaggedMemory:
        if not isinstance(payload, dict):
            raise ValueError("memory payload must be an object")
        content = str(payload.get("content", "")).strip()
        if not content:
            raise ValueError("memory payload content is empty")
        now = time.time()
        timestamp = _bounded_float(payload.get("timestamp"), now, minimum=0.0, maximum=now + 60.0)
        return TaggedMemory(
            id=str(payload.get("id") or f"mem_{time.time_ns()}_{uuid.uuid4().hex[:8]}"),
            content=content[:800],
            timestamp=timestamp,
            emotional_valence=_bounded_float(payload.get("emotional_valence"), 0.0, minimum=-1.0, maximum=1.0),
            importance=_bounded_float(payload.get("importance"), 0.5, minimum=0.0, maximum=1.0),
            decay_rate=_bounded_float(payload.get("decay_rate"), 0.02, minimum=0.0001, maximum=1.0),
            last_rehearsed=_bounded_float(payload.get("last_rehearsed"), timestamp, minimum=0.0, maximum=now + 60.0),
            tags=self._normalize_tags(payload.get("tags")),
        )

    def _memory_to_payload(self, memory: TaggedMemory) -> dict[str, Any]:
        return {
            "id": memory.id,
            "content": memory.content,
            "timestamp": memory.timestamp,
            "emotional_valence": memory.emotional_valence,
            "importance": memory.importance,
            "decay_rate": memory.decay_rate,
            "last_rehearsed": memory.last_rehearsed,
            "tags": list(memory.tags),
        }

    def _normalize_tags(self, tags: Any) -> list[str]:
        if tags is None:
            return []
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, (list, tuple, set)):
            return []
        normalized = []
        for tag in tags:
            text = str(tag).strip().lower()
            if text and text not in normalized:
                normalized.append(text[:64])
        return normalized[:20]

    def _quarantine_corrupt_store(self, cause: BaseException) -> None:
        if not self.db_path.exists():
            return
        quarantine_path = self.db_path.with_name(f"{self.db_path.name}.corrupt.{int(time.time())}")
        try:
            self.db_path.replace(quarantine_path)
            logger.warning("Quarantined corrupt long-term memory store at %s", quarantine_path)
        except OSError as exc:
            _record_ltm_degradation(
                exc,
                action="left corrupt long-term memory store in place after quarantine failed",
                severity="degraded",
                extra={"path": str(self.db_path), "cause": type(cause).__name__},
            )


_memory_instance = None
_instance_lock = None


async def get_long_term_memory_engine():
    """Thread-safe singleton for engine access."""
    global _instance_lock, _memory_instance
    if _instance_lock is None:
        _instance_lock = asyncio.Lock()
    async with _instance_lock:
        if _memory_instance is None:
            _memory_instance = LongTermMemoryEngine()
        return _memory_instance
