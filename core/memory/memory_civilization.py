"""core/memory/memory_civilization.py — Memory Civilization Coordinator.

Manages all memory types (episodic, semantic, procedural, etc.) with metadata, provenance, and privacy structures.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Aura.MemoryCivilization")


@dataclass
class MemoryItem:
    """A discrete unit of memory inside Aura's multi-scale memory civilization."""
    memory_id: str
    memory_type: str  # "episodic", "semantic", "procedural", "project", "relationship", "world", "research", "codebase", "tool", "failure", "preference", "mission", "self_modification", "policy"
    content: str
    source: str
    confidence: float
    privacy: str = "private"  # "private", "anonymized", "public"
    expiry: Optional[float] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    last_verified: float = field(default_factory=time.time)
    contradiction_links: List[str] = field(default_factory=list)
    user_controls: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "privacy": self.privacy,
            "expiry": self.expiry,
            "provenance": self.provenance,
            "last_verified": self.last_verified,
            "contradiction_links": self.contradiction_links,
            "user_controls": self.user_controls,
        }


class MemoryCivilization:
    """Coordinates memory structures across episodic, semantic, and procedural layers."""

    def __init__(self) -> None:
        self.memories: Dict[str, MemoryItem] = {}
        self._initialized = False

    async def initialize(self) -> None:
        self._initialized = True
        logger.info("Memory Civilization Engine initialized.")

    def record_memory(
        self,
        memory_id: str,
        memory_type: str,
        content: str,
        source: str,
        confidence: float = 1.0,
        privacy: str = "private",
        expiry: Optional[float] = None,
        provenance: Dict[str, Any] = None,
    ) -> MemoryItem:
        """Stores a new memory unit in the civilization ledger."""
        item = MemoryItem(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            source=source,
            confidence=confidence,
            privacy=privacy,
            expiry=expiry,
            provenance=provenance or {},
        )
        self.memories[memory_id] = item
        logger.info("Recorded memory: id=%s type=%s content_len=%d", memory_id, memory_type, len(content))
        return item

    async def record_mission_outcome(self, objective: str, result: Dict[str, Any]) -> None:
        """Saves a mission's performance and lessons to memory."""
        mem_id = f"mission_{int(time.time())}"
        content = f"Objective: {objective} -> Result: {result.get('ok', False)} error: {result.get('error', 'none')}"
        self.record_memory(
            memory_id=mem_id,
            memory_type="mission",
            content=content,
            source="mission_engine",
            confidence=1.0,
            provenance={"result": result},
        )

    def search_memories(self, query: str, memory_type: Optional[str] = None) -> List[MemoryItem]:
        """Simple keyword filter across memory civilization contents."""
        results = []
        for item in self.memories.values():
            if memory_type and item.memory_type != memory_type:
                continue
            if query.lower() in item.content.lower():
                results.append(item)
        return results


_civilization_instance: MemoryCivilization | None = None


def get_memory_civilization() -> MemoryCivilization:
    global _civilization_instance
    if _civilization_instance is None:
        _civilization_instance = MemoryCivilization()
    return _civilization_instance
