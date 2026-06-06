from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


class ContentType(Enum):
    UNKNOWN = auto()
    PERCEPTUAL = auto()
    AFFECTIVE = auto()
    MEMORIAL = auto()
    INTENTIONAL = auto()
    LINGUISTIC = auto()
    SOMATIC = auto()
    SOCIAL = auto()
    META = auto()


@dataclass
class CognitiveCandidate:
    content: str
    source: str
    priority: float
    content_type: ContentType = ContentType.UNKNOWN
    affect_weight: float = 0.0
    focus_bias: float = 0.0
    submitted_at: float = field(default_factory=time.time)

    @property
    def effective_priority(self) -> float:
        age = time.time() - self.submitted_at
        recency = max(0.0, 1.0 - (age / 10.0))
        scaled = (self.priority + self.affect_weight * 0.3 + self.focus_bias) * (0.7 + 0.3 * recency)
        return min(1.0, scaled)


@dataclass
class BroadcastRecord:
    winner: CognitiveCandidate
    losers: list[str]
    timestamp: float = field(default_factory=time.time)


ProcessorFn = Callable[[CognitiveCandidate], Optional[Awaitable[Any]]]


class GlobalWorkspace:
    _INHIBIT_TICKS = 3
    _MAX_CANDIDATES = 20
    _IGNITION_THRESHOLD = 0.6
    _PHI_PRIORITY_BOOST = 0.15

    def __init__(self, attention_schema: Any = None):
        self._lock: Optional[asyncio.Lock] = None
        self._candidates: list[CognitiveCandidate] = []
        self._inhibited: dict[str, int] = {}
        self._processors: list[ProcessorFn] = []
        self._history: list[BroadcastRecord] = []
        self._tick = 0
        self.attention_schema = attention_schema
        self.last_winner: Optional[CognitiveCandidate] = None
        self.ignition_level = 0.0
        self.ignited = False
        self._ignition_count = 0
        self._current_phi = 0.0

    async def submit(self, candidate: CognitiveCandidate) -> bool:
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            if self._inhibited.get(candidate.source, 0) > 0:
                return False

            if self._current_phi > 0.1:
                phi_boost = min(self._PHI_PRIORITY_BOOST, self._current_phi * 0.1)
                candidate.focus_bias = min(1.0, candidate.focus_bias + phi_boost)

            if len(self._candidates) >= self._MAX_CANDIDATES:
                return False

            self._candidates = [item for item in self._candidates if item.source != candidate.source]
            self._candidates.append(candidate)
            return True

    def register_processor(self, fn: ProcessorFn) -> None:
        self._processors.append(fn)

    async def run_competition(self) -> Optional[CognitiveCandidate]:
        self._tick += 1

        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            self._inhibited = {src: ticks - 1 for src, ticks in self._inhibited.items() if ticks > 1}
            if not self._candidates:
                return None

            self._candidates.sort(key=lambda item: item.effective_priority, reverse=True)
            winner = self._candidates[0]
            losers = self._candidates[1:]
            for loser in losers:
                self._inhibited[loser.source] = self._INHIBIT_TICKS

            self._candidates = []
            self._history.append(BroadcastRecord(winner=winner, losers=[item.source for item in losers]))
            self._history = self._history[-100:]
            self.last_winner = winner

        winner_priority = winner.effective_priority
        self.ignition_level = min(1.0, winner_priority / self._IGNITION_THRESHOLD)
        was_ignited = self.ignited
        self.ignited = winner_priority >= self._IGNITION_THRESHOLD
        if self.ignited and not was_ignited:
            self._ignition_count += 1

        if self.attention_schema and hasattr(self.attention_schema, "set_focus"):
            focus_result = self.attention_schema.set_focus(
                content=winner.content,
                source=winner.source,
                priority=winner.effective_priority,
            )
            if inspect.isawaitable(focus_result):
                await focus_result

        if self._processors:
            await asyncio.gather(*(self._safe_call(proc, winner) for proc in self._processors))

        logger.debug(
            "workspace tick=%s winner=%s priority=%.3f ignition=%s",
            self._tick,
            winner.source,
            winner.effective_priority,
            self.ignited,
        )
        return winner

    async def _safe_call(self, fn: ProcessorFn, winner: CognitiveCandidate) -> None:
        result = fn(winner)
        if inspect.isawaitable(result):
            await result

    def get_snapshot(self) -> dict[str, Any]:
        last = self.last_winner
        return {
            "tick": self._tick,
            "last_winner": last.source if last else None,
            "last_content": last.content if last else None,
            "last_priority": round(last.effective_priority, 3) if last else 0.0,
            "pending_candidates": len(self._candidates),
            "inhibited_sources": sorted(self._inhibited.keys()),
            "broadcast_history_len": len(self._history),
            "ignition_level": round(self.ignition_level, 3),
            "ignited": self.ignited,
            "ignition_count": self._ignition_count,
            "phi": round(self._current_phi, 4),
        }

    def update_phi(self, phi: float) -> None:
        self._current_phi = max(0.0, float(phi))
