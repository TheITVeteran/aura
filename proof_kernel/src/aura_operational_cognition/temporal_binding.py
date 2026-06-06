from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TemporalEvent:
    content: str
    source: str
    valence: float = 0.0
    significance: float = 0.5
    timestamp: float = field(default_factory=time.time)
    bound_to_prev: bool = False

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def as_narrative_line(self) -> str:
        age = self.age_seconds()
        if age < 60:
            when = f"{age:.0f}s ago"
        elif age < 3600:
            when = f"{age / 60:.1f}m ago"
        else:
            when = f"{age / 3600:.1f}h ago"
        valence_symbol = "positive" if self.valence > 0.2 else ("negative" if self.valence < -0.2 else "neutral")
        return f"[{when} | {valence_symbol} | sig={self.significance:.1f}] {self.content[:80]}"


class TemporalBindingEngine:
    _PRESENT_WINDOW_SECS = 300
    _MAX_EVENTS = 200
    _NARRATIVE_REFRESH_TICKS = 5
    _MAX_ANCHORS = 10

    def __init__(self):
        self._lock = asyncio.Lock()
        self._events: deque[TemporalEvent] = deque(maxlen=self._MAX_EVENTS)
        self._anchors: list[str] = []
        self._current_narrative = ""
        self._narrative_age = 0.0
        self._tick_count = 0
        self._birth_time = time.time()

    async def record_event(
        self,
        content: str,
        source: str,
        valence: float = 0.0,
        significance: float = 0.5,
    ) -> None:
        async with self._lock:
            event = TemporalEvent(
                content=content,
                source=source,
                valence=valence,
                significance=significance,
            )
            if self._events:
                event.bound_to_prev = True
            self._events.append(event)

    async def maybe_refresh_narrative(self, tick: int) -> None:
        self._tick_count = tick
        if tick % self._NARRATIVE_REFRESH_TICKS == 0:
            await self._rebuild_narrative()

    async def get_narrative(self) -> str:
        async with self._lock:
            return self._current_narrative or self._build_minimal_narrative()

    def get_snapshot(self) -> dict[str, Any]:
        present = [item for item in self._events if item.age_seconds() < self._PRESENT_WINDOW_SECS]
        return {
            "total_events": len(self._events),
            "present_window_events": len(present),
            "anchors": len(self._anchors),
            "narrative_age_secs": round(time.time() - self._narrative_age, 1),
            "uptime_hours": round((time.time() - self._birth_time) / 3600, 2),
            "avg_valence": round(sum(item.valence for item in present) / len(present), 3) if present else 0.0,
        }

    async def _rebuild_narrative(self) -> None:
        async with self._lock:
            now = time.time()
            present = [item for item in self._events if item.age_seconds() < self._PRESENT_WINDOW_SECS]
            past = [item for item in self._events if item.age_seconds() >= self._PRESENT_WINDOW_SECS]

            if past:
                top_past = sorted(past, key=lambda item: item.significance, reverse=True)
                for event in top_past[:2]:
                    anchor = f"Earlier: {event.content[:60]} (sig={event.significance:.1f})"
                    if anchor not in self._anchors:
                        if len(self._anchors) < self._MAX_ANCHORS:
                            self._anchors.append(anchor)
                        else:
                            self._anchors.pop(0)
                            self._anchors.append(anchor)

            if present:
                avg_valence = sum(item.valence for item in present) / len(present)
                affect_desc = "positive" if avg_valence > 0.2 else ("negative" if avg_valence < -0.2 else "neutral")
                dominant = max(present, key=lambda item: item.significance)
                dominant_desc = dominant.content[:80]
            else:
                affect_desc = "neutral"
                dominant_desc = "nothing in particular"

            lines = [
                f"[AUTOBIOGRAPHICAL PRESENT - {((now - self._birth_time) / 3600.0):.1f}h uptime]",
                f"Current affective tone: {affect_desc}.",
                f"Most salient recent focus: '{dominant_desc}'.",
            ]
            if present:
                lines.append(f"Recent stream ({len(present)} events):")
                for event in sorted(present, key=lambda item: item.timestamp)[-5:]:
                    lines.append(f"  {event.as_narrative_line()}")
            if self._anchors:
                lines.append("Temporal anchors (compressed past):")
                for anchor in self._anchors[-3:]:
                    lines.append(f"  {anchor}")

            self._current_narrative = "\n".join(lines)
            self._narrative_age = now

    def _build_minimal_narrative(self) -> str:
        uptime = time.time() - self._birth_time
        return f"[AUTOBIOGRAPHICAL PRESENT] System is {uptime:.0f}s old. No events recorded yet."
