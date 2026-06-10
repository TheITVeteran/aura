"""Recall telemetry: memory retrieval quality as a measured quantity.

"Black-hole memory expands effective model capacity" is only a real
claim if recall quality is measurable. This module instruments every
RAG-bridge retrieval with hit/miss, candidate and kept counts, and
latency, keeps a bounded in-memory window plus rolling aggregates, and
exposes a snapshot for health/diagnostics surfaces.

Thread-safe, dependency-free, and cheap: one lock and a deque append
per retrieval. No I/O on the hot path.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

_WINDOW_MAX = 200


@dataclass(frozen=True)
class RecallEvent:
    at: float
    query_words: int
    candidates: int
    kept: int
    hit: bool
    latency_ms: float
    skipped_reason: str = ""


class RecallTelemetry:
    """Bounded window of retrieval events with rolling aggregates."""

    def __init__(self, window: int = _WINDOW_MAX):
        self._lock = threading.Lock()
        self._events: deque[RecallEvent] = deque(maxlen=max(10, window))
        self._total = 0
        self._hits = 0
        self._skips = 0
        self._latency_sum_ms = 0.0

    def record(
        self,
        *,
        query_words: int,
        candidates: int,
        kept: int,
        latency_ms: float,
        skipped_reason: str = "",
    ) -> None:
        hit = kept > 0 and not skipped_reason
        event = RecallEvent(
            at=time.time(),
            query_words=int(query_words),
            candidates=int(candidates),
            kept=int(kept),
            hit=hit,
            latency_ms=float(latency_ms),
            skipped_reason=str(skipped_reason or ""),
        )
        with self._lock:
            self._events.append(event)
            self._total += 1
            if skipped_reason:
                self._skips += 1
            elif hit:
                self._hits += 1
            self._latency_sum_ms += event.latency_ms

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)
            total = self._total
            hits = self._hits
            skips = self._skips
            latency_sum = self._latency_sum_ms

        attempted = max(0, total - skips)
        window_attempted = [e for e in events if not e.skipped_reason]
        window_hits = sum(1 for e in window_attempted if e.hit)
        window_latency = sorted(e.latency_ms for e in window_attempted)

        def _pct(values: list[float], q: float) -> float:
            if not values:
                return 0.0
            index = min(len(values) - 1, max(0, int(q * (len(values) - 1))))
            return round(values[index], 1)

        return {
            "lifetime": {
                "retrievals": total,
                "attempted": attempted,
                "hits": hits,
                "skips": skips,
                "hit_rate": round(hits / attempted, 3) if attempted else None,
                "mean_latency_ms": round(latency_sum / total, 1) if total else 0.0,
            },
            "window": {
                "size": len(events),
                "attempted": len(window_attempted),
                "hits": window_hits,
                "hit_rate": (
                    round(window_hits / len(window_attempted), 3)
                    if window_attempted
                    else None
                ),
                "latency_p50_ms": _pct(window_latency, 0.50),
                "latency_p95_ms": _pct(window_latency, 0.95),
            },
            "recent": [asdict(e) for e in events[-5:]],
        }


_TELEMETRY = RecallTelemetry()


def get_recall_telemetry() -> RecallTelemetry:
    return _TELEMETRY
