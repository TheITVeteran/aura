"""What recall actually did, recorded as it happens.

The gap this closes
-------------------
``tools/fit_actr_retrieval.py`` fitted the ACT-R retrieval curve by generating
synthetic trace ages and importances, running Aura's own ``_static_rank`` over
them, labelling the top-k as "recalled", and fitting to those labels. That is
internal calibration of the ranker against itself on invented inputs. It is
worth having and it is not evidence about Aura's memory, still less about human
memory, and the distinction was not being drawn sharply enough.

The reason it had to be synthetic is that nothing observed real recalls. The
outcome ledger in ``retrieval_outcomes.py`` records what happened to memories
that *were* returned — retrieved, helpful, harmful — which cannot support a
retrieval-probability curve, because the curve is about the candidates that
were considered and NOT returned. Those were never written down anywhere.

So this records them. Every ranked recall contributes one observation per
candidate: its activation, the rank it came out at, and how many candidates it
was competing against. From that, "was it recalled" is recoverable for any
top-k the caller applied, over the real population rather than a generated one.

Cost and safety
---------------
This sits on the recall path, so it is bounded and cheap by construction: a
fixed-size ring of plain floats, no content, no keys, no allocation beyond the
ring. Nothing here can grow, and nothing here holds anything that could
identify a memory — an activation and a rank are not a memory. Recording is
best-effort and never raises into recall.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "RecallObservation",
    "RecallObservationRing",
    "get_recall_observations",
    "record_ranking",
]

#: Observations retained. One ranked recall of 40 candidates contributes 40, so
#: this is a few thousand recalls' worth — enough to fit a two-parameter curve
#: many times over, and small enough to be invisible in memory.
_RING_CAPACITY = 20000


@dataclass(frozen=True, slots=True)
class RecallObservation:
    """One candidate's fate in one ranked recall."""

    activation: float
    rank: int
    candidates: int

    def recalled_at(self, top_k: int) -> bool:
        return self.rank < top_k


class RecallObservationRing:
    """Bounded, thread-safe record of ranked recalls."""

    def __init__(self, capacity: int = _RING_CAPACITY) -> None:
        self._lock = threading.Lock()
        self._ring: deque[RecallObservation] = deque(maxlen=capacity)
        self._rankings = 0

    def record(self, activations: Sequence[float]) -> None:
        """Record one ranked recall. ``activations`` must be in ranked order."""
        total = len(activations)
        if total < 2:
            # A single candidate is not a competition and carries no
            # information about a retrieval threshold.
            return
        with self._lock:
            self._rankings += 1
            for rank, activation in enumerate(activations):
                if activation != activation:  # NaN
                    continue
                self._ring.append(
                    RecallObservation(
                        activation=float(activation), rank=rank, candidates=total
                    )
                )

    def observations(self) -> list[RecallObservation]:
        with self._lock:
            return list(self._ring)

    def samples(self, *, top_k_fraction: float = 0.2) -> list[tuple[float, int]]:
        """``(activation, recalled)`` pairs, using each recall's own top-k.

        ``top_k_fraction`` reproduces the slice callers actually apply. It is
        per-ranking rather than global, so a recall of 5 candidates and one of
        50 each contribute their own notion of "returned" instead of being
        forced onto one threshold.
        """
        out: list[tuple[float, int]] = []
        for obs in self.observations():
            top_k = max(1, int(obs.candidates * top_k_fraction))
            out.append((obs.activation, int(obs.recalled_at(top_k))))
        return out

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "observations": len(self._ring),
                "rankings": self._rankings,
                "capacity": self._ring.maxlen,
                "saturated": len(self._ring) == self._ring.maxlen,
            }

    def clear(self) -> None:
        with self._lock:
            self._ring.clear()
            self._rankings = 0


_ring: RecallObservationRing | None = None
_ring_lock = threading.Lock()


def get_recall_observations() -> RecallObservationRing:
    global _ring
    if _ring is None:
        with _ring_lock:
            if _ring is None:
                _ring = RecallObservationRing()
    return _ring


def record_ranking(activations: Iterable[float]) -> None:
    """Best-effort record of one ranked recall. Never raises into recall."""
    try:
        get_recall_observations().record(list(activations))
    except (RuntimeError, ValueError, TypeError, MemoryError):
        # Observation is diagnostics. It must never be able to fail a recall,
        # and a degradation receipt per recall would be louder than the signal.
        pass  # no-op: intentional
