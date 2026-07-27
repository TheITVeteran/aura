"""Why the last generation came back empty.

The router defers background inference by returning ``""``. That is the right
behaviour — the local substrate holds one 32B model and a foreground turn must
win — but the empty string carries no reason, and every caller downstream has
to guess at one. The guesses are wrong in a specific, expensive way:

    RuntimeError: LLM returned no Python source; the model returned nothing at
    all.

The model was never asked. The reconstruction lane then reported "0/14 held-out
positions reproduced", blaming verification for a generation that never ran, and
the user was told the build had failed on quality when it had failed on
admission. That is the same failure class as a good answer discarded by a gate
and then reported as an infrastructure fault: the true cause exists, briefly, in
one function, and is thrown away before anyone who could report it sees it.

So the router writes the reason down here on its way out, and any caller holding
an unexplained empty generation can ask. Deliberately tiny: a process-local
last-value per lane, no history, no lock contention on the hot path. It explains
a failure; it is not telemetry and nothing branches on it.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

# Older than this and the deferral almost certainly belongs to some other call.
# An empty generation is reported within milliseconds of the deferral that
# caused it; a stale reason attached to an unrelated emptiness would be a
# confident wrong answer, which is worse than no answer.
_FRESHNESS_S = 20.0


@dataclass(frozen=True)
class Deferral:
    """One refusal to run inference, with the reason the refuser gave."""

    origin: str
    reason: str
    at: float

    def describe(self) -> str:
        return (
            f"inference was deferred, not run: {self.reason}"
            + (f" (origin {self.origin})" if self.origin else "")
        )


_lock = threading.Lock()
_last: Deferral | None = None


def record_deferral(*, origin: str, reason: str) -> None:
    """Called by whoever returned the empty string, at the moment it did."""
    global _last
    cleaned = " ".join(str(reason or "").split())[:200]
    if not cleaned:
        return
    with _lock:
        _last = Deferral(origin=" ".join(str(origin or "").split())[:80], at=time.time(), reason=cleaned)


def last_deferral(*, now: float | None = None) -> Deferral | None:
    """The most recent deferral, if it is recent enough to explain a failure."""
    with _lock:
        entry = _last
    if entry is None:
        return None
    stamp = float(now if now is not None else time.time())
    return entry if stamp - entry.at <= _FRESHNESS_S else None


def explain_empty_generation(*, now: float | None = None) -> str:
    """A cause to append to an empty-generation error, or "" if unknown.

    Returns a clause, not a sentence, so callers keep their own phrasing and
    this only ever adds the part they could not know.
    """
    entry = last_deferral(now=now)
    return entry.describe() if entry else ""


def reset_for_test() -> None:
    global _last
    with _lock:
        _last = None


__all__ = [
    "Deferral",
    "explain_empty_generation",
    "last_deferral",
    "record_deferral",
    "reset_for_test",
]
