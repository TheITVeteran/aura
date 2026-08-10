"""core/brain/llm/context_window_evidence.py — a window size that says where it came from.

Clean-room adoption of Ouroboros's ``capability_evidence`` (MIT; mechanism
reimplemented). They deleted a static per-model context-window table and
replaced every window claim with EVIDENCE carrying a status and a source,
scoped to a route fingerprint, failing closed when unknown.

Aura has already been bitten by the failure that motivates it. The
indefinite-coherence sawtooth was the context budget being sized against a
docstring that claimed ~8K while the real window was 32,768 — a 4x
underestimate that made her discard continuity to defend a budget she was
using about 2% of.

The registry now reads the model's real ``config.json``, which fixed that.
What it still cannot do is tell a caller *whether it actually found
anything*. ``get_model_context_window`` returns ``32768`` from three
different dead ends — an unreadable config, a non-Path model location, no
usable field anywhere — and that number is indistinguishable from a
32,768 that was measured. A guess wearing a measurement's clothes is
exactly the shape of the original defect, and the fix for it is not a
better guess.

So the size now travels with its provenance:

``MEASURED``
    ``max_position_embeddings`` from the model's own ``config.json``. The
    architectural limit, from the artifact that defines it.
``DERIVED``
    A sliding-window or tokenizer-advertised maximum. Real, but a step
    removed from the architecture — some tokenizers advertise windows that
    need rope scaling actually switched on.
``ASSUMED``
    Nothing was readable. The value is a default and nobody measured
    anything.

``ASSUMED`` records a degradation the first time it is hit for a given
model, so an unreadable artifact becomes visible instead of silently
sizing every prompt for the rest of the process's life. It does not raise:
refusing to answer would take the runtime down over a missing file, and
the number is survivable — being *unaware* of it is what was not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.runtime.errors import record_degradation

__all__ = [
    "WindowSource",
    "ContextWindowEvidence",
    "assumed",
    "measured",
    "derived",
    "note_assumption",
]


class WindowSource(StrEnum):
    MEASURED = "measured"
    DERIVED = "derived"
    ASSUMED = "assumed"


@dataclass(frozen=True)
class ContextWindowEvidence:
    """A context window with the provenance of the number attached."""

    tokens: int
    source: WindowSource
    detail: str = ""
    model: str = ""

    @property
    def is_measured(self) -> bool:
        """Whether this number came from the artifact rather than a default."""
        return self.source is not WindowSource.ASSUMED

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "source": str(self.source),
            "detail": self.detail,
            "model": self.model,
            "is_measured": self.is_measured,
        }

    def __int__(self) -> int:
        return int(self.tokens)


def measured(tokens: int, *, model: str = "", detail: str = "config.json") -> ContextWindowEvidence:
    return ContextWindowEvidence(int(tokens), WindowSource.MEASURED, detail, model)


def derived(tokens: int, *, model: str = "", detail: str = "") -> ContextWindowEvidence:
    return ContextWindowEvidence(int(tokens), WindowSource.DERIVED, detail, model)


def assumed(tokens: int, *, model: str = "", detail: str = "") -> ContextWindowEvidence:
    return ContextWindowEvidence(int(tokens), WindowSource.ASSUMED, detail, model)


#: Models already reported, so an unreadable artifact degrades once rather
#: than on every prompt assembly.
_REPORTED: set[str] = set()


def note_assumption(evidence: ContextWindowEvidence) -> ContextWindowEvidence:
    """Report an assumed window once per model, then stay quiet.

    Returns the evidence unchanged so callers can wrap in place.
    """
    if evidence.is_measured:
        return evidence
    key = evidence.model or "unknown"
    if key in _REPORTED:
        return evidence
    _REPORTED.add(key)
    record_degradation(
        "context_window_evidence",
        RuntimeError(
            f"context window for {key!r} could not be measured "
            f"({evidence.detail or 'no readable source'}); "
            f"assuming {evidence.tokens} tokens"
        ),
        severity="warning",
        action=(
            "sized the context budget from a default; the number is a guess and "
            "every prompt assembled for this model inherits it"
        ),
    )
    return evidence


def reset_for_test() -> None:
    _REPORTED.clear()
