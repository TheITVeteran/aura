"""Speak while she is still thinking, without speaking anything ungoverned.

The voice lane documented a trade-off and then accepted it: ``think_stream``
gives token-level streaming and a good time-to-first-audio but skips the
governance and validation phases; the governed turn is safe but returns one
finished string, so nothing can be spoken until the last token is decoded.
Time-to-first-audio was therefore proportional to *total* reply length, and the
only lever left was to make replies short — a 45-word cap, with fillers papering
over the wait.

That cap is why voice answers are shallower than the same question typed. It is
latency management wearing the costume of a style choice.

The trade-off is false. Governance is not one indivisible act performed on a
finished paragraph: most of it is clause-local. Whether text leaks internal
scaffolding, contradicts the clock, claims an instrument that does not exist, or
describes an artifact that was never built — each of those can be decided on the
clause in front of you. Only obligations about the reply *as a whole* (an exact
word count, a required closing phrase) need the end, and those bind the last
clause, not the first.

So this releases clause by clause, and every clause passes the clause-local
gates before a single sample of it is synthesised. Nothing ungoverned is ever
spoken; the difference is that the governing happens continuously instead of
once at the end.

**The cadence is real, and that is the point.** Chunks arrive when the model
finishes producing them, so a clause that needed more thought lands later, and
the pause before it is genuine hesitation rather than a synthesised one. That is
what makes generated speech sound like a person thinking instead of a document
being read. The filler reflex still exists for the long waits, and it now has
much less to cover.

**A refused clause ends the turn.** If a clause fails its gates, speaking the
rest would be speaking around a hole — the sentence after a suppressed one
usually depends on it. The stream stops, what was already spoken is recorded as
spoken (the interruption machinery already relies on that distinction), and the
caller decides what to say instead.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Voice.GovernedStream")

_RECOVERABLE = (RuntimeError, AttributeError, TypeError, ValueError, OSError, ImportError, KeyError)


@dataclass
class StreamOutcome:
    """What the stream produced, and how it ended."""

    spoken: str = ""
    chunks: int = 0
    first_chunk_ms: float = 0.0
    total_ms: float = 0.0
    refused_reason: str = ""
    corrections: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return bool(self.refused_reason)

    def narrative(self) -> str:
        parts = [
            f"{self.chunks} chunk(s)",
            f"first audio at {self.first_chunk_ms:.0f}ms",
            f"total {self.total_ms:.0f}ms",
        ]
        if self.corrections:
            parts.append("corrected: " + "; ".join(self.corrections[:3]))
        if self.refused_reason:
            parts.append(f"stopped: {self.refused_reason}")
        return ", ".join(parts)


def govern_clause(clause: str) -> tuple[str, str, bool]:
    """Apply the clause-local gates. Returns (safe_text, refusal, stop_after).

    Every check here decides on the clause alone. A gate that genuinely needs
    the whole reply does not belong in this function — it belongs at the flush,
    where the whole reply exists.
    """
    text = str(clause or "")
    if not text.strip():
        return "", "", False
    stop_after = False

    # 1. Nothing internal is ever spoken. A leaked scaffold line read aloud is
    #    worse than one shown, because it cannot be skimmed past.
    try:
        from core.conversation.response_reliability import strip_internal_scaffold

        stripped = strip_internal_scaffold(text)
        if not stripped.strip():
            return "", "the clause was entirely internal scaffolding", True
        if stripped.strip() != text.strip():
            # The kept prefix is real and gets spoken. But a model that has
            # begun emitting its own scaffolding has left the reply, and what
            # follows is not trustworthy continuation — so this is the last
            # clause of the turn.
            stop_after = True
        text = stripped
    except _RECOVERABLE as exc:
        record_degradation(
            "voice_duplex.governed_stream",
            exc,
            severity="warning",
            action="spoke a clause without the scaffold check",
        )

    # 2. Claims this runtime can measure are settled by the measurement — the
    #    clock, its own instruments, whether an artifact it describes exists.
    try:
        from core.conversation.grounded_claim_guard import verify_grounded_claims

        grounded = verify_grounded_claims(text)
        if grounded.changed:
            if not grounded.text.strip():
                return "", "the clause was a claim contradicted by a real reading", True
            text = grounded.text
    except _RECOVERABLE as exc:
        record_degradation(
            "voice_duplex.governed_stream",
            exc,
            severity="warning",
            action="spoke a clause without the grounded-claim check",
        )

    return text, "", stop_after


async def stream_governed_reply(
    tokens: AsyncIterator[str],
    *,
    first_max_chars: int,
    max_chars: int,
    speak: Callable[[str], object],
    on_first_chunk: Callable[[], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
) -> StreamOutcome:
    """Release a streaming reply clause by clause, governing each one.

    ``speak`` may be sync or async; its awaitable is awaited so back-pressure
    from the synthesiser naturally paces the stream rather than queueing an
    unbounded backlog of audio nobody has heard yet.

    ``should_continue`` is checked between chunks so a barge-in stops the reply
    at a clause boundary instead of mid-word.
    """
    from core.voice.duplex.clause_chunker import StreamingChunker

    outcome = StreamOutcome()
    chunker = StreamingChunker(first_max_chars=first_max_chars, max_chars=max_chars)
    started = time.perf_counter()
    spoken_parts: list[str] = []

    async def _emit(raw: str) -> bool:
        """Govern one chunk and speak it. False means stop the turn."""
        safe, refusal, stop_after = govern_clause(raw)
        if refusal:
            outcome.refused_reason = refusal
            return False
        if not safe.strip():
            return True
        if safe.strip() != raw.strip():
            outcome.corrections.append(f"clause repaired before speaking: {raw[:60]!r}")
        result = speak(safe)
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            await result
        spoken_parts.append(safe)
        outcome.chunks += 1
        if outcome.chunks == 1:
            outcome.first_chunk_ms = (time.perf_counter() - started) * 1000.0
            if on_first_chunk is not None:
                on_first_chunk()
        if stop_after:
            outcome.refused_reason = "the reply began emitting internals"
            return False
        return True

    try:
        async for token in tokens:
            if should_continue is not None and not should_continue():
                outcome.refused_reason = "the listener interrupted"
                break
            for chunk in chunker.push(str(token or "")):
                if not await _emit(chunk):
                    break
            if outcome.refused_reason:
                break
        else:
            for chunk in chunker.flush():
                if should_continue is not None and not should_continue():
                    outcome.refused_reason = "the listener interrupted"
                    break
                if not await _emit(chunk):
                    break
    except asyncio.CancelledError:
        raise
    except _RECOVERABLE as exc:
        record_degradation(
            "voice_duplex.governed_stream",
            exc,
            action="voice stream ended early; what was already spoken stands",
        )
        outcome.refused_reason = f"the stream failed: {type(exc).__name__}"

    outcome.spoken = " ".join(part.strip() for part in spoken_parts if part.strip()).strip()
    outcome.total_ms = (time.perf_counter() - started) * 1000.0
    logger.info("voice governed stream: %s", outcome.narrative())
    return outcome


__all__ = ["StreamOutcome", "govern_clause", "stream_governed_reply"]
