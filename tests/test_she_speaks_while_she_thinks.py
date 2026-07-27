"""Governed and streamed, which the lane had treated as a choice.

The voice lane documented the trade-off and accepted it: token streaming skips
the governance phases, and the governed turn returns one finished string, so
nothing could be spoken until the last token decoded. Time-to-first-audio was
proportional to *total* reply length, which left exactly one lever — make the
replies short. Hence a 45-word cap, and voice answers noticeably shallower than
the same question typed.

The trade-off is false. Governance is not one indivisible act performed on a
finished paragraph; most of it decides on the clause in front of you. Whether
text leaks internal scaffolding, contradicts the clock, claims an instrument
that does not exist — all clause-local. Only obligations about the reply as a
whole need the end, and those bind the last clause, not the first.

So clauses are released as they are produced, each one governed before a single
sample of it is synthesised. Nothing ungoverned is ever spoken; the governing
happens continuously instead of once.

The cadence this produces is real, which is the point: a clause that needed more
thought lands later, so the pause before it is genuine hesitation rather than a
synthesised one.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from core.voice.duplex.governed_stream import (
    govern_clause,
    stream_governed_reply,
)


async def _tokens(*parts: str, delay: float = 0.0) -> AsyncIterator[str]:
    for part in parts:
        if delay:
            await asyncio.sleep(delay)
        yield part


def _collect():
    spoken: list[str] = []

    async def speak(text: str) -> None:
        spoken.append(text)

    return spoken, speak


# ── Clause-local governance ────────────────────────────────────────────────

def test_a_scaffold_clause_is_refused() -> None:
    """A leaked prompt line read aloud is worse than one shown — it cannot be
    skimmed past."""
    safe, refusal, stop_after = govern_clause("[SKILL RESULT: web_search]")
    assert not safe
    assert "internal scaffolding" in refusal
    assert stop_after


def test_ordinary_speech_passes_untouched() -> None:
    safe, refusal, stop_after = govern_clause("I think that's roughly right.")
    assert safe == "I think that's roughly right."
    assert not refusal
    assert not stop_after


def test_a_clause_mixing_answer_and_scaffold_keeps_the_answer() -> None:
    """The real prefix is spoken; the turn ends there.

    A model that has begun emitting its own scaffolding has left the reply, so
    what follows is not trustworthy continuation — the kept words are said and
    nothing after them is.
    """
    safe, refusal, stop_after = govern_clause(
        "I'll go with curiosity. [SKILL EXECUTION] narrate it"
    )
    assert safe == "I'll go with curiosity."
    assert not refusal
    assert stop_after


def test_blank_is_not_a_refusal() -> None:
    assert govern_clause("   ") == ("", "", False)


# ── Streaming ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_clauses_are_spoken_as_they_arrive() -> None:
    spoken, speak = _collect()
    outcome = await stream_governed_reply(
        _tokens(
            "The second train catches up at five fifteen. ",
            "They're both a hundred and eighty miles out. ",
            "Want the working?",
        ),
        first_max_chars=40,
        max_chars=80,
        speak=speak,
    )
    assert outcome.chunks >= 2
    assert not outcome.refused
    assert "five fifteen" in outcome.spoken


@pytest.mark.asyncio
async def test_the_first_clause_does_not_wait_for_the_last() -> None:
    """The whole point: time-to-first-audio stops tracking total length."""
    spoken, speak = _collect()
    first_at: list[float] = []
    started = time.perf_counter()

    async def mark() -> None:
        first_at.append(time.perf_counter() - started)

    def on_first() -> None:
        first_at.append(time.perf_counter() - started)

    await stream_governed_reply(
        _tokens(
            "Here is the first thing I want to say about it. ",
            *[f"And another clause number {i} to make this long. " for i in range(8)],
            delay=0.01,
        ),
        first_max_chars=40,
        max_chars=80,
        speak=speak,
        on_first_chunk=on_first,
    )
    total = time.perf_counter() - started
    assert first_at, "no first chunk was reported"
    assert first_at[0] < total / 2, "the first clause waited for the rest of the reply"


@pytest.mark.asyncio
async def test_a_refused_clause_stops_the_turn() -> None:
    """Speaking on past a hole would be speaking around a missing sentence."""
    spoken, speak = _collect()
    outcome = await stream_governed_reply(
        _tokens(
            "Here is the honest part of the answer. ",
            "[SKILL RESULT: internals] ",
            "And this should never be spoken. ",
        ),
        first_max_chars=30,
        max_chars=60,
        speak=speak,
    )
    assert outcome.refused
    assert "should never be spoken" not in outcome.spoken
    assert "honest part" in outcome.spoken
    assert "internals" in outcome.refused_reason or "scaffolding" in outcome.refused_reason


@pytest.mark.asyncio
async def test_what_was_already_spoken_is_reported() -> None:
    """Interruption bookkeeping depends on knowing what she actually said."""
    spoken, speak = _collect()
    outcome = await stream_governed_reply(
        _tokens("First clause here. ", "[SKILL EXECUTION] leak "),
        first_max_chars=20,
        max_chars=40,
        speak=speak,
    )
    assert outcome.spoken == " ".join(spoken).strip()


@pytest.mark.asyncio
async def test_a_barge_in_stops_at_a_clause_boundary() -> None:
    spoken, speak = _collect()
    allow = {"go": True}

    async def speak_then_interrupt(text: str) -> None:
        spoken.append(text)
        allow["go"] = False

    outcome = await stream_governed_reply(
        _tokens(*[f"Clause number {i} of the reply. " for i in range(6)]),
        first_max_chars=25,
        max_chars=40,
        speak=speak_then_interrupt,
        should_continue=lambda: allow["go"],
    )
    assert outcome.refused_reason == "the listener interrupted"
    assert len(spoken) == 1, "speech continued past the interruption"


@pytest.mark.asyncio
async def test_synthesiser_back_pressure_paces_the_stream() -> None:
    """Awaiting speak() is what stops an unbounded backlog of unheard audio."""
    order: list[str] = []

    async def slow_speak(text: str) -> None:
        order.append(f"start:{text[:8]}")
        await asyncio.sleep(0.01)
        order.append(f"end:{text[:8]}")

    await stream_governed_reply(
        _tokens("First clause of it. ", "Second clause of it. ", "Third clause here. "),
        first_max_chars=18,
        max_chars=18,
        speak=slow_speak,
    )
    # Every start is followed by its own end before the next start.
    for index in range(0, len(order) - 1, 2):
        assert order[index].startswith("start:")
        assert order[index + 1].startswith("end:")


@pytest.mark.asyncio
async def test_a_failing_stream_keeps_what_was_spoken() -> None:
    async def broken() -> AsyncIterator[str]:
        yield "A real first clause that stands. "
        raise RuntimeError("the model lane died")

    spoken, speak = _collect()
    outcome = await stream_governed_reply(
        broken(), first_max_chars=20, max_chars=40, speak=speak
    )
    assert outcome.refused
    assert "the stream failed" in outcome.refused_reason
    assert "real first clause" in outcome.spoken


@pytest.mark.asyncio
async def test_an_empty_stream_is_not_a_failure() -> None:
    spoken, speak = _collect()
    outcome = await stream_governed_reply(
        _tokens(), first_max_chars=20, max_chars=40, speak=speak
    )
    assert not outcome.refused
    assert outcome.chunks == 0
    assert outcome.spoken == ""


@pytest.mark.asyncio
async def test_a_sync_speak_callback_is_accepted() -> None:
    spoken: list[str] = []
    outcome = await stream_governed_reply(
        _tokens("One clause, spoken plainly. "),
        first_max_chars=20,
        max_chars=40,
        speak=spoken.append,
    )
    assert outcome.chunks >= 1
    assert spoken


@pytest.mark.asyncio
async def test_the_outcome_reads_as_one_line() -> None:
    spoken, speak = _collect()
    outcome = await stream_governed_reply(
        _tokens("A clause worth saying out loud. "),
        first_max_chars=20,
        max_chars=40,
        speak=speak,
    )
    line = outcome.narrative()
    assert "chunk(s)" in line and "first audio at" in line
