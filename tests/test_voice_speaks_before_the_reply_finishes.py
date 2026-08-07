"""The latency claim, end to end, through the real session.

Every other test in this area checks a piece: that a channel carries chunks,
that a clause is governed, that a reconciliation detects a revision. None of
them would fail if the session went on blocking for the finished string —
which is exactly the state the code was in for months, with
`governed_stream.py` fully tested and never called.

So this test asserts the property that actually matters and that unit tests
cannot see: **audio goes out before cognition returns.** The responder here
publishes clauses and then sits on the finished reply for a long time. If the
session speaks during that gap, streaming is wired. If it waits, it is not,
however green everything else is.
"""
from __future__ import annotations

import asyncio

import pytest

from core.voice.duplex.mind_bridge import MindBridge
from core.voice.duplex.session import DuplexVoiceSession

# How long the responder sits on the finished reply after the last clause.
# Long enough that "spoke during the gap" cannot be a scheduling coincidence.
FINAL_DELAY_S = 0.6


def _session(responder, **kwargs):
    events: list[dict] = []
    audio: list[bytes] = []

    async def send_json(payload):
        events.append(payload)

    async def send_binary(payload):
        audio.append(payload)

    session = DuplexVoiceSession(
        session_id="stream-e2e",
        send_json=send_json,
        send_binary=send_binary,
        mind=MindBridge(session_id="stream-e2e", responder=responder),
        **kwargs,
    )
    return session, events, audio


async def _warm(session) -> None:
    """Load the synthesiser before the turn, exactly as production does.

    ``DuplexVoiceSession.start()`` warms every model precisely so that nobody
    pays cold start inside a live turn — Kokoro is 635 ms cold against 190 ms
    warm on this host, and the first load also initialises the phonemiser.
    A test that skips the warm-up measures model loading rather than the
    streaming path, and would report "audio arrived after cognition returned"
    for a reason that has nothing to do with what is being tested.

    Only the TTS is warmed here. `start()` also loads Whisper, which is 13–35 s
    and irrelevant to a typed turn.
    """
    await session._tts.warm_up(session._prosody_spec())


async def _with_a_listener(session, coro):
    """Run a turn with something on the other end that is actually playing.

    The session deliberately stays in SPEAKING until the audio has been
    *heard*, not merely sent — Kokoro synthesises several times faster than
    realtime, so returning to LISTENING at send-time would disarm barge-in for
    the rest of the reply and the user would talk over her with nothing
    stopping it. The client's reported playback position is what ends the
    wait; the wall clock is only the fallback for a client that went quiet.

    A test with no client falls to that fallback and waits out the full
    duration of every utterance in real time, which is not a hang and is not
    what production does. This stands in for the browser, reporting playback
    as fast as it arrives.
    """
    played = asyncio.Event()

    async def listener() -> None:
        while not played.is_set():
            track = session._speaking
            if track is not None:
                session._client_played_s = track.sent_duration_s
            await asyncio.sleep(0.01)

    task = asyncio.create_task(listener())
    try:
        return await coro
    finally:
        played.set()
        await task


def test_the_reply_is_consumed_while_cognition_is_still_running() -> None:
    """The property, stated as a fact about ordering rather than a race.

    The tempting version of this test starts a stopwatch and asserts that
    audio went out before the responder returned. That measures Kokoro. Real
    synthesis takes hundreds of milliseconds to seconds, so against a fake
    responder that finishes in under a second the assertion fails on a fast,
    correctly wired system — and the fix would be to pad the fake until it
    passed, which is a test tuned to agree with itself.

    What actually distinguishes "wired" from "not wired" is whether the
    surface had *already begun consuming the reply* at the moment cognition
    finished. Before this was wired that count was necessarily zero: nothing
    read the channel until the finished string came back. It is checked from
    inside the responder, at the last instant before it returns.
    """

    async def exercise() -> None:
        consumed_at_return: list[int] = []

        async def responder(
            transcript, *, effective_message, session_id, timeout_s, reply_stream=None
        ):
            for clause in (
                "Yeah, the short answer is yes. ",
                "The longer version is that it depends on the workload. ",
                "For a read-heavy one it is clearly worth it.",
            ):
                if reply_stream is not None:
                    reply_stream.publish(clause)
                await asyncio.sleep(0.02)
            # The model has said everything; the governed turn is still
            # finishing its whole-reply work. This gap is the latency that
            # used to be paid in full before the first syllable.
            await asyncio.sleep(FINAL_DELAY_S)
            if reply_stream is not None:
                consumed_at_return.append(reply_stream.stats.consumed)
            return (
                "Yeah, the short answer is yes. The longer version is that it "
                "depends on the workload. For a read-heavy one it is clearly "
                "worth it."
            )

        session, events, audio = _session(responder)
        await _warm(session)
        await session.handle_command({"command": "text", "text": "is it worth it?"})
        turn = session._turn_task
        assert turn is not None
        await _with_a_listener(session, asyncio.wait_for(turn, timeout=60.0))
        await session.close()

        assert consumed_at_return, "the responder never returned"
        assert consumed_at_return[0] > 0, (
            "cognition finished with the reply channel untouched — the surface "
            "was still waiting for the finished string, so the streaming path "
            "is not wired"
        )
        assert audio, "no audio was ever sent"

    asyncio.run(exercise())


def test_a_responder_that_does_not_stream_still_gets_spoken() -> None:
    """Most cognition paths return their reply in one piece.

    They must not become silent turns, and they must not report a synthesis
    failure for a stream that simply never happened.
    """

    async def exercise() -> None:
        async def responder(
            transcript, *, effective_message, session_id, timeout_s, reply_stream=None
        ):
            return "All at once, no streaming here."

        session, events, audio = _session(responder)
        await _warm(session)
        await session.handle_command({"command": "text", "text": "hello"})
        turn = session._turn_task
        assert turn is not None
        await _with_a_listener(session, asyncio.wait_for(turn, timeout=60.0))
        await session.close()

        replies = [e for e in events if e.get("type") == "voice.reply"]
        assert replies, "the finished reply was never spoken"
        assert replies[-1]["text"] == "All at once, no streaming here."
        errors = [e for e in events if e.get("type") == "voice.error"]
        assert not errors, f"a non-streaming turn reported an error: {errors}"

    asyncio.run(exercise())


def test_a_reply_revised_after_it_was_spoken_produces_a_correction() -> None:
    """Governance changing text the listener already heard is said out loud.

    Continuing as though they heard the corrected version is the failure the
    reconciliation exists to prevent.
    """

    async def exercise() -> None:
        async def responder(
            transcript, *, effective_message, session_id, timeout_s, reply_stream=None
        ):
            if reply_stream is not None:
                reply_stream.publish("The answer is definitely yes. ")
            await asyncio.sleep(0.05)
            # The whole-reply pass disagreed with what was already spoken.
            return "The answer is actually no."

        session, events, _audio = _session(responder)
        await _warm(session)
        await session.handle_command({"command": "text", "text": "well?"})
        turn = session._turn_task
        assert turn is not None
        await _with_a_listener(session, asyncio.wait_for(turn, timeout=60.0))
        await session.close()

        spoken = " ".join(
            str(e.get("text", "")) for e in events if e.get("type") == "voice.reply"
        )
        assert "actually no" in spoken, f"no correction was spoken: {spoken!r}"

    asyncio.run(exercise())


@pytest.mark.parametrize("clauses", [1, 3, 8])
def test_every_streamed_clause_reaches_the_listener(clauses: int) -> None:
    """Nothing is dropped between governance and the speaker."""

    async def exercise() -> None:
        # Written as words, because the synthesiser reads digits aloud as
        # words and the caption follows the audio. Asserting on "0" here
        # would be asserting that she mispronounces numbers.
        names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
        parts = [f"Clause {names[i]} here. " for i in range(clauses)]

        async def responder(
            transcript, *, effective_message, session_id, timeout_s, reply_stream=None
        ):
            for part in parts:
                if reply_stream is not None:
                    reply_stream.publish(part)
                await asyncio.sleep(0.01)
            return "".join(parts).strip()

        session, events, _audio = _session(responder)
        await _warm(session)
        await session.handle_command({"command": "text", "text": "go"})
        turn = session._turn_task
        assert turn is not None
        await _with_a_listener(session, asyncio.wait_for(turn, timeout=60.0))
        await session.close()

        heard = " ".join(
            str(e.get("text", "")) for e in events if e.get("type") == "voice.chunk"
        )
        for i in range(clauses):
            assert f"Clause {names[i]}" in heard, f"clause {i} never reached the listener"

    asyncio.run(exercise())
