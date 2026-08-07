"""The reply as it forms, and the failures she has to explain.

Two modules, one property between them: neither may ever put words in her
mouth or take words out of her mouth without saying so.

``reply_stream`` exists so a surface can start speaking before the turn is
finished. The danger it introduces is obvious once stated — the text spoken
early is pre-stabilisation, so it can disagree with the answer the turn
finally stands behind. Most of these tests are about that disagreement being
detected and surfaced rather than smoothed over.

``failure_context`` exists so that a capability failing produces *facts* for
her to narrate rather than a sentence written months in advance. Its tests
are largely about what it refuses to do: leak across turns, invent readings,
or emit anything that reads like dialogue.
"""
from __future__ import annotations

import asyncio

import pytest

from core.conversation.failure_context import (
    CapabilityFailure,
    bind_failure_ledger,
    pending_failure_context,
    record_capability_failure,
    render_failure_block,
)
from core.conversation.reply_stream import (
    ReplyStreamChannel,
    active_reply_stream,
    bind_reply_stream,
    publish_reply_chunk,
    reconcile,
)

# ── the channel ──────────────────────────────────────────────────────────


def test_chunks_reach_the_bound_channel_and_nowhere_else() -> None:
    """The binding is per-async-context, which is the whole safety argument.

    The telemetry topic these chunks ride on is global. If a surface could
    receive another surface's reply, a voice lane would occasionally speak
    the desktop's answer — a far worse defect than the latency this fixes.
    """

    async def exercise() -> None:
        channel = ReplyStreamChannel()
        publish_reply_chunk("this one has nowhere to go")
        assert channel.stats.published == 0

        with bind_reply_stream(channel):
            assert active_reply_stream() is channel
            publish_reply_chunk("hello")
            publish_reply_chunk(" there")
        assert active_reply_stream() is None

        received = [chunk async for chunk in channel]
        assert received == ["hello", " there"]

    asyncio.run(exercise())


def test_a_sibling_task_cannot_reach_this_turns_channel() -> None:
    """Context is copied into a task at creation, never shared sideways."""

    async def exercise() -> None:
        mine = ReplyStreamChannel()
        theirs = ReplyStreamChannel()

        async def other_turn() -> None:
            with bind_reply_stream(theirs):
                publish_reply_chunk("theirs")

        with bind_reply_stream(mine):
            publish_reply_chunk("mine")
            await asyncio.create_task(other_turn())

        assert [c async for c in mine] == ["mine"]
        assert [c async for c in theirs] == ["theirs"]

    asyncio.run(exercise())


def test_the_channel_always_closes_even_when_the_turn_raises() -> None:
    """A consumer waiting on a producer that died is a hang, not an error."""

    async def exercise() -> None:
        channel = ReplyStreamChannel()
        with pytest.raises(RuntimeError):
            with bind_reply_stream(channel):
                publish_reply_chunk("partial")
                raise RuntimeError("cognition blew up")
        assert channel.closed
        assert [c async for c in channel] == ["partial"]

    asyncio.run(exercise())


def test_a_stalled_consumer_loses_chunks_rather_than_slowing_cognition() -> None:
    """Publishing must never apply back-pressure to the mind producing it.

    The stream is an accelerator. The finished reply still arrives by the
    ordinary path, so dropping chunks costs latency; blocking the turn to
    preserve them would cost the answer.
    """

    async def exercise() -> None:
        channel = ReplyStreamChannel()
        with bind_reply_stream(channel):
            for i in range(4096):
                publish_reply_chunk(f"chunk-{i}")
        assert channel.stats.dropped > 0
        assert not channel.stats.lossless

    asyncio.run(exercise())


def test_drain_gives_up_rather_than_waiting_forever() -> None:
    """A producer that wedges mid-reply must not wedge the surface."""

    async def exercise() -> None:
        channel = ReplyStreamChannel()
        channel.attach_loop(asyncio.get_running_loop())
        channel.publish("first")
        got = [chunk async for chunk in channel.drain(timeout_s=0.05)]
        assert got == ["first"]
        assert not channel.closed  # it timed out; it did not end

    asyncio.run(exercise())


# ── reconciliation ───────────────────────────────────────────────────────


def test_a_prefix_leaves_only_the_remainder_to_say() -> None:
    verdict = reconcile("The short answer is yes.", "The short answer is yes. Here is why.")
    assert verdict.consistent
    assert verdict.remainder == "Here is why."


def test_whitespace_and_case_are_not_divergence() -> None:
    """The stabiliser re-flows text; nobody heard the difference."""
    assert reconcile("the answer  is\nyes", "The answer is yes, mostly.").consistent


def test_a_changed_word_is_divergence_and_she_is_told() -> None:
    """The listener is holding a sentence she no longer stands behind.

    Silence is not an option here: continuing as though they heard the
    corrected version is the exact failure the reconciliation exists for.
    """
    verdict = reconcile("The answer is yes.", "The answer is no.")
    assert verdict.diverged
    assert "The answer is yes." in verdict.correction_context
    assert "The answer is no." in verdict.correction_context
    # It hands her the situation, not a script.
    assert "in your own words" in verdict.correction_context


def test_speaking_then_returning_nothing_is_divergence_not_success() -> None:
    verdict = reconcile("I already said this out loud.", "")
    assert verdict.diverged
    assert "did not survive" in verdict.correction_context


def test_an_empty_stream_is_the_ordinary_non_streaming_turn() -> None:
    verdict = reconcile("", "The whole reply, produced at once.")
    assert verdict.empty
    assert verdict.remainder == "The whole reply, produced at once."


# ── failure context ──────────────────────────────────────────────────────


def test_failures_only_collect_inside_a_bound_turn() -> None:
    """A background task's failure belongs to nobody's reply.

    Injecting it into the next unrelated turn would have her explain
    something the user never asked about, which is worse than dropping it.
    """
    assert record_capability_failure("web", intent="search", cause="offline") is None
    with bind_failure_ledger() as ledger:
        assert record_capability_failure("web", intent="search", cause="offline") is not None
        assert len(ledger.records) == 1


def test_the_ledger_does_not_leak_between_turns() -> None:
    with bind_failure_ledger():
        record_capability_failure("web", intent="search", cause="offline")
    with bind_failure_ledger():
        assert pending_failure_context() == ""


def test_an_unknown_cause_is_recorded_as_failed_not_invented() -> None:
    failure = CapabilityFailure(
        capability="x", intent="do a thing", cause="banana-flavoured"
    )
    assert failure.cause == "failed"


def test_the_block_carries_readings_and_never_a_sentence_to_say() -> None:
    """The division of labour this whole module exists for.

    The runtime supplies facts; she supplies words. A field that starts
    reading like dialogue is a canned response with extra steps.
    """
    block = render_failure_block(
        [
            CapabilityFailure(
                capability="media_playback",
                intent="play Kind of Blue",
                cause="offline",
                detail="probe to 1.1.1.1:53 failing for 240s",
                still_possible=("903 local files are playable",),
            )
        ]
    )
    assert "probe to 1.1.1.1:53 failing for 240s" in block
    assert "903 local files are playable" in block
    assert "in your own words" in block
    # No phrasing she is expected to repeat.
    for canned in ("I'm sorry", "I am unable", "I can't help", "Unfortunately"):
        assert canned.lower() not in block.lower()


def test_what_still_works_travels_with_what_broke() -> None:
    """A failure report listing only the failure invites over-generalising.

    "I'm offline" when one host is unreachable is a bigger claim than the
    evidence supports, and it is the claim a model will reach for if the
    remaining capability is not in front of it.
    """
    with bind_failure_ledger():
        record_capability_failure(
            "web_search",
            intent="look up the train times",
            cause="offline",
            still_possible=("everything already in memory", "local files"),
        )
        block = pending_failure_context()
    assert "still works" in block
    assert "local files" in block


def test_records_are_bounded_so_a_prompt_cannot_become_a_log_dump() -> None:
    with bind_failure_ledger() as ledger:
        for i in range(50):
            record_capability_failure(f"cap{i}", intent="x", cause="failed")
        assert len(ledger.records) <= 6
        # The most recent are what explain what just happened.
        assert ledger.records[-1].capability == "cap49"
