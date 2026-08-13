"""A closed reply stream always releases its consumer under backpressure."""

from __future__ import annotations

import pytest

from core.conversation import reply_stream

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_full_queue_close_drains_without_waiting_for_lost_sentinel(monkeypatch) -> None:
    monkeypatch.setattr(reply_stream, "_MAX_PENDING_CHUNKS", 2)
    channel = reply_stream.ReplyStreamChannel()
    channel.publish("one")
    channel.publish("two")
    channel.close()

    chunks = [chunk async for chunk in channel.drain(timeout_s=0.05)]

    assert chunks == ["one", "two"]
    assert channel.closed


@pytest.mark.asyncio
async def test_close_remains_idempotent_after_backpressure() -> None:
    channel = reply_stream.ReplyStreamChannel()
    channel.close()
    channel.close()
    assert [chunk async for chunk in channel.drain(timeout_s=0.05)] == []
