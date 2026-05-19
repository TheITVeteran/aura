import asyncio

import pytest

from interface import server


@pytest.mark.asyncio
async def test_ws_broadcaster_unsubscribes_when_shutdown_is_requested(monkeypatch):
    events = []

    class Bus:
        async def subscribe(self):
            events.append("subscribed")
            return asyncio.Queue()

        async def unsubscribe(self, queue):
            events.append(("unsubscribed", queue.empty()))

    monkeypatch.setattr(server, "broadcast_bus", Bus())
    monkeypatch.setattr(server, "is_shutdown_requested", lambda: True)

    await server._ws_broadcaster()

    assert events == ["subscribed", ("unsubscribed", True)]
