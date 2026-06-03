import asyncio
import inspect

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


def test_websocket_timeout_path_does_not_direct_generate_raw_fallback():
    source = inspect.getsource(server.websocket_endpoint)

    assert "gate.generate(" not in source
    assert "instead of fabricating a recovered answer" in source


def test_websocket_chat_uses_desktop_cognitive_engine_trace_metadata():
    source = inspect.getsource(server.websocket_endpoint)

    assert 'origin="desktop-ui"' in source
    assert 'source="desktop_websocket"' in source
    assert "desktop WebSocket chat path requires CognitiveEngine" in source
