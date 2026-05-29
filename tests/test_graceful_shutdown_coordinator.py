import asyncio

import pytest


@pytest.mark.asyncio
async def test_graceful_shutdown_runs_canonical_shutdown_coordinator(monkeypatch):
    from core.graceful_shutdown import GracefulShutdown
    from core.runtime.shutdown_coordinator import (
        clear_shutdown_request,
        get_shutdown_coordinator,
        reset_shutdown_coordinator,
    )

    events: list[str] = []

    class _Container:
        async def shutdown(self):
            events.append("container")

    import core.container as container_module

    monkeypatch.setattr(container_module, "get_container", lambda: _Container())
    clear_shutdown_request()
    reset_shutdown_coordinator()
    GracefulShutdown._hooks = []
    GracefulShutdown._is_shutting_down = False
    GracefulShutdown._shutdown_event = asyncio.Event()

    get_shutdown_coordinator().register(
        lambda: events.append("coordinator"),
        phase="actors",
        name="test_handler",
        timeout=0.5,
    )

    await GracefulShutdown.trigger_shutdown("test")

    assert events == ["coordinator", "container"]
    assert GracefulShutdown._shutdown_event.is_set()

    GracefulShutdown._hooks = []
    GracefulShutdown._is_shutting_down = False
    GracefulShutdown._shutdown_event = None
    reset_shutdown_coordinator()
    clear_shutdown_request()
