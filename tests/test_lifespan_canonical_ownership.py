from __future__ import annotations

import asyncio

import pytest

from core.runtime import lifespan as lifespan_module
from core.runtime.shutdown_coordinator import (
    ShutdownCoordinator,
    clear_shutdown_request,
    is_shutdown_requested,
    request_shutdown,
)


@pytest.fixture(autouse=True)
def _clear_shutdown_latch() -> None:
    clear_shutdown_request()
    yield
    clear_shutdown_request()


@pytest.mark.asyncio
async def test_startup_uses_only_explicit_hooks_and_never_resolves_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = lifespan_module.LifespanManager()
    calls: list[str] = []

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("lifespan startup must not cold-resolve services")
        ),
    )
    manager.register_startup(lambda: calls.append("startup"))

    await manager.startup()

    assert calls == ["startup"]
    assert manager.running is True


@pytest.mark.asyncio
async def test_shutdown_latches_before_hook_and_replays_one_canonical_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = ShutdownCoordinator(phases=("actors",))
    manager = lifespan_module.LifespanManager()
    observations: list[bool] = []
    monkeypatch.setattr(lifespan_module, "get_shutdown_coordinator", lambda: coordinator)
    manager.register_shutdown(
        lambda: observations.append(is_shutdown_requested()),
        name="compatibility-hook",
    )

    first = await manager.shutdown(reason="first")
    second = await manager.shutdown(reason="second")

    assert first.clean is True
    assert second.clean is True
    assert observations == [True]
    assert second.repeated_call_count == 1
    assert manager.get_status()["last_shutdown_report"]["clean"] is True


@pytest.mark.asyncio
async def test_startup_refuses_without_invoking_hook_after_process_latch() -> None:
    manager = lifespan_module.LifespanManager()
    calls: list[str] = []
    manager.register_startup(lambda: calls.append("started"))
    request_shutdown("unit-test")

    with pytest.raises(RuntimeError, match="runtime_shutdown"):
        await manager.startup()

    assert calls == []
    assert manager.running is False


@pytest.mark.asyncio
async def test_container_stop_hook_delegates_to_canonical_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = ShutdownCoordinator(phases=("actors",))
    manager = lifespan_module.LifespanManager()
    calls: list[str] = []
    coordinator.register(lambda: calls.append("stopped"), phase="actors", name="owner")
    monkeypatch.setattr(lifespan_module, "get_shutdown_coordinator", lambda: coordinator)

    await manager.on_stop_async()
    await manager.on_stop_async()

    assert calls == ["stopped"]
    assert is_shutdown_requested() is True


def test_shutdown_registration_is_rejected_after_teardown_has_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = ShutdownCoordinator(phases=("actors",))
    manager = lifespan_module.LifespanManager()
    monkeypatch.setattr(lifespan_module, "get_shutdown_coordinator", lambda: coordinator)

    asyncio.run(coordinator.shutdown())

    with pytest.raises(RuntimeError, match="teardown has started"):
        manager.register_shutdown(lambda: None, name="late")
