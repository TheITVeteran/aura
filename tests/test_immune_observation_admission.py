"""Heavy immune observation must not silently occupy the owner loop.

CP126 93ff2643: observe_error and observe_signature run PCA, clustering and
durable writes synchronously, and asynchronous runtime code calls them — the
same class as the five-second embedding load that hung the app on launch.

It cannot simply decline on the loop the way a retrieval can: the caller
needs the response, and dropping an immune observation to protect latency
trades a visible stall for an invisible blind spot. So the work happens, an
async path exists, and an on-loop call is recorded.
"""
from __future__ import annotations

import asyncio

import pytest

from core.adaptation import adaptive_immunity as mod
from core.adaptation.adaptive_immunity import get_adaptive_immune_system


@pytest.fixture()
def immune():
    return get_adaptive_immune_system()


def test_the_loop_probe_is_accurate():
    assert mod._on_event_loop() is False
    assert asyncio.run(_probe()) is True


async def _probe():
    return mod._on_event_loop()


def test_an_async_path_exists_for_both_observers(immune):
    assert callable(immune.observe_error_async)
    assert callable(immune.observe_signature_async)


def test_the_async_path_returns_the_same_shape(immune):
    sync = immune.observe_signature("memory", "ValueError")
    got = asyncio.run(immune.observe_signature_async("memory", "ValueError"))

    assert type(got) is type(sync)


def test_the_async_path_does_not_run_on_the_loop(immune, monkeypatch):
    """It offloads, so the observation must not see itself on the loop."""
    seen = []
    real = mod._on_event_loop
    monkeypatch.setattr(
        mod, "_on_event_loop", lambda: seen.append(real()) or real()
    )

    asyncio.run(immune.observe_error_async(ValueError("x"), {"component": "memory"}))

    assert seen and not any(seen)


def test_an_on_loop_sync_call_is_recorded(immune, monkeypatch):
    """The remaining offenders must be findable rather than silent."""
    recorded = []
    monkeypatch.setattr(
        mod, "_record_adaptive_immunity_degradation",
        lambda _exc, **kw: recorded.append(kw.get("action", "")),
    )
    monkeypatch.setattr(mod, "_on_event_loop", lambda: True)

    immune.observe_signature("memory", "ValueError")

    assert any("event loop" in action or "async" in action for action in recorded)


def test_an_off_loop_sync_call_is_not_flagged(immune, monkeypatch):
    recorded = []
    monkeypatch.setattr(
        mod, "_record_adaptive_immunity_degradation",
        lambda _exc, **kw: recorded.append(kw),
    )

    immune.observe_signature("memory", "ValueError")

    assert recorded == []


def test_the_observation_still_happens_on_the_loop(immune, monkeypatch):
    """Dropping it would trade a visible stall for an invisible blind spot."""
    monkeypatch.setattr(mod, "_on_event_loop", lambda: True)
    monkeypatch.setattr(
        mod, "_record_adaptive_immunity_degradation", lambda *a, **k: None
    )

    response = immune.observe_signature("memory", "ValueError")

    assert response is not None
    assert response.coverage_report is not None


def test_both_observers_share_one_body():
    import inspect

    for name in ("observe_error", "observe_signature"):
        source = inspect.getsource(getattr(mod.AdaptiveImmuneSystem, name))
        assert "self._observe_event(" in source
