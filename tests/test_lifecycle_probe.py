from __future__ import annotations

import asyncio

import pytest

from core.runtime import lifecycle_probe


def test_probe_is_disabled_without_explicit_enable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_SHUTDOWN_PROBE_TARGET", "container")
    monkeypatch.setenv("AURA_SHUTDOWN_PROBE_HOLD_SECONDS", "1")

    assert lifecycle_probe.shutdown_probe_hold_seconds("container") == 0.0


def test_probe_target_is_exact_and_hold_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_SHUTDOWN_PROBE_ENABLED", "1")
    monkeypatch.setenv("AURA_SHUTDOWN_PROBE_TARGET", "coordinator:state_vault")
    monkeypatch.setenv("AURA_SHUTDOWN_PROBE_HOLD_SECONDS", "99")

    assert lifecycle_probe.shutdown_probe_hold_seconds("container") == 0.0
    assert (
        lifecycle_probe.shutdown_probe_hold_seconds("coordinator:state_vault")
        == 2.0
    )


def test_async_probe_returns_after_configured_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_SHUTDOWN_PROBE_ENABLED", "1")
    monkeypatch.setenv("AURA_SHUTDOWN_PROBE_TARGET", "container")
    monkeypatch.setenv("AURA_SHUTDOWN_PROBE_HOLD_SECONDS", "0.05")

    assert asyncio.run(lifecycle_probe.hold_shutdown_probe_async("container")) == 0.05
