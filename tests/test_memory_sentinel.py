from __future__ import annotations

import os
import resource

import pytest

import aura_main
from core.container import ServiceContainer
from tools.memory_sentinel import should_kill_for_memory


@pytest.fixture(autouse=True)
def isolated_container():
    ServiceContainer.clear()
    yield
    ServiceContainer.clear()


def test_memory_sentinel_waits_for_normal_lethal_confirmation():
    assert (
        should_kill_for_memory(
            managed_mb=46_500.0,
            lethal_mb=46_000.0,
            consecutive_over=1,
        )
        is False
    )
    assert (
        should_kill_for_memory(
            managed_mb=46_500.0,
            lethal_mb=46_000.0,
            consecutive_over=2,
        )
        is True
    )


def test_memory_sentinel_kills_large_overshoot_immediately():
    assert (
        should_kill_for_memory(
            managed_mb=54_000.0,
            lethal_mb=46_000.0,
            consecutive_over=1,
        )
        is True
    )


def test_desktop_boot_memory_protection_registers_armed_external_sentinel(monkeypatch):
    popen_calls: list[list[str]] = []

    class FakePopen:
        pid = os.getpid()

        def __init__(self, args, **kwargs):
            popen_calls.append([str(arg) for arg in args])
            stdout = kwargs.get("stdout")
            if stdout is not None:
                stdout.close()

        def poll(self):
            return None

    monkeypatch.setenv("AURA_MEMORY_SENTINEL", "1")
    monkeypatch.setenv("AURA_MEMWATCH_LETHAL_MB", "46080")
    monkeypatch.setenv("AURA_MEMORY_SENTINEL_INTERVAL_S", "0.5")
    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    monkeypatch.setattr(resource, "setrlimit", lambda _kind, _limits: None)
    monkeypatch.setattr(aura_main.subprocess, "Popen", FakePopen)

    aura_main._install_systemwide_memory_protection()

    sentinel = ServiceContainer.get("external_memory_sentinel")
    assert sentinel.is_armed() is True
    assert sentinel.get_status()["armed"] is True
    assert sentinel.get_status()["lethal_mb"] == 46080.0
    assert popen_calls
    assert "memory_sentinel.py" in " ".join(popen_calls[0])
    assert "--pid" in popen_calls[0]
    assert str(os.getpid()) in popen_calls[0]


def test_disabled_external_memory_sentinel_is_never_reported_armed(monkeypatch):
    monkeypatch.setenv("AURA_MEMORY_SENTINEL", "0")
    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    monkeypatch.setattr(resource, "setrlimit", lambda _kind, _limits: None)

    aura_main._install_systemwide_memory_protection()

    sentinel = ServiceContainer.get("external_memory_sentinel")
    assert sentinel.is_armed() is False
    assert sentinel.get_status()["armed"] is False
