from __future__ import annotations

import os
import resource

import pytest

import aura_main
from core.container import ServiceContainer
from tools.memory_sentinel import should_kill_for_memory


@pytest.fixture(autouse=True)
def isolated_container():
    env_keys = (
        "AURA_MLX_MEMORY_LIMIT_GB",
        "AURA_PROCESS_RSS_LIMIT_GB",
    )
    previous_env = {key: os.environ.get(key) for key in env_keys}
    ServiceContainer.clear()
    yield
    ServiceContainer.clear()
    for key, value in previous_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


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


def test_memory_sentinel_default_ceiling_is_host_safe_on_64gb_node(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)

    ceiling = aura_main._bounded_memory_ceiling_mb(64 * 1024.0)

    assert ceiling == 45_875.2


def test_memory_sentinel_clamps_excessive_env_override(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)

    ceiling = aura_main._bounded_memory_ceiling_mb(64 * 1024.0, "120000")

    assert ceiling == 45_875.2


def test_memory_sentinel_allows_explicit_unsafe_override_only_when_opted_in(monkeypatch):
    monkeypatch.setenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", "1")

    ceiling = aura_main._bounded_memory_ceiling_mb(64 * 1024.0, "120000")

    assert ceiling == 120000.0


def test_memory_sentinel_malformed_override_falls_back_to_safe_ceiling(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)

    ceiling = aura_main._bounded_memory_ceiling_mb(64 * 1024.0, "not-a-number")

    assert ceiling == 45_875.2


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
    assert sentinel.get_status()["lethal_mb"] == pytest.approx(45_875.2)
    assert popen_calls
    assert "memory_sentinel.py" in " ".join(popen_calls[0])
    assert "--pid" in popen_calls[0]
    assert str(os.getpid()) in popen_calls[0]
    assert "--lethal-mb" in popen_calls[0]
    assert str(45_875.2) in popen_calls[0]


def test_disabled_external_memory_sentinel_is_never_reported_armed(monkeypatch):
    monkeypatch.setenv("AURA_MEMORY_SENTINEL", "0")
    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    monkeypatch.setattr(resource, "setrlimit", lambda _kind, _limits: None)

    aura_main._install_systemwide_memory_protection()

    sentinel = ServiceContainer.get("external_memory_sentinel")
    assert sentinel.is_armed() is False
    assert sentinel.get_status()["armed"] is False
