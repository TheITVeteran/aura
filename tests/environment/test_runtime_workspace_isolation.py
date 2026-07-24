"""Contract: environment runtime state never touches the live store from tests.

The learning sidecars under ``~/.aura/data/environment_runtime`` are the live
organism's memory. These tests pin the two-layer isolation: the explicit
``AURA_ENV_RUNTIME_DIR`` override always wins, and under pytest with no
override the resolver redirects to a per-process temporary workspace instead
of the live data directory — including for calls made before any fixture ran.
"""
from __future__ import annotations

from pathlib import Path

from core.environment import runtime_workspace
from core.environment.runtime_workspace import environment_runtime_dir


def test_explicit_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_ENV_RUNTIME_DIR", str(tmp_path / "explicit"))
    resolved = environment_runtime_dir("terminal_grid:nethack", purpose="learning")
    assert str(resolved).startswith(str((tmp_path / "explicit").resolve()))
    assert resolved.is_dir()


def test_pytest_without_override_never_resolves_into_live_data_dir(monkeypatch):
    monkeypatch.delenv("AURA_ENV_RUNTIME_DIR", raising=False)
    from core.config import config

    live_root = Path(config.paths.data_dir) / "environment_runtime"
    resolved = environment_runtime_dir("terminal_grid:nethack", purpose="learning")
    assert not str(resolved).startswith(str(live_root)), (
        "test-process resolution reached the live organism's learning store"
    )
    assert resolved.is_dir()


def test_isolation_root_is_stable_within_the_process(monkeypatch):
    monkeypatch.delenv("AURA_ENV_RUNTIME_DIR", raising=False)
    first = environment_runtime_dir("env_a", purpose="learning")
    second = environment_runtime_dir("env_b", purpose="learning")
    # Same per-process root, distinct per-environment subtrees.
    assert first.parent.parent == second.parent.parent
    assert first != second


def test_guard_detects_pytest_via_sys_modules():
    # The guard must be active in exactly this situation: running under
    # pytest with no explicit override.
    assert runtime_workspace._test_isolation_root() is not None
