"""Runtime mode helper regressions."""

from __future__ import annotations

from core.runtime.mode import is_simulation


def test_is_simulation_accepts_simulated_mode(monkeypatch):
    monkeypatch.setenv("AURA_MODE", "simulated")

    assert is_simulation() is True


def test_is_simulation_rejects_non_simulated_mode(monkeypatch):
    monkeypatch.setenv("AURA_MODE", "live")

    assert is_simulation() is False
