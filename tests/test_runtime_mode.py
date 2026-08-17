"""Runtime mode helper regressions."""

from __future__ import annotations

from core.runtime.mode import MODE_MANIFESTS, is_simulation, mode_context


def test_is_simulation_accepts_simulated_mode(monkeypatch):
    monkeypatch.setenv("AURA_MODE", "simulated")

    assert is_simulation() is True


def test_is_simulation_rejects_non_simulated_mode(monkeypatch):
    monkeypatch.setenv("AURA_MODE", "live")

    assert is_simulation() is False


def test_runtime_modes_expose_no_cloud_model_capability(monkeypatch):
    for manifest in MODE_MANIFESTS.values():
        assert "allows_cloud" not in manifest

    monkeypatch.setenv("AURA_MODE", "production")
    assert "allows_cloud" not in mode_context()
