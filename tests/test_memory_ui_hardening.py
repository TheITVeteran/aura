from __future__ import annotations

from types import SimpleNamespace

import pytest

from interface import memory_ui


class _EmptyEpisodic:
    async def recall_recent_async(self, *, limit: int):
        return []


class _FailingHorcrux:
    def __init__(self) -> None:
        self.checked = False

    async def check_shards(self):
        self.checked = True
        raise RuntimeError("shard probe failed")


class _FailingMemoryVault:
    def __init__(self) -> None:
        self.read_attempted = False

    @property
    def memories(self):
        self.read_attempted = True
        raise RuntimeError("memory store unavailable")


@pytest.mark.asyncio
async def test_memory_ui_reports_degraded_when_horcrux_probe_fails(monkeypatch):
    horcrux = _FailingHorcrux()
    vault = SimpleNamespace(memories=[], horcrux=horcrux)
    facade = SimpleNamespace(vector=vault, episodic=_EmptyEpisodic(), setup=lambda: None)

    monkeypatch.setattr(
        memory_ui,
        "get_runtime_service",
        lambda _name, default=None: facade,
    )

    result = await memory_ui.get_vault_stats()

    assert result["status"] == "degraded"
    assert result["degraded"] is True
    assert "Horcrux shard check failed" in result["degradation_reasons"]
    assert result["horcrux"] == "UNKNOWN"
    assert horcrux.checked is True


@pytest.mark.asyncio
async def test_memory_ui_reports_degraded_when_memory_read_fails(monkeypatch):
    vault = _FailingMemoryVault()
    facade = SimpleNamespace(vector=vault, episodic=_EmptyEpisodic(), setup=lambda: None)

    monkeypatch.setattr(
        memory_ui,
        "get_runtime_service",
        lambda _name, default=None: facade,
    )

    result = await memory_ui.get_vault_stats()

    assert result["status"] == "degraded"
    assert result["degraded"] is True
    assert "Memory vault read failed" in result["degradation_reasons"]
    assert result["total_nodes"] == 0
    assert vault.read_attempted is True
