from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.kernel.upgrades_10x import NativeMultimodalBridge
from core.runtime.connectivity import ConnectivityProbe, render_connectivity_prompt_block
from core.state.aura_state import AuraState


def test_connectivity_probe_supports_forced_offline_without_canned_reply(monkeypatch):
    monkeypatch.setenv("AURA_FORCE_OFFLINE", "1")
    probe = ConnectivityProbe(target="203.0.113.1", port=9, timeout_s=0.01, ttl_s=0)

    status = probe.status(force=True)
    block = render_connectivity_prompt_block(status)

    assert status.online is False
    assert status.mode == "forced_offline"
    assert "do not fabricate live sources" in block
    assert "no internet" not in block.lower()


def test_connectivity_prompt_block_marks_online_status():
    block = render_connectivity_prompt_block(
        {"online": True, "mode": "tcp_probe", "target": "1.1.1.1:53", "latency_ms": 12.3}
    )

    assert "External internet appears available" in block
    assert "governance" in block


@pytest.mark.asyncio
async def test_native_multimodal_bridge_binds_connectivity_to_state(monkeypatch):
    state = AuraState()
    monkeypatch.setattr(
        "core.runtime.connectivity.get_connectivity_status",
        lambda: SimpleNamespace(to_dict=lambda: {"online": False, "mode": "forced_offline", "target": "test"}),
    )
    bridge = NativeMultimodalBridge(SimpleNamespace(organs={}))

    await bridge.execute(state, "Perception")

    assert state.world.facts["connectivity"]["online"] is False
    assert state.response_modifiers["connectivity"]["mode"] == "forced_offline"
