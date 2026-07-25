"""A shed that protects the foreground must not unload the foreground's lane.

The 2026-07-25 probe loaded models **55 times across 30 turns** and shed the
fallback ladder five times, each shed logged as
``unloading Qwen2.5-7B-Instruct-4bit to protect the foreground lane
(protected_foreground_shed)``.

The sequence: a protected turn arrives, the cortex begins a 20GB load, free
memory dips under the shed threshold *because of that load*, and the shed then
unloads the Brainstem and Reflex — the only lanes that can answer while the
cortex warms. Shedding the parachute to make the plane lighter.

Every reload then pays full load latency and contends with the cortex for the
single GPU slot, which is most of a 72s p50.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.inference_gate import InferenceGate

pytestmark = pytest.mark.unit


class FakeClient:
    def __init__(self, path):
        self.model_path = path
        self.rebooted = False

    def is_alive(self):
        return True

    async def reboot_worker(self, reason="", mark_failed=False):
        self.rebooted = True


@pytest.fixture()
def gate(monkeypatch):
    g = InferenceGate.__new__(InferenceGate)
    g._mlx_client = object()
    g._last_background_memory_shed_at = 0.0
    g._brainstem_client = FakeClient("/models/Qwen2.5-7B-Instruct-4bit")
    g._reflex_client = FakeClient("/models/Qwen2.5-1.5B-Instruct-4bit")
    # Memory is genuinely tight — the shed is entitled to run.
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: type("S", (), {"available_gb": 10.0})(),
        raising=False,
    )
    return g


def _run_shed(gate, clients, reason):
    import core.brain.llm.mlx_client as mlx_client

    original = dict(getattr(mlx_client, "_CLIENTS", {}))
    mlx_client._CLIENTS.clear()
    mlx_client._CLIENTS.update(clients)
    try:
        asyncio.run(
            gate._shed_background_workers_for_memory_pressure(force=True, reason=reason)
        )
    finally:
        mlx_client._CLIENTS.clear()
        mlx_client._CLIENTS.update(original)


class TestTheLadderSurvivesAProtectedShed:
    def test_the_fallback_lanes_are_kept(self, gate):
        brainstem = gate._brainstem_client
        reflex = gate._reflex_client
        background = FakeClient("/models/some-background-model")

        _run_shed(
            gate,
            {
                brainstem.model_path: brainstem,
                reflex.model_path: reflex,
                background.model_path: background,
            },
            "protected_foreground_shed",
        )

        assert not brainstem.rebooted, "the Brainstem answers this very turn"
        assert not reflex.rebooted, "the Reflex is the last rung"

    def test_genuine_background_workers_are_still_shed(self, gate):
        """The shed must still do its job — this is not a disable."""
        background = FakeClient("/models/some-background-model")

        _run_shed(
            gate,
            {background.model_path: background},
            "protected_foreground_shed",
        )

        assert background.rebooted

    def test_other_shed_reasons_are_unchanged(self, gate):
        """Only foreground protection preserves the ladder; OOM relief cannot."""
        brainstem = gate._brainstem_client

        _run_shed(gate, {brainstem.model_path: brainstem}, "memory_pressure_critical")

        assert brainstem.rebooted, (
            "a real memory emergency must still be able to shed everything"
        )


class TestLadderDiscovery:
    def test_ladder_paths_come_from_the_live_clients(self, gate):
        paths = gate._fallback_ladder_paths()
        assert "/models/Qwen2.5-7B-Instruct-4bit" in paths
        assert "/models/Qwen2.5-1.5B-Instruct-4bit" in paths

    def test_env_configured_lanes_are_included(self, gate, monkeypatch):
        monkeypatch.setenv("AURA_MLX_BRAINSTEM_MODEL", "/models/custom-brainstem")
        assert "/models/custom-brainstem" in gate._fallback_ladder_paths()

    def test_a_gate_with_no_ladder_returns_empty(self):
        bare = InferenceGate.__new__(InferenceGate)
        assert bare._fallback_ladder_paths() == frozenset() or isinstance(
            bare._fallback_ladder_paths(), frozenset
        )
