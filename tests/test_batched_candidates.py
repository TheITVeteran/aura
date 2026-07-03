"""Batched best-of-N candidate lane — resolution, fallback, and wiring.

The lane only ever makes best-of-N cheaper: every miss (disabled, no live
client, worker error, timeout) returns None/[] and the amplifier keeps its
serial sampling path.
"""
from __future__ import annotations

import asyncio

import pytest

import core.brain.llm.batch_candidates as bc


class _Client:
    def __init__(self, model_path: str, alive: bool = True, texts=None):
        self.model_path = model_path
        self._alive = alive
        self.calls: list[dict] = []
        self._texts = texts if texts is not None else ["a", "b", "c"]

    def is_alive(self):
        return self._alive

    async def generate_batch_async(self, prompt, *, n, max_tokens, temperature, timeout_s):
        self.calls.append({"prompt": prompt, "n": n})
        return self._texts


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("AURA_BATCHED_CANDIDATES", "0")
    assert asyncio.run(bc.generate_candidates_batched("p", 4)) is None


def test_single_sample_never_batches(monkeypatch):
    monkeypatch.setenv("AURA_BATCHED_CANDIDATES", "1")
    assert asyncio.run(bc.generate_candidates_batched("p", 1)) is None


def test_prefers_heavy_lane_client(monkeypatch):
    monkeypatch.setenv("AURA_BATCHED_CANDIDATES", "1")
    light = _Client("/models/Qwen2.5-1.5B-Instruct-4bit")
    heavy = _Client("/models/Qwen2.5-32B-Instruct-4bit")
    monkeypatch.setattr(
        "core.brain.llm.mlx_client._CLIENTS", {"l": light, "h": heavy}
    )
    out = asyncio.run(bc.generate_candidates_batched("p", 4))
    assert out == ["a", "b", "c"]
    assert heavy.calls and not light.calls


def test_dead_clients_yield_serial_fallback(monkeypatch):
    monkeypatch.setenv("AURA_BATCHED_CANDIDATES", "1")
    dead = _Client("/models/Qwen2.5-32B-Instruct-4bit", alive=False)
    monkeypatch.setattr("core.brain.llm.mlx_client._CLIENTS", {"d": dead})
    assert asyncio.run(bc.generate_candidates_batched("p", 4)) is None


def test_empty_batch_yields_serial_fallback(monkeypatch):
    monkeypatch.setenv("AURA_BATCHED_CANDIDATES", "1")
    empty = _Client("/models/Qwen2.5-32B-Instruct-4bit", texts=[])
    monkeypatch.setattr("core.brain.llm.mlx_client._CLIENTS", {"e": empty})
    assert asyncio.run(bc.generate_candidates_batched("p", 4)) is None


def test_amplifier_uses_batched_lane_before_serial(monkeypatch):
    """_generate_candidates returns the batched pool when the lane delivers."""
    import time as _time

    from core.brain.reasoning_amplifier_v2 import (
        ProblemRepresentation,
        build_amplifier_v2,
    )

    serial_calls: list[int] = []

    async def fake_generate(prompt, temperature):
        serial_calls.append(1)
        return "serial"

    amp = build_amplifier_v2(fake_generate)

    async def fake_batched(prompt, n, *, timeout_s):
        return [f"candidate-{i}" for i in range(n)]

    monkeypatch.setattr(
        "core.brain.llm.batch_candidates.generate_candidates_batched", fake_batched
    )

    problem = ProblemRepresentation(objective="compute 2+2", task_type="math")
    out = asyncio.run(
        amp._generate_candidates(problem, "", 4, _time.monotonic() + 30.0)
    )

    assert out == ["candidate-0", "candidate-1", "candidate-2", "candidate-3"]
    assert not serial_calls, "batched hit must not invoke serial sampling"


def test_worker_source_contract():
    import inspect

    import core.brain.llm.mlx_worker as worker

    src = inspect.getsource(worker)
    assert 'elif action == "generate_batch":' in src
    assert "batch_generate(" in src
    # Raw candidates by design: no sentinel or quality gates in the batch branch.
    assert "RAW (no sentinel/quality gates)" in src
