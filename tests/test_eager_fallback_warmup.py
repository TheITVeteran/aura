"""Eager fallback warmup — the brainstem is resident before it is needed.

Live 2026-07-15 soak: once the cortex dropped under sustained load, the demand-
loaded brainstem cold-load was aborted at the squeezed per-turn budget and never
became resident, so 22% of turns returned no reply (canonical_chat_no_reply).
Warming the brainstem at boot guarantees the conversation path always has a fast
answer tier to fall to.
"""
from __future__ import annotations

import asyncio

import pytest

import core.brain.inference_gate as ig_mod
from core.brain.inference_gate import InferenceGate


class _FakeBrainstem:
    def __init__(self, alive: bool = False) -> None:
        self._alive = alive
        self.warmup_calls = 0

    def is_alive(self) -> bool:
        return self._alive

    async def warmup(self) -> None:
        self.warmup_calls += 1
        self._alive = True


def _bare_gate() -> InferenceGate:
    gate = InferenceGate.__new__(InferenceGate)
    gate._eager_fallback_task = None
    return gate


def _patch_client(monkeypatch, fake: _FakeBrainstem) -> None:
    import core.brain.llm.mlx_client as mlx_mod
    import core.brain.llm.model_registry as reg_mod

    monkeypatch.setattr(mlx_mod, "get_mlx_client", lambda *a, **k: fake, raising=False)
    monkeypatch.setattr(reg_mod, "get_brainstem_path", lambda: "/tmp/brainstem", raising=False)


@pytest.mark.asyncio
async def test_eager_warmup_makes_brainstem_resident(monkeypatch):
    monkeypatch.setenv("AURA_EAGER_FALLBACK_WARMUP", "1")
    fake = _FakeBrainstem(alive=False)
    _patch_client(monkeypatch, fake)
    gate = _bare_gate()
    gate._schedule_eager_fallback_warmup(delay=0.0)
    assert gate._eager_fallback_task is not None
    await gate._eager_fallback_task
    assert fake.warmup_calls == 1


@pytest.mark.asyncio
async def test_eager_warmup_skips_when_disabled(monkeypatch):
    monkeypatch.setenv("AURA_EAGER_FALLBACK_WARMUP", "0")
    fake = _FakeBrainstem(alive=False)
    _patch_client(monkeypatch, fake)
    gate = _bare_gate()
    gate._schedule_eager_fallback_warmup(delay=0.0)
    assert gate._eager_fallback_task is None
    assert fake.warmup_calls == 0


@pytest.mark.asyncio
async def test_eager_warmup_skips_already_resident_brainstem(monkeypatch):
    monkeypatch.setenv("AURA_EAGER_FALLBACK_WARMUP", "1")
    fake = _FakeBrainstem(alive=True)
    _patch_client(monkeypatch, fake)
    gate = _bare_gate()
    gate._schedule_eager_fallback_warmup(delay=0.0)
    await gate._eager_fallback_task
    assert fake.warmup_calls == 0


@pytest.mark.asyncio
async def test_eager_warmup_does_not_double_schedule(monkeypatch):
    monkeypatch.setenv("AURA_EAGER_FALLBACK_WARMUP", "1")
    fake = _FakeBrainstem(alive=False)
    _patch_client(monkeypatch, fake)
    gate = _bare_gate()

    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_warmup() -> None:
        fake.warmup_calls += 1
        started.set()
        await release.wait()

    fake.warmup = _slow_warmup  # type: ignore[method-assign]
    gate._schedule_eager_fallback_warmup(delay=0.0)
    first = gate._eager_fallback_task
    await started.wait()
    # A second call while the first task is still in flight must not re-schedule.
    gate._schedule_eager_fallback_warmup(delay=0.0)
    assert gate._eager_fallback_task is first
    release.set()
    await first
    assert fake.warmup_calls == 1
