"""Tests for idle reasoning pre-computation (compute-amortization loophole)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from core.brain import reasoning_precompute as rpc
from core.brain import reasoning_solved_cache as rsc
from core.brain.reasoning_precompute import (
    PrecomputeQueue,
    get_precompute_queue,
    reset_precompute_queue,
)


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    cache = rsc.ReasoningSolvedCache(tmp_path / "c.json")
    # Patch the symbol in BOTH the source module and where reasoning_precompute
    # bound it at import time, so enqueue/tick see the temp cache.
    monkeypatch.setattr(rsc, "get_reasoning_solved_cache", lambda: cache)
    monkeypatch.setattr(rpc, "get_reasoning_solved_cache", lambda: cache)
    return cache


@dataclass
class _FakeResult:
    verified: bool
    answer: str = ""


def test_enqueue_only_cacheable_types(fresh_cache):
    q = PrecomputeQueue()
    assert q.enqueue("compute 99 * 99", "math") is True
    assert q.enqueue("where is the boot path", "repo_audit") is False  # source-dependent
    assert q.pending() == 1


def test_enqueue_dedup(fresh_cache):
    q = PrecomputeQueue()
    assert q.enqueue("solve it", "math") is True
    assert q.enqueue("Solve  it", "math") is False  # normalized dup
    assert q.pending() == 1
    assert q.stats()["deduped"] == 1


def test_enqueue_skips_already_cached(fresh_cache):
    fresh_cache.put("known problem", "math", answer="42", confidence=0.9, mode="deep", verified=True)
    q = PrecomputeQueue()
    assert q.enqueue("known problem", "math") is False
    assert q.stats()["already_cached"] == 1


def test_queue_bounded(fresh_cache):
    q = PrecomputeQueue(max_queue=8)
    for i in range(40):
        q.enqueue(f"problem {i}", "math")
    assert q.pending() <= 8


def test_tick_solves_verified(fresh_cache):
    q = PrecomputeQueue()
    q.enqueue("hard math problem", "math")

    async def _solve(objective, task_type):
        return _FakeResult(verified=True, answer="solved")

    solved = asyncio.run(q.tick(_solve, max_items=1))
    assert solved == 1
    assert q.stats()["solved"] == 1
    assert q.pending() == 0


def test_tick_counts_failures(fresh_cache):
    q = PrecomputeQueue()
    q.enqueue("unsolvable", "math")

    async def _solve(objective, task_type):
        return _FakeResult(verified=False)

    solved = asyncio.run(q.tick(_solve, max_items=1))
    assert solved == 0
    assert q.stats()["failed"] == 1


def test_tick_disabled_by_flag(fresh_cache, monkeypatch):
    monkeypatch.setenv("AURA_REASONING_PRECOMPUTE", "0")
    q = PrecomputeQueue()
    q.enqueue("p", "math")

    calls = []

    async def _solve(objective, task_type):
        calls.append((objective, task_type))
        raise AssertionError("should not be called when disabled")

    assert asyncio.run(q.tick(_solve)) == 0
    assert calls == []


def test_singleton_reset():
    a = get_precompute_queue()
    reset_precompute_queue()
    b = get_precompute_queue()
    assert a is not b


def test_register_reasoning_jobs_idempotent():
    from core.brain.reasoning_background import register_reasoning_jobs

    registered = []

    class _FakeConductor:
        def register(self, name, interval_s, fn, run_immediately=False, policy="maintenance"):
            registered.append(name)

    c = _FakeConductor()
    assert register_reasoning_jobs(c) is True
    assert register_reasoning_jobs(c) is False  # idempotent
    assert "reasoning_idle_precompute" in registered
    assert "reasoning_self_improve" in registered
    assert "reasoning_nonparametric_ingest" in registered


@pytest.mark.asyncio
async def test_nonparametric_background_never_spawns_model_for_maintenance(monkeypatch):
    from types import SimpleNamespace

    from core.brain import reasoning_background
    from core.brain.llm import mlx_client

    class _ColdClient:
        def is_alive(self):
            return False

        async def ingest_nonparametric_async(self, **_kwargs):
            raise AssertionError("cold maintenance must not start worker ingestion")

    monkeypatch.setattr(mlx_client, "get_mlx_client", lambda: _ColdClient())
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False),
    )

    result = await reasoning_background._job_nonparametric_ingest()

    assert result == {
        "status": "skipped_worker_not_resident",
        "spawned_worker": False,
    }
