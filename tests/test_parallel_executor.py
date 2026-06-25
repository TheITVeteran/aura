"""Tests for parallel (forked) fluid execution."""
from __future__ import annotations

import asyncio

import pytest

from core.agency.parallel_executor import ParallelExecutor, ParallelTask
from core.skills.fluid_executor import FluidExecutor, Step


class _OkVerifier:
    async def verify(self, predicate, args=None):
        from types import SimpleNamespace

        return SimpleNamespace(success=True, detail="ok")


def _factory():
    # real asyncio.sleep so timeouts/concurrency are exercised honestly
    return FluidExecutor(verifier=_OkVerifier())


async def _quick():
    await asyncio.sleep(0.02)


def _task(name, n=2):
    return ParallelTask(goal=name, steps=[Step(f"{name}-{i}", _quick) for i in range(n)])


@pytest.mark.asyncio
async def test_all_tasks_complete_concurrently():
    ex = ParallelExecutor(max_concurrency=4, executor_factory=_factory)
    receipt = await ex.run([_task("a"), _task("b"), _task("c")])
    assert receipt.all_completed
    assert receipt.completed_count == 3


@pytest.mark.asyncio
async def test_concurrency_is_bounded():
    peak = {"v": 0, "cur": 0}

    async def _track():
        peak["cur"] += 1
        peak["v"] = max(peak["v"], peak["cur"])
        await asyncio.sleep(0.05)
        peak["cur"] -= 1

    def factory():
        return FluidExecutor(verifier=_OkVerifier())

    tasks = [ParallelTask(goal=f"g{i}", steps=[Step(f"s{i}", _track)]) for i in range(8)]
    ex = ParallelExecutor(max_concurrency=3, executor_factory=factory)
    receipt = await ex.run(tasks)
    assert receipt.completed_count == 8
    assert peak["v"] <= 3                    # never more than the bound running at once
    assert receipt.peak_concurrency <= 3


@pytest.mark.asyncio
async def test_one_task_timeout_does_not_kill_others():
    async def _slow():
        await asyncio.sleep(5.0)

    ex = ParallelExecutor(max_concurrency=4, per_task_timeout_s=0.1, executor_factory=_factory)
    tasks = [
        _task("fast1"),
        ParallelTask(goal="slow", steps=[Step("slow", _slow)]),
        _task("fast2"),
    ]
    receipt = await ex.run(tasks)
    assert receipt.completed_count == 2          # the two fast tasks still finished
    assert receipt.stalled_count == 1            # the slow one timed out, isolated
    by_goal = {t.goal: t for t in receipt.tasks}
    assert by_goal["slow"].stalled and not by_goal["slow"].completed
    assert by_goal["fast1"].completed


@pytest.mark.asyncio
async def test_failing_task_isolated():
    async def _boom():
        await asyncio.sleep(0)
        raise RuntimeError("worker blew up")

    ex = ParallelExecutor(max_concurrency=4, executor_factory=_factory)
    tasks = [_task("ok"), ParallelTask(goal="boom", steps=[Step("boom", _boom, max_retries=0)])]
    receipt = await ex.run(tasks)
    assert receipt.completed_count == 1
    by_goal = {t.goal: t for t in receipt.tasks}
    assert not by_goal["boom"].completed


@pytest.mark.asyncio
async def test_empty_tasks():
    ex = ParallelExecutor(executor_factory=_factory)
    receipt = await ex.run([])
    assert receipt.completed_count == 0 and not receipt.all_completed
