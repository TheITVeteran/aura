"""Tests for autonomous goal pursuit (the orchestrator over fluid + parallel)."""
from __future__ import annotations

import pytest

from core.agency.goal_pursuit import GoalPursuitEngine
from core.agency.parallel_executor import ParallelExecutor, ParallelTask
from core.skills.fluid_executor import FluidExecutor, Step


class _Verifier:
    def __init__(self, results=None):
        self._results = {k: list(v) for k, v in (results or {}).items()}

    async def verify(self, predicate, args=None):
        from types import SimpleNamespace

        seq = self._results.get(predicate)
        ok = seq.pop(0) if seq else True
        return SimpleNamespace(success=ok, detail=str(ok))


async def _noop():
    return None


def _engine(verifier=None, **kw):
    ex = FluidExecutor(verifier=verifier or _Verifier(), sleep=lambda _s: _an())
    pex = ParallelExecutor(executor_factory=lambda: FluidExecutor(verifier=verifier or _Verifier(), sleep=lambda _s: _an()))
    return GoalPursuitEngine(executor=ex, parallel=pex, **kw)


async def _an():
    return None


@pytest.mark.asyncio
async def test_pursue_sequential_to_completion():
    eng = _engine(_Verifier({"file_exists": [True]}))
    out = await eng.pursue("make file", [Step("write", _noop, verify="file_exists")])
    assert out.completed and out.attempts == 1 and not out.deferred


@pytest.mark.asyncio
async def test_pursue_parallel_to_completion():
    eng = _engine()
    tasks = [
        ParallelTask("research-a", [Step("a", _noop)]),
        ParallelTask("research-b", [Step("b", _noop)]),
    ]
    out = await eng.pursue("research two things", tasks, parallel=True)
    assert out.completed
    assert out.receipts[0].all_completed


@pytest.mark.asyncio
async def test_timing_gate_defers():
    eng = _engine()
    out = await eng.pursue("ping user", [Step("x", _noop)], timing_ok=lambda: False)
    assert out.deferred and not out.completed and "timing" in out.reason


@pytest.mark.asyncio
async def test_timing_gate_async_allows():
    eng = _engine()

    async def _ok():
        return True

    out = await eng.pursue("act", [Step("x", _noop)], timing_ok=_ok)
    assert out.completed


@pytest.mark.asyncio
async def test_follow_through_replans_after_stall():
    # first plan's step fails verification; replan supplies a working step.
    eng = _engine(_Verifier({"bad": [False, False, False], "good": [True]}), max_replans=1)
    replans = []

    def _replan(receipt):
        replans.append(receipt)
        return [Step("recovered", _noop, verify="good")]

    out = await eng.pursue("ship it", [Step("broken", _noop, verify="bad", max_retries=2)], replan=_replan)
    assert out.completed and out.attempts == 2 and len(replans) == 1


@pytest.mark.asyncio
async def test_no_replan_returns_not_completed():
    eng = _engine(_Verifier({"bad": [False, False, False]}))
    out = await eng.pursue("ship it", [Step("broken", _noop, verify="bad", max_retries=2)])
    assert not out.completed and not out.deferred and out.attempts == 1
    assert out.receipts and not out.receipts[0].completed
