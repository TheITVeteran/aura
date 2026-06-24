"""Tests for the goal planner (goal text → executable plan)."""
from __future__ import annotations

import pytest

from core.agency.goal_planner import GoalPlanner
from core.skills.fluid_executor import FluidExecutor


async def _an():
    return None


def _executor():
    return FluidExecutor(verifier=None, sleep=lambda _s: _an())


# ── classification ────────────────────────────────────────────────────────

def test_classify_computational():
    assert GoalPlanner().classify("calculate 47 * 89") == "computational"
    assert GoalPlanner().classify("solve x^2 - 5*x + 6 = 0 for x") == "computational"


def test_classify_general_is_reasoning():
    assert GoalPlanner().classify("figure out the best approach to X") == "reasoning"
    assert GoalPlanner().classify("decide what to prioritize") == "reasoning"


def test_classify_empty_is_none():
    assert GoalPlanner().classify("") == "none"


# ── computational plan: exact answer, no model ────────────────────────────

@pytest.mark.asyncio
async def test_computational_plan_solves_exactly():
    results = []
    planner = GoalPlanner(on_result=lambda g, k, a: results.append((k, a)))
    steps = await planner.plan("what is 47 * 89")
    assert len(steps) == 1
    receipt = await _executor().run("compute", steps)
    assert receipt.completed
    assert results and results[0][0] == "computational" and "4183" in results[0][1]


# ── reasoning plan: deliberates via injected model ────────────────────────

@pytest.mark.asyncio
async def test_reasoning_plan_deliberates():
    async def _gen(prompt, temp):
        return "Answer: organize by topic"

    captured = []
    planner = GoalPlanner(generate=_gen, on_result=lambda g, k, a: captured.append((k, a)))
    steps = await planner.plan("how should I organize my notes?")
    assert steps[0].name == "reason"
    receipt = await _executor().run("reason", steps)
    assert receipt.completed
    assert captured and captured[0][0] == "reasoning"
    assert "organize by topic" in captured[0][1]


# ── reach plan: governed ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_reach_plan_uses_governed_gateway():
    from core.skills.reach_gateway import ReachGateway, ReachPolicy

    class _FakeResp:
        status_code = 200
        text = "service ok"

    class _FakeHttp:
        async def request(self, method, url, json=None, headers=None):
            return _FakeResp()

    gw = ReachGateway(policy=ReachPolicy(read_hosts=frozenset({"api.example.com"})), http=_FakeHttp())
    captured = []
    planner = GoalPlanner(reach=gw, on_result=lambda g, k, a: captured.append((k, a)))
    assert planner.classify("fetch https://api.example.com/status") == "reach"
    steps = await planner.plan("fetch https://api.example.com/status")
    receipt = await _executor().run("reach", steps)
    assert receipt.completed
    assert captured and captured[0][0] == "reach" and "service ok" in captured[0][1]


@pytest.mark.asyncio
async def test_empty_goal_yields_no_steps():
    assert await GoalPlanner().plan("") == []
