"""Tests for tool-augmented reasoning (exact subproblem offloading)."""
from __future__ import annotations

import pytest

from core.brain.tool_augmented_reasoning import (
    looks_computational,
    solve_exact,
    tool_augmented_answer,
)


# ── detection ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("q", [
    "what is 47 * 89",
    "calculate 1234 + 5678",
    "compute 2^10",
    "47*89",
    "solve x^2 - 5*x + 6 = 0 for x",
])
def test_recognizes_computational_queries(q):
    assert looks_computational(q)


@pytest.mark.parametrize("q", [
    "how are you feeling today?",
    "what is the capital of France",
    "tell me about the ocean",
    "what is love",
])
def test_ignores_non_computational(q):
    assert not looks_computational(q)


# ── exact solving ─────────────────────────────────────────────────────────

def test_exact_arithmetic():
    r = solve_exact("what is 47 * 89")
    assert r.ok and r.method == "evaluate"
    assert "4183" in r.answer                     # 47*89 = 4183, exactly


def test_exact_large_arithmetic_beats_guessing():
    r = solve_exact("123456 * 789")
    assert r.ok and r.answer == "97406784"        # exact — a value the LLM would fumble


def test_exact_power():
    r = solve_exact("compute 2^10")
    assert r.ok and "1024" in r.answer


def test_solve_quadratic():
    r = solve_exact("solve x^2 - 5*x + 6 = 0 for x")
    assert r.ok and r.method == "solve_equation"
    assert "2" in r.answer and "3" in r.answer     # roots are 2 and 3


def test_non_computational_returns_not_ok():
    assert not solve_exact("how are you?").ok
    assert tool_augmented_answer("how are you?") is None


# ── causal: the reasoning strategy uses the exact path ────────────────────

@pytest.mark.asyncio
async def test_strategy_execute_uses_tool_fastpath():
    from core.brain.reasoning_strategies import ReasoningStrategies

    llm_called = {"n": 0}

    async def _gen(prompt, **kw):
        llm_called["n"] += 1
        return "I think it's around 4000"          # the LLM would GUESS wrong

    rs = ReasoningStrategies(_gen)
    result = await rs.execute("what is 47 * 89")
    assert "4183" in result.content                # exact tool answer, not the guess
    assert result.metadata.get("tool_augmented") is True
    assert llm_called["n"] == 0                     # the LLM was bypassed entirely
