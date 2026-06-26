"""The reasoning amplifier v2 must be live on the reasoning_strategies hard-task path."""
from __future__ import annotations

import pytest

from core.brain.reasoning_strategies import ReasoningStrategies, StrategyType


def _strategies(answer: str) -> ReasoningStrategies:
    async def generate(prompt: str, **kwargs) -> str:
        return answer

    return ReasoningStrategies(generate)


@pytest.mark.asyncio
async def test_math_query_routes_through_amplifier_v2():
    strat = _strategies("The answer: 6 * 7 = 42")
    result = await strat.execute("Please compute 6 times 7 carefully", strategy=StrategyType.CONSISTENCY)
    assert result.metadata.get("amplifier_v2") is True
    assert "reasoning_receipt" in result.metadata
    assert result.metadata["reasoning_receipt"]["task_type"] == "math"


@pytest.mark.asyncio
async def test_arithmetic_error_marked_unverified_live():
    # Phrased so the exact-computation fast-path does not pre-empt it; the generated
    # answer carries a calculation error the truth engine must catch.
    strat = _strategies("Working it out: 2 + 2 = 5, so the total is five.")
    result = await strat.execute(
        "Compute the sum and show the arithmetic for the two values", strategy=StrategyType.CONSISTENCY
    )
    assert result.metadata.get("amplifier_v2") is True
    assert result.metadata.get("verified") is False
    assert result.confidence < 0.6


@pytest.mark.asyncio
async def test_code_query_routes_through_amplifier_v2():
    strat = _strategies("```python\ndef f(x):\n    return x + 1\n```")
    result = await strat.execute("write a function that increments x", strategy=StrategyType.DECOMPOSE)
    assert result.metadata.get("amplifier_v2") is True
    assert result.metadata.get("verified") is True


@pytest.mark.asyncio
async def test_casual_query_does_not_amplify():
    strat = _strategies("Hey, good to see you!")
    result = await strat.execute("hi how are you today", strategy=StrategyType.DIRECT)
    assert not result.metadata.get("amplifier_v2")


@pytest.mark.asyncio
async def test_bypass_amplifier_flag(monkeypatch):
    strat = _strategies("compute result 3 * 3 = 9")
    result = await strat.execute("compute 3 times 3", strategy=StrategyType.CONSISTENCY, bypass_amplifier=True)
    assert not result.metadata.get("amplifier_v2")


@pytest.mark.asyncio
async def test_env_disable(monkeypatch):
    monkeypatch.setenv("AURA_REASONING_AMPLIFIER_V2", "0")
    strat = _strategies("compute 3 * 3 = 9")
    result = await strat.execute("compute 3 times 3", strategy=StrategyType.CONSISTENCY)
    assert not result.metadata.get("amplifier_v2")
