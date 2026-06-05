import asyncio
from pathlib import Path

import pytest


class SlowThinkingEngine:
    def __init__(self, delay_s: float):
        self.delay_s = delay_s
        self.calls = []

    async def think(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(self.delay_s)
        return type("Thought", (), {"content": "finished", "action": None})()


def test_live_thinking_watchdog_uses_current_runtime_budget():
    source = Path("core/orchestrator/mixins/incoming_logic.py").read_text(encoding="utf-8")

    assert "timeout=300.0" in source
    assert "Thinking task exceeded 300s limit" in source


@pytest.mark.asyncio
async def test_timeout_guard_allows_work_below_budget():
    engine = SlowThinkingEngine(delay_s=0.01)

    thought = await asyncio.wait_for(engine.think(objective="long analysis"), timeout=0.25)

    assert thought.content == "finished"
    assert engine.calls == [{"objective": "long analysis"}]


@pytest.mark.asyncio
async def test_timeout_guard_cancels_work_above_budget():
    engine = SlowThinkingEngine(delay_s=0.25)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(engine.think(objective="stalled analysis"), timeout=0.01)
