import asyncio
import inspect

import pytest

from core.orchestrator.mixins import incoming_logic


class SlowThinkingEngine:
    def __init__(self, delay_s: float):
        self.delay_s = delay_s
        self.calls = []

    async def think(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(self.delay_s)
        return type("Thought", (), {"content": "finished", "action": None})()


def test_live_thinking_watchdog_uses_current_runtime_budget():
    source = inspect.getsource(incoming_logic.IncomingLogicMixin._original_handle_incoming_logic)

    assert incoming_logic.THINKING_WATCHDOG_TIMEOUT_S == 300.0
    assert "timeout=THINKING_WATCHDOG_TIMEOUT_S" in source


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
