import pytest

import core.mind_tick as mind_module
from core.mind_tick import MindTick, _schedule_mind_task


class ClosingAwaitable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True

    def __await__(self):
        if False:
            yield None
        return None


class FailingTracker:
    def create_task(self, _awaitable, *, name=None):
        self.last_name = name
        raise RuntimeError(f"{name}: loop unavailable")


class Watchdog:
    def register_component(self, *_args, **_kwargs):
        return None


def test_mind_scheduler_closes_unscheduled_awaitable():
    awaitable = ClosingAwaitable()

    task = _schedule_mind_task(awaitable, name="mind.contract", tracker=FailingTracker())

    assert task is None
    assert awaitable.closed is True


@pytest.mark.asyncio
async def test_mind_tick_start_rolls_back_when_loop_cannot_be_scheduled(monkeypatch):
    monkeypatch.setattr(mind_module, "get_task_tracker", lambda: FailingTracker())
    monkeypatch.setattr("infrastructure.watchdog.get_watchdog", lambda: Watchdog())

    tick = MindTick.__new__(MindTick)
    tick._running = False
    tick._task = None

    async def run_loop():
        return None

    tick._run_loop = run_loop

    await tick.start()

    assert tick._running is False
    assert tick._task is None
