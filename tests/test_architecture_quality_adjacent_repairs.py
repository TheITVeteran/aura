import asyncio

import pytest

from core.intent.belief_extractor import BeliefExtractor
from core.runtime.runtime_relaunch import schedule_relaunch, supervisor_will_restart
from core.self import will_engine as will_engine_module
from core.self.will_engine import WillEngine
from core.world.perception_scheduler import PerceptionScheduler


@pytest.mark.asyncio
async def test_belief_extractor_uses_live_revision_engine_contract() -> None:
    calls: list[dict[str, object]] = []

    class RevisionEngine:
        async def process_new_claim(self, **claim):
            calls.append(claim)
            return {"ok": True}

    extractor = BeliefExtractor(RevisionEngine())
    count = await extractor.extract_and_integrate(
        "BELIEF: Octopuses explore novel objects [Domain: WORLD, Confidence: 0.8]",
        source="tool_result",
    )

    assert count == 1
    assert calls == [
        {
            "claim": "Octopuses explore novel objects",
            "domain": "world",
            "source": "tool_result",
            "confidence": 0.8,
        }
    ]


@pytest.mark.asyncio
async def test_will_engine_lifecycle_uses_governed_task_owner(monkeypatch) -> None:
    created: list[tuple[str, asyncio.Task[None]]] = []

    def create(coro, *, name: str):
        task = asyncio.create_task(coro, name=name)
        created.append((name, task))
        return task

    monkeypatch.setattr(will_engine_module, "create_tracked_task", create)
    engine = WillEngine(tick_interval=3600.0)

    await engine.initialize()
    assert created[0][0] == "aura.will_engine"
    assert engine._tick_task is created[0][1]

    await engine.shutdown()
    assert created[0][1].done()


def test_perception_scheduler_owns_injected_hub() -> None:
    class Hub:
        async def perceive(self, query: str):
            return {"query": query}

    hub = Hub()
    scheduler = PerceptionScheduler(perception_hub=hub)

    assert scheduler._perception_hub is hub


def test_runtime_relaunch_exports_real_control_surface() -> None:
    assert callable(schedule_relaunch)
    assert callable(supervisor_will_restart)
