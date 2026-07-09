from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.autonomy.autonomous_initiative_loop import (
    _MAX_MISSION_ADVANCES_PER_CYCLE,
    AutonomousInitiativeLoop,
)
from core.container import ServiceContainer


async def _hold_until_cancelled(marker: list[str], name: str) -> None:
    try:
        await asyncio.Event().wait()
    finally:
        marker.append(name)


def _install_held_loops(loop: AutonomousInitiativeLoop, marker: list[str]) -> None:
    loop._world_watcher_loop = lambda: _hold_until_cancelled(marker, "world")
    loop._knowledge_gap_monitor_loop = lambda: _hold_until_cancelled(marker, "knowledge")
    loop._self_development_loop = lambda: _hold_until_cancelled(marker, "self_development")
    loop._social_interaction_loop = lambda: _hold_until_cancelled(marker, "social")
    loop._mission_watcher_loop = lambda: _hold_until_cancelled(marker, "mission")
    loop._discovery_loop = lambda: _hold_until_cancelled(marker, "frontier_discovery")


@pytest.mark.asyncio
async def test_start_keeps_core_loops_when_event_subscription_fails(monkeypatch):
    class BrokenBus:
        async def subscribe(self, _topic: str):
            self.topic = _topic
            if self.topic:
                raise RuntimeError("event bus offline")
            return asyncio.Queue()

    marker: list[str] = []
    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    _install_held_loops(loop, marker)
    monkeypatch.setattr(
        "core.autonomy.autonomous_initiative_loop.optional_service",
        lambda *_args, **_kwargs: BrokenBus(),
    )

    status = await loop.start()
    await asyncio.sleep(0)

    assert status == {
        "ok": True,
        "already_running": False,
        "core_tasks": {
            "world": True,
            "knowledge": True,
            "self_development": True,
            "social": True,
            "mission": True,
            "frontier_discovery": True,
        },
        "event_subscription": False,
    }
    assert all(getattr(task, "_aura_supervised", False) for task in loop._core_tasks())

    await loop.stop()
    assert set(marker) == {
        "world",
        "knowledge",
        "self_development",
        "social",
        "mission",
        "frontier_discovery",
    }


@pytest.mark.asyncio
async def test_start_is_idempotent_while_core_tasks_are_alive(monkeypatch):
    marker: list[str] = []
    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    _install_held_loops(loop, marker)
    monkeypatch.setattr(
        "core.autonomy.autonomous_initiative_loop.optional_service",
        lambda *_args, **_kwargs: None,
    )

    first = await loop.start()
    await asyncio.sleep(0)
    first_world_task = loop._world_task
    second = await loop.start()

    assert first["already_running"] is False
    assert second["already_running"] is True
    assert loop._world_task is first_world_task

    await loop.stop()
    assert set(marker) == {
        "world",
        "knowledge",
        "self_development",
        "social",
        "mission",
        "frontier_discovery",
    }


@pytest.mark.asyncio
async def test_stop_awaits_background_task_cancellation(monkeypatch):
    marker: list[str] = []
    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    _install_held_loops(loop, marker)
    monkeypatch.setattr(
        "core.autonomy.autonomous_initiative_loop.optional_service",
        lambda *_args, **_kwargs: None,
    )

    await loop.start()
    await asyncio.sleep(0)
    await loop.stop()

    assert loop.running is False
    assert all(task.done() for task in loop._core_tasks())
    assert set(marker) == {
        "world",
        "knowledge",
        "self_development",
        "social",
        "mission",
        "frontier_discovery",
    }


@pytest.mark.asyncio
async def test_proactive_initiation_accepts_event_bus_priority_envelope(monkeypatch):
    emitted: list[tuple[str, str, str]] = []
    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())

    monkeypatch.setattr(
        AutonomousInitiativeLoop,
        "_emit_feed",
        staticmethod(
            lambda title, content, *, category: emitted.append((title, content, category))
        ),
    )

    await loop._on_proactive_initiation(
        (
            30,
            7,
            {
                "topic": "aura.proactive.initiation",
                "data": {
                    "content": "Investigate sustained CPU pressure before starting optional work.",
                    "source": "jarvis_anticipation",
                },
            },
        )
    )

    assert emitted == [
        (
            "Proactive Initiation",
            "Investigate sustained CPU pressure before starting optional work.",
            "Initiative",
        )
    ]


@pytest.mark.asyncio
async def test_event_listener_survives_malformed_envelope_then_processes_next(monkeypatch):
    emitted: list[str] = []
    queue: asyncio.Queue = asyncio.Queue()
    loop = AutonomousInitiativeLoop(orchestrator=SimpleNamespace())
    loop.running = True

    monkeypatch.setattr(
        AutonomousInitiativeLoop,
        "_emit_feed",
        staticmethod(lambda _title, content, *, category: emitted.append(content)),
    )
    monkeypatch.setattr(
        "core.autonomy.autonomous_initiative_loop._record_initiative_degradation",
        lambda *_args, **_kwargs: None,
    )

    listener = asyncio.create_task(loop._event_listener_loop(queue))
    await queue.put((30, 1, {"topic": "aura.proactive.initiation", "data": "bad"}))
    await queue.put(
        (
            30,
            2,
            {
                "topic": "aura.proactive.initiation",
                "data": {"content": "Valid follow-up event still arrives."},
            },
        )
    )

    await asyncio.wait_for(queue.join(), timeout=1.0)
    assert emitted == ["Valid follow-up event still arrives."]
    assert not listener.done()

    loop.running = False
    listener.cancel()
    await asyncio.wait_for(listener, timeout=1.0)


@pytest.mark.asyncio
async def test_mission_watcher_advances_bounded_ready_missions(monkeypatch):
    from core.planning.mission_state import MissionStatus

    class MissionState:
        def __init__(self) -> None:
            self.advanced: list[str] = []
            self.missions = [
                SimpleNamespace(
                    mission_id=f"mission-{idx}",
                    objective=f"objective {idx}",
                    status=MissionStatus.ACTIVE,
                    graph=SimpleNamespace(is_complete=False),
                )
                for idx in range(_MAX_MISSION_ADVANCES_PER_CYCLE + 2)
            ]

        def list_active_missions(self):
            return self.missions

        async def advance_mission(self, mission_id):
            self.advanced.append(mission_id)
            return SimpleNamespace(description="next step", action="open_app")

    mission_state = MissionState()
    ServiceContainer.register_instance("mission_state", mission_state, required=False)
    try:
        monkeypatch.setattr(
            AutonomousInitiativeLoop,
            "_emit_feed",
            staticmethod(lambda *_args, **_kwargs: None),
        )

        advanced = await AutonomousInitiativeLoop(orchestrator=SimpleNamespace())._advance_active_missions_once()
    finally:
        ServiceContainer.clear()

    assert advanced == _MAX_MISSION_ADVANCES_PER_CYCLE
    assert mission_state.advanced == [
        f"mission-{idx}" for idx in range(_MAX_MISSION_ADVANCES_PER_CYCLE)
    ]
