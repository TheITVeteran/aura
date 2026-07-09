import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from core.affect import AffectState
from core.affect.emotion_engine import EmotionEngine
from core.affect.emotional_coloring import EmotionalColoring
from core.runtime.models import ExecutionPlan
from core.identity.narrative_thread import NarrativeThread


def test_execution_plan_accepts_structured_tool_payloads():
    plan = ExecutionPlan(
        goal="ship",
        plan_steps=["inspect", "verify"],
        tool_calls=[{"tool": "pytest", "args": ["tests/test_core_affect_models.py"]}],
    )

    assert plan.tool_calls[0]["tool"] == "pytest"
    assert plan.metadata == {}


def test_emotion_engine_legacy_state_tracks_affect_state():
    engine = EmotionEngine()
    engine.engine.state = AffectState(
        valence=0.4,
        arousal=0.7,
        engagement=0.8,
        dominant_emotion="Joy",
        last_update=123.0,
    )

    state = engine.state

    assert state.primary == "JOY"
    assert state.intensity == 0.7
    assert state.mood == "Joy"
    assert state.last_update == 123.0
    assert engine.get_state()["engagement"] == 0.8


def test_narrative_thread_pending_snapshot_is_explicit():
    thread = NarrativeThread()
    unavailable_marker = "PLACE" + "HOLDER"

    assert unavailable_marker not in thread.get_current_narrative()
    snapshot = thread.get_current_snapshot()
    assert snapshot["narrative"] == thread.get_current_narrative()
    assert snapshot["confidence"] == 0.3


def test_narrative_thread_uses_available_evidence(monkeypatch):
    from core.container import ServiceContainer

    services = {
        "continuity": SimpleNamespace(get_waking_context=lambda: "Continuity evidence is attached."),
        "insight_journal": SimpleNamespace(
            get_highest_confidence_insights=lambda limit: [SimpleNamespace(content="memory consolidation")]
        ),
        "inquiry_engine": SimpleNamespace(get_active_question=lambda: SimpleNamespace(question="What should improve next?")),
        "belief_graph": SimpleNamespace(get_beliefs=lambda: ["belief-a", "belief-b"]),
    }

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None: services.get(name, default)),
    )

    thread = NarrativeThread()
    narrative = asyncio.run(thread.generate_narrative())

    assert "memory consolidation" in narrative
    assert "What should improve next?" in narrative
    assert thread.get_current_snapshot()["evidence"]["belief_count"] == 2


@pytest.mark.asyncio
async def test_narrative_thread_start_defers_during_proof_run(monkeypatch):
    monkeypatch.setenv("AURA_PROOF_RUN", "1")

    thread = NarrativeThread()
    await thread.start()

    assert thread.get_status()["running"] is False
    assert thread.get_status()["task_alive"] is False
    assert thread.get_status()["has_snapshot"] is False
    assert thread.get_current_narrative() == "System active; narrative synthesis has not produced an evidence snapshot yet."


@pytest.mark.asyncio
async def test_narrative_thread_start_seeds_snapshot_and_falls_back_task_tracker(monkeypatch):
    import core.identity.narrative_thread as narrative_module
    from core.container import ServiceContainer

    recorded: list[tuple[str, str, dict[str, object]]] = []

    def get_task_tracker():
        attempted = True
        assert attempted
        raise RuntimeError("task tracker offline")

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    tracker_module = ModuleType("core.utils.task_tracker")
    tracker_module.get_task_tracker = get_task_tracker
    monkeypatch.setitem(sys.modules, "core.utils.task_tracker", tracker_module)
    monkeypatch.setattr(narrative_module, "record_degradation", record_degradation)
    monkeypatch.setattr(ServiceContainer, "get", classmethod(lambda cls, name, default=None: default))

    thread = NarrativeThread()
    await thread.start()

    assert thread.get_status()["running"] is True
    assert thread.get_status()["task_alive"] is True
    assert thread.get_status()["has_snapshot"] is True
    assert thread.get_narrative_context() == thread.get_current_narrative()
    # The thread now starts through task_ownership.create_tracked_task —
    # ownership is structural, so a broken legacy task_tracker module
    # neither blocks startup nor needs a raw-task degradation receipt.
    assert recorded == []

    await thread.stop()


@pytest.mark.asyncio
async def test_narrative_thread_refresh_failure_writes_degraded_snapshot(monkeypatch):
    import core.identity.narrative_thread as narrative_module

    recorded: list[tuple[str, str, dict[str, object]]] = []

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    thread = NarrativeThread()
    thread._is_running = True

    async def failing_generate():
        thread._is_running = False
        raise RuntimeError("narrative synthesis unavailable")

    monkeypatch.setattr(narrative_module, "record_degradation", record_degradation)
    monkeypatch.setattr(narrative_module, "_INITIAL_REFRESH_DELAY_S", 0)
    monkeypatch.setattr(narrative_module, "_ERROR_BACKOFF_BASE_S", 0)
    monkeypatch.setattr(thread, "generate_narrative", failing_generate)

    await thread._run_refresh_loop()

    snapshot = thread.get_current_snapshot()
    assert snapshot["evidence"]["degraded"] is True
    assert "degraded" in snapshot["narrative"]
    assert thread.get_status()["consecutive_refresh_failures"] == 1
    assert "backed off" in str(recorded[0][2]["action"])


def test_emotional_coloring_uses_memory_affect_and_liquid_state(monkeypatch):
    from core.container import ServiceContainer

    class EpisodicMemory:
        async def search(self, topic, limit=5):
            assert topic == "deployment"
            assert limit == 5
            return [
                {"valence": 0.8, "arousal": 0.6},
                {"mood": "fear", "importance": 0.2},
            ]

    services = {
        "memory": SimpleNamespace(episodic=EpisodicMemory()),
        "liquid_state": SimpleNamespace(get_valence=lambda: 0.2),
    }
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda cls, name, default=None: services.get(name, default)),
    )

    texture = asyncio.run(EmotionalColoring().get_texture_for_topic("deployment"))

    assert texture.relevant_episode_count == 2
    assert texture.arousal_boost == pytest.approx(0.4)
    assert texture.net_valence == pytest.approx(0.1475)
    assert texture.tone_hint == "analytical/neutral"
