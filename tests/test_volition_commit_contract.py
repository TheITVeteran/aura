"""CP126: volition must authorize the CONCRETE action and only then advance state.

Two defect classes are pinned here:

1. The Unified Will used to approve a generic ``volition_tick`` with a fixed
   priority BEFORE any goal was selected, so it never saw the objective it was
   authorizing and could not refuse a particular initiative.
2. Timers, cooldowns, and outreach counters advanced at goal GENERATION or
   SELECTION time. A goal that lost selection — or was refused, or failed —
   still looked like completed work (activity advanced) while suppressing its
   own retry (cooldown recorded), so a broken initiative path went silently
   dormant.
"""
from __future__ import annotations

import asyncio
import types

import pytest

import core.volition as volition_module
from core.volition import VolitionEngine


class _Status:
    running = True


class _Orchestrator:
    def __init__(self):
        self.status = _Status()
        self.conversation_history = []
        self.soul = None
        self.cognitive_engine = None


class _Decision:
    def __init__(self, approved: bool, receipt_id: str = "receipt-1", reason: str = ""):
        self._approved = approved
        self.receipt_id = receipt_id
        self.reason = reason

    def is_approved(self) -> bool:
        return self._approved


class _RecordingWill:
    """Captures every Will decision so the test can inspect what was authorized."""

    def __init__(self, approve_action: bool = True):
        self.calls: list[dict] = []
        self.approve_action = approve_action

    def decide(self, *, content, source, domain, priority):
        self.calls.append({"content": content, "priority": priority, "source": source})
        if content.startswith("volition_action:"):
            return _Decision(self.approve_action, reason="policy" if not self.approve_action else "")
        return _Decision(True)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    paths = types.SimpleNamespace(
        home_dir=tmp_path,
        brain_dir=tmp_path / "brain",
        data_dir=tmp_path / "data",
    )
    paths.brain_dir.mkdir(parents=True, exist_ok=True)
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(volition_module.config, "paths", paths)
    eng = VolitionEngine(_Orchestrator())
    eng.general_interests = ["testing"]
    eng.fun_interests = []
    eng.technical_interests = []
    return eng


def _install_will(monkeypatch, will):
    monkeypatch.setattr("core.will.get_will", lambda: will)


def _only_goal(engine, goal):
    async def _search():
        return [goal]

    engine._search_for_autonomous_goals = _search


def test_will_authorizes_the_concrete_action_not_a_generic_tick(engine, monkeypatch):
    will = _RecordingWill(approve_action=True)
    _install_will(monkeypatch, will)
    _only_goal(
        engine,
        {
            "objective": "Reach out about the deployment plan.",
            "id": "g1",
            "origin": "intrinsic_connection",
            "complexity": 0.3,
        },
    )

    result = asyncio.run(engine.tick(None))

    assert result is not None
    action_calls = [c for c in will.calls if c["content"].startswith("volition_action:")]
    assert action_calls, "the selected action must get its own Will decision"
    authorized = action_calls[0]
    # The Will sees the real objective and origin, not an abstract tick.
    assert "intrinsic_connection" in authorized["content"]
    assert "deployment plan" in authorized["content"]
    # And the real priority, not a hardcoded 0.4.
    assert authorized["priority"] == pytest.approx(0.3)
    # The receipt travels with the goal so effectors can prove authorization.
    assert result["will_receipt"] == "receipt-1"


def test_refused_action_advances_no_state(engine, monkeypatch):
    will = _RecordingWill(approve_action=False)
    _install_will(monkeypatch, will)
    goal = {
        "objective": "Say something spontaneous.",
        "id": "g2",
        "origin": "impulse_question",
        "complexity": 0.2,
        "speak": True,
        volition_module._COMMIT_EFFECTS_KEY: {"speech": True},
    }
    _only_goal(engine, goal)

    before_action = engine.last_action_time
    before_impulse = engine.last_impulse_time

    result = asyncio.run(engine.tick(None))

    assert result is None, "a refused action must not be returned"
    # Nothing may look like it happened.
    assert engine.unanswered_speak_count == 0
    assert engine.speak_backoff_multiplier == 1.0
    assert engine.last_action_time == before_action
    assert engine.last_impulse_time == before_impulse
    assert engine.last_speak_time == 0.0
    # And the refused objective must NOT be on cooldown — a refusal is not
    # completed work, so the goal stays retryable.
    assert goal["objective"] not in engine._goal_cooldowns


def test_admitted_action_commits_state_once(engine, monkeypatch):
    will = _RecordingWill(approve_action=True)
    _install_will(monkeypatch, will)
    goal = {
        "objective": "Share a thought.",
        "id": "g3",
        "origin": "impulse_share",
        "complexity": 0.2,
        "speak": True,
        volition_module._COMMIT_EFFECTS_KEY: {"speech": True},
    }
    _only_goal(engine, goal)

    result = asyncio.run(engine.tick(None))

    assert result is not None
    assert engine.unanswered_speak_count == 1
    assert engine.last_action_time > 0.0
    assert engine.last_impulse_time > 0.0
    assert result["objective"] in engine._goal_cooldowns
    # The private commit key never leaks to downstream consumers.
    assert volition_module._COMMIT_EFFECTS_KEY not in result


def test_inquiry_cooldown_only_burns_on_admission(engine, monkeypatch):
    will = _RecordingWill(approve_action=False)
    _install_will(monkeypatch, will)
    goal = {
        "objective": "Advance active inquiry with grounded research: why.",
        "id": "g4",
        "origin": "intrinsic_inquiry",
        "complexity": 0.6,
        volition_module._COMMIT_EFFECTS_KEY: {"inquiry": True},
    }
    _only_goal(engine, goal)

    asyncio.run(engine.tick(None))
    assert engine.last_inquiry_goal_time == 0.0, (
        "a refused inquiry must not burn the cooldown that blocks the next one"
    )

    will.approve_action = True
    asyncio.run(engine.tick(None))
    assert engine.last_inquiry_goal_time > 0.0
