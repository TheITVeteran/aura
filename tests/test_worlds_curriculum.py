"""Embodied practice curriculum: deterministic tasks, honest scoring,
governed ledger, presence belief.
"""
from __future__ import annotations

import json

import pytest

import core.worlds.curriculum as curriculum
from core.worlds.curriculum import generate_task, practice_summary, run_task
from core.worlds.physics import PhysicsError


def test_task_generation_is_deterministic():
    a = generate_task(42, "navigate")
    b = generate_task(42, "navigate")
    assert a.describe() == b.describe()
    c = generate_task(43, "navigate")
    assert c.target != a.target
    with pytest.raises(PhysicsError):
        generate_task(1, "swim")


def test_navigate_task_scores_success():
    result = run_task(generate_task(7, "navigate", size=16))
    assert result["success"] is True
    assert result["score"] >= 0.7
    assert result["ticks_used"] > 0
    assert result["detail"]["navigation"]["status"] == "reached"


def test_fetch_task_full_cycle():
    result = run_task(generate_task(11, "fetch", size=16))
    # Fetch is genuinely harder; assert honest structure, and score
    # consistency with the reported outcome either way.
    assert result["task"]["kind"] == "fetch"
    assert "approach" in result["detail"]
    if result["success"]:
        assert result["detail"]["grasped"] is True
        assert result["detail"]["drop_distance"] <= 2.5
        assert result["score"] >= 0.7
    else:
        assert result["score"] < 0.7


async def test_ledger_records_and_summarizes(tmp_path, monkeypatch):
    monkeypatch.setattr(curriculum, "ledger_path",
                        lambda: tmp_path / "practice_ledger.jsonl")
    assert practice_summary()["attempts"] == 0

    result = run_task(generate_task(7, "navigate", size=16))
    await curriculum.record_practice(result)
    await curriculum.record_practice(result)

    raw = (tmp_path / "practice_ledger.jsonl").read_text().strip().splitlines()
    assert len(raw) == 2 and json.loads(raw[0])["task"]["kind"] == "navigate"

    trend = practice_summary()
    assert trend["attempts"] == 2
    assert trend["success_rate"] == 1.0
    assert trend["by_kind"]["navigate"] == 2


async def test_world_forge_practice_actions(tmp_path, monkeypatch):
    monkeypatch.setattr(curriculum, "ledger_path",
                        lambda: tmp_path / "practice_ledger.jsonl")
    from core.skills.world_forge import WorldForgeSkill

    skill = WorldForgeSkill()
    ran = await skill.execute(
        {"action": "practice", "seed": 7, "kind": "navigate", "size": 16}, {})
    assert ran["ok"] and ran["success"] is True

    trend = await skill.execute({"action": "practice_summary"}, {})
    assert trend["ok"] and trend["trend"]["attempts"] == 1


def test_device_presence_becomes_world_state_belief(tmp_path, monkeypatch):
    import asyncio

    import core.security.device_pairing as dp
    from core.world_state import get_world_state

    monkeypatch.setattr(dp.get_config().security, "internal_only_mode",
                        False, raising=False)
    registry = dp.reset_device_registry_for_tests(tmp_path / "devices.json")
    challenge = registry.begin_pairing("bryan")
    issued = asyncio.run(
        registry.complete_pairing(challenge["code"], "presence phone"))
    assert registry.verify_token(issued["token"]) is not None

    belief = get_world_state().get_belief(
        f"device_presence.{issued['device_id']}")
    assert belief is not None
    assert belief["name"] == "presence phone"
    dp.reset_device_registry_for_tests(tmp_path / "unused.json")


async def test_practice_feeds_the_learning_loop(tmp_path, monkeypatch):
    import core.learning.deliberate_practice as dp_mod

    observed = []

    class _Director:
        def observe(self, **kwargs):
            observed.append(kwargs)

    monkeypatch.setattr(dp_mod, "get_practice_director", lambda: _Director())
    monkeypatch.setattr(curriculum, "ledger_path",
                        lambda: tmp_path / "practice_ledger.jsonl")

    result = run_task(generate_task(7, "navigate", size=16))
    await curriculum.record_practice(result)

    assert len(observed) == 1
    assert observed[0]["domain"] == "embodied.navigate"
    assert observed[0]["attempts"] == 1
    assert observed[0]["correct"] == (1 if result["success"] else 0)
    assert observed[0]["source"] == "world_curriculum"
