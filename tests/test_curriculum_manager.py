from __future__ import annotations

import json

import pytest

from core.memory.knowledge.curriculum import CurriculumManager
from core.skills.curiosity import CuriositySkill


def test_missing_curriculum_initializes_a_nonempty_durable_library(tmp_path):
    path = tmp_path / "curriculum" / "media_recommendations.json"

    manager = CurriculumManager(path)

    assert path.is_file()
    assert manager.get_all_categories()
    assert manager.get_suggestion() is not None
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema"] == "aura.curriculum.v1"
    assert persisted["categories"]


def test_invalid_entries_are_rejected_without_losing_valid_ones(tmp_path):
    path = tmp_path / "curriculum.json"
    path.write_text(
        json.dumps(
            {
                "categories": [
                    {
                        "name": "Valid",
                        "items": [
                            {"name": "Keep", "description": "A valid item", "status": "new"},
                            {"name": "Drop", "description": "Bad state", "status": "invented"},
                        ],
                    },
                    "not-an-object",
                ]
            }
        ),
        encoding="utf-8",
    )

    manager = CurriculumManager(path)

    assert manager.get_all_categories() == ["Valid"]
    assert manager.get_suggestion()["item"]["name"] == "Keep"


def test_failed_completion_write_does_not_claim_or_mutate_success(tmp_path, monkeypatch):
    manager = CurriculumManager(tmp_path / "curriculum.json")
    item = manager.get_suggestion()["item"]
    monkeypatch.setattr(manager, "_save_data", lambda: False)

    message = manager.mark_complete(item["name"])

    assert "Could not persist" in message
    assert manager.get_suggestion()["item"]["name"] == item["name"]


@pytest.mark.asyncio
async def test_curiosity_fetches_one_suggestion_per_request(monkeypatch):
    skill = object.__new__(CuriositySkill)
    calls = 0

    def suggestion(_category=None):
        nonlocal calls
        calls += 1
        return "one suggestion"

    monkeypatch.setattr(skill, "get_suggestion", suggestion)

    result = await skill.execute({"action": "get_suggestion"})

    assert calls == 1
    assert result == {
        "ok": True,
        "result": "one suggestion",
        "summary": "one suggestion",
    }
