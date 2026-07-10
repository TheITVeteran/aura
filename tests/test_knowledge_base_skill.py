from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from skills.knowledge_base import KnowledgeBaseInput, KnowledgeBaseSkill


@pytest.mark.asyncio
async def test_knowledge_base_full_lifecycle_is_integrity_verified(tmp_path: Path):
    store = tmp_path / "knowledge"
    skill = KnowledgeBaseSkill(store)
    assert not store.exists()

    created = await skill.execute(
        {
            "action": "create",
            "content": "A release must pass its rollback drill before promotion.",
            "summary": "Release rollback policy",
            "title": "Release Safety",
        }
    )
    assert created["ok"] is True
    assert created["status"] == "created"

    duplicate = await skill.execute(
        {"action": "create", "content": "replacement", "title": "Release Safety"}
    )
    assert duplicate == {
        "error": "knowledge item already exists; use upsert to replace it",
        "item_id": created["item_id"],
        "ok": False,
        "status": "conflict",
    }

    read = await skill.execute({"action": "read", "title": "Release Safety"})
    assert read["ok"] is True
    assert read["status"] == "verified"
    assert read["content_sha256"] == created["content_sha256"]

    search = await skill.execute({"action": "search", "query": "rollback drill"})
    assert search["ok"] is True
    assert search["count"] == 1
    assert search["results"][0]["item_id"] == created["item_id"]

    updated = await skill.execute(
        {
            "action": "upsert",
            "content": "Promotion requires rollback and restore drills.",
            "title": "Release Safety",
        }
    )
    assert updated["ok"] is True
    assert updated["status"] == "updated"
    assert updated["content_sha256"] != created["content_sha256"]

    listed = await skill.execute({"action": "list", "limit": 10})
    assert listed["ok"] is True
    assert listed["total"] == 1
    assert listed["items"][0]["item_id"] == created["item_id"]

    deleted = await skill.execute({"action": "delete", "title": "Release Safety"})
    assert deleted["ok"] is True
    assert deleted["status"] == "deleted"
    assert (await skill.execute({"action": "read", "title": "Release Safety"}))["ok"] is False


@pytest.mark.asyncio
async def test_knowledge_base_detects_body_tampering(tmp_path: Path):
    skill = KnowledgeBaseSkill(tmp_path / "knowledge")
    created = await skill.execute(
        {"action": "create", "content": "trusted body", "title": "Integrity"}
    )
    item_path = skill._item_path(created["item_id"])
    item_path.write_text("tampered body", encoding="utf-8")

    read = await skill.execute({"action": "read", "title": "Integrity"})
    search = await skill.execute({"action": "search", "query": "tampered"})

    assert read["ok"] is False
    assert read["status"] == "integrity_failed"
    assert search["ok"] is False
    assert search["status"] == "partial_integrity_failure"
    assert search["integrity_failures"] == [created["item_id"]]


@pytest.mark.asyncio
async def test_knowledge_base_compensates_failed_catalog_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    skill = KnowledgeBaseSkill(tmp_path / "knowledge")
    original = await skill.execute(
        {"action": "create", "content": "original", "title": "Compensation"}
    )
    item_path = skill._item_path(original["item_id"])

    def fail_catalog_write(_catalog):
        raise OSError("injected catalog fsync failure")

    monkeypatch.setattr(skill, "_write_catalog", fail_catalog_write)
    failed_update = await skill.execute(
        {"action": "upsert", "content": "uncommitted", "title": "Compensation"}
    )
    failed_create = await skill.execute(
        {"action": "create", "content": "uncommitted", "title": "New Item"}
    )

    assert failed_update["ok"] is False
    assert failed_update["status"] == "transaction_failed"
    assert item_path.read_text(encoding="utf-8") == "original"
    assert failed_create["ok"] is False
    assert not skill._item_path(skill._item_id("New Item")).exists()


@pytest.mark.asyncio
async def test_knowledge_base_serializes_concurrent_updates(tmp_path: Path):
    skill = KnowledgeBaseSkill(tmp_path / "knowledge")
    first, second = await asyncio.gather(
        skill.execute({"action": "create", "content": "alpha", "title": "Alpha"}),
        skill.execute({"action": "create", "content": "beta", "title": "Beta"}),
    )
    listed = await skill.execute({"action": "list", "limit": 10})

    assert first["ok"] is True
    assert second["ok"] is True
    assert listed["total"] == 2
    assert {item["title"] for item in listed["items"]} == {"Alpha", "Beta"}


def test_knowledge_base_schema_requires_action_specific_fields():
    schema = KnowledgeBaseInput.model_json_schema()
    assert schema["properties"]["action"]["enum"] == [
        "create",
        "upsert",
        "read",
        "search",
        "list",
        "delete",
    ]
    with pytest.raises(ValueError, match="create requires title and content"):
        KnowledgeBaseInput(action="create", title="Missing body")
