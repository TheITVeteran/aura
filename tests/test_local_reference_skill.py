"""Search never dead-ends: the local-reference lane and its fallbacks.

Pins the three layers built after a live autonomous-goal failure
(web_search FAILED during the self-repair goal):
1. the planner only advertises web_search when the skill exists, and
   always offers local_reference_search;
2. the search-intent shortcut falls back to the local corpus lane;
3. EnhancedWebSearchSkill degrades to the local corpus at runtime with
   explicit provenance when the web fails.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from core.knowledge.local_corpus import LocalCorpusStore


def _seeded_store(tmp_path):
    store = LocalCorpusStore(tmp_path / "corpus.db")
    store.add_documents([
        (
            "Hafele-Keating experiment",
            "The Hafele-Keating experiment flew atomic clocks around the "
            "world on commercial airliners to test time dilation predicted "
            "by special and general relativity.",
            "wikipedia",
        ),
    ])
    return store


class TestLocalReferenceSkill:
    def test_answers_from_corpus_with_provenance(self, tmp_path, monkeypatch):
        import core.knowledge.local_corpus as corpus_mod
        from core.skills.local_reference import LocalReferenceSearchSkill

        monkeypatch.setattr(corpus_mod, "_store", _seeded_store(tmp_path))
        out = asyncio.run(
            LocalReferenceSearchSkill().execute(
                {"query": "atomic clocks airliners time dilation"}, {}
            )
        )
        assert out["success"] is True
        assert out["results"][0]["title"].startswith("Hafele")
        assert out["results"][0]["provenance"] == "local_corpus"

    def test_empty_corpus_is_reported_not_invented(self, tmp_path, monkeypatch):
        import core.knowledge.local_corpus as corpus_mod
        from core.skills.local_reference import LocalReferenceSearchSkill

        monkeypatch.setattr(
            corpus_mod, "_store", LocalCorpusStore(tmp_path / "empty.db")
        )
        out = asyncio.run(
            LocalReferenceSearchSkill().execute({"query": "anything"}, {})
        )
        assert out["success"] is False
        assert out["results"] == []
        assert "ingest_wikipedia" in out["message"]


class TestPlannerToolMenuHonesty:
    def _planner(self, registry):
        from core.planning.planner import Planner

        return Planner(cognitive_engine=object(), registry=registry)

    def test_web_search_not_advertised_without_the_skill(self):
        planner = self._planner(registry=None)
        schemas = planner._load_tool_schemas()
        assert "web_search" not in schemas
        assert "local_reference_search" in schemas

    def test_web_search_advertised_when_skill_exists(self):
        registry = SimpleNamespace(skills={"web_search": object()})
        schemas = self._planner(registry)._load_tool_schemas()
        assert "web_search" in schemas
        assert "local_reference_search" in schemas

    def test_search_shortcut_falls_back_to_local_lane(self):
        planner = self._planner(registry=None)
        shortcut = planner._detect_intent("search for the speed of sound")
        assert shortcut is not None
        assert shortcut["tool_calls"][0].tool == "local_reference_search"

    def test_search_shortcut_prefers_web_when_available(self):
        registry = SimpleNamespace(skills={"web_search": object()})
        planner = self._planner(registry)
        shortcut = planner._detect_intent("search for the speed of sound")
        assert shortcut is not None
        assert shortcut["tool_calls"][0].tool == "web_search"


class TestWebSearchOfflineFallback:
    def test_web_failure_degrades_to_local_corpus(self, tmp_path, monkeypatch):
        import core.knowledge.local_corpus as corpus_mod
        from core.skills.web_search import EnhancedWebSearchSkill

        monkeypatch.setattr(corpus_mod, "_store", _seeded_store(tmp_path))
        skill = EnhancedWebSearchSkill()
        skill.pipeline.search = AsyncMock(
            return_value={"ok": False, "error": "network unreachable"}
        )
        out = asyncio.run(
            skill.execute({"query": "atomic clocks time dilation"}, {})
        )
        assert out["ok"] is True
        assert out["provenance"] == "local_corpus"
        assert out["offline_fallback"] is True
        assert out["web_error"] == "network unreachable"
        assert out["results"][0]["title"].startswith("Hafele")

    def test_no_corpus_keeps_the_honest_web_failure(self, tmp_path, monkeypatch):
        import core.knowledge.local_corpus as corpus_mod
        from core.skills.web_search import EnhancedWebSearchSkill

        monkeypatch.setattr(
            corpus_mod, "_store", LocalCorpusStore(tmp_path / "empty.db")
        )
        skill = EnhancedWebSearchSkill()
        skill.pipeline.search = AsyncMock(
            return_value={"ok": False, "error": "network unreachable"}
        )
        out = asyncio.run(skill.execute({"query": "anything"}, {}))
        assert out["ok"] is False
        assert out.get("error") == "network unreachable"
