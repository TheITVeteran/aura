from __future__ import annotations

import asyncio
import sys
import types

from core.memory import semantic_defrag as semantic_module
from core.memory.semantic_defrag import SemanticDefragmenter


def _install_container(monkeypatch, memory, llm=None):
    def get(name, default=None):
        if name == "vector_memory":
            return memory
        if name == "llm_router":
            return llm
        return default

    monkeypatch.setattr(semantic_module.ServiceContainer, "get", staticmethod(get))


def test_semantic_defrag_deterministic_merge_when_llm_unavailable(monkeypatch):
    docs = [f"alpha memory detail {index}" for index in range(10)]
    ids = [f"m{index}" for index in range(10)]

    class Collection:
        def __init__(self):
            self.deleted: list[list[str]] = []

        def get(self, **_kwargs):
            return {
                "ids": ids,
                "documents": docs,
                "metadatas": [{"id": memory_id, "valence": 0.5} for memory_id in ids],
            }

        def delete(self, *, ids):
            self.deleted.append(list(ids))

    class Memory:
        _fallback_mode = False

        def __init__(self):
            self._collection = Collection()
            self.added: list[tuple[str, dict[str, object]]] = []

        def search_similar(self, content, limit=5):
            if content == docs[0]:
                return [
                    {"id": ids[1], "content": docs[1], "distance": 0.04, "metadata": {"id": ids[1], "valence": 0.5}},
                    {"id": ids[2], "content": docs[2], "distance": 0.05, "metadata": {"id": ids[2], "valence": 0.5}},
                ]
            return []

        def add_memory(self, content, *, metadata):
            self.added.append((content, metadata))

    memory = Memory()
    _install_container(monkeypatch, memory, llm=None)

    result = asyncio.run(SemanticDefragmenter().run_defrag_cycle())

    assert result["status"] == "completed"
    assert result["merged"] == 1
    assert memory.added
    assert memory.added[0][1]["merge_method"] == "deterministic"
    assert memory.added[0][1]["source_ids"] == ["m0", "m1", "m2"]
    assert memory._collection.deleted == [["m0", "m1", "m2"]]


def test_semantic_defrag_collection_failure_records_receipt(monkeypatch):
    recorded: list[tuple[str, str, dict[str, object]]] = []

    class Collection:
        def get(self, **_kwargs):
            attempted = True
            assert attempted
            raise RuntimeError("collection offline")

    class Memory:
        _fallback_mode = False
        _collection = Collection()

    def record_degradation(module, exc, **kwargs):
        recorded.append((module, type(exc).__name__, kwargs))

    monkeypatch.setattr(semantic_module, "record_degradation", record_degradation)
    _install_container(monkeypatch, Memory(), llm=None)

    result = asyncio.run(SemanticDefragmenter().run_defrag_cycle())

    assert result["status"] == "failed"
    assert recorded[0][0] == "semantic_defrag"
    assert recorded[0][1] == "RuntimeError"
    assert recorded[0][2]["receipt_required"] is True
    assert "without deleting source memories" in str(recorded[0][2]["action"])


def test_semantic_defrag_start_stop_uses_task_tracker(monkeypatch):
    created: list[str] = []

    class Tracker:
        def create_task(self, coro, name=None):
            created.append(name or "")
            return asyncio.create_task(coro)

    tracker_module = types.ModuleType("core.utils.task_tracker")
    tracker_module.get_task_tracker = lambda: Tracker()
    monkeypatch.setitem(sys.modules, "core.utils.task_tracker", tracker_module)

    async def scenario():
        defragger = SemanticDefragmenter(interval_s=1.0)
        assert defragger.start() is True
        assert created == ["semantic_defrag.scheduler"]
        defragger.stop()
        await asyncio.sleep(0)
        assert defragger._running is False

    asyncio.run(scenario())
