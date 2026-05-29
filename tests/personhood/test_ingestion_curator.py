"""tests/personhood/test_ingestion_curator.py
===========================================
Unit tests verifying the IngestionLoop and KnowledgeCurator services:
  1. IngestionLoop trigger conditions and text chunking.
  2. KnowledgeCurator defrag triggering and memory promotion/purging.
"""

import pytest
from types import SimpleNamespace
from core.memory.ingestion_loop import IngestionLoop
from core.memory.knowledge_curator import KnowledgeCurator
from core.container import ServiceContainer


class DefragmenterDouble:
    def __init__(self):
        self.calls = 0

    async def run_defrag_cycle(self):
        self.calls += 1
        return {"clusters_consolidated": 2}


class CollectionDouble:
    def __init__(self, payload):
        self.payload = payload
        self.update_calls = []
        self.delete_calls = []

    def get(self, **_kwargs):
        return self.payload

    def update(self, **kwargs):
        self.update_calls.append(kwargs)

    def delete(self, **kwargs):
        self.delete_calls.append(kwargs)


def test_text_chunking():
    """Verify text chunking in IngestionLoop."""
    loop = IngestionLoop()
    text = "one two three four five six seven eight nine ten"
    
    # Chunk size of 4, overlap of 1
    chunks = loop._chunk_text(text, chunk_size=4, overlap=1)
    
    # Expected chunks:
    # 1: "one two three four"
    # 2: "four five six seven"
    # 3: "seven eight nine ten"
    assert len(chunks) == 3
    assert chunks[0] == "one two three four"
    assert chunks[1] == "four five six seven"
    assert chunks[2] == "seven eight nine ten"


@pytest.mark.asyncio
async def test_ingestion_loop_early_exit():
    """Verify that IngestionLoop returns early if curiosity is low and no active goals."""
    loop = IngestionLoop()
    
    substrate = SimpleNamespace(current=SimpleNamespace(curiosity=0.4))
    ServiceContainer.register_instance("liquid_substrate", substrate)
    
    goal_engine = SimpleNamespace(get_goals=lambda: [])
    ServiceContainer.register_instance("goal_engine", goal_engine)
    
    chunk_calls = []

    def record_chunk_call(*args, **kwargs):
        chunk_calls.append((args, kwargs))
        return []

    loop._chunk_text = record_chunk_call
    await loop._ingest_cycle()
    assert chunk_calls == []


@pytest.mark.asyncio
async def test_knowledge_curator_consolidation():
    """Verify KnowledgeCurator triggers consolidation and optimizes memories."""
    curator = KnowledgeCurator()
    
    curator._defragmenter = DefragmenterDouble()
    
    # Set up test memory items
    # Item 1: 5 days old, very low importance (should be purged)
    # Item 2: High importance, not promoted yet (should be promoted to core_knowledge)
    now = 1716940800.0
    collection = CollectionDouble({
        "ids": ["mem-stale", "mem-important"],
        "metadatas": [
            {"timestamp": now - 300000.0, "importance": 0.1},
            {"timestamp": now - 1000.0, "importance": 0.9, "category": "general"}
        ],
        "documents": [
            "low relevance old info",
            "highly important foundational knowledge"
        ]
    })
    memory = SimpleNamespace(_fallback_mode=False, _collection=collection)
    
    ServiceContainer.register_instance("vector_memory", memory)
    
    import time

    original_time = time.time
    time.time = lambda: now
    try:
        res = await curator.consolidate_and_curate()
    finally:
        time.time = original_time

    assert curator._defragmenter.calls == 1
    assert collection.delete_calls == []
    assert len(collection.update_calls) == 2
    promote_kwargs = collection.update_calls[0]
    purge_kwargs = collection.update_calls[1]
    assert promote_kwargs["ids"] == ["mem-important"]
    assert promote_kwargs["metadatas"][0]["category"] == "core_knowledge"
    assert purge_kwargs["ids"] == ["mem-stale"]
    assert purge_kwargs["metadatas"][0]["candidate_for_purge"] is True

    assert res["purged"] == 0
    assert res["purge_candidates"] == 1
    assert res["promoted"] == 1
