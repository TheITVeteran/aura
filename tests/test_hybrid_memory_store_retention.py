import asyncio
import json

from core.memory.retention_policy import MemoryRetentionPolicy
from core.memory.hybrid_store import HybridMemoryStore


def test_hybrid_store_prunes_without_deadlock_and_preserves_salient_entries(tmp_path) -> None:
    async def scenario() -> list[dict]:
        store = HybridMemoryStore(str(tmp_path))
        store.retention_policy = MemoryRetentionPolicy(
            max_items=4,
            prune_keep_fraction=0.95,
            basis="test",
        )
        store.prune_threshold = store.retention_policy.max_items

        await store.store("old low confidence", {"confidence": 0.1, "source": "noise"})
        await store.store("protected identity", {"confidence": 0.2, "protected": True})
        await store.store("high confidence fact", {"confidence": 0.99, "importance": 0.9})
        await store.store("recent one", {"confidence": 0.3})
        await store.store("recent two", {"confidence": 0.3})

        with store.episodic_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    entries = asyncio.run(scenario())
    contents = {entry["content"] for entry in entries}

    assert len(entries) == 4
    assert "protected identity" in contents
    assert "high confidence fact" in contents
    assert "recent two" in contents
