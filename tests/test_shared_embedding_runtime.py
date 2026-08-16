from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.memory.embedding_runtime import SharedEmbeddingRuntime


@dataclass
class _Engine:
    close_count: int = 0
    marker: str = "shared"

    def close(self) -> None:
        self.close_count += 1


def test_all_owners_share_one_engine_until_the_last_release() -> None:
    created: list[_Engine] = []

    def factory() -> _Engine:
        engine = _Engine()
        created.append(engine)
        return engine

    runtime = SharedEmbeddingRuntime(factory)
    memory = runtime.acquire("vector-memory")
    rag = runtime.acquire("rag")
    relevance = runtime.acquire("evidence-relevance")

    assert len(created) == 1
    assert memory.marker == rag.marker == relevance.marker == "shared"
    assert runtime.snapshot() == {
        "engine_live": True,
        "lease_count": 3,
        "owners": ("evidence-relevance", "rag", "vector-memory"),
    }

    memory.close()
    memory.close()
    rag.close()
    assert created[0].close_count == 0
    assert relevance.marker == "shared"

    relevance.close()
    assert created[0].close_count == 1
    assert runtime.snapshot()["engine_live"] is False
    with pytest.raises(RuntimeError, match="shared_embedding_lease_released"):
        _ = relevance.marker


def test_new_owner_after_full_release_gets_a_fresh_engine() -> None:
    created: list[_Engine] = []

    def factory() -> _Engine:
        engine = _Engine(marker=f"engine-{len(created) + 1}")
        created.append(engine)
        return engine

    runtime = SharedEmbeddingRuntime(factory)
    first = runtime.acquire("first")
    assert first.marker == "engine-1"
    first.close()

    second = runtime.acquire("second")
    assert second.marker == "engine-2"
    assert [engine.close_count for engine in created] == [1, 0]
    second.close()
    assert [engine.close_count for engine in created] == [1, 1]


def test_runtime_close_invalidates_all_leases_and_closes_once() -> None:
    engine = _Engine()
    runtime = SharedEmbeddingRuntime(lambda: engine)
    left = runtime.acquire("left")
    right = runtime.acquire("right")

    runtime.close()
    runtime.close()

    assert engine.close_count == 1
    assert runtime.snapshot() == {
        "engine_live": False,
        "lease_count": 0,
        "owners": (),
    }
    with pytest.raises(RuntimeError, match="shared_embedding_lease_released"):
        _ = left.marker
    with pytest.raises(RuntimeError, match="shared_embedding_lease_released"):
        _ = right.marker
