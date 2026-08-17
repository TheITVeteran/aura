from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from core.memory.embedding_runtime import SharedEmbeddingRuntime


@dataclass
class _Engine:
    close_count: int = 0
    marker: str = "shared"
    embed_count: int = 0

    def close(self) -> None:
        self.close_count += 1

    def embed(self, _text: str):
        self.embed_count += 1
        return [0.0, 1.0, 0.0]


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


def test_process_prewarm_is_idempotent_and_retained_until_close(monkeypatch) -> None:
    import core.memory.embedding_runtime as embedding_runtime

    engine = _Engine()
    runtime = SharedEmbeddingRuntime(lambda: engine)
    monkeypatch.setattr(embedding_runtime, "_RUNTIME", runtime)
    monkeypatch.setattr(embedding_runtime, "_PREWARM_LEASE", None)

    first = embedding_runtime.prewarm_shared_embedding_runtime()
    second = embedding_runtime.prewarm_shared_embedding_runtime()

    assert first["vector_dimensions"] == 3
    assert second["lease_count"] == 1
    assert runtime.snapshot()["owners"] == ("runtime-prewarm",)
    assert engine.embed_count == 2

    embedding_runtime.close_shared_embedding_runtime()
    assert engine.close_count == 1
    assert runtime.snapshot()["engine_live"] is False


def test_server_prewarm_waits_for_cortex_readiness(monkeypatch) -> None:
    import core.consciousness.unified_self as unified_self_module
    import core.memory.embedding_runtime as embedding_runtime
    import core.memory.profile_manager as profile_manager_module
    import core.self.self_condition as self_condition_module
    from interface import server

    class _Gate:
        ready_events: list[tuple[bool, str]] = []

        def get_conversation_status(self):
            return {"conversation_ready": True}

        def set_chat_dependencies_ready(self, ready, *, blocker=""):
            self.ready_events.append((bool(ready), str(blocker)))

    gate = _Gate()

    calls: list[str] = []
    monkeypatch.setattr(
        server.ServiceContainer,
        "get",
        classmethod(
            lambda _cls, key, default=None: gate
            if key == "inference_gate"
            else default
        ),
    )
    monkeypatch.setattr(server, "is_shutdown_requested", lambda: False)
    monkeypatch.setattr(
        embedding_runtime,
        "prewarm_shared_embedding_runtime",
        lambda: calls.append("prewarmed")
        or {"vector_dimensions": 1024, "lease_count": 1},
    )
    async def _profile():
        calls.append("profile")
        return object()

    async def _self():
        calls.append("unified_self")
        return object()

    monkeypatch.setattr(
        profile_manager_module.ProfileManager,
        "get_instance",
        _profile,
    )
    monkeypatch.setattr(unified_self_module, "get_unified_self", _self)
    monkeypatch.setattr(
        self_condition_module,
        "build_self_condition_projection",
        lambda: type("Projection", (), {"evidence_id": "condition-proof"})(),
    )

    asyncio.run(
        server._prewarm_chat_dependencies_after_cortex_ready(
            readiness_timeout_s=0.1,
            poll_interval_s=0.01,
        )
    )

    assert set(calls) == {"prewarmed", "profile", "unified_self"}
    assert gate.ready_events == [
        (False, "chat_dependencies_warming"),
        (True, ""),
    ]
