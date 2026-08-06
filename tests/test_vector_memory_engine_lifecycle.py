from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.memory.vector_memory_engine import VectorMemoryEngine


@pytest.mark.asyncio
async def test_vector_memory_engine_closes_each_owned_backend_once() -> None:
    closed: list[str] = []
    engine = VectorMemoryEngine.__new__(VectorMemoryEngine)
    engine.embedder = SimpleNamespace(close=lambda: closed.append("embedder"))
    engine.vault = SimpleNamespace(close=lambda: closed.append("vault"))
    engine._closed = False

    await engine.on_stop_async()
    await engine.on_stop_async()

    assert closed == ["embedder", "vault"]


def test_vector_memory_engine_closes_vault_when_embedder_close_fails() -> None:
    closed: list[str] = []
    engine = VectorMemoryEngine.__new__(VectorMemoryEngine)

    def fail_embedder_close() -> None:
        closed.append("embedder")
        raise RuntimeError("embedder close failed")

    engine.embedder = SimpleNamespace(close=fail_embedder_close)
    engine.vault = SimpleNamespace(close=lambda: closed.append("vault"))
    engine._closed = False

    with pytest.raises(RuntimeError, match="embedder close failed"):
        engine.close()

    assert closed == ["embedder", "vault"]
