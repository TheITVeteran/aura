"""One process-wide embedding model with explicit subsystem ownership.

Vector memory, semantic RAG, and evidence routing all use the same encoder.
Historically each surface constructed its own ``EmbeddingEngine`` and the
first foreground turn could therefore load the same checkpoint three times.
This runtime gives every consumer an independently closable lease while
keeping exactly one underlying engine alive until the final owner releases it.
"""

from __future__ import annotations

import atexit
import logging
import uuid
from collections.abc import Callable
from typing import Any

from core.runtime.lockdep import checked_lock

logger = logging.getLogger("Aura.EmbeddingRuntime")


class SharedEmbeddingLease:
    """A transparent, independently closable view of the shared engine."""

    __slots__ = ("_closed", "_runtime", "_token")

    def __init__(self, runtime: SharedEmbeddingRuntime, token: str) -> None:
        self._runtime = runtime
        self._token = token
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime.resolve(self._token), name)

    def __enter__(self) -> SharedEmbeddingLease:
        self._runtime.resolve(self._token)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._runtime.release(self._token)


class SharedEmbeddingRuntime:
    """Reference-counted owner of one lazily loaded embedding engine."""

    def __init__(self, engine_factory: Callable[[], Any]) -> None:
        self._engine_factory = engine_factory
        self._lock = checked_lock("embedding_runtime.lifecycle", reentrant=True)
        self._engine: Any | None = None
        self._owners: dict[str, str] = {}

    def acquire(self, owner: str) -> SharedEmbeddingLease:
        owner_name = str(owner or "unknown").strip() or "unknown"
        with self._lock:
            if self._engine is None:
                self._engine = self._engine_factory()
                logger.info("Embedding runtime created shared engine")
            token = uuid.uuid4().hex
            self._owners[token] = owner_name
        return SharedEmbeddingLease(self, token)

    def resolve(self, token: str) -> Any:
        with self._lock:
            if token not in self._owners or self._engine is None:
                raise RuntimeError("shared_embedding_lease_released")
            return self._engine

    def release(self, token: str) -> None:
        engine: Any | None = None
        with self._lock:
            if self._owners.pop(token, None) is None:
                return
            if not self._owners:
                engine, self._engine = self._engine, None
        if engine is not None:
            close = getattr(engine, "close", None)
            if callable(close):
                close()
            logger.info("Embedding runtime closed engine after final owner release")

    def close(self) -> None:
        """Invalidate every lease and close the engine exactly once."""
        with self._lock:
            self._owners.clear()
            engine, self._engine = self._engine, None
        if engine is not None:
            close = getattr(engine, "close", None)
            if callable(close):
                close()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "engine_live": self._engine is not None,
                "lease_count": len(self._owners),
                "owners": tuple(sorted(self._owners.values())),
            }


def _new_embedding_engine() -> Any:
    # Lazy import avoids a module cycle while VectorMemoryEngine itself acquires
    # a lease during construction.
    from core.memory.vector_memory_engine import EmbeddingEngine

    return EmbeddingEngine()


_RUNTIME = SharedEmbeddingRuntime(_new_embedding_engine)


def acquire_shared_embedding_engine(owner: str) -> SharedEmbeddingLease:
    return _RUNTIME.acquire(owner)


def shared_embedding_runtime_snapshot() -> dict[str, Any]:
    return _RUNTIME.snapshot()


def close_shared_embedding_runtime() -> None:
    _RUNTIME.close()


atexit.register(close_shared_embedding_runtime)
