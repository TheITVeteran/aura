# core/memory package (Digital Metabolism: Shims Restored)
from .rag import chunk_text, retrieve_memories, retrieve_memories_v2, tokenize

__all__ = [
    "MemoryFacade",
    "chunk_text",
    "retrieve_memories",
    "retrieve_memories_v2",
    "tokenize",
]


def __getattr__(name: str):
    if name == "MemoryFacade":
        from .memory_facade import MemoryFacade

        return MemoryFacade
    raise AttributeError(f"module 'core.memory' has no attribute {name!r}")
