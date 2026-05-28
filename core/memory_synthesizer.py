"""core/memory_synthesizer.py -- Compatibility Facade for MemorySynthesizer

All actual implementation has been consolidated under the memory subsystem:
core/memory/memory_synthesizer.py

This module re-exports all elements to ensure complete backward-compatibility.
"""
from __future__ import annotations

import sys
import types
from typing import Any

from core.memory import memory_synthesizer as _canonical

WorldviewSnapshot = _canonical.WorldviewSnapshot
MemorySynthesizer = _canonical.MemorySynthesizer
get_memory_synthesizer = _canonical.get_memory_synthesizer
get_task_tracker = _canonical.get_task_tracker

__all__ = [
    "WorldviewSnapshot",
    "MemorySynthesizer",
    "get_memory_synthesizer",
    "get_task_tracker",
]


def __getattr__(name: str) -> Any:
    return getattr(_canonical, name)


class _MemorySynthesizerFacadeModule(types.ModuleType):
    """Propagate legacy monkeypatches to the canonical memory module."""

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith("__") and hasattr(_canonical, name):
            setattr(_canonical, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _MemorySynthesizerFacadeModule
