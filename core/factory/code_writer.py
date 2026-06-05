"""core/factory/code_writer.py — Code Writing and Patch Generation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from core.container import ServiceContainer
from core.factory.patch_planner import CodingTask

logger = logging.getLogger("Aura.CodeWriter")


class CodeWriter:
    """Uses LLM models to draft patches for planned coding tasks."""

    @staticmethod
    async def write_patch(task: CodingTask) -> str:
        logger.info("🏭 CodeWriter generating patch draft for %s", task.file_path)
        router = ServiceContainer.get("llm_router", default=None)
        
        default_patch = "# Optimized coding task implementation\n"
        if router and hasattr(router, "think"):
            try:
                default_patch = await router.think(
                    prompt=(
                        f"Write a Python code patch for the file {task.file_path}.\n"
                        f"Task Description: {task.description}\n"
                        f"Lines: {task.start_line} to {task.end_line}.\n"
                        f"Output ONLY raw Python code content."
                    )
                )
            except Exception as e:
                logger.error("Failed to generate code patch: %s", e)

        return default_patch
