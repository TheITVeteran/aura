"""core/factory/code_writer.py — Code Writing and Patch Generation.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from core.container import ServiceContainer
from core.factory.patch_planner import CodingTask
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.CodeWriter")
_CODE_WRITER_RECOVERABLE_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class CodeWriter:
    """Uses LLM models to draft patches for planned coding tasks."""

    @staticmethod
    async def write_patch(task: CodingTask) -> str:
        logger.info("🏭 CodeWriter generating patch draft for %s", task.file_path)
        router = ServiceContainer.get("llm_router", default=None)
        if not router or not hasattr(router, "think"):
            raise RuntimeError("code_writer_unavailable:llm_router_missing")

        try:
            patch = await router.think(
                prompt=(
                    f"Write a Python code patch for the file {task.file_path}.\n"
                    f"Task Description: {task.description}\n"
                    f"Lines: {task.start_line} to {task.end_line}.\n"
                    f"Output ONLY raw Python code content."
                )
            )
        except _CODE_WRITER_RECOVERABLE_ERRORS as e:
            record_degradation(
                "code_writer",
                e,
                action="failed closed patch generation instead of returning placeholder code",
                extra={"file_path": task.file_path},
            )
            logger.error("Failed to generate code patch: %s", e)
            raise RuntimeError("code_writer_patch_generation_failed") from e

        patch_text = str(patch or "").strip()
        if not patch_text:
            raise RuntimeError("code_writer_empty_patch")
        return patch_text
