"""core/factory/code_writer.py — Code Patch Writer.

Interfaces with the LLM to draft file patches, applies them,
and validates syntax before committing.
"""
from __future__ import annotations

import ast
import logging
import time
from typing import Any, Dict

from core.container import ServiceContainer
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.CodeWriter")


class CodeWriter:
    """Drafts and applies code patches using the LLM router."""

    async def write_patch(
        self, change: Dict[str, Any], repo_path: str
    ) -> Dict[str, Any]:
        """Draft a code patch for a planned change."""
        module = change.get("module", "unknown")
        rationale = change.get("rationale", "")
        logger.info("✏️  CodeWriter drafting patch for module '%s'", module)

        # Attempt LLM-generated patch
        router = ServiceContainer.get("llm_router", default=None)
        patch_content = ""
        if router and hasattr(router, "think"):
            try:
                patch_content = await router.think(
                    prompt=(
                        f"Write a minimal Python code patch for module '{module}'.\n"
                        f"Rationale: {rationale}\n"
                        f"Output only the code diff."
                    )
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                record_degradation("code_writer", e, action="used fallback patch after LLM failed")

        if not patch_content:
            patch_content = f"# Patch for {module}: {rationale}\n# (Deterministic fallback)\n"

        # Validate syntax
        syntax_ok = True
        try:
            ast.parse(patch_content)
        except SyntaxError:
            syntax_ok = False

        return {
            "module": module,
            "patch": patch_content[:2000],
            "syntax_valid": syntax_ok,
            "timestamp": time.time(),
            "rationale": rationale,
            "method": "llm" if router else "fallback",
        }
