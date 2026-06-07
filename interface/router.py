"""Compatibility adapter for legacy interface-level skill routing.

The canonical routing owner is ``core.cognitive.router.IntentRouter``. This
module keeps the historical ``interface.router.IntentRouter`` import path alive
without preserving the old direct background-dispatch lane.
"""

from __future__ import annotations

import logging
from typing import Any

from core.cognitive.router import IntentRouter as CanonicalIntentRouter
from core.security.input_sanitizer import input_sanitizer

logger = logging.getLogger("Aura.InterfaceRouter")


class IntentRouter(CanonicalIntentRouter):
    """Legacy import shim that delegates execution to the canonical router."""

    async def route_execution(
        self,
        skill_name: str,
        params: dict[str, Any],
        engine: Any,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sanitized_params, is_safe = input_sanitizer.validate_params(dict(params or {}))
        if not is_safe:
            logger.error("Security blocked execution of %r due to unsafe parameters.", skill_name)
            return {
                "ok": False,
                "status": "REJECTED_SECURITY",
                "skill": str(skill_name or ""),
                "message": f"Execution of {skill_name} blocked by InputSanitizer.",
            }
        sanitized_context, context_is_safe = input_sanitizer.validate_params(dict(context or {}))
        if not context_is_safe:
            logger.error("Security blocked execution context for %r.", skill_name)
            return {
                "ok": False,
                "status": "REJECTED_SECURITY",
                "skill": str(skill_name or ""),
                "message": f"Execution context for {skill_name} was blocked by InputSanitizer.",
            }

        return await super().route_execution(
            str(skill_name or "").strip(),
            sanitized_params,
            engine,
            context=sanitized_context,
        )
