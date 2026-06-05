"""core/tools/tool_registry.py — Tool Registry.

Stores, catalogs, and invokes all verified and forged tool classes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("Aura.ToolRegistry")


class ToolRegistry:
    """Central directory containing all operational tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Any] = {}

    def register_tool(self, name: str, manifest: Any) -> None:
        self._tools[name] = manifest
        logger.info("📦 ToolRegistry: registered '%s'", name)

    def get_tool(self, name: str) -> Optional[Any]:
        return self._tools.get(name)

    async def execute_tool(self, name: str, *args, **kwargs) -> Dict[str, Any]:
        """Invoke a registered tool dynamically."""
        manifest = self.get_tool(name)
        if not manifest:
            logger.error("🚫 ToolRegistry: tool '%s' not found", name)
            return {"ok": False, "error": f"tool_not_found:{name}"}

        # Dynamically instantiate and execute if class exists in code
        try:
            local_vars: Dict[str, Any] = {}
            exec(manifest.code, globals(), local_vars)

            # Find class or function matching the tool name
            cls = local_vars.get(name)

            # If the test defines a simple function / main block
            if not cls and "def main" in manifest.code:
                main_func = local_vars.get("main")
                if main_func:
                    res = main_func(*args, **kwargs)
                    return {"ok": True, "result": res}

            if cls:
                instance = cls()
                if hasattr(instance, "run"):
                    res = instance.run(*args, **kwargs)
                    return {"ok": True, "result": res}

            return {"ok": False, "error": "run_method_or_main_not_found"}
        except Exception as e:
            logger.error("Error executing tool %s: %s", name, e)
            return {"ok": False, "error": str(e)}


_tool_registry_instance: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _tool_registry_instance
    if _tool_registry_instance is None:
        _tool_registry_instance = ToolRegistry()
    return _tool_registry_instance
