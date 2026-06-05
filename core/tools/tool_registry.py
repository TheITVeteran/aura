"""core/tools/tool_registry.py — Tool Registry.

Stores, catalogs, and invokes all verified and forged tool classes.
"""
from __future__ import annotations

import logging
import ast
from typing import Any, Dict, Optional

from core.sandbox.runner import run_untrusted

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
        """Invoke a registered tool in the isolated tool sandbox."""
        manifest = self.get_tool(name)
        if not manifest:
            logger.error("🚫 ToolRegistry: tool '%s' not found", name)
            return {"ok": False, "error": f"tool_not_found:{name}"}

        try:
            driver = _build_sandbox_driver(str(manifest.code), name, args, kwargs)
            sandbox_result = run_untrusted(driver)
            if sandbox_result.get("status") != "ok":
                return {
                    "ok": False,
                    "error": sandbox_result.get("repr")
                    or sandbox_result.get("stderr")
                    or sandbox_result.get("status"),
                }
            stdout = str(sandbox_result.get("stdout") or "").strip()
            if not stdout:
                return {"ok": False, "error": "tool_returned_no_result"}
            parsed = ast.literal_eval(stdout.splitlines()[-1])
            return {"ok": True, "result": parsed}
        except (AttributeError, SyntaxError, TypeError, ValueError) as e:
            logger.error("Error executing tool %s: %s", name, e)
            return {"ok": False, "error": str(e)}


def _build_sandbox_driver(code: str, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    if not name.isidentifier():
        raise ValueError(f"tool name is not a valid Python identifier: {name}")
    return "\n".join(
        [
            code,
            f"_aura_args = {args!r}",
            f"_aura_kwargs = {kwargs!r}",
            "try:",
            f"    _aura_cls = {name}",
            "except NameError:",
            "    _aura_cls = None",
            "try:",
            "    _aura_main = main",
            "except NameError:",
            "    _aura_main = None",
            "if _aura_cls is not None:",
            "    _aura_result = _aura_cls().run(*_aura_args, **_aura_kwargs)",
            "elif _aura_main is not None:",
            "    _aura_result = _aura_main(*_aura_args, **_aura_kwargs)",
            "else:",
            "    raise Exception('run_method_or_main_not_found')",
            "print(repr(_aura_result))",
            "",
        ]
    )


_tool_registry_instance: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _tool_registry_instance
    if _tool_registry_instance is None:
        _tool_registry_instance = ToolRegistry()
    return _tool_registry_instance
