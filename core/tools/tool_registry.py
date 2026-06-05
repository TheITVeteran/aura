"""core/tools/tool_registry.py — Universal Tool Registry."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple

from core.runtime.action_executor import ActionExecutor
from core.tools.tool_manifest import ToolManifest
from core.tools.tool_permission import ToolPermissionGuard
from core.tools.tool_sandbox import ToolSandbox
from core.tools.tool_verifier import ToolVerifier
from core.will import ActionDomain
from core.config import config

logger = logging.getLogger("Aura.ToolRegistry")
_TOOL_REGISTRY_RECOVERABLE_ERRORS = (
    json.JSONDecodeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


class ToolRegistry:
    """Manages installation, security checking, and execution of workspace tools."""

    def __init__(self) -> None:
        self.in_memory_tools: Dict[str, Tuple[str, ToolManifest]] = {}
        self.tools_dir = config.paths.data_dir / "tools"

    async def install_tool(
        self,
        manifest_data: Dict[str, Any],
        code: str,
        source: str = "tool_registry",
    ) -> bool:
        """Verifies, registers, and persists a tool under the security workspace."""
        try:
            manifest = ToolManifest.from_dict(manifest_data)
            
            # 1. Verify integrity & cryptographic signature
            if not ToolVerifier.verify_integrity(code, manifest):
                logger.error("Rejecting tool %s due to integrity/signature validation failure", manifest.name)
                return False

            # 2. Persist tool manifest and source code using governed ActionExecutor
            manifest_path = self.tools_dir / f"{manifest.name}.manifest.json"
            code_path = self.tools_dir / f"{manifest.name}.py"

            # Save manifest
            await ActionExecutor.execute(
                domain=ActionDomain.FILE_WRITE,
                action_name="tool_registry.save_manifest",
                params={"path": str(manifest_path), "obj": manifest.to_dict()},
                source=source,
            )

            # Save code
            await ActionExecutor.execute(
                domain=ActionDomain.FILE_WRITE,
                action_name="tool_registry.save_code",
                params={"path": str(code_path), "text": code},
                source=source,
            )

            # 3. Store in memory cache
            self.in_memory_tools[manifest.name] = (code, manifest)
            logger.info("Successfully installed tool: %s", manifest.name)
            return True
        except _TOOL_REGISTRY_RECOVERABLE_ERRORS as e:
            logger.error("Failed to install tool: %s", e, exc_info=True)
            return False

    async def get_tool(self, name: str) -> Optional[Tuple[str, ToolManifest]]:
        """Retrieves a tool from memory cache or local persistence."""
        if name in self.in_memory_tools:
            return self.in_memory_tools[name]

        # Try to load from disk
        manifest_path = self.tools_dir / f"{name}.manifest.json"
        code_path = self.tools_dir / f"{name}.py"

        if manifest_path.exists() and code_path.exists():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                code = code_path.read_text(encoding="utf-8")

                manifest = ToolManifest.from_dict(manifest_data)
                self.in_memory_tools[name] = (code, manifest)
                return code, manifest
            except _TOOL_REGISTRY_RECOVERABLE_ERRORS as e:
                logger.error("Error loading tool %s from disk: %s", name, e)
                return None
        return None

    async def execute_tool(self, name: str, params: Dict[str, Any], source: str = "tool_registry") -> Dict[str, Any]:
        """Runs a registered tool, enforcing permissions and sandboxing constraints."""
        tool_data = await self.get_tool(name)
        if not tool_data:
            return {"ok": False, "error": f"tool_not_found:{name}"}

        code, manifest = tool_data

        # 1. Enforce directory/domain permission matching
        target_path = params.get("target_path")
        if target_path and not ToolPermissionGuard.is_directory_allowed(manifest, target_path):
            return {"ok": False, "error": f"permission_denied:directory_access_blocked:{target_path}"}

        target_domain = params.get("target_domain")
        if target_domain and not ToolPermissionGuard.is_domain_allowed(manifest, target_domain):
            return {"ok": False, "error": f"permission_denied:domain_access_blocked:{target_domain}"}

        # 2. Execute within the tool sandbox wrapper
        result = ToolSandbox.run(code, manifest, params)
        return result


_registry_instance: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolRegistry()
    return _registry_instance
