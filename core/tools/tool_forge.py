"""core/tools/tool_forge.py — Tool Forge and Compiler."""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List

from core.tools.tool_registry import get_tool_registry

logger = logging.getLogger("Aura.ToolForge")


class ToolForge:
    """Dynamically generates, signs, and registers new tools inside Aura."""

    @staticmethod
    async def forge_and_install(
        name: str,
        code: str,
        owner: str = "AuraLeviathan",
        risk_tier: str = "medium",
        permissions: List[str] = None,
        allowed_domains: List[str] = None,
        allowed_directories: List[str] = None,
    ) -> bool:
        """Assembles a valid manifest, computes signatures, and registers the tool."""
        permissions = permissions or ["all"]
        allowed_domains = allowed_domains or ["*"]
        allowed_directories = allowed_directories or ["/"]

        # Compute SHA-256 hash of code
        code_bytes = code.encode("utf-8")
        hash_sha256 = hashlib.sha256(code_bytes).hexdigest()

        # Generate signature
        signature = f"{owner}_signed_{hash_sha256[:16]}"

        manifest_data = {
            "name": name,
            "version": "1.0.0",
            "owner": owner,
            "hash_sha256": hash_sha256,
            "signature": signature,
            "risk_tier": risk_tier,
            "allowed_domains": allowed_domains,
            "allowed_directories": allowed_directories,
            "permissions": permissions,
            "sandbox_required": True,
        }

        logger.info("Forging new tool: %s (risk_tier=%s)", name, risk_tier)
        registry = get_tool_registry()
        success = await registry.install_tool(manifest_data, code, source="tool_forge")
        return success
