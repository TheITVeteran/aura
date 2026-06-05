"""core/factory/patch_planner.py — Patch Planning Engine.

Coordinates change plans from objectives and repo maps,
ensuring minimal edits with clear rationale.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("Aura.PatchPlanner")


class PatchPlanner:
    """Decomposes an objective into specific file-level changes."""

    async def create_plan(
        self, objective: str, repo_map: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Produce a structured change plan targeting the objective."""
        logger.info("📋 PatchPlanner: creating plan for '%s'", objective[:60])

        changes: List[Dict[str, Any]] = []
        modules = repo_map.get("modules", [])

        # Determine which modules are likely affected
        objective_lower = objective.lower()
        affected = [m for m in modules if any(
            kw in m.lower() for kw in objective_lower.split()[:5]
        )]

        if not affected and modules:
            # Default: target the first module
            affected = modules[:1]

        for mod in affected:
            changes.append({
                "module": mod,
                "action": "modify",
                "rationale": f"Address objective: {objective[:40]}",
                "estimated_lines": 20,
                "risk": "low",
            })

        return {
            "objective": objective,
            "changes": changes,
            "affected_modules": affected,
            "estimated_total_lines": sum(c.get("estimated_lines", 0) for c in changes),
        }
