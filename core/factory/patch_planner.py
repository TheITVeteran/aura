"""core/factory/patch_planner.py — Software Factory Patch Planner.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger("Aura.PatchPlanner")


@dataclass
class CodingTask:
    file_path: str
    description: str
    start_line: int
    end_line: int


class PatchPlanner:
    """Plans coding tasks to fix bugs or implement features with minimal changes."""

    @staticmethod
    def plan_changes(issue_desc: str, codebase_map: dict) -> List[CodingTask]:
        logger.info("🏭 Planning code edits for: '%s'", issue_desc[:80])
        # Simple rule parser: find target files mentioned or guess based on matches
        target_file = "core/utils/helper.py"
        for f in codebase_map.get("files", []):
            if "helper" in f or "utility" in f:
                target_file = f
                break

        return [
            CodingTask(
                file_path=target_file,
                description=f"Refactor to resolve issue: {issue_desc}",
                start_line=1,
                end_line=10,
            )
        ]
