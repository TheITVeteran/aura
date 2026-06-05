"""core/actuation/file_actuator.py — File and Repository Actuator."""
from __future__ import annotations

from typing import Any, Dict
from core.actuation.world_actuator import get_world_actuator


class FileActuator:
    """Wrapper for file writes and repository git operations."""

    @classmethod
    async def write_file(cls, path: str, content: str, source: str = "file_actuator") -> Dict[str, Any]:
        return await get_world_actuator().actuate(
            category="local_files",
            action_name="write_file",
            params={"path": path, "text": content},
            source=source,
        )

    @classmethod
    async def modify_repo(cls, repo_path: str, action: str, params: Dict[str, Any], source: str = "file_actuator") -> Dict[str, Any]:
        act_params = {"repo_path": repo_path, "action": action, **params}
        return await get_world_actuator().actuate(
            category="code_repos",
            action_name="modify_repo",
            params=act_params,
            source=source,
            high_risk_flag=action in ("publish_code", "push"),
        )
