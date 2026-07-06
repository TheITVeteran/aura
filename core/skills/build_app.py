"""Build-an-app skill.

Live capability surface: Aura builds a real, runnable single-file web app from a
natural spec, validates that it actually works, and writes it to disk so it can
be opened and used. General across app kinds (games, tools, toys).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill


class BuildAppInput(BaseModel):
    spec: str = Field(..., description="What app to build, e.g. 'a playable checkers game'.")
    out_dir: str = Field("artifacts/live_apps", description="Where to write the app file.")
    max_tokens: int = Field(6000, description="Generation budget for the app.")


class BuildAppSkill(BaseSkill):
    name = "build_app"
    description = (
        "Build a real, runnable single-file web app (game, tool, or toy) from a natural "
        "description, validate that it actually works, and write it to disk to open and use."
    )
    input_model = BuildAppInput
    timeout_seconds = 240.0
    metabolic_cost = 3
    effect_scope = "read_write_artifacts"
    requires_approval = False

    async def execute(self, params: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(params, dict):
            params = BuildAppInput(**params)
        elif not isinstance(params, BuildAppInput):
            params = BuildAppInput.model_validate(params)

        # Self-taught, test-driven build: recall prior lessons, research the task
        # (concepts + reference code), generate, FUNCTIONALLY test that it works,
        # feed the exact failure back, persist, and retain the general lesson.
        from core.capabilities.self_taught_builder import build_app_verified

        result = await build_app_verified(
            params.spec, out_dir=params.out_dir, max_tokens=params.max_tokens,
        )
        payload = result.to_dict()
        return {
            "ok": bool(result.ok),
            "skill": self.name,
            "spec": result.spec,
            "path": result.path,
            "playable": result.playable,
            "result": payload,
            "summary": (
                f"Built '{result.spec}' -> {result.path}: "
                f"playable={result.playable} after {result.iterations} test-driven iteration(s); "
                f"researched {len(result.research_used)} source(s), recalled "
                f"{len(result.recalled_lessons)} prior lesson(s). Lesson retained."
            ),
        }


__all__ = ["BuildAppInput", "BuildAppSkill"]
