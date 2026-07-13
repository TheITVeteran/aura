"""core/skills/world_forge.py
──────────────────────────
Persistent spatial worlds as a governed skill.

Lets Aura's cognition create, re-enter, simulate, and act inside her
own persistent physical worlds: procedurally generated terrain and
props, deterministic rigid-body dynamics, an event journal that makes
each world a place with a remembered history rather than a throwaway
simulation.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from core.skills.base_skill import BaseSkill

WorldAction = Literal[
    "create",
    "list",
    "inspect",
    "step",
    "impulse",
    "spawn_agent",
    "agent",
    "fork",
    "compare",
    "practice",
    "practice_summary",
]
AgentCommand = Literal[
    "proprioception",
    "look",
    "walk",
    "jump",
    "grasp",
    "throw",
    "navigate",
]
WorldTheme = Literal["plains", "highlands", "arena"]
_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"


class WorldForgeInput(BaseModel):  # type: ignore[misc]
    """Typed action contract for persistent world operations."""

    model_config = ConfigDict(extra="forbid")

    action: WorldAction = "list"
    world_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    name: str = Field(default="", max_length=120)
    seed: int = Field(default=0, ge=0, le=(1 << 64) - 1)
    size: int = Field(default=32, ge=8, le=128)
    theme: WorldTheme = "plains"
    ticks: int = Field(default=120, ge=1, le=10_000)
    body_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    impulse: tuple[float, float, float] | None = None
    agent_id: str = Field(default="agent", pattern=_ID_PATTERN)
    command: AgentCommand = "proprioception"
    heading: float | None = Field(default=None, allow_inf_nan=False)
    rays: int = Field(default=8, ge=1, le=64)
    max_distance: float = Field(default=30.0, gt=0.0, le=1_000.0, allow_inf_nan=False)
    speed: float = Field(default=6.0, ge=0.0, le=25.0, allow_inf_nan=False)
    pitch: float = Field(
        default=0.35,
        ge=-math.pi / 2.0,
        le=math.pi / 2.0,
        allow_inf_nan=False,
    )
    target: tuple[float, float] | None = None
    tolerance: float = Field(default=1.0, gt=0.0, le=10.0, allow_inf_nan=False)
    max_ticks: int = Field(default=12_000, ge=1, le=36_000)
    new_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    other_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    top: int = Field(default=8, ge=1, le=64)
    recent_events: int = Field(default=10, ge=0, le=500)
    kind: Literal["navigate", "fetch"] = "navigate"

    @field_validator("impulse", "target")  # type: ignore[untyped-decorator]
    @classmethod
    def vectors_must_be_finite(cls, value: tuple[float, ...] | None) -> tuple[float, ...] | None:
        if value is not None and not all(math.isfinite(component) for component in value):
            raise ValueError("vector components must be finite")
        return value

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def validate_action_requirements(self) -> WorldForgeInput:
        if self.action not in {"list", "practice", "practice_summary"} and self.world_id is None:
            raise ValueError(f"world_id is required for action {self.action!r}")
        if self.action == "impulse" and (self.body_id is None or self.impulse is None):
            raise ValueError("impulse action requires body_id and impulse")
        if self.action == "fork" and self.new_id is None:
            raise ValueError("fork action requires new_id")
        if self.action == "compare" and self.other_id is None:
            raise ValueError("compare action requires other_id")
        if self.action == "agent" and self.command == "grasp" and self.body_id is None:
            raise ValueError("agent grasp command requires body_id")
        if self.action == "agent" and self.command == "navigate" and self.target is None:
            raise ValueError("agent navigate command requires target")
        if self.action == "agent" and self.command == "walk" and self.ticks > 6_000:
            raise ValueError("agent walk command supports at most 6000 ticks")
        return self


_SIMULATION_SCOPE = (
    "Bounded deterministic classical simulation with translational rigid-body dynamics and "
    "rotational sphere dynamics; oriented-box rotation, a VR renderer, and physical-world "
    "transfer are not yet implemented."
)


class WorldForgeSkill(BaseSkill):  # type: ignore[misc]
    name = "world_forge"
    description = (
        "Create and inhabit persistent spatial physics worlds: procedural terrain, "
        "translational dynamics plus rotational sphere dynamics (gravity, collisions, "
        "friction), an embodied agent "
        "body (walk, look via raycasts, jump, grasp, throw, navigate with A*), "
        "counterfactual world-forking, and a journal of what happened. Worlds "
        "survive restarts."
    )
    effect_scope = "state_mutation"
    input_model = WorldForgeInput
    output = "World summaries with state digests and journal excerpts"
    execution_profile = "cpu"
    memory_mb_estimate = 384
    metabolic_cost = 2
    timeout_seconds = 120.0

    def match(self, goal: dict[str, Any]) -> bool:
        objective = str(goal.get("objective", "")).lower()
        keywords = (
            "physics world",
            "simulate a world",
            "virtual world",
            "spatial world",
            "world forge",
            "procedural world",
            "persistent world",
            "drop a ball",
            "rigid body",
        )
        return any(keyword in objective for keyword in keywords)

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        from core.worlds import PhysicsError, get_world_host

        try:
            request = (
                params
                if isinstance(params, WorldForgeInput)
                else WorldForgeInput.model_validate(params if isinstance(params, dict) else {})
            )
        except ValidationError as exc:
            return {"ok": False, "error": f"invalid world parameters: {exc}"}
        values = request.model_dump(exclude_none=True)
        action = request.action
        host = get_world_host()
        try:
            if action == "create":
                summary = await host.create_world(
                    request.world_id or "",
                    seed=request.seed,
                    size=request.size,
                    theme=request.theme,
                    name=request.name,
                )
                return self._ok(
                    action,
                    durable=True,
                    world=summary,
                    summary=(
                        f"Created world '{summary['world_id']}' "
                        f"({summary['bodies']} bodies, theme {summary['theme']})."
                    ),
                )
            if action == "list":
                worlds = host.list_worlds()
                return self._ok(
                    action,
                    worlds=worlds,
                    summary=f"{len(worlds)} persistent world(s).",
                )
            if action == "inspect":
                detail = host.inspect(
                    request.world_id or "",
                    recent_events=request.recent_events,
                )
                return self._ok(
                    action,
                    world=detail,
                    summary=(
                        f"World '{detail['world_id']}' at tick {detail['tick']}, "
                        f"energy {detail['kinetic_energy']}."
                    ),
                )
            if action == "step":
                summary = await host.step_world(request.world_id or "", request.ticks)
                return self._ok(
                    action,
                    durable=True,
                    world=summary,
                    summary=(
                        f"Advanced '{summary['world_id']}' to tick {summary['tick']} "
                        f"({summary['asleep']} bodies at rest)."
                    ),
                )
            if action == "impulse":
                summary = await host.apply_impulse(
                    request.world_id or "",
                    request.body_id or "",
                    request.impulse or (0.0, 0.0, 0.0),
                )
                return self._ok(
                    action,
                    durable=True,
                    world=summary,
                    summary=(f"Applied impulse to '{request.body_id}' in '{summary['world_id']}'."),
                )
            if action == "spawn_agent":
                body = await host.spawn_agent(
                    request.world_id or "",
                    request.agent_id,
                )
                return self._ok(
                    action,
                    durable=True,
                    agent=body,
                    summary=(f"Embodied agent '{body['agent_id']}' spawned at {body['position']}."),
                )
            if action == "agent":
                forwarded = {
                    key: value
                    for key, value in values.items()
                    if key
                    in {
                        "heading",
                        "ticks",
                        "rays",
                        "max_distance",
                        "body_id",
                        "speed",
                        "pitch",
                        "target",
                        "tolerance",
                        "max_ticks",
                    }
                }
                result = await host.agent_command(
                    request.world_id or "",
                    request.agent_id,
                    request.command,
                    **forwarded,
                )
                mutating = request.command not in {"proprioception", "look"}
                return {
                    "action": action,
                    "durable": mutating,
                    "simulation_scope": _SIMULATION_SCOPE,
                    **result,
                    "summary": (
                        f"Agent '{request.agent_id}' ran '{request.command}' "
                        f"(ok={result.get('ok')})."
                    ),
                }
            if action == "fork":
                summary = await host.fork_world(
                    request.world_id or "",
                    request.new_id or "",
                )
                return self._ok(
                    action,
                    durable=True,
                    world=summary,
                    summary=(
                        f"Forked '{request.world_id}' into '{summary['world_id']}' "
                        "for counterfactual runs."
                    ),
                )
            if action == "compare":
                report = host.compare_worlds(
                    request.world_id or "",
                    request.other_id or "",
                    top=request.top,
                )
                return self._ok(
                    action,
                    comparison=report,
                    summary=(
                        f"{report['bodies_diverged']} body states diverged across "
                        f"{report['bodies_compared']} shared bodies; "
                        f"identical={report['identical']}."
                    ),
                )
            if action == "practice":
                from core.worlds.curriculum import (
                    generate_task,
                    record_practice,
                    run_task,
                )

                task = generate_task(request.seed, request.kind, size=request.size)
                result = run_task(task, max_ticks=request.max_ticks)
                await record_practice(result)
                return self._ok(
                    action,
                    **result,
                    summary=(
                        f"Practice {task.kind} (seed {task.seed}): "
                        f"{'success' if result['success'] else 'failure'}, "
                        f"score {result['score']} in {result['ticks_used']} ticks."
                    ),
                )
            if action == "practice_summary":
                from core.worlds.curriculum import practice_summary

                trend = practice_summary()
                return self._ok(
                    action,
                    trend=trend,
                    summary=(
                        f"{trend['attempts']} attempts, success rate "
                        f"{trend['success_rate']}, recent "
                        f"{trend.get('recent_success_rate')}."
                        if trend["attempts"] else "No practice recorded yet."
                    ),
                )
            return {"ok": False, "error": f"Unknown world_forge action '{action}'"}
        except (OverflowError, PhysicsError, TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    @staticmethod
    def _ok(action: str, **payload: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "action": action,
            "simulation_scope": _SIMULATION_SCOPE,
            **payload,
        }
