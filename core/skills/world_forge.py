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

from typing import Any, Dict

from core.skills.base_skill import BaseSkill


class WorldForgeSkill(BaseSkill):
    name = "world_forge"
    description = (
        "Create and inhabit persistent 3D physics worlds: procedural terrain, "
        "rigid-body dynamics (gravity, collisions, friction), impulses on "
        "objects, and a journal of what happened. Worlds survive restarts."
    )
    effect_scope = "state_mutation"
    inputs = {
        "action": "one of: create, list, inspect, step, impulse",
        "world_id": "lowercase identifier of the world",
        "seed": "create: generation seed (int)",
        "size": "create: terrain size 8..128 (default 32)",
        "theme": "create: plains | highlands | arena",
        "ticks": "step: physics ticks to advance (1..10000)",
        "body_id": "impulse: target body",
        "impulse": "impulse: [ix, iy, iz] Newton-seconds",
    }
    output = "World summaries with state digests and journal excerpts"

    def match(self, goal: Dict[str, Any]) -> bool:
        objective = str(goal.get("objective", "")).lower()
        keywords = (
            "physics world", "simulate a world", "virtual world", "spatial world",
            "world forge", "procedural world", "persistent world", "drop a ball",
            "rigid body",
        )
        return any(keyword in objective for keyword in keywords)

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        from core.worlds import PhysicsError, get_world_host

        params = params if isinstance(params, dict) else {}
        action = str(params.get("action", "list") or "list").strip().lower()
        host = get_world_host()
        try:
            if action == "create":
                summary = await host.create_world(
                    str(params.get("world_id", "") or ""),
                    seed=int(params.get("seed", 0) or 0),
                    size=int(params.get("size", 32) or 32),
                    theme=str(params.get("theme", "plains") or "plains"),
                    name=str(params.get("name", "") or ""),
                )
                return {"ok": True, "action": action, "world": summary,
                        "summary": f"Created world '{summary['world_id']}' "
                                   f"({summary['bodies']} bodies, theme {summary['theme']})."}
            if action == "list":
                worlds = host.list_worlds()
                return {"ok": True, "action": action, "worlds": worlds,
                        "summary": f"{len(worlds)} persistent world(s)."}
            if action == "inspect":
                detail = host.inspect(str(params.get("world_id", "") or ""))
                return {"ok": True, "action": action, "world": detail,
                        "summary": f"World '{detail['world_id']}' at tick {detail['tick']}, "
                                   f"energy {detail['kinetic_energy']}."}
            if action == "step":
                summary = await host.step_world(
                    str(params.get("world_id", "") or ""),
                    int(params.get("ticks", 120) or 120),
                )
                return {"ok": True, "action": action, "world": summary,
                        "summary": f"Advanced '{summary['world_id']}' to tick "
                                   f"{summary['tick']} ({summary['asleep']} bodies at rest)."}
            if action == "impulse":
                impulse = params.get("impulse") or (0.0, 0.0, 0.0)
                summary = await host.apply_impulse(
                    str(params.get("world_id", "") or ""),
                    str(params.get("body_id", "") or ""),
                    tuple(float(x) for x in impulse),
                )
                return {"ok": True, "action": action, "world": summary,
                        "summary": f"Applied impulse to '{params.get('body_id')}' in "
                                   f"'{summary['world_id']}'."}
            return {"ok": False, "error": f"Unknown world_forge action '{action}'"}
        except PhysicsError as exc:
            return {"ok": False, "error": str(exc)}
