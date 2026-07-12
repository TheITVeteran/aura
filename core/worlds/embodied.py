"""core/worlds/embodied.py
───────────────────────
An embodied agent inside Aura's physics worlds.

This closes the perception–action loop the wishlist calls "robust
perception and embodiment" and "high-level task execution": a body with
real dynamics, senses that measure the world (raycasts, proprioception),
effectors that change it (locomotion, jumping, grasping, throwing), and
a navigator that plans over generated terrain and reports success or
failure honestly.

Everything stays deterministic: rays are exact analytic intersections,
locomotion is force-based through the same physics step as every other
body, and A* runs on the blueprint heightfield with a fixed expansion
order.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.worlds.generation import WorldBlueprint
from core.worlds.physics import Body, PhysicsError, PhysicsWorld

AGENT_RADIUS = 0.45
AGENT_MASS = 60.0
_WALK_ACCEL = 24.0  # m/s² of drive while grounded
_AIR_ACCEL = 4.0  # weak air control
_MAX_WALK_SPEED = 3.0  # m/s horizontal
_JUMP_SPEED = 4.5  # m/s vertical
_GROUND_EPSILON = 0.12
_GRASP_RANGE = 1.6
_GRASP_MAX_MASS = 20.0
_MAX_WALK_TICKS = 6_000
_MAX_NAV_TICKS = 36_000  # 5 simulated minutes at 120 Hz
_WAYPOINT_RADIUS = 0.6
_MAX_CLIMB = 1.25  # traversable height step between grid cells


@dataclass
class RayHit:
    body_id: str
    distance: float
    point: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_id": self.body_id,
            "distance": round(self.distance, 6),
            "point": [round(p, 6) for p in self.point],
        }


@dataclass
class AgentState:
    agent_id: str
    yaw: float = 0.0  # radians, 0 = +x
    held_body: str | None = None
    last_navigation: dict[str, Any] = field(default_factory=dict)


class EmbodiedAgent:
    """A dynamic body plus senses and effectors, bound to one world."""

    def __init__(
        self, world: PhysicsWorld, blueprint: WorldBlueprint | None, agent_id: str = "agent"
    ):
        self.world = world
        self.blueprint = blueprint
        self.state = AgentState(agent_id=agent_id)

    # ── lifecycle ──────────────────────────────────────────────

    @classmethod
    def spawn(
        cls,
        world: PhysicsWorld,
        blueprint: WorldBlueprint | None,
        *,
        agent_id: str = "agent",
        position: tuple[float, float, float] | None = None,
    ) -> EmbodiedAgent:
        if agent_id in world.bodies:
            raise PhysicsError(f"body '{agent_id}' already exists in this world")
        if position is None:
            spawn = blueprint.spawn_point if blueprint else [0.0, 0.0, 1.0]
            position = (spawn[0], spawn[1], spawn[2] + AGENT_RADIUS)
        world.add_body(
            Body(
                body_id=agent_id,
                shape="sphere",
                position=np.asarray(position, dtype=np.float64),
                velocity=np.zeros(3, dtype=np.float64),
                mass=AGENT_MASS,
                radius=AGENT_RADIUS,
                restitution=0.0,
                friction=0.9,
                # A body, not a ball: locomotion must not spin it into an
                # uncontrolled rolling sphere under contact friction.
                angular_locked=True,
            )
        )
        return cls(world, blueprint, agent_id)

    @property
    def body(self) -> Body:
        return self.world.body(self.state.agent_id)

    # ── proprioception ─────────────────────────────────────────

    def grounded(self) -> bool:
        """Standing on something: a support contact within epsilon below."""
        hit = self.raycast(direction=(0.0, 0.0, -1.0), max_distance=AGENT_RADIUS + _GROUND_EPSILON)
        return hit is not None

    def proprioception(self) -> dict[str, Any]:
        body = self.body
        return {
            "agent_id": self.state.agent_id,
            "position": [round(float(x), 4) for x in body.position],
            "velocity": [round(float(x), 4) for x in body.velocity],
            "speed": round(float(np.linalg.norm(body.velocity)), 4),
            "yaw": round(self.state.yaw, 4),
            "grounded": self.grounded(),
            "holding": self.state.held_body,
        }

    # ── exteroception: exact analytic raycasts ─────────────────

    def raycast(
        self,
        *,
        direction: tuple[float, float, float],
        max_distance: float = 50.0,
        origin: tuple[float, float, float] | None = None,
    ) -> RayHit | None:
        d = np.asarray(direction, dtype=np.float64)
        if d.shape != (3,) or not np.all(np.isfinite(d)):
            raise PhysicsError("ray direction must be a finite 3-vector")
        max_distance = float(max_distance)
        if not math.isfinite(max_distance) or max_distance <= 0.0:
            raise PhysicsError("ray max_distance must be finite and positive")
        norm = float(np.linalg.norm(d))
        if norm <= 1e-12:
            raise PhysicsError("ray direction must be nonzero")
        d = d / norm
        o = (
            np.asarray(origin, dtype=np.float64)
            if origin is not None
            else self.body.position.copy()
        )
        if o.shape != (3,) or not np.all(np.isfinite(o)):
            raise PhysicsError("ray origin must be a finite 3-vector")

        best: RayHit | None = None
        for key in sorted(self.world.bodies):
            if key == self.state.agent_id or key == self.state.held_body:
                continue
            body = self.world.bodies[key]
            t = self._intersect(o, d, body)
            if t is not None and 1e-9 < t <= max_distance:
                if best is None or t < best.distance:
                    point = o + t * d
                    best = RayHit(key, float(t), [float(p) for p in point])
        return best

    @staticmethod
    def _intersect(o: np.ndarray, d: np.ndarray, body: Body) -> float | None:
        if body.shape == "sphere":
            oc = o - body.position
            b = float(oc @ d)
            c = float(oc @ oc) - body.radius**2
            disc = b * b - c
            if disc < 0.0:
                return None
            t = -b - math.sqrt(disc)
            return t if t > 0.0 else None
        if body.shape == "plane":
            if abs(d[2]) < 1e-12:
                return None
            t = (body.plane_height - o[2]) / d[2]
            return t if t > 0.0 else None
        if body.shape == "box":
            lo = body.position - body.half_extents
            hi = body.position + body.half_extents
            t_near, t_far = -math.inf, math.inf
            for axis in range(3):
                if abs(d[axis]) < 1e-12:
                    if not lo[axis] <= o[axis] <= hi[axis]:
                        return None
                    continue
                t1 = (lo[axis] - o[axis]) / d[axis]
                t2 = (hi[axis] - o[axis]) / d[axis]
                t1, t2 = min(t1, t2), max(t1, t2)
                t_near, t_far = max(t_near, t1), min(t_far, t2)
            if t_near > t_far or t_far <= 0.0:
                return None
            return t_near if t_near > 0.0 else None
        return None

    def look(self, *, rays: int = 8, max_distance: float = 30.0) -> list[dict[str, Any]]:
        """A horizontal sweep of rays around the current yaw — a cheap,
        exact depth scan of the surroundings."""
        if isinstance(rays, bool) or not isinstance(rays, int) or not 1 <= rays <= 64:
            raise PhysicsError("look rays must be an integer in [1, 64]")
        if not math.isfinite(max_distance) or not 0.0 < max_distance <= 1_000.0:
            raise PhysicsError("look max_distance must be finite and in (0, 1000]")
        observations = []
        for i in range(rays):
            angle = self.state.yaw + (i - rays // 2) * (math.pi / rays)
            hit = self.raycast(
                direction=(math.cos(angle), math.sin(angle), 0.0),
                max_distance=max_distance,
            )
            observations.append(
                {
                    "bearing": round(angle, 4),
                    "hit": hit.to_dict() if hit else None,
                }
            )
        return observations

    # ── effectors ──────────────────────────────────────────────

    def walk(self, *, heading: float | None = None, ticks: int = 1) -> None:
        """Drive toward ``heading`` (radians) for ``ticks`` physics steps.
        Force-based: acceleration is applied through velocity like any
        other dynamics, capped at walking speed, weak in the air."""
        if heading is not None:
            if not math.isfinite(heading):
                raise PhysicsError("walk heading must be finite")
            self.state.yaw = float(heading)
        if (
            isinstance(ticks, bool)
            or not isinstance(ticks, int)
            or not 1 <= ticks <= _MAX_WALK_TICKS
        ):
            raise PhysicsError(f"walk ticks must be an integer in [1, {_MAX_WALK_TICKS}]")
        for _ in range(ticks):
            body = self.body
            accel = _WALK_ACCEL if self.grounded() else _AIR_ACCEL
            drive = (
                np.array([math.cos(self.state.yaw), math.sin(self.state.yaw), 0.0])
                * accel
                * self.world.dt
            )
            body.velocity = body.velocity + drive
            horizontal = body.velocity[:2]
            speed = float(np.linalg.norm(horizontal))
            if speed > _MAX_WALK_SPEED:
                body.velocity[:2] = horizontal * (_MAX_WALK_SPEED / speed)
            body.sleeping = False
            body.still_ticks = 0
            self.world.step()
            self._carry_held_body()

    def jump(self) -> bool:
        if not self.grounded():
            return False
        body = self.body
        body.velocity[2] = _JUMP_SPEED
        body.sleeping = False
        return True

    def grasp(self, body_id: str) -> bool:
        """Pick up a nearby, light, dynamic body (kinematic carry)."""
        if self.state.held_body is not None:
            raise PhysicsError(f"already holding '{self.state.held_body}'")
        target = self.world.body(body_id)
        if target.is_static:
            raise PhysicsError(f"'{body_id}' is static")
        if target.mass > _GRASP_MAX_MASS:
            raise PhysicsError(f"'{body_id}' is too heavy to carry")
        gap = float(np.linalg.norm(target.position - self.body.position))
        if gap > _GRASP_RANGE:
            return False
        self.state.held_body = body_id
        self._carry_held_body()
        return True

    def throw(self, *, speed: float = 6.0, pitch: float = 0.35) -> str:
        """Release the held body with a velocity along the current yaw."""
        if self.state.held_body is None:
            raise PhysicsError("not holding anything")
        speed = float(speed)
        pitch = float(pitch)
        if not math.isfinite(speed) or not 0.0 <= speed <= 25.0:
            raise PhysicsError("throw speed must be finite and in [0, 25]")
        if not math.isfinite(pitch) or not -math.pi / 2.0 <= pitch <= math.pi / 2.0:
            raise PhysicsError("throw pitch must be finite and in [-pi/2, pi/2]")
        held = self.world.body(self.state.held_body)
        held.velocity = (
            np.array(
                [
                    math.cos(self.state.yaw) * math.cos(pitch) * speed,
                    math.sin(self.state.yaw) * math.cos(pitch) * speed,
                    math.sin(pitch) * speed,
                ]
            )
            + self.body.velocity
        )
        held.sleeping = False
        held.still_ticks = 0
        released = self.state.held_body
        self.state.held_body = None
        return released

    def _carry_held_body(self) -> None:
        if self.state.held_body is None:
            return
        held = self.world.body(self.state.held_body)
        offset = np.array([math.cos(self.state.yaw), math.sin(self.state.yaw), 0.4]) * (
            AGENT_RADIUS + held.radius + 0.1
        )
        held.position = self.body.position + offset
        held.velocity = self.body.velocity.copy()
        held.sleeping = False

    # ── navigation: plan on the heightfield, walk the plan ─────

    def plan_path(self, target_xy: tuple[float, float]) -> list[tuple[float, float]] | None:
        """A* over the blueprint heightfield with a climb constraint.
        Returns world-frame waypoints, or None when unreachable."""
        if self.blueprint is None:
            raise PhysicsError("navigation needs a generated world (no blueprint)")
        heights = np.asarray(self.blueprint.heightfield, dtype=np.float64)
        size = self.blueprint.size
        target = np.asarray(target_xy, dtype=np.float64)
        if target.shape != (2,) or not np.all(np.isfinite(target)):
            raise PhysicsError("navigation target must be a finite 2-vector")
        lower = -size / 2.0
        upper = size / 2.0
        if not lower <= target[0] < upper or not lower <= target[1] < upper:
            raise PhysicsError(f"navigation target must lie inside world bounds [{lower}, {upper})")

        def to_grid(x: float, y: float) -> tuple[int, int]:
            return (
                int(np.clip(x + size / 2.0, 0, size - 1)),
                int(np.clip(y + size / 2.0, 0, size - 1)),
            )

        def to_world(i: int, j: int) -> tuple[float, float]:
            return (i - size / 2.0 + 0.5, j - size / 2.0 + 0.5)

        start = to_grid(float(self.body.position[0]), float(self.body.position[1]))
        goal = to_grid(float(target_xy[0]), float(target_xy[1]))
        frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost: dict[tuple[int, int], float] = {start: 0.0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            ci, cj = current
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < size and 0 <= nj < size):
                    continue
                climb = abs(float(heights[ni, nj]) - float(heights[ci, cj]))
                if climb > _MAX_CLIMB:
                    continue
                new_cost = cost[current] + 1.0 + climb
                neighbor = (ni, nj)
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    heuristic = abs(goal[0] - ni) + abs(goal[1] - nj)
                    heapq.heappush(frontier, (new_cost + heuristic, neighbor))
                    came_from[neighbor] = current
        if goal not in came_from:
            return None
        path: list[tuple[float, float]] = []
        node: tuple[int, int] | None = goal
        while node is not None:
            path.append(to_world(*node))
            node = came_from[node]
        path.reverse()
        return path

    def navigate_to(
        self,
        target_xy: tuple[float, float],
        *,
        tolerance: float = 1.0,
        max_ticks: int = _MAX_NAV_TICKS,
    ) -> dict[str, Any]:
        """Plan and execute a walk to ``target_xy``. Reports the outcome
        honestly: reached / no_path / timed_out, with the distance left."""
        if (
            isinstance(max_ticks, bool)
            or not isinstance(max_ticks, int)
            or not 1 <= max_ticks <= _MAX_NAV_TICKS
        ):
            raise PhysicsError(f"navigation max_ticks must be an integer in [1, {_MAX_NAV_TICKS}]")
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or not 0.0 < tolerance <= 10.0:
            raise PhysicsError("navigation tolerance must be finite and in (0, 10]")
        path = self.plan_path(target_xy)
        if path is None:
            result = {"status": "no_path", "target": list(target_xy), "ticks_used": 0}
            self.state.last_navigation = result
            return result

        ticks_used = 0
        waypoint_index = 0
        target = np.array([target_xy[0], target_xy[1]])
        while ticks_used < max_ticks:
            position = self.body.position[:2]
            if float(np.linalg.norm(position - target)) <= tolerance:
                result = {
                    "status": "reached",
                    "target": list(target_xy),
                    "ticks_used": ticks_used,
                    "final_distance": round(float(np.linalg.norm(position - target)), 4),
                }
                self.state.last_navigation = result
                return result
            while (
                waypoint_index < len(path) - 1
                and float(np.linalg.norm(position - np.array(path[waypoint_index])))
                < _WAYPOINT_RADIUS
            ):
                waypoint_index += 1
            waypoint = np.array(path[waypoint_index])
            delta = waypoint - position
            heading = math.atan2(float(delta[1]), float(delta[0]))
            self.walk(heading=heading, ticks=4)
            ticks_used += 4
        result = {
            "status": "timed_out",
            "target": list(target_xy),
            "ticks_used": ticks_used,
            "final_distance": round(float(np.linalg.norm(self.body.position[:2] - target)), 4),
        }
        self.state.last_navigation = result
        return result
