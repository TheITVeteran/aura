"""core/worlds/hosting.py
──────────────────────
Persistent world hosting: named worlds that survive restarts.

Each world is a governed JSON document under ``data/worlds/`` holding
its generation blueprint, full dynamical state, and an event journal of
what has happened inside it (drops, collisions, impulses, rests). A
world reloaded after a crash resumes from its exact persisted state —
same digest, same history. This is the substrate for Aura's persistent
subjective worlds: places with continuity, not throwaway sims.

Bounds: step budget per call, journal cap, world-count cap. All writes
go through the file_write_gateway under a governed scope.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.config import get_config
from core.governance_context import local_internal_governed_scope
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway
from core.worlds.embodied import EmbodiedAgent
from core.worlds.generation import WorldBlueprint, generate_world
from core.worlds.physics import PhysicsError, PhysicsWorld

logger = logging.getLogger("Aura.Worlds")

MAX_TICKS_PER_STEP = 10_000
MAX_WORLDS = 64
_JOURNAL_CAP = 500
_IMPULSE_JOURNAL_THRESHOLD = 2.0
_SCHEMA_VERSION = 1
_MAX_WORLD_DOCUMENT_BYTES = 32 * 1024 * 1024
_WORLD_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_HOST_ERRORS = (OSError, RuntimeError, TypeError, ValueError, KeyError)


class WorldPersistenceError(PhysicsError):
    """A world transition could not be committed durably."""


class HostedWorld:
    def __init__(
        self,
        world_id: str,
        blueprint: WorldBlueprint,
        physics: PhysicsWorld,
        journal: list[dict[str, Any]] | None = None,
        created_at: float | None = None,
    ):
        self.world_id = world_id
        self.blueprint = blueprint
        self.physics = physics
        self.journal: list[dict[str, Any]] = list(journal or [])
        self.created_at = created_at if created_at is not None else time.time()
        self.updated_at = self.created_at
        self.agents: dict[str, EmbodiedAgent] = {}

    def agent(self, agent_id: str) -> EmbodiedAgent:
        agent = self.agents.get(str(agent_id))
        if agent is None:
            raise PhysicsError(f"no agent '{agent_id}' in world '{self.world_id}'")
        return agent

    def record(self, kind: str, detail: dict[str, Any]) -> None:
        self.journal.append(
            {
                "tick": self.physics.tick,
                "at": time.time(),
                "kind": kind,
                **detail,
            }
        )
        if len(self.journal) > _JOURNAL_CAP:
            del self.journal[: len(self.journal) - _JOURNAL_CAP]

    def summary(self) -> dict[str, Any]:
        sleeping = sum(
            1 for body in self.physics.bodies.values() if not body.is_static and body.sleeping
        )
        dynamic = sum(1 for body in self.physics.bodies.values() if not body.is_static)
        return {
            "world_id": self.world_id,
            "name": self.blueprint.name,
            "theme": self.blueprint.theme,
            "seed": self.blueprint.seed,
            "tick": self.physics.tick,
            "bodies": len(self.physics.bodies),
            "dynamic_bodies": dynamic,
            "asleep": sleeping,
            "kinetic_energy": round(self.physics.total_kinetic_energy(), 6),
            "agents": sorted(self.agents),
            "state_digest": self.physics.state_digest(),
            "journal_entries": len(self.journal),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_document(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "world_id": self.world_id,
            "blueprint": self.blueprint.to_dict(),
            "physics": self.physics.to_dict(),
            "journal": self.journal,
            "agents": {
                agent_id: {
                    "yaw": agent.state.yaw,
                    "held_body": agent.state.held_body,
                    "last_navigation": agent.state.last_navigation,
                }
                for agent_id, agent in self.agents.items()
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> HostedWorld:
        if not isinstance(document, dict):
            raise PhysicsError("hosted world document must be an object")
        if "payload" in document:
            if document.get("schema_name") not in {None, "hosted_world"}:
                raise PhysicsError("hosted world document has the wrong schema")
            if int(document.get("schema_version", 0) or 0) != _SCHEMA_VERSION:
                raise PhysicsError("hosted world document schema version is unsupported")
        payload = document.get("payload", document)
        if not isinstance(payload, dict):
            raise PhysicsError("hosted world payload must be an object")
        world = cls(
            world_id=str(payload["world_id"]),
            blueprint=WorldBlueprint.from_dict(payload["blueprint"]),
            physics=PhysicsWorld.from_dict(payload["physics"]),
            journal=list(payload.get("journal", [])),
            created_at=float(payload.get("created_at", time.time())),
        )
        world.updated_at = float(payload.get("updated_at", world.created_at))
        agent_rows = payload.get("agents") or {}
        if not isinstance(agent_rows, dict):
            raise PhysicsError("hosted world agents must be an object")
        for agent_id, row in agent_rows.items():
            if not isinstance(row, dict) or str(agent_id) not in world.physics.bodies:
                raise PhysicsError(f"hosted world agent {agent_id!r} is invalid")
            agent = EmbodiedAgent(world.physics, world.blueprint, str(agent_id))
            agent.state.yaw = float(row.get("yaw", 0.0))
            agent.state.held_body = row.get("held_body")
            if (
                agent.state.held_body is not None
                and agent.state.held_body not in world.physics.bodies
            ):
                raise PhysicsError(f"hosted world agent {agent_id!r} holds a missing body")
            navigation = row.get("last_navigation") or {}
            if not isinstance(navigation, dict):
                raise PhysicsError("hosted world navigation state must be an object")
            agent.state.last_navigation = dict(navigation)
            world.agents[str(agent_id)] = agent
        return world

    def clone(self) -> HostedWorld:
        return HostedWorld.from_document(self.to_document())

    def replace_from(self, replacement: HostedWorld) -> None:
        """Publish one fully persisted staged generation without changing identity."""

        if replacement.world_id != self.world_id:
            raise PhysicsError("cannot publish a different world identity")
        self.blueprint = replacement.blueprint
        self.physics = replacement.physics
        self.journal = replacement.journal
        self.created_at = replacement.created_at
        self.updated_at = replacement.updated_at
        self.agents = replacement.agents


class WorldHost:
    """Registry of live worlds with governed persistence."""

    def __init__(self, root: Path):
        self.root = root
        self._worlds: dict[str, HostedWorld] = {}
        self._lock = threading.Lock()
        self._world_locks: dict[str, asyncio.Lock] = {}
        self._reserved_world_ids: set[str] = set()

    # ── lifecycle ──────────────────────────────────────────────

    async def create_world(
        self,
        world_id: str,
        *,
        seed: int,
        size: int = 32,
        theme: str = "plains",
        name: str = "",
    ) -> dict[str, Any]:
        world_id = self._validate_id(world_id)
        self._reserve_new_world(world_id)
        try:
            blueprint = generate_world(seed, size=size, theme=theme, name=name or world_id)
            world = HostedWorld(world_id, blueprint, blueprint.to_physics_world())
            world.record(
                "genesis",
                {
                    "seed": seed,
                    "theme": theme,
                    "size": size,
                    "blueprint_digest": blueprint.digest(),
                },
            )
            cancellation_pending = await self._persist(world)
            with self._lock:
                self._worlds[world_id] = world
            if cancellation_pending:
                raise asyncio.CancelledError
            return world.summary()
        finally:
            self._release_reservation(world_id)

    def load_world(self, world_id: str) -> HostedWorld:
        world_id = self._validate_id(world_id)
        with self._lock:
            cached = self._worlds.get(world_id)
            if cached is not None:
                return cached
            if world_id in self._reserved_world_ids:
                raise PhysicsError(f"world '{world_id}' is being created")
        path = self._path(world_id)
        if not path.exists():
            raise PhysicsError(f"unknown world '{world_id}'")
        try:
            if path.stat().st_size > _MAX_WORLD_DOCUMENT_BYTES:
                raise PhysicsError("hosted world document exceeds the size limit")
            document = json.loads(path.read_text(encoding="utf-8"))
            world = HostedWorld.from_document(document)
            if world.world_id != world_id:
                raise PhysicsError("hosted world identity does not match its filename")
        except _HOST_ERRORS as exc:
            record_degradation("worlds.load", exc)
            raise PhysicsError(f"world '{world_id}' could not be loaded") from exc
        with self._lock:
            self._worlds.setdefault(world_id, world)
            return self._worlds[world_id]

    def list_worlds(self) -> list[dict[str, Any]]:
        summaries: dict[str, dict[str, Any]] = {}
        with self._lock:
            for world in self._worlds.values():
                summaries[world.world_id] = world.summary()
        if self.root.exists():
            for path in sorted(self.root.glob("*.json")):
                world_id = path.stem
                if world_id not in summaries and _WORLD_ID_PATTERN.match(world_id):
                    try:
                        summaries[world_id] = self.load_world(world_id).summary()
                    except _HOST_ERRORS as exc:
                        record_degradation("worlds.list", exc)
                        summaries[world_id] = {"world_id": world_id, "error": "unreadable"}
        return [summaries[key] for key in sorted(summaries)]

    # ── interaction ────────────────────────────────────────────

    async def step_world(self, world_id: str, ticks: int) -> dict[str, Any]:
        if not 1 <= ticks <= MAX_TICKS_PER_STEP:
            raise PhysicsError(f"ticks must be in [1, {MAX_TICKS_PER_STEP}]")
        world_id = self._validate_id(world_id)
        async with self._mutation_lock(world_id):
            world = self.load_world(world_id)
            staged = world.clone()
            energy_before = staged.physics.total_kinetic_energy()
            notable_hits = 0
            for _ in range(ticks):
                staged.physics.step()
                for agent in staged.agents.values():
                    agent._carry_held_body()
                for contact in staged.physics.last_contacts:
                    if contact.impulse >= _IMPULSE_JOURNAL_THRESHOLD:
                        notable_hits += 1
                        staged.record(
                            "impact",
                            {
                                "bodies": [contact.body_a, contact.body_b],
                                "impulse": round(contact.impulse, 4),
                            },
                        )
            staged.record(
                "stepped",
                {
                    "ticks": ticks,
                    "energy_before": round(energy_before, 6),
                    "energy_after": round(staged.physics.total_kinetic_energy(), 6),
                    "notable_impacts": notable_hits,
                },
            )
            staged.updated_at = time.time()
            await self._commit_existing(world, staged)
            return world.summary()

    async def apply_impulse(
        self, world_id: str, body_id: str, impulse: tuple[float, float, float]
    ) -> dict[str, Any]:
        vector = np.asarray(impulse, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise PhysicsError("impulse must be a finite 3-vector")
        magnitude = float(np.linalg.norm(vector))
        if magnitude > 1e4:
            raise PhysicsError("impulse magnitude cap is 1e4")
        world_id = self._validate_id(world_id)
        async with self._mutation_lock(world_id):
            world = self.load_world(world_id)
            staged = world.clone()
            body = staged.physics.body(body_id)
            if body.is_static:
                raise PhysicsError(f"body '{body_id}' is static")
            body.velocity = body.velocity + vector / body.mass
            body.sleeping = False
            body.still_ticks = 0
            staged.record(
                "impulse",
                {
                    "body": body_id,
                    "impulse": [float(x) for x in vector],
                    "magnitude": round(magnitude, 4),
                },
            )
            staged.updated_at = time.time()
            await self._commit_existing(world, staged)
            return world.summary()

    # ── embodiment ─────────────────────────────────────────────

    async def spawn_agent(self, world_id: str, agent_id: str = "agent") -> dict[str, Any]:
        world_id = self._validate_id(world_id)
        agent_id = self._validate_id(agent_id)
        async with self._mutation_lock(world_id):
            world = self.load_world(world_id)
            staged = world.clone()
            if agent_id in staged.agents:
                raise PhysicsError(f"agent '{agent_id}' already exists")
            agent = EmbodiedAgent.spawn(
                staged.physics,
                staged.blueprint,
                agent_id=agent_id,
            )
            staged.agents[agent_id] = agent
            state = agent.proprioception()
            staged.record("agent_spawned", {"agent": agent_id, "position": state["position"]})
            staged.updated_at = time.time()
            await self._commit_existing(world, staged)
            return state

    async def agent_command(
        self, world_id: str, agent_id: str, command: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Execute one embodied action and journal it. Commands:
        proprioception | look | walk | jump | grasp | throw | navigate."""
        world_id = self._validate_id(world_id)
        agent_id = self._validate_id(agent_id)
        command = str(command or "").strip().lower()
        async with self._mutation_lock(world_id):
            world = self.load_world(world_id)
            if command == "proprioception":
                return {"ok": True, **world.agent(agent_id).proprioception()}
            if command == "look":
                return {
                    "ok": True,
                    "observations": world.agent(agent_id).look(
                        rays=int(kwargs.get("rays", 8)),
                        max_distance=float(kwargs.get("max_distance", 30.0)),
                    ),
                }

            staged = world.clone()
            agent = staged.agent(agent_id)
            result: dict[str, Any]
            if command == "walk":
                heading = kwargs.get("heading")
                agent.walk(
                    heading=float(heading) if heading is not None else None,
                    ticks=int(kwargs.get("ticks", 60)),
                )
                result = {"ok": True, **agent.proprioception()}
            elif command == "jump":
                result = {"ok": agent.jump(), **agent.proprioception()}
            elif command == "grasp":
                grabbed = agent.grasp(str(kwargs.get("body_id", "") or ""))
                result = {"ok": grabbed, **agent.proprioception()}
            elif command == "throw":
                released = agent.throw(
                    speed=float(kwargs.get("speed", 6.0)),
                    pitch=float(kwargs.get("pitch", 0.35)),
                )
                result = {"ok": True, "released": released, **agent.proprioception()}
            elif command == "navigate":
                target = kwargs.get("target") or (0.0, 0.0)
                if not isinstance(target, (list, tuple)) or len(target) != 2:
                    raise PhysicsError("navigation target must be a two-coordinate sequence")
                outcome = agent.navigate_to(
                    (float(target[0]), float(target[1])),
                    tolerance=float(kwargs.get("tolerance", 1.0)),
                    max_ticks=int(kwargs.get("max_ticks", 12000)),
                )
                result = {
                    "ok": outcome["status"] == "reached",
                    "navigation": outcome,
                    **agent.proprioception(),
                }
            else:
                raise PhysicsError(f"unknown agent command '{command}'")
            staged.record(
                "agent_action",
                {
                    "agent": agent_id,
                    "command": command,
                    "ok": bool(result.get("ok")),
                },
            )
            staged.updated_at = time.time()
            await self._commit_existing(world, staged)
            return result

    # ── counterfactual forking ─────────────────────────────────

    async def fork_world(self, source_id: str, new_id: str) -> dict[str, Any]:
        """Copy a world's exact state into a new world — the substrate for
        counterfactual reasoning: fork, intervene, compare."""
        source_id = self._validate_id(source_id)
        new_id = self._validate_id(new_id)
        async with self._mutation_lock(source_id):
            self._reserve_new_world(new_id)
            try:
                source = self.load_world(source_id)
                document = source.to_document()
                document["world_id"] = new_id
                fork = HostedWorld.from_document(document)
                fork.record(
                    "forked_from",
                    {
                        "source": source_id,
                        "source_tick": source.physics.tick,
                        "source_digest": source.physics.state_digest(),
                    },
                )
                cancellation_pending = await self._persist(fork)
                with self._lock:
                    self._worlds[new_id] = fork
                if cancellation_pending:
                    raise asyncio.CancelledError
                return fork.summary()
            finally:
                self._release_reservation(new_id)

    def compare_worlds(self, id_a: str, id_b: str, *, top: int = 8) -> dict[str, Any]:
        """Divergence report between two worlds (typically a fork pair)."""
        if isinstance(top, bool) or not isinstance(top, int) or not 1 <= top <= 64:
            raise PhysicsError("comparison top must be an integer in [1, 64]")
        world_a, world_b = self.load_world(id_a), self.load_world(id_b)
        with self._lock:
            ids_a = set(world_a.physics.bodies)
            ids_b = set(world_b.physics.bodies)
            shared = sorted(ids_a & ids_b)
            deltas: list[dict[str, Any]] = []
            for key in shared:
                body_a = world_a.physics.bodies[key]
                body_b = world_b.physics.bodies[key]
                position_delta = float(np.linalg.norm(body_a.position - body_b.position))
                velocity_delta = float(np.linalg.norm(body_a.velocity - body_b.velocity))
                attributes_changed = body_a.to_dict() != body_b.to_dict()
                if position_delta > 1e-9 or velocity_delta > 1e-9 or attributes_changed:
                    deltas.append(
                        {
                            "body_id": key,
                            "position_delta": round(position_delta, 6),
                            "velocity_delta": round(velocity_delta, 6),
                            "attributes_changed": attributes_changed,
                        }
                    )
            deltas.sort(
                key=lambda row: (
                    -max(row["position_delta"], row["velocity_delta"]),
                    row["body_id"],
                )
            )

            def _agent_state(world: HostedWorld) -> dict[str, dict[str, Any]]:
                return {
                    agent_id: {
                        "yaw": agent.state.yaw,
                        "held_body": agent.state.held_body,
                        "last_navigation": agent.state.last_navigation,
                    }
                    for agent_id, agent in sorted(world.agents.items())
                }

            agents_a = _agent_state(world_a)
            agents_b = _agent_state(world_b)
            added_bodies = sorted(ids_b - ids_a)
            removed_bodies = sorted(ids_a - ids_b)
            simulation_identical = (
                world_a.physics.to_dict() == world_b.physics.to_dict() and agents_a == agents_b
            )
            return {
                "identical": simulation_identical,
                "state_digest_a": world_a.physics.state_digest(),
                "state_digest_b": world_b.physics.state_digest(),
                "tick_a": world_a.physics.tick,
                "tick_b": world_b.physics.tick,
                "bodies_compared": len(shared),
                "bodies_diverged": len(deltas) + len(added_bodies) + len(removed_bodies),
                "bodies_added": added_bodies,
                "bodies_removed": removed_bodies,
                "agents_diverged": agents_a != agents_b,
                "largest_divergences": deltas[:top],
            }

    def render_state(self, world_id: str) -> dict[str, Any]:
        """Complete geometric state for a renderer: every body with its
        shape, pose, and dimensions. Read-only."""
        world = self.load_world(world_id)
        bodies = []
        for key in sorted(world.physics.bodies):
            body = world.physics.bodies[key]
            bodies.append({
                "body_id": key,
                "shape": body.shape,
                "position": [round(float(x), 5) for x in body.position],
                "orientation": [round(float(x), 6) for x in body.orientation],
                "radius": body.radius,
                "half_extents": [round(float(x), 5) for x in body.half_extents],
                "plane_height": body.plane_height,
                "static": body.is_static,
                "sleeping": body.sleeping,
                "is_agent": key in world.agents,
            })
        return {
            "world_id": world.world_id,
            "name": world.blueprint.name,
            "theme": world.blueprint.theme,
            "tick": world.physics.tick,
            "size": world.blueprint.size,
            "spawn_point": world.blueprint.spawn_point,
            "state_digest": world.physics.state_digest(),
            "bodies": bodies,
        }

    def inspect(self, world_id: str, *, recent_events: int = 10) -> dict[str, Any]:
        world = self.load_world(world_id)
        with self._lock:
            return {
                **world.summary(),
                "spawn_point": world.blueprint.spawn_point,
                "recent_events": world.journal[-max(0, recent_events) :],
                "bodies_preview": [
                    world.physics.bodies[key].to_dict() for key in sorted(world.physics.bodies)[:16]
                ],
            }

    # ── internals ──────────────────────────────────────────────

    def _validate_id(self, world_id: str) -> str:
        raw = str(world_id or "").strip()
        cleaned = raw.lower()
        if raw != cleaned:
            raise PhysicsError("world_id must use lowercase characters")
        if not _WORLD_ID_PATTERN.match(cleaned):
            raise PhysicsError("world_id must be 1-64 chars of lowercase letters, digits, _ or -")
        return cleaned

    def _path(self, world_id: str) -> Path:
        return self.root / f"{world_id}.json"

    def _mutation_lock(self, world_id: str) -> asyncio.Lock:
        with self._lock:
            return self._world_locks.setdefault(world_id, asyncio.Lock())

    def _reserve_new_world(self, world_id: str) -> None:
        with self._lock:
            known = set(self._worlds) | self._reserved_world_ids
            if self.root.exists():
                known.update(
                    path.stem
                    for path in self.root.glob("*.json")
                    if _WORLD_ID_PATTERN.fullmatch(path.stem)
                )
            if world_id in known or self._path(world_id).exists():
                raise PhysicsError(f"world '{world_id}' already exists")
            if len(known) >= MAX_WORLDS:
                raise PhysicsError(f"world cap ({MAX_WORLDS}) reached")
            self._reserved_world_ids.add(world_id)

    def _release_reservation(self, world_id: str) -> None:
        with self._lock:
            self._reserved_world_ids.discard(world_id)

    async def _commit_existing(self, world: HostedWorld, staged: HostedWorld) -> None:
        cancellation_pending = await self._persist(staged)
        with self._lock:
            world.replace_from(staged)
        if cancellation_pending:
            raise asyncio.CancelledError

    async def _persist(self, world: HostedWorld) -> bool:
        async def _write() -> None:
            gateway = get_file_write_gateway()
            with local_internal_governed_scope(
                "worlds.host",
                domain="file_write",
                receipt_prefix="world-host",
            ):
                await gateway.ensure_directory_async(self.root, source="worlds.host")
                await gateway.write_json_async(
                    self._path(world.world_id),
                    world.to_document(),
                    schema_version=_SCHEMA_VERSION,
                    schema_name="hosted_world",
                    source="worlds.host",
                )

        write_task = asyncio.create_task(_write(), name=f"world-persist:{world.world_id}")
        cancellation_pending = False
        try:
            while not write_task.done():
                try:
                    await asyncio.shield(write_task)
                except asyncio.CancelledError:
                    cancellation_pending = True
            write_task.result()
        except _HOST_ERRORS as exc:
            record_degradation("worlds.persist", exc)
            logger.error("Failed to persist world '%s': %s", world.world_id, exc)
            raise WorldPersistenceError(
                f"world '{world.world_id}' could not be committed durably"
            ) from exc
        return cancellation_pending


# ── module singleton ─────────────────────────────────────────────

_host: WorldHost | None = None
_host_lock = threading.Lock()


def worlds_root() -> Path:
    return Path(get_config().paths.data_dir) / "worlds"


def get_world_host() -> WorldHost:
    global _host
    if _host is None:
        with _host_lock:
            if _host is None:
                _host = WorldHost(worlds_root())
    return _host


def reset_world_host_for_tests(root: Path | None = None) -> WorldHost:
    global _host
    with _host_lock:
        _host = WorldHost(root or worlds_root())
        return _host
