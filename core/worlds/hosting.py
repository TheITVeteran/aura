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
from core.worlds.generation import WorldBlueprint, generate_world
from core.worlds.physics import PhysicsError, PhysicsWorld

logger = logging.getLogger("Aura.Worlds")

MAX_TICKS_PER_STEP = 10_000
MAX_WORLDS = 64
_JOURNAL_CAP = 500
_IMPULSE_JOURNAL_THRESHOLD = 2.0
_SCHEMA_VERSION = 1
_WORLD_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_HOST_ERRORS = (OSError, RuntimeError, TypeError, ValueError, KeyError)


class HostedWorld:
    def __init__(self, world_id: str, blueprint: WorldBlueprint, physics: PhysicsWorld,
                 journal: list[dict[str, Any]] | None = None,
                 created_at: float | None = None):
        self.world_id = world_id
        self.blueprint = blueprint
        self.physics = physics
        self.journal: list[dict[str, Any]] = list(journal or [])
        self.created_at = created_at if created_at is not None else time.time()
        self.updated_at = self.created_at

    def record(self, kind: str, detail: dict[str, Any]) -> None:
        self.journal.append({
            "tick": self.physics.tick,
            "at": time.time(),
            "kind": kind,
            **detail,
        })
        if len(self.journal) > _JOURNAL_CAP:
            del self.journal[: len(self.journal) - _JOURNAL_CAP]

    def summary(self) -> dict[str, Any]:
        sleeping = sum(
            1 for body in self.physics.bodies.values()
            if not body.is_static and body.sleeping
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
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "HostedWorld":
        payload = document.get("payload", document)
        world = cls(
            world_id=str(payload["world_id"]),
            blueprint=WorldBlueprint.from_dict(payload["blueprint"]),
            physics=PhysicsWorld.from_dict(payload["physics"]),
            journal=list(payload.get("journal", [])),
            created_at=float(payload.get("created_at", time.time())),
        )
        world.updated_at = float(payload.get("updated_at", world.created_at))
        return world


class WorldHost:
    """Registry of live worlds with governed persistence."""

    def __init__(self, root: Path):
        self.root = root
        self._worlds: dict[str, HostedWorld] = {}
        self._lock = threading.Lock()

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
        with self._lock:
            if world_id in self._worlds or self._path(world_id).exists():
                raise PhysicsError(f"world '{world_id}' already exists")
            if len(self._worlds) >= MAX_WORLDS:
                raise PhysicsError(f"world cap ({MAX_WORLDS}) reached")
        blueprint = generate_world(seed, size=size, theme=theme, name=name or world_id)
        world = HostedWorld(world_id, blueprint, blueprint.to_physics_world())
        world.record("genesis", {
            "seed": seed, "theme": theme, "size": size,
            "blueprint_digest": blueprint.digest(),
        })
        with self._lock:
            self._worlds[world_id] = world
        await self._persist(world)
        return world.summary()

    def load_world(self, world_id: str) -> HostedWorld:
        world_id = self._validate_id(world_id)
        with self._lock:
            cached = self._worlds.get(world_id)
            if cached is not None:
                return cached
        path = self._path(world_id)
        if not path.exists():
            raise PhysicsError(f"unknown world '{world_id}'")
        import json

        document = json.loads(path.read_text(encoding="utf-8"))
        world = HostedWorld.from_document(document)
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
        world = self.load_world(world_id)
        energy_before = world.physics.total_kinetic_energy()
        notable_hits = 0
        for _ in range(ticks):
            world.physics.step()
            for contact in world.physics.last_contacts:
                if contact.impulse >= _IMPULSE_JOURNAL_THRESHOLD:
                    notable_hits += 1
                    world.record("impact", {
                        "bodies": [contact.body_a, contact.body_b],
                        "impulse": round(contact.impulse, 4),
                    })
        world.record("stepped", {
            "ticks": ticks,
            "energy_before": round(energy_before, 6),
            "energy_after": round(world.physics.total_kinetic_energy(), 6),
            "notable_impacts": notable_hits,
        })
        world.updated_at = time.time()
        await self._persist(world)
        return world.summary()

    async def apply_impulse(
        self, world_id: str, body_id: str, impulse: tuple[float, float, float]
    ) -> dict[str, Any]:
        world = self.load_world(world_id)
        body = world.physics.body(body_id)
        if body.is_static:
            raise PhysicsError(f"body '{body_id}' is static")
        vector = np.asarray(impulse, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise PhysicsError("impulse must be a finite 3-vector")
        magnitude = float(np.linalg.norm(vector))
        if magnitude > 1e4:
            raise PhysicsError("impulse magnitude cap is 1e4")
        body.velocity = body.velocity + vector / body.mass
        body.sleeping = False
        body.still_ticks = 0
        world.record("impulse", {
            "body": body_id,
            "impulse": [float(x) for x in vector],
            "magnitude": round(magnitude, 4),
        })
        world.updated_at = time.time()
        await self._persist(world)
        return world.summary()

    def inspect(self, world_id: str, *, recent_events: int = 10) -> dict[str, Any]:
        world = self.load_world(world_id)
        return {
            **world.summary(),
            "spawn_point": world.blueprint.spawn_point,
            "recent_events": world.journal[-max(0, recent_events):],
            "bodies_preview": [
                world.physics.bodies[key].to_dict()
                for key in sorted(world.physics.bodies)[:16]
            ],
        }

    # ── internals ──────────────────────────────────────────────

    def _validate_id(self, world_id: str) -> str:
        cleaned = str(world_id or "").strip().lower()
        if not _WORLD_ID_PATTERN.match(cleaned):
            raise PhysicsError(
                "world_id must be 1-64 chars of lowercase letters, digits, _ or -"
            )
        return cleaned

    def _path(self, world_id: str) -> Path:
        return self.root / f"{world_id}.json"

    async def _persist(self, world: HostedWorld) -> None:
        try:
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
        except _HOST_ERRORS as exc:
            record_degradation("worlds.persist", exc)
            logger.error("Failed to persist world '%s': %s", world.world_id, exc)


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
