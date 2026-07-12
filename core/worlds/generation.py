"""core/worlds/generation.py
─────────────────────────
Seeded procedural world generation.

Worlds are reproducible artifacts: the same (seed, size, theme) always
produces the same blueprint, byte for byte — ``digest()`` is the proof.
Terrain is multi-octave value noise (pure numpy, no dependencies);
entities are placed by the same seeded generator. A blueprint can be
realized into a live PhysicsWorld: terrain becomes static box columns
(coarse but honest voxel terrain the collision engine actually
supports), props become dynamic spheres and boxes.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.worlds.physics import Body, PhysicsError, PhysicsWorld

MAX_WORLD_SIZE = 128
MAX_ENTITIES = 256

THEMES = {
    "plains": {"relief": 1.5, "prop_density": 0.02, "prop_kinds": ("rock", "crate")},
    "highlands": {"relief": 6.0, "prop_density": 0.03, "prop_kinds": ("rock", "boulder")},
    "arena": {"relief": 0.0, "prop_density": 0.05, "prop_kinds": ("crate", "ball", "pillar")},
}


def _value_noise(size: int, seed: int, *, octaves: int = 4) -> np.ndarray:
    """Deterministic multi-octave value noise in [0, 1], shape (size, size)."""
    rng = np.random.default_rng(seed)
    field_sum = np.zeros((size, size), dtype=np.float64)
    amplitude, total_amplitude = 1.0, 0.0
    for octave in range(octaves):
        lattice = max(2, 2 ** (octave + 1))
        coarse = rng.random((lattice + 1, lattice + 1))
        # Bilinear upsample of the coarse lattice to the full grid.
        xs = np.linspace(0.0, lattice, size)
        x0 = np.clip(xs.astype(np.intp), 0, lattice - 1)
        fx = xs - x0
        rows = (
            coarse[x0, :] * (1.0 - fx)[:, None] + coarse[x0 + 1, :] * fx[:, None]
        )
        layer = (
            rows[:, x0] * (1.0 - fx)[None, :] + rows[:, x0 + 1] * fx[None, :]
        )
        field_sum += amplitude * layer
        total_amplitude += amplitude
        amplitude *= 0.5
    return field_sum / total_amplitude


@dataclass
class WorldBlueprint:
    seed: int
    size: int
    theme: str
    heightfield: list[list[float]]
    entities: list[dict[str, Any]]
    spawn_point: list[float]
    schema_version: int = 1
    name: str = ""

    def digest(self) -> str:
        payload = {
            "seed": self.seed,
            "size": self.size,
            "theme": self.theme,
            "heightfield": self.heightfield,
            "entities": self.entities,
            "spawn_point": self.spawn_point,
            "schema_version": self.schema_version,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "size": self.size,
            "theme": self.theme,
            "heightfield": self.heightfield,
            "entities": self.entities,
            "spawn_point": self.spawn_point,
            "schema_version": self.schema_version,
            "name": self.name,
            "digest": self.digest(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorldBlueprint":
        return cls(
            seed=int(payload["seed"]),
            size=int(payload["size"]),
            theme=str(payload["theme"]),
            heightfield=payload["heightfield"],
            entities=payload["entities"],
            spawn_point=list(payload["spawn_point"]),
            schema_version=int(payload.get("schema_version", 1)),
            name=str(payload.get("name", "")),
        )

    # ── realization ────────────────────────────────────────────

    def to_physics_world(
        self, *, terrain_columns: int = 8, dt: float = 1.0 / 120.0
    ) -> PhysicsWorld:
        """Realize the blueprint as a live physics world.

        Terrain enters as ``terrain_columns²`` static box columns sampled
        from the heightfield; props enter as dynamic bodies dropped just
        above the surface."""
        world = PhysicsWorld(dt=dt)
        world.add_body(Body(
            body_id="ground",
            shape="plane",
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            mass=0.0,
            plane_height=0.0,
            restitution=0.3,
            friction=0.8,
        ))
        heights = np.asarray(self.heightfield, dtype=np.float64)
        columns = max(1, min(int(terrain_columns), self.size))
        cell = self.size / columns
        for i in range(columns):
            for j in range(columns):
                sample_x = min(int((i + 0.5) * cell), self.size - 1)
                sample_y = min(int((j + 0.5) * cell), self.size - 1)
                height = float(heights[sample_x, sample_y])
                if height < 0.05:
                    continue
                world.add_body(Body(
                    body_id=f"terrain_{i}_{j}",
                    shape="box",
                    position=(
                        (i + 0.5) * cell - self.size / 2.0,
                        (j + 0.5) * cell - self.size / 2.0,
                        height / 2.0,
                    ),
                    velocity=(0.0, 0.0, 0.0),
                    mass=0.0,
                    half_extents=(cell / 2.0, cell / 2.0, max(height / 2.0, 1e-3)),
                    restitution=0.2,
                    friction=0.9,
                ))
        for entity in self.entities:
            shape = str(entity.get("shape", "sphere"))
            body = Body(
                body_id=str(entity["entity_id"]),
                shape=shape,
                position=entity["position"],
                velocity=(0.0, 0.0, 0.0),
                mass=float(entity.get("mass", 1.0)),
                radius=float(entity.get("radius", 0.4)),
                half_extents=entity.get("half_extents", (0.4, 0.4, 0.4)),
                restitution=float(entity.get("restitution", 0.4)),
                friction=float(entity.get("friction", 0.5)),
                rolling_resistance=float(entity.get("rolling_resistance", 0.0)),
            )
            world.add_body(body)
        return world


def generate_world(
    seed: int,
    *,
    size: int = 32,
    theme: str = "plains",
    name: str = "",
) -> WorldBlueprint:
    """Deterministically generate a world blueprint from a seed."""
    if not 8 <= size <= MAX_WORLD_SIZE:
        raise PhysicsError(f"size must be in [8, {MAX_WORLD_SIZE}]")
    if theme not in THEMES:
        raise PhysicsError(f"theme must be one of {sorted(THEMES)}")
    config = THEMES[theme]
    rng = np.random.default_rng(seed)

    relief = float(config["relief"])
    noise = _value_noise(size, seed)
    heights = np.round(noise * relief, 6)
    heightfield = [[float(h) for h in row] for row in heights]

    prop_count = min(
        MAX_ENTITIES, max(0, int(size * size * float(config["prop_density"])))
    )
    entities: list[dict[str, Any]] = []
    kinds = tuple(config["prop_kinds"])
    for index in range(prop_count):
        kind = kinds[int(rng.integers(0, len(kinds)))]
        x = float(rng.uniform(-size / 2.0, size / 2.0))
        y = float(rng.uniform(-size / 2.0, size / 2.0))
        grid_x = min(int(x + size / 2.0), size - 1)
        grid_y = min(int(y + size / 2.0), size - 1)
        surface = float(heights[grid_x, grid_y])
        if kind in {"ball", "rock"}:
            radius = float(np.round(rng.uniform(0.2, 0.6), 4))
            entities.append({
                "entity_id": f"{kind}_{index}",
                "kind": kind,
                "shape": "sphere",
                "radius": radius,
                "mass": float(np.round(4.19 * radius ** 3, 4)),
                "position": [round(x, 4), round(y, 4), round(surface + radius + 0.5, 4)],
                "restitution": 0.55 if kind == "ball" else 0.25,
                "rolling_resistance": 0.02,
            })
        elif kind == "boulder":
            radius = float(np.round(rng.uniform(0.6, 1.4), 4))
            entities.append({
                "entity_id": f"boulder_{index}",
                "kind": kind,
                "shape": "sphere",
                "radius": radius,
                "mass": float(np.round(8.0 * radius ** 3, 4)),
                "position": [round(x, 4), round(y, 4), round(surface + radius + 0.5, 4)],
                "restitution": 0.15,
                "rolling_resistance": 0.03,
            })
        else:  # crate / pillar → boxes
            half = 0.4 if kind == "crate" else 0.3
            height_half = half if kind == "crate" else 1.2
            entities.append({
                "entity_id": f"{kind}_{index}",
                "kind": kind,
                "shape": "box",
                "half_extents": [half, half, height_half],
                "mass": 0.0 if kind == "pillar" else float(np.round(rng.uniform(0.5, 3.0), 4)),
                "position": [round(x, 4), round(y, 4), round(surface + height_half + 0.5, 4)],
                "restitution": 0.3,
            })

    center = size // 2
    spawn_height = float(heights[center, center]) + 1.0
    return WorldBlueprint(
        seed=int(seed),
        size=int(size),
        theme=theme,
        heightfield=heightfield,
        entities=entities,
        spawn_point=[0.0, 0.0, round(spawn_height, 6)],
        name=name or f"{theme}-{seed}",
    )
