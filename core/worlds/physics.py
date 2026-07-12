"""core/worlds/physics.py
───────────────────────
Deterministic 3D rigid-body physics (rotational-sphere v2).

Engineering contract:
- Fixed timestep, semi-implicit (symplectic) Euler integration. The
  discrete trajectory has a closed form under constant gravity —
  x_n = x_0 + n·v_0·dt + g·dt²·n(n+1)/2 — and the tests assert it to
  1e-9, not "roughly parabolic".
- Impulse-based contact resolution with restitution and Coulomb
  friction; positional projection with a slop tolerance kills sink-in.
- Strict determinism: bodies iterate in sorted-id order, contacts
  resolve in a canonical order, no wall-clock or global RNG anywhere.
  ``state_digest()`` is the proof surface — identical worlds stepped
  identically produce identical digests.
- Sleeping: bodies below the velocity epsilon for a full damp window
  stop integrating; any contact impulse wakes them.

Shapes: dynamic spheres, dynamic axis-aligned boxes, static planes
(z = height, normal +z), static boxes. Z is up.

Rotational dynamics (v2): spheres carry full angular state — quaternion
orientation, world-frame angular velocity, solid-sphere inertia
I = (2/5)mr². Contact friction acts at the contact point, exchanging
linear and angular momentum, so a sliding ball spins up and converges
to rolling without slipping at exactly 5/7 of its initial speed (the
classical result; the tests assert it). Optional rolling resistance
lets rollers come to rest.

Remaining declared limitation: boxes stay axis-aligned and do not
rotate (their inverse inertia is zero). Oriented-box dynamics with SAT
contact manifolds is v3 territory, stated plainly.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

_SLEEP_SPEED = 5e-3
_SLEEP_TICKS = 30
_PENETRATION_SLOP = 1e-4
_POSITION_CORRECTION = 0.8
# Below this approach speed a contact stops bouncing (restitution 0):
# the standard resting-contact treatment that lets bodies settle instead
# of micro-bouncing forever on gravity's per-tick velocity kick.
_RESTITUTION_SPEED_THRESHOLD = 0.15


class PhysicsError(ValueError):
    """Invalid body construction or world operation."""


type FloatArray = NDArray[np.float64]


def _vec(value: Iterable[float], name: str) -> FloatArray:
    arr = np.asarray(tuple(value), dtype=np.float64)
    if arr.shape != (3,):
        raise PhysicsError(f"{name} must be a 3-vector")
    if not np.all(np.isfinite(arr)):
        raise PhysicsError(f"{name} must be finite")
    return cast(FloatArray, arr.copy())


@dataclass
class Body:
    body_id: str
    shape: str  # "sphere" | "box" | "plane"
    position: np.ndarray
    velocity: np.ndarray
    mass: float = 1.0  # 0.0 → static
    radius: float = 0.5
    half_extents: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))
    plane_height: float = 0.0
    restitution: float = 0.5
    friction: float = 0.4
    rolling_resistance: float = 0.0
    angular_locked: bool = False  # e.g. the embodied agent: never spins
    orientation: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0])
    )  # quaternion (w, x, y, z)
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sleeping: bool = False
    still_ticks: int = 0

    def __post_init__(self) -> None:
        if self.shape not in {"sphere", "box", "plane"}:
            raise PhysicsError(f"unknown shape '{self.shape}'")
        self.position = _vec(self.position, "position")
        self.velocity = _vec(self.velocity, "velocity")
        self.half_extents = _vec(self.half_extents, "half_extents")
        if self.mass < 0.0 or not math.isfinite(self.mass):
            raise PhysicsError("mass must be >= 0 (0 means static)")
        if self.shape == "sphere" and self.radius <= 0.0:
            raise PhysicsError("sphere radius must be positive")
        if self.shape == "box" and np.any(self.half_extents <= 0.0):
            raise PhysicsError("box half_extents must be positive")
        if self.shape == "plane" and self.mass != 0.0:
            raise PhysicsError("planes must be static (mass 0)")
        if not 0.0 <= self.restitution <= 1.0:
            raise PhysicsError("restitution must be in [0, 1]")
        if self.friction < 0.0:
            raise PhysicsError("friction must be >= 0")
        if self.rolling_resistance < 0.0:
            raise PhysicsError("rolling_resistance must be >= 0")
        self.orientation = np.asarray(self.orientation, dtype=np.float64)
        if self.orientation.shape != (4,):
            raise PhysicsError("orientation must be a quaternion (w, x, y, z)")
        norm = float(np.linalg.norm(self.orientation))
        if norm <= 1e-12:
            raise PhysicsError("orientation quaternion must be nonzero")
        self.orientation = self.orientation / norm
        self.angular_velocity = _vec(self.angular_velocity, "angular_velocity")

    @property
    def is_static(self) -> bool:
        return self.mass == 0.0

    @property
    def inverse_mass(self) -> float:
        return 0.0 if self.is_static else 1.0 / self.mass

    @property
    def inertia(self) -> float:
        """Scalar (isotropic) moment of inertia. Only spheres rotate."""
        if self.shape != "sphere" or self.is_static or self.angular_locked:
            return 0.0
        return 0.4 * self.mass * self.radius ** 2

    @property
    def inverse_inertia(self) -> float:
        inertia = self.inertia
        return 0.0 if inertia <= 0.0 else 1.0 / inertia

    def kinetic_energy(self) -> float:
        if self.is_static:
            return 0.0
        linear = 0.5 * self.mass * float(self.velocity @ self.velocity)
        angular = 0.5 * self.inertia * float(
            self.angular_velocity @ self.angular_velocity
        )
        return linear + angular

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_id": self.body_id,
            "shape": self.shape,
            "position": [float(x) for x in self.position],
            "velocity": [float(x) for x in self.velocity],
            "mass": self.mass,
            "radius": self.radius,
            "half_extents": [float(x) for x in self.half_extents],
            "plane_height": self.plane_height,
            "restitution": self.restitution,
            "friction": self.friction,
            "rolling_resistance": self.rolling_resistance,
            "angular_locked": self.angular_locked,
            "orientation": [float(x) for x in self.orientation],
            "angular_velocity": [float(x) for x in self.angular_velocity],
            "sleeping": self.sleeping,
            "still_ticks": self.still_ticks,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Body:
        return cls(
            body_id=str(row["body_id"]),
            shape=str(row["shape"]),
            position=row["position"],
            velocity=row["velocity"],
            mass=float(row.get("mass", 1.0)),
            radius=float(row.get("radius", 0.5)),
            half_extents=row.get("half_extents", (0.5, 0.5, 0.5)),
            plane_height=float(row.get("plane_height", 0.0)),
            restitution=float(row.get("restitution", 0.5)),
            friction=float(row.get("friction", 0.4)),
            rolling_resistance=float(row.get("rolling_resistance", 0.0)),
            angular_locked=bool(row.get("angular_locked", False)),
            orientation=row.get("orientation", (1.0, 0.0, 0.0, 0.0)),
            angular_velocity=row.get("angular_velocity", (0.0, 0.0, 0.0)),
            sleeping=bool(row.get("sleeping", False)),
            still_ticks=int(row.get("still_ticks", 0)),
        )


@dataclass
class Contact:
    body_a: str
    body_b: str
    normal: np.ndarray  # from a toward b
    penetration: float
    impulse: float = 0.0


class PhysicsWorld:
    """A deterministic rigid-body world with a fixed timestep."""

    def __init__(self, *, gravity: float = -9.81, dt: float = 1.0 / 120.0):
        if dt <= 0.0 or dt > 0.1:
            raise PhysicsError("dt must be in (0, 0.1]")
        self.gravity = np.array([0.0, 0.0, float(gravity)], dtype=np.float64)
        self.dt = float(dt)
        self.bodies: dict[str, Body] = {}
        self.tick = 0
        self.last_contacts: list[Contact] = []

    # ── population ─────────────────────────────────────────────

    def add_body(self, body: Body) -> Body:
        if body.body_id in self.bodies:
            raise PhysicsError(f"duplicate body_id '{body.body_id}'")
        self.bodies[body.body_id] = body
        return body

    def remove_body(self, body_id: str) -> bool:
        return self.bodies.pop(body_id, None) is not None

    def body(self, body_id: str) -> Body:
        try:
            return self.bodies[body_id]
        except KeyError as exc:
            raise PhysicsError(f"unknown body '{body_id}'") from exc

    # ── stepping ───────────────────────────────────────────────

    def step(self, ticks: int = 1) -> None:
        if ticks < 1:
            raise PhysicsError("ticks must be >= 1")
        for _ in range(ticks):
            self._integrate()
            contacts = self._detect_contacts()
            self._resolve_contacts(contacts)
            self._update_sleep_state()
            self.last_contacts = contacts
            self.tick += 1

    def _dynamic_bodies(self) -> list[Body]:
        return [
            self.bodies[key]
            for key in sorted(self.bodies)
            if not self.bodies[key].is_static
        ]

    def _integrate(self) -> None:
        for body in self._dynamic_bodies():
            if body.sleeping:
                continue
            body.velocity = body.velocity + self.gravity * self.dt
            body.position = body.position + body.velocity * self.dt
            if body.inverse_inertia > 0.0 and float(
                body.angular_velocity @ body.angular_velocity
            ) > 0.0:
                # q̇ = ½ ω ⊗ q with ω as a pure quaternion (world frame).
                w, x, y, z = body.orientation
                ox, oy, oz = body.angular_velocity
                delta = 0.5 * self.dt * np.array([
                    -ox * x - oy * y - oz * z,
                    ox * w + oy * z - oz * y,
                    oy * w + oz * x - ox * z,
                    oz * w + ox * y - oy * x,
                ])
                body.orientation = body.orientation + delta
                body.orientation /= float(np.linalg.norm(body.orientation))

    # ── collision detection ────────────────────────────────────

    def _detect_contacts(self) -> list[Contact]:
        contacts: list[Contact] = []
        keys = sorted(self.bodies)
        for i, key_a in enumerate(keys):
            for key_b in keys[i + 1:]:
                a, b = self.bodies[key_a], self.bodies[key_b]
                if a.is_static and b.is_static:
                    continue
                contact = self._pair_contact(a, b)
                if contact is not None:
                    contacts.append(contact)
        return contacts

    def _pair_contact(self, a: Body, b: Body) -> Contact | None:
        # Order the pair canonically by shape so each case is handled once.
        if (a.shape, b.shape) in {("plane", "sphere"), ("plane", "box"), ("box", "sphere")}:
            flipped = self._pair_contact(b, a)
            if flipped is not None:
                flipped.body_a, flipped.body_b = flipped.body_b, flipped.body_a
                flipped.normal = -flipped.normal
            return flipped

        if a.shape == "sphere" and b.shape == "sphere":
            delta = b.position - a.position
            dist = float(np.linalg.norm(delta))
            overlap = a.radius + b.radius - dist
            if overlap <= 0.0:
                return None
            normal = delta / dist if dist > 1e-12 else np.array([0.0, 0.0, 1.0])
            return Contact(a.body_id, b.body_id, normal, overlap)

        if a.shape == "sphere" and b.shape == "plane":
            gap = (a.position[2] - a.radius) - b.plane_height
            if gap >= 0.0:
                return None
            return Contact(a.body_id, b.body_id, np.array([0.0, 0.0, -1.0]), -gap)

        if a.shape == "sphere" and b.shape == "box":
            closest = np.clip(
                a.position, b.position - b.half_extents, b.position + b.half_extents
            )
            delta = a.position - closest
            dist = float(np.linalg.norm(delta))
            if dist >= a.radius:
                return None
            if dist > 1e-12:
                normal = -delta / dist  # from sphere toward box
                return Contact(a.body_id, b.body_id, normal, a.radius - dist)
            # Sphere center inside the box: push out along the axis of
            # least penetration (deterministic tie-break by axis order).
            face_gaps = np.concatenate([
                (b.position + b.half_extents) - a.position,
                a.position - (b.position - b.half_extents),
            ])
            axis = int(np.argmin(face_gaps))
            normal = np.zeros(3)
            normal[axis % 3] = -1.0 if axis < 3 else 1.0
            return Contact(a.body_id, b.body_id, normal, float(face_gaps[axis]) + a.radius)

        if a.shape == "box" and b.shape == "box":
            gaps = (a.half_extents + b.half_extents) - np.abs(b.position - a.position)
            if np.any(gaps <= 0.0):
                return None
            axis = int(np.argmin(gaps))
            normal = np.zeros(3)
            normal[axis] = math.copysign(1.0, float(b.position[axis] - a.position[axis]) or 1.0)
            return Contact(a.body_id, b.body_id, normal, float(gaps[axis]))

        if a.shape == "box" and b.shape == "plane":
            gap = (a.position[2] - a.half_extents[2]) - b.plane_height
            if gap >= 0.0:
                return None
            return Contact(a.body_id, b.body_id, np.array([0.0, 0.0, -1.0]), -gap)

        return None

    # ── collision response ─────────────────────────────────────

    def _resolve_contacts(self, contacts: list[Contact]) -> None:
        for contact in contacts:
            a, b = self.bodies[contact.body_a], self.bodies[contact.body_b]
            inv_mass_sum = a.inverse_mass + b.inverse_mass
            if inv_mass_sum <= 0.0:
                continue
            normal = contact.normal
            relative_velocity = b.velocity - a.velocity
            approach_speed = float(relative_velocity @ normal)

            if approach_speed < 0.0:
                restitution = (
                    min(a.restitution, b.restitution)
                    if -approach_speed > _RESTITUTION_SPEED_THRESHOLD
                    else 0.0
                )
                impulse = -(1.0 + restitution) * approach_speed / inv_mass_sum
                impulse_vector = impulse * normal
                a.velocity = a.velocity - impulse_vector * a.inverse_mass
                b.velocity = b.velocity + impulse_vector * b.inverse_mass
                contact.impulse = impulse
                # Only a genuine hit wakes bodies; the per-tick micro-impulse
                # of a resting contact must not reset the sleep counter.
                if -approach_speed > _RESTITUTION_SPEED_THRESHOLD:
                    self._wake(a)
                    self._wake(b)

                # Coulomb friction at the CONTACT POINT: the relative
                # velocity includes each sphere's surface motion (ω × r),
                # and the impulse exchanges linear and angular momentum —
                # this is what makes sliding balls spin up into rolling.
                r_a = a.radius * normal if a.inverse_inertia > 0.0 else None
                r_b = -b.radius * normal if b.inverse_inertia > 0.0 else None
                contact_velocity = b.velocity - a.velocity
                if r_b is not None:
                    contact_velocity = contact_velocity + np.cross(b.angular_velocity, r_b)
                if r_a is not None:
                    contact_velocity = contact_velocity - np.cross(a.angular_velocity, r_a)
                tangent_velocity = contact_velocity - float(contact_velocity @ normal) * normal
                tangent_speed = float(np.linalg.norm(tangent_velocity))
                if tangent_speed > 1e-9:
                    tangent = tangent_velocity / tangent_speed
                    friction = math.sqrt(a.friction * b.friction)
                    # Effective mass along the tangent, with rotational terms.
                    k_tangent = inv_mass_sum
                    if r_a is not None:
                        arm = np.cross(r_a, tangent)
                        k_tangent += a.inverse_inertia * float(arm @ arm)
                    if r_b is not None:
                        arm = np.cross(r_b, tangent)
                        k_tangent += b.inverse_inertia * float(arm @ arm)
                    friction_impulse = min(tangent_speed / k_tangent, friction * impulse)
                    friction_vector = friction_impulse * tangent
                    a.velocity = a.velocity + friction_vector * a.inverse_mass
                    b.velocity = b.velocity - friction_vector * b.inverse_mass
                    if r_a is not None:
                        a.angular_velocity = a.angular_velocity + (
                            a.inverse_inertia * np.cross(r_a, friction_vector)
                        )
                    if r_b is not None:
                        b.angular_velocity = b.angular_velocity - (
                            b.inverse_inertia * np.cross(r_b, friction_vector)
                        )

                # Rolling resistance: a small decelerating budget against
                # sustained rolling, proportional to the normal impulse.
                self._apply_rolling_resistance(a, b, normal, impulse)

            # Positional projection so resting bodies don't sink.
            correction = max(contact.penetration - _PENETRATION_SLOP, 0.0)
            if correction > 0.0:
                offset = (_POSITION_CORRECTION * correction / inv_mass_sum) * normal
                a.position = a.position - offset * a.inverse_mass
                b.position = b.position + offset * b.inverse_mass

    @staticmethod
    def _apply_rolling_resistance(a: Body, b: Body, normal: np.ndarray, impulse: float) -> None:
        """Modeled (not exact) rolling drag: scales the tangential linear
        and angular velocity down in proportion to the normal impulse, so
        rollers with a nonzero coefficient eventually stop and sleep."""
        for body in (a, b):
            if body.inverse_inertia <= 0.0 or body.rolling_resistance <= 0.0:
                continue
            tangential = body.velocity - float(body.velocity @ normal) * normal
            speed = float(np.linalg.norm(tangential))
            if speed <= 1e-9:
                continue
            decel = min(speed, body.rolling_resistance * impulse * body.inverse_mass)
            scale = 1.0 - decel / speed
            body.velocity = body.velocity - tangential * (1.0 - scale)
            body.angular_velocity = body.angular_velocity * scale

    def _wake(self, body: Body) -> None:
        body.sleeping = False
        body.still_ticks = 0

    def _update_sleep_state(self) -> None:
        for body in self._dynamic_bodies():
            speed = float(np.linalg.norm(body.velocity))
            if body.inverse_inertia > 0.0:
                # A spinning body is not at rest even if its center is.
                speed += float(np.linalg.norm(body.angular_velocity)) * body.radius
            if speed < _SLEEP_SPEED:
                body.still_ticks += 1
                if body.still_ticks >= _SLEEP_TICKS:
                    body.sleeping = True
                    body.velocity[:] = 0.0
                    body.angular_velocity[:] = 0.0
            else:
                body.still_ticks = 0
                body.sleeping = False

    # ── observability ──────────────────────────────────────────

    def total_kinetic_energy(self) -> float:
        return sum(body.kinetic_energy() for body in self.bodies.values())

    def total_momentum(self) -> np.ndarray:
        momentum = np.zeros(3)
        for body in self.bodies.values():
            if not body.is_static:
                momentum += body.mass * body.velocity
        return momentum

    def state_digest(self) -> str:
        """Canonical hash of the dynamical state — the determinism proof."""
        payload = {
            "tick": self.tick,
            "bodies": [
                {
                    "id": key,
                    "p": [round(float(x), 12) for x in self.bodies[key].position],
                    "v": [round(float(x), 12) for x in self.bodies[key].velocity],
                    "w": [round(float(x), 12) for x in self.bodies[key].angular_velocity],
                    "q": [round(float(x), 12) for x in self.bodies[key].orientation],
                    "sleeping": self.bodies[key].sleeping,
                }
                for key in sorted(self.bodies)
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "gravity": float(self.gravity[2]),
            "dt": self.dt,
            "tick": self.tick,
            "bodies": [self.bodies[key].to_dict() for key in sorted(self.bodies)],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PhysicsWorld:
        world = cls(gravity=float(payload["gravity"]), dt=float(payload["dt"]))
        world.tick = int(payload.get("tick", 0))
        for row in payload.get("bodies", []):
            world.add_body(Body.from_dict(row))
        return world
