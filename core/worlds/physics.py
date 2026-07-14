"""core/worlds/physics.py
───────────────────────
Deterministic 3D rigid-body physics (oriented-body v3).

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

Rotational dynamics: spheres and opted-in oriented boxes carry quaternion
orientation and world-frame angular velocity. Sphere inertia is isotropic;
box inertia is a rotated tensor. Contact-point impulses exchange linear and
angular momentum. Oriented boxes use SAT with clipped manifolds, while boxes
that do not opt in retain the deterministic axis-aligned path. Optional
rolling resistance lets rotating bodies come to rest.
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
# Explicit wake requires a decisively fast impact. Slow contacts that
# actually move a body still keep it awake through the sleep-state speed
# check; this threshold only stops the penetration/separation limit
# cycle of settled stacks (~2 ticks of free fall) from faking hits.
_WAKE_SPEED_THRESHOLD = 0.3


class PhysicsError(ValueError):
    """Invalid body construction or world operation."""


type FloatArray = NDArray[np.float64]


def _vec(value: Iterable[float], name: str) -> FloatArray:
    try:
        arr = np.asarray(tuple(value), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhysicsError(f"{name} must be a numeric 3-vector") from exc
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
    oriented: bool = False  # boxes: opt into full 6-DoF rotation (SAT)
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
        """Scalar (isotropic) moment of inertia — spheres only."""
        if self.shape != "sphere" or self.is_static or self.angular_locked:
            return 0.0
        return 0.4 * self.mass * self.radius ** 2

    @property
    def inverse_inertia(self) -> float:
        inertia = self.inertia
        return 0.0 if inertia <= 0.0 else 1.0 / inertia

    @property
    def rotates(self) -> bool:
        if self.is_static or self.angular_locked:
            return False
        return self.shape == "sphere" or (self.shape == "box" and self.oriented)

    def rotation_matrix(self) -> np.ndarray:
        from core.worlds.obb import quat_to_matrix

        return cast(np.ndarray, quat_to_matrix(self.orientation))

    def inverse_inertia_world(self) -> np.ndarray:
        """World-frame inverse inertia tensor (3×3). Isotropic for
        spheres; box tensor I=m/3·diag(hy²+hz², …) rotated into world;
        zeros for static/locked/translational bodies."""
        if not self.rotates:
            return np.zeros((3, 3))
        if self.shape == "sphere":
            return cast(np.ndarray, np.eye(3) * self.inverse_inertia)
        hx, hy, hz = (float(v) for v in self.half_extents)
        diagonal = (self.mass / 3.0) * np.array([
            hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy,
        ])
        rotation = self.rotation_matrix()
        return cast(np.ndarray, rotation @ np.diag(1.0 / diagonal) @ rotation.T)

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
            "oriented": self.oriented,
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
            oriented=bool(row.get("oriented", False)),
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
    point: np.ndarray | None = None  # world contact point (manifold paths)
    manifold_index: int = 0          # position within the pair's manifold
    impulse: float = 0.0            # accumulated normal impulse this tick
    friction_impulse: np.ndarray = field(default_factory=lambda: np.zeros(3))
    restitution_bias: float = 0.0   # target separating speed (e · approach)
    k_normal: float = 0.0           # effective mass along the normal
    r_a: np.ndarray | None = None   # contact arm on a (rotating bodies)
    r_b: np.ndarray | None = None
    inv_inertia_a: np.ndarray | None = None  # cached world tensors
    inv_inertia_b: np.ndarray | None = None


class PhysicsWorld:
    """A deterministic rigid-body world with a fixed timestep."""

    def __init__(
        self,
        *,
        gravity: float = -9.81,
        dt: float = 1.0 / 120.0,
        velocity_iterations: int = 8,
    ):
        if dt <= 0.0 or dt > 0.1:
            raise PhysicsError("dt must be in (0, 0.1]")
        if not 1 <= int(velocity_iterations) <= 64:
            raise PhysicsError("velocity_iterations must be in [1, 64]")
        self.gravity = np.array([0.0, 0.0, float(gravity)], dtype=np.float64)
        self.dt = float(dt)
        self.velocity_iterations = int(velocity_iterations)
        self.bodies: dict[str, Body] = {}
        self.tick = 0
        self.last_contacts: list[Contact] = []
        # Warm-start cache: last tick's accumulated impulses per pair.
        # Re-applying them as the solver's starting point is what makes
        # stacked bodies converge instead of jittering (Box2D technique).
        self._contact_cache: dict[tuple[str, str, int], tuple[float, np.ndarray]] = {}

    # ── population ─────────────────────────────────────────────

    def add_body(self, body: Body) -> Body:
        if body.body_id in self.bodies:
            raise PhysicsError(f"duplicate body_id '{body.body_id}'")
        self.bodies[body.body_id] = body
        return body

    def remove_body(self, body_id: str) -> bool:
        removed = self.bodies.pop(body_id, None) is not None
        if removed:
            # Purge warm-start entries: a future body reusing this id must
            # not inherit a ghost impulse from its predecessor.
            self._contact_cache = {
                pair: value for pair, value in self._contact_cache.items()
                if body_id not in pair[:2]
            }
        return removed

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
            self._update_sleep_state(contacts)
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
            if body.rotates and float(
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
                if (a.shape == "box" and a.oriented) or (
                    b.shape == "box" and b.oriented
                ):
                    contacts.extend(self._oriented_pair_contacts(a, b))
                else:
                    contact = self._pair_contact(a, b)
                    if contact is not None:
                        contacts.append(contact)
        return contacts

    def _oriented_pair_contacts(self, a: Body, b: Body) -> list[Contact]:
        """Manifold contacts for pairs involving an oriented box (SAT)."""
        from core.worlds import obb

        # Canonicalize so the oriented box is X; flip normals if swapped.
        flipped = not (a.shape == "box" and a.oriented)
        box, other = (b, a) if flipped else (a, b)
        rotation = box.rotation_matrix()
        if other.shape == "plane":
            manifold = obb.obb_vs_plane(
                box.position, rotation, box.half_extents, other.plane_height)
        elif other.shape == "sphere":
            manifold = obb.obb_vs_sphere(
                box.position, rotation, box.half_extents,
                other.position, other.radius)
        else:  # box (axis-aligned static or oriented)
            other_rotation = (
                other.rotation_matrix() if other.oriented else np.eye(3))
            manifold = obb.obb_vs_obb(
                box.position, rotation, box.half_extents,
                other.position, other_rotation, other.half_extents)
        if manifold is None:
            return []
        contacts: list[Contact] = []
        for index, contact_point in enumerate(manifold.points):
            normal = manifold.normal if not flipped else -manifold.normal
            contacts.append(Contact(
                body_a=a.body_id,
                body_b=b.body_id,
                normal=normal.copy(),
                penetration=float(contact_point.penetration),
                point=contact_point.point.copy(),
                manifold_index=index,
            ))
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
        """Sequential impulse solver (Box2D-style): warm-started
        accumulated impulses iterated to convergence, friction clamped to
        the Coulomb disk of the accumulated normal impulse, restitution as
        a velocity bias from the pre-solve approach speed."""
        solvable: list[Contact] = []
        for contact in contacts:
            a, b = self.bodies[contact.body_a], self.bodies[contact.body_b]
            inv_mass_sum = a.inverse_mass + b.inverse_mass
            if inv_mass_sum <= 0.0:
                continue
            # Island sleeping: a contact whose every dynamic participant is
            # asleep is skipped outright — no warm-start nudges, no
            # correction creep. Settled stacks stay settled.
            if (a.is_static or a.sleeping) and (b.is_static or b.sleeping):
                continue
            if contact.point is not None:
                contact.r_a = (contact.point - a.position) if a.rotates else None
                contact.r_b = (contact.point - b.position) if b.rotates else None
            else:
                contact.r_a = a.radius * contact.normal if a.rotates else None
                contact.r_b = -b.radius * contact.normal if b.rotates else None
            contact.inv_inertia_a = a.inverse_inertia_world() if contact.r_a is not None else None
            contact.inv_inertia_b = b.inverse_inertia_world() if contact.r_b is not None else None
            contact.k_normal = inv_mass_sum
            if contact.r_a is not None:
                arm = np.cross(contact.r_a, contact.normal)
                contact.k_normal += float(
                    contact.normal @ np.cross(contact.inv_inertia_a @ arm, contact.r_a))
            if contact.r_b is not None:
                arm = np.cross(contact.r_b, contact.normal)
                contact.k_normal += float(
                    contact.normal @ np.cross(contact.inv_inertia_b @ arm, contact.r_b))
            approach_velocity = b.velocity - a.velocity
            if contact.r_b is not None:
                approach_velocity = approach_velocity + np.cross(b.angular_velocity, contact.r_b)
            if contact.r_a is not None:
                approach_velocity = approach_velocity - np.cross(a.angular_velocity, contact.r_a)
            approach_speed = float(approach_velocity @ contact.normal)
            if -approach_speed > _RESTITUTION_SPEED_THRESHOLD:
                contact.restitution_bias = (
                    -min(a.restitution, b.restitution) * approach_speed
                )
            if -approach_speed > _WAKE_SPEED_THRESHOLD:
                # A genuine hit wakes bodies; resting micro-contacts and
                # settling limit cycles don't.
                self._wake(a)
                self._wake(b)
            solvable.append(contact)

        # Warm start AFTER all approach speeds are read: applying cached
        # impulses mid-setup would contaminate the next contact's reading
        # and fake a "hit" every tick, defeating sleep.
        for contact in solvable:
            cached = self._contact_cache.get(
                (contact.body_a, contact.body_b, contact.manifold_index))
            if cached is not None:
                cached_normal, cached_friction = cached
                contact.impulse = cached_normal
                contact.friction_impulse = cached_friction.copy()
                self._apply_contact_impulse(
                    self.bodies[contact.body_a],
                    self.bodies[contact.body_b],
                    contact,
                    cached_normal * contact.normal + cached_friction,
                )

        for _ in range(self.velocity_iterations):
            for contact in solvable:
                a, b = self.bodies[contact.body_a], self.bodies[contact.body_b]
                normal = contact.normal

                # Normal constraint: reach the restitution target speed,
                # accumulated impulse clamped non-negative (no pulling).
                # Contact-point velocity includes rotation for bodies with
                # arms (off-center box contacts create torque).
                normal_velocity = b.velocity - a.velocity
                if contact.r_b is not None:
                    normal_velocity = normal_velocity + np.cross(
                        b.angular_velocity, contact.r_b)
                if contact.r_a is not None:
                    normal_velocity = normal_velocity - np.cross(
                        a.angular_velocity, contact.r_a)
                normal_speed = float(normal_velocity @ normal)
                delta = -(normal_speed - contact.restitution_bias) / contact.k_normal
                new_total = max(0.0, contact.impulse + delta)
                delta = new_total - contact.impulse
                contact.impulse = new_total
                if delta != 0.0:
                    self._apply_contact_impulse(a, b, contact, delta * normal)

                # Friction: drive the contact-point tangential velocity to
                # zero, accumulated impulse clamped to the Coulomb disk.
                contact_velocity = b.velocity - a.velocity
                if contact.r_b is not None:
                    contact_velocity = contact_velocity + np.cross(
                        b.angular_velocity, contact.r_b)
                if contact.r_a is not None:
                    contact_velocity = contact_velocity - np.cross(
                        a.angular_velocity, contact.r_a)
                tangent_velocity = (
                    contact_velocity - float(contact_velocity @ normal) * normal
                )
                tangent_speed = float(np.linalg.norm(tangent_velocity))
                if tangent_speed <= 1e-12:
                    continue
                tangent = tangent_velocity / tangent_speed
                k_tangent = a.inverse_mass + b.inverse_mass
                if contact.r_a is not None:
                    arm = np.cross(contact.r_a, tangent)
                    k_tangent += float(
                        tangent @ np.cross(contact.inv_inertia_a @ arm, contact.r_a))
                if contact.r_b is not None:
                    arm = np.cross(contact.r_b, tangent)
                    k_tangent += float(
                        tangent @ np.cross(contact.inv_inertia_b @ arm, contact.r_b))
                desired = contact.friction_impulse - (tangent_speed / k_tangent) * tangent
                # Project onto the tangent plane and clamp to μ·λ_normal.
                desired = desired - float(desired @ normal) * normal
                budget = math.sqrt(a.friction * b.friction) * contact.impulse
                magnitude = float(np.linalg.norm(desired))
                if magnitude > budget:
                    desired = desired * (budget / magnitude if magnitude > 0.0 else 0.0)
                delta_vec = desired - contact.friction_impulse
                contact.friction_impulse = desired
                if float(delta_vec @ delta_vec) > 0.0:
                    self._apply_contact_impulse(a, b, contact, delta_vec)

        # Post-solve, once per contact: rolling drag + position projection.
        self._contact_cache = {}
        for contact in solvable:
            a, b = self.bodies[contact.body_a], self.bodies[contact.body_b]
            self._contact_cache[
                (contact.body_a, contact.body_b, contact.manifold_index)
            ] = (contact.impulse, contact.friction_impulse.copy())
            if contact.impulse > 0.0:
                self._apply_rolling_resistance(a, b, contact.normal, contact.impulse)
            correction = max(contact.penetration - _PENETRATION_SLOP, 0.0)
            if correction > 0.0:
                offset = (
                    _POSITION_CORRECTION * correction / contact.k_normal
                ) * contact.normal
                a.position = a.position - offset * a.inverse_mass
                b.position = b.position + offset * b.inverse_mass

    @staticmethod
    def _apply_contact_impulse(
        a: Body, b: Body, contact: Contact, impulse_on_b: np.ndarray
    ) -> None:
        """Apply an impulse (+ on b, − on a) at the contact, updating
        angular velocity through each sphere's contact arm."""
        a.velocity = a.velocity - impulse_on_b * a.inverse_mass
        b.velocity = b.velocity + impulse_on_b * b.inverse_mass
        if contact.r_a is not None and contact.inv_inertia_a is not None:
            a.angular_velocity = a.angular_velocity - (
                contact.inv_inertia_a @ np.cross(contact.r_a, impulse_on_b))
        if contact.r_b is not None and contact.inv_inertia_b is not None:
            b.angular_velocity = b.angular_velocity + (
                contact.inv_inertia_b @ np.cross(contact.r_b, impulse_on_b))

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

    @staticmethod
    def _body_motion_speed(body: Body) -> float:
        speed = float(np.linalg.norm(body.velocity))
        if body.rotates:
            # A spinning body is not at rest even if its center is.
            reach = body.radius if body.shape == "sphere" else float(
                np.max(body.half_extents))
            speed += float(np.linalg.norm(body.angular_velocity)) * reach
        return speed

    def _update_sleep_state(self, contacts: list[Contact]) -> None:
        """Island sleeping (Box2D discipline): bodies connected by contacts
        sleep and wake as a unit. A stack sleeps only when every member is
        still; one moving member keeps — or wakes — the whole island."""
        dynamic = self._dynamic_bodies()
        parent: dict[str, str] = {body.body_id: body.body_id for body in dynamic}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for contact in contacts:
            if contact.body_a in parent and contact.body_b in parent:
                parent[find(contact.body_a)] = find(contact.body_b)

        islands: dict[str, list[Body]] = {}
        for body in dynamic:
            islands.setdefault(find(body.body_id), []).append(body)

        for members in islands.values():
            if all(self._body_motion_speed(b) < _SLEEP_SPEED for b in members):
                for body in members:
                    body.still_ticks += 1
                if all(b.still_ticks >= _SLEEP_TICKS for b in members):
                    for body in members:
                        body.sleeping = True
                        body.velocity[:] = 0.0
                        body.angular_velocity[:] = 0.0
            else:
                for body in members:
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
            # Warm-start state is dynamical state: without it, a resumed
            # world's solver walks a different intra-tick path than the
            # uninterrupted one and digests diverge.
            "contact_cache": [
                [key[0], key[1], key[2], impulse, [float(v) for v in friction]]
                for key, (impulse, friction) in sorted(self._contact_cache.items())
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PhysicsWorld:
        world = cls(gravity=float(payload["gravity"]), dt=float(payload["dt"]))
        world.tick = int(payload.get("tick", 0))
        for row in payload.get("bodies", []):
            world.add_body(Body.from_dict(row))
        raw_cache = payload.get("contact_cache", [])
        if not isinstance(raw_cache, list):
            raise PhysicsError("contact_cache must be a list")
        restored_cache: dict[
            tuple[str, str, int], tuple[float, FloatArray]
        ] = {}
        for entry in raw_cache:
            if not isinstance(entry, list) or len(entry) != 5:
                raise PhysicsError("contact_cache entries must have five fields")
            body_a, body_b = entry[0], entry[1]
            if not isinstance(body_a, str) or not isinstance(body_b, str):
                raise PhysicsError("contact_cache body references must be strings")
            if body_a == body_b:
                raise PhysicsError("contact_cache cannot contain a self-contact")
            if body_a not in world.bodies or body_b not in world.bodies:
                raise PhysicsError("contact_cache references an unknown body")
            raw_index = entry[2]
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise PhysicsError("contact_cache manifold index must be an integer")
            if raw_index < 0:
                raise PhysicsError("contact_cache manifold index must be non-negative")
            try:
                impulse = float(entry[3])
            except (TypeError, ValueError, OverflowError) as exc:
                raise PhysicsError(
                    "contact_cache impulse must be finite and non-negative"
                ) from exc
            if not math.isfinite(impulse) or impulse < 0.0:
                raise PhysicsError(
                    "contact_cache impulse must be finite and non-negative"
                )
            friction = _vec(entry[4], "contact_cache friction")
            key = (body_a, body_b, raw_index)
            if key in restored_cache:
                raise PhysicsError("contact_cache contains a duplicate manifold key")
            restored_cache[key] = (impulse, friction)
        world._contact_cache = restored_cache
        return world
