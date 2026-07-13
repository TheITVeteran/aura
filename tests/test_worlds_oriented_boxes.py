"""Oriented-box (6-DoF) dynamics: SAT manifolds driving real torque.

The signature behaviors only full rotational dynamics can produce:
an edge-balanced box TOPPLES onto a face; a tilted box settles flat;
off-center hits impart spin.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.worlds import Body, PhysicsError, PhysicsWorld
from core.worlds.obb import quat_to_matrix


def _plane(friction=0.7):
    return Body(body_id="ground", shape="plane", position=(0, 0, 0),
                velocity=(0, 0, 0), mass=0.0, plane_height=0.0,
                restitution=0.1, friction=friction)


def _obox(body_id="crate", pos=(0, 0, 2.0), quat=(1, 0, 0, 0),
          spin=(0, 0, 0), half=(0.5, 0.5, 0.5), mass=2.0):
    return Body(body_id=body_id, shape="box", position=pos,
                velocity=(0, 0, 0), mass=mass, half_extents=half,
                restitution=0.1, friction=0.7, oriented=True,
                orientation=quat, angular_velocity=spin)


def _tilt_y(angle):
    return (math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0)


def _up_alignment(body) -> float:
    """|cos| of the angle between the box's nearest axis and world z —
    1.0 means a face is flat against the ground."""
    rotation = quat_to_matrix(body.orientation)
    return float(np.max(np.abs(rotation.T @ np.array([0.0, 0.0, 1.0]))))


def test_tilted_box_settles_flat():
    world = PhysicsWorld(dt=1.0 / 240.0)
    world.add_body(_plane())
    world.add_body(_obox(pos=(0, 0, 0.9), quat=_tilt_y(0.35)))
    world.step(4800)  # 20 s
    crate = world.body("crate")
    assert _up_alignment(crate) > 0.99, _up_alignment(crate)
    assert float(crate.position[2]) == pytest.approx(0.5, abs=0.05)
    assert crate.sleeping


def test_edge_balanced_box_topples():
    # Balanced nearly on its edge (just past 45°): must fall to a face,
    # not levitate — the canonical torque-from-contact test.
    world = PhysicsWorld(dt=1.0 / 240.0)
    world.add_body(_plane())
    tilt = math.pi / 4.0 + 0.06
    start_height = math.sqrt(2.0) * 0.5 + 0.01
    world.add_body(_obox(pos=(0, 0, start_height), quat=_tilt_y(tilt)))
    world.step(4800)
    crate = world.body("crate")
    assert _up_alignment(crate) > 0.98
    assert float(crate.position[2]) == pytest.approx(0.5, abs=0.06)


def test_spinning_box_conserves_angular_velocity_in_free_flight():
    world = PhysicsWorld(gravity=0.0, dt=1.0 / 120.0)
    world.add_body(_obox(pos=(0, 0, 10.0), spin=(0.0, 0.0, 2.0)))
    world.step(600)
    crate = world.body("crate")
    np.testing.assert_allclose(crate.angular_velocity, [0, 0, 2.0], atol=1e-12)
    assert float(np.linalg.norm(crate.orientation)) == pytest.approx(1.0, abs=1e-9)


def test_off_center_hit_spins_the_box():
    world = PhysicsWorld(gravity=0.0, dt=1.0 / 240.0)
    world.add_body(_obox(pos=(0, 0, 0), mass=4.0))
    world.add_body(Body(body_id="bullet", shape="sphere",
                        position=(-2.0, 0.35, 0.0), velocity=(8.0, 0, 0),
                        mass=0.5, radius=0.15, restitution=0.4, friction=0.3))
    world.step(240)
    crate = world.body("crate")
    # Struck off-axis: it must acquire yaw spin AND linear momentum.
    assert abs(float(crate.angular_velocity[2])) > 0.1
    assert float(crate.velocity[0]) > 0.1


def test_oriented_stack_settles():
    world = PhysicsWorld(dt=1.0 / 240.0)
    world.add_body(_plane())
    for index in range(2):
        world.add_body(_obox(
            body_id=f"crate{index}",
            pos=(0.02 * index, 0, 0.5 + 1.02 * index),
            quat=_tilt_y(0.05 * index)))
    world.step(4800)
    for index in range(2):
        crate = world.body(f"crate{index}")
        assert _up_alignment(crate) > 0.98
        assert crate.sleeping, index
    assert float(world.body("crate1").position[2]) == pytest.approx(1.5, abs=0.1)


def test_oriented_state_survives_serialization():
    import json

    def build():
        world = PhysicsWorld(dt=1.0 / 240.0)
        world.add_body(_plane())
        world.add_body(_obox(pos=(0, 0, 1.2), quat=_tilt_y(0.3),
                             spin=(0.1, 0.2, 0.3)))
        return world

    reference = build()
    reference.step(400)
    split = build()
    split.step(200)
    resumed = PhysicsWorld.from_dict(json.loads(json.dumps(split.to_dict())))
    resumed.step(200)
    assert resumed.state_digest() == reference.state_digest()
    assert resumed.body("crate").oriented is True


@pytest.mark.parametrize(
    "entry",
    [
        ["ground", "crate", 0, float("nan"), [0.0, 0.0, 0.0]],
        ["ground", "missing", 0, 0.1, [0.0, 0.0, 0.0]],
        ["ground", "crate", -1, 0.1, [0.0, 0.0, 0.0]],
        ["ground", "crate", 0, 0.1, [0.0, 0.0]],
        ["ground", "crate"],
    ],
)
def test_oriented_restart_rejects_malformed_warm_start_cache(entry):
    world = PhysicsWorld(dt=1.0 / 240.0)
    world.add_body(_plane())
    world.add_body(_obox())
    payload = world.to_dict()
    payload["contact_cache"] = [entry]

    with pytest.raises(PhysicsError):
        PhysicsWorld.from_dict(payload)
