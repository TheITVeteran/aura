"""Rotational dynamics v2: closed-form verification.

The headline law: a sphere sliding on a plane with friction (no bounce)
converges to rolling without slipping at exactly v = (5/7)·v₀, having
kept exactly 5/7 of its kinetic energy. Both are classical results with
no free parameters — the engine must hit them, not approximate them.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.worlds import Body, PhysicsError, PhysicsWorld


def _plane(friction=0.6):
    return Body(body_id="ground", shape="plane", position=(0, 0, 0),
                velocity=(0, 0, 0), mass=0.0, plane_height=0.0,
                restitution=0.0, friction=friction)


def _ball(vel=(4.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0), friction=0.6,
          rolling_resistance=0.0, radius=0.5):
    return Body(body_id="ball", shape="sphere", position=(0, 0, radius),
                velocity=vel, mass=1.0, radius=radius, restitution=0.0,
                friction=friction, rolling_resistance=rolling_resistance,
                angular_velocity=spin)


def test_sliding_sphere_converges_to_five_sevenths_speed():
    world = PhysicsWorld(dt=1.0 / 240.0)
    world.add_body(_plane())
    ball = world.add_body(_ball(vel=(4.0, 0.0, 0.0)))
    world.step(2400)  # 10 seconds — ample time to reach pure rolling
    v = float(ball.velocity[0])
    omega_y = float(ball.angular_velocity[1])
    assert v == pytest.approx(4.0 * 5.0 / 7.0, rel=0.01)
    # Rolling without slipping: contact-point velocity is zero.
    assert v - omega_y * ball.radius == pytest.approx(0.0, abs=0.01)


def test_sliding_sphere_keeps_five_sevenths_of_energy():
    world = PhysicsWorld(dt=1.0 / 240.0)
    world.add_body(_plane())
    ball = world.add_body(_ball(vel=(4.0, 0.0, 0.0)))
    initial = ball.kinetic_energy()
    world.step(2400)
    assert ball.kinetic_energy() == pytest.approx(initial * 5.0 / 7.0, rel=0.02)


def test_backspin_ball_reverses_direction():
    """A ball sliding forward with strong backspin ends up rolling
    BACKWARD — the classic billiards draw shot. Final v = (5v₀ + 2ω₀r)/7
    with ω₀ negative for backspin."""
    r = 0.5
    v0, w0 = 2.0, -12.0  # backspin: surface velocity adds to slide
    world = PhysicsWorld(dt=1.0 / 240.0)
    world.add_body(_plane())
    ball = world.add_body(_ball(vel=(v0, 0.0, 0.0), spin=(0.0, w0, 0.0), radius=r))
    world.step(2400)
    expected = (5.0 * v0 + 2.0 * w0 * r) / 7.0  # = (10 - 12)/7 < 0
    assert expected < 0.0
    assert float(ball.velocity[0]) == pytest.approx(expected, rel=0.02)


def test_spin_is_conserved_in_free_flight():
    world = PhysicsWorld(gravity=0.0, dt=1.0 / 120.0)
    ball = world.add_body(_ball(vel=(0.0, 0.0, 0.0), spin=(1.0, 2.0, 3.0)))
    world.step(600)
    np.testing.assert_allclose(ball.angular_velocity, [1.0, 2.0, 3.0], atol=1e-12)
    # Orientation advanced and stayed a unit quaternion.
    assert float(np.linalg.norm(ball.orientation)) == pytest.approx(1.0, abs=1e-9)
    assert abs(float(ball.orientation[0]) - 1.0) > 1e-3  # actually rotated


def test_rolling_resistance_brings_roller_to_rest():
    world = PhysicsWorld(dt=1.0 / 120.0)
    world.add_body(_plane())
    ball = world.add_body(_ball(vel=(3.0, 0.0, 0.0), rolling_resistance=0.05))
    world.step(6000)  # 50 seconds
    assert ball.sleeping
    assert float(np.linalg.norm(ball.velocity)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.linalg.norm(ball.angular_velocity)) == pytest.approx(0.0, abs=1e-9)


def test_pure_roller_without_resistance_keeps_rolling():
    world = PhysicsWorld(dt=1.0 / 120.0)
    world.add_body(_plane())
    ball = world.add_body(_ball(vel=(3.0, 0.0, 0.0), rolling_resistance=0.0))
    world.step(2400)
    # Ideal surface: once rolling, it stays rolling (≈5/7·v₀), never sleeps.
    assert not ball.sleeping
    assert float(ball.velocity[0]) == pytest.approx(3.0 * 5.0 / 7.0, rel=0.02)


def test_angular_locked_body_never_spins():
    world = PhysicsWorld(dt=1.0 / 240.0)
    world.add_body(_plane())
    walker = world.add_body(Body(
        body_id="walker", shape="sphere", position=(0, 0, 0.5),
        velocity=(4.0, 0.0, 0.0), mass=60.0, radius=0.5,
        restitution=0.0, friction=0.9, angular_locked=True))
    world.step(1200)
    np.testing.assert_allclose(walker.angular_velocity, np.zeros(3), atol=1e-12)
    assert walker.inverse_inertia == 0.0


def test_rotation_state_survives_serialization_deterministically():
    def build():
        world = PhysicsWorld(dt=1.0 / 120.0)
        world.add_body(_plane())
        world.add_body(_ball(vel=(3.0, 1.0, 0.0), spin=(0.5, -0.2, 0.1)))
        return world

    import json
    reference = build()
    reference.step(400)

    split = build()
    split.step(200)
    resumed = PhysicsWorld.from_dict(json.loads(json.dumps(split.to_dict())))
    resumed.step(200)
    assert resumed.state_digest() == reference.state_digest()


def test_invalid_rotational_construction_rejected():
    with pytest.raises(PhysicsError):
        _ball(spin=(0.0, 0.0))  # not a 3-vector
    with pytest.raises(PhysicsError):
        Body(body_id="x", shape="sphere", position=(0, 0, 0),
             velocity=(0, 0, 0), orientation=(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(PhysicsError):
        Body(body_id="x", shape="sphere", position=(0, 0, 0),
             velocity=(0, 0, 0), rolling_resistance=-1.0)
