"""MuJoCo backend: high-fidelity seam behind the same interfaces.

Skips cleanly when mujoco is not installed; asserts real dynamics when
it is (it is installed on this host by owner approval, 2026-07-13).
"""
from __future__ import annotations

import math

import pytest

from core.worlds.generation import generate_world
from core.worlds.mujoco_backend import (
    MujocoAgentSimulator,
    blueprint_to_mjcf,
    mujoco_available,
    simulate_blueprint,
)

pytestmark = pytest.mark.skipif(not mujoco_available(), reason="mujoco not installed")


def test_blueprint_compiles_and_settles():
    blueprint = generate_world(9, size=16, theme="arena")
    xml = blueprint_to_mjcf(blueprint)
    assert xml.startswith("<mujoco") and "terrain_" in xml or "ball" in xml

    report = simulate_blueprint(blueprint, seconds=2.0)
    assert report["engine"] == "mujoco"
    assert report["steps"] > 0
    # Dynamic props ended at finite, plausible heights (not exploded).
    for name, pos in report["bodies"].items():
        assert all(math.isfinite(v) for v in pos), name
        assert -1.0 <= pos[2] <= 60.0, (name, pos)


def test_agent_simulator_honors_interface_contract():
    sim = MujocoAgentSimulator()
    obs = sim.reset(seed=3)
    assert obs.distance > 0.0
    vec = obs.to_vector()
    assert vec.shape == (7,)

    # PD-drive toward the target: pure pursuit orbits under real inertia;
    # adding velocity damping converges — exactly what genuine dynamics
    # (not a scripted lerp) must exhibit.
    for _ in range(600):
        dx = obs.target[0] - obs.position[0]
        dy = obs.target[1] - obs.position[1]
        ax = 1.4 * dx - 0.9 * obs.velocity[0]
        ay = 1.4 * dy - 0.9 * obs.velocity[1]
        obs = sim.step((max(-1.0, min(1.0, ax)), max(-1.0, min(1.0, ay))))
    assert obs.distance < 0.4, obs.distance


def test_agent_simulator_is_seed_reproducible():
    a = MujocoAgentSimulator().reset(seed=11)
    b = MujocoAgentSimulator().reset(seed=11)
    assert a.position == b.position and a.target == b.target
