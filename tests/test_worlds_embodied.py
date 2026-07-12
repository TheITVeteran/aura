"""Embodied agent + counterfactual forking verification.

Raycasts are checked against analytic intersection distances; locomotion,
navigation, grasping, and forking are checked by observable outcome.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import core.worlds.hosting as hosting
from core.worlds import (
    Body,
    EmbodiedAgent,
    PhysicsError,
    PhysicsWorld,
    generate_world,
)
from core.worlds.embodied import AGENT_RADIUS


def _flat_world() -> PhysicsWorld:
    world = PhysicsWorld(dt=1.0 / 120.0)
    world.add_body(Body(body_id="ground", shape="plane", position=(0, 0, 0),
                        velocity=(0, 0, 0), mass=0.0, plane_height=0.0,
                        restitution=0.1, friction=0.9))
    return world


def _spawned(world=None) -> EmbodiedAgent:
    world = world or _flat_world()
    return EmbodiedAgent.spawn(world, None, position=(0, 0, AGENT_RADIUS))


# ── Exact senses ─────────────────────────────────────────────────

def test_raycast_sphere_distance_is_analytic():
    world = _flat_world()
    world.add_body(Body(body_id="ball", shape="sphere", position=(5, 0, AGENT_RADIUS),
                        velocity=(0, 0, 0), mass=1.0, radius=0.5))
    agent = _spawned(world)
    hit = agent.raycast(direction=(1, 0, 0))
    assert hit is not None and hit.body_id == "ball"
    assert hit.distance == pytest.approx(5.0 - 0.5, abs=1e-9)


def test_raycast_box_and_plane_distances():
    world = _flat_world()
    world.add_body(Body(body_id="wall", shape="box", position=(0, 4, 1),
                        velocity=(0, 0, 0), mass=0.0, half_extents=(3, 0.5, 1)))
    agent = _spawned(world)
    hit = agent.raycast(direction=(0, 1, 0))
    assert hit is not None and hit.body_id == "wall"
    assert hit.distance == pytest.approx(3.5, abs=1e-9)

    down = agent.raycast(direction=(0, 0, -1))
    assert down is not None and down.body_id == "ground"
    assert down.distance == pytest.approx(AGENT_RADIUS, abs=1e-6)


def test_raycast_reports_nearest_and_misses_honestly():
    world = _flat_world()
    world.add_body(Body(body_id="near", shape="sphere", position=(3, 0, AGENT_RADIUS),
                        velocity=(0, 0, 0), mass=1.0, radius=0.4))
    world.add_body(Body(body_id="far", shape="sphere", position=(8, 0, AGENT_RADIUS),
                        velocity=(0, 0, 0), mass=1.0, radius=0.4))
    agent = _spawned(world)
    assert agent.raycast(direction=(1, 0, 0)).body_id == "near"
    assert agent.raycast(direction=(-1, 0, 0)) is None
    with pytest.raises(PhysicsError):
        agent.raycast(direction=(0, 0, 0))


def test_look_returns_bearing_sweep():
    world = _flat_world()
    world.add_body(Body(body_id="pillar", shape="box", position=(4, 0, 1),
                        velocity=(0, 0, 0), mass=0.0, half_extents=(0.3, 0.3, 1)))
    agent = _spawned(world)
    scan = agent.look(rays=9)
    assert len(scan) == 9
    hits = [entry for entry in scan if entry["hit"]]
    assert any(entry["hit"]["body_id"] == "pillar" for entry in hits)


# ── Proprioception + locomotion ──────────────────────────────────

def test_agent_is_grounded_and_walks_where_told():
    agent = _spawned()
    assert agent.grounded()
    agent.walk(heading=0.0, ticks=240)  # 2 seconds east
    state = agent.proprioception()
    assert state["position"][0] > 2.0
    assert abs(state["position"][1]) < 0.2
    assert state["speed"] <= 3.01  # walk speed cap holds


def test_jump_only_from_ground():
    agent = _spawned()
    assert agent.jump() is True
    agent.world.step(10)
    assert not agent.grounded()
    assert agent.jump() is False  # no double jump


# ── Grasp and throw ──────────────────────────────────────────────

def test_grasp_carry_throw_cycle():
    world = _flat_world()
    world.add_body(Body(body_id="rock", shape="sphere", position=(1.0, 0, 0.3),
                        velocity=(0, 0, 0), mass=2.0, radius=0.3))
    agent = _spawned(world)
    assert agent.grasp("rock") is True
    agent.walk(heading=0.0, ticks=120)
    rock = world.body("rock")
    # Carried: the rock moved with the agent.
    assert float(rock.position[0]) > 1.5
    released = agent.throw(speed=8.0, pitch=0.4)
    assert released == "rock"
    assert agent.state.held_body is None
    assert float(rock.velocity[0]) > 3.0 and float(rock.velocity[2]) > 1.0


def test_grasp_limits_are_enforced():
    world = _flat_world()
    world.add_body(Body(body_id="anvil", shape="sphere", position=(1.0, 0, 0.5),
                        velocity=(0, 0, 0), mass=500.0, radius=0.5))
    world.add_body(Body(body_id="distant", shape="sphere", position=(9.0, 0, 0.3),
                        velocity=(0, 0, 0), mass=1.0, radius=0.3))
    agent = _spawned(world)
    with pytest.raises(PhysicsError):
        agent.grasp("anvil")  # too heavy
    assert agent.grasp("distant") is False  # out of range
    with pytest.raises(PhysicsError):
        agent.grasp("ground")  # static
    with pytest.raises(PhysicsError):
        agent.throw()  # holding nothing


# ── Navigation ───────────────────────────────────────────────────

def test_navigation_reaches_target_on_generated_terrain():
    blueprint = generate_world(21, size=24, theme="plains")
    world = blueprint.to_physics_world()
    agent = EmbodiedAgent.spawn(world, blueprint)
    outcome = agent.navigate_to((6.0, 5.0), tolerance=1.2, max_ticks=24000)
    assert outcome["status"] == "reached", outcome
    position = agent.proprioception()["position"]
    assert math.hypot(position[0] - 6.0, position[1] - 5.0) <= 1.5


def test_navigation_reports_no_path_honestly():
    blueprint = generate_world(5, size=16, theme="plains")
    # Wall off the target with an impassable synthetic cliff.
    for row in range(16):
        blueprint.heightfield[10][row] = 50.0
    world = blueprint.to_physics_world()
    agent = EmbodiedAgent.spawn(world, blueprint, position=(0, 0, AGENT_RADIUS + 0.2))
    outcome = agent.navigate_to((6.5, 0.0), max_ticks=2000)
    assert outcome["status"] == "no_path"


# ── Hosted embodiment + forking ──────────────────────────────────

@pytest.fixture
def host(tmp_path):
    yield hosting.reset_world_host_for_tests(tmp_path / "worlds")
    hosting.reset_world_host_for_tests(tmp_path / "unused")


async def test_hosted_agent_survives_restart(host, tmp_path):
    await host.create_world("home", seed=9, size=16, theme="plains")
    spawned = await host.spawn_agent("home", "aura")
    await host.agent_command("home", "aura", "walk", heading=0.0, ticks=120)
    moved = await host.agent_command("home", "aura", "proprioception")

    reloaded = hosting.reset_world_host_for_tests(tmp_path / "worlds")
    state = await reloaded.agent_command("home", "aura", "proprioception")
    assert state["position"] == moved["position"]
    assert state["position"][0] > spawned["position"][0]


async def test_fork_intervene_compare(host):
    await host.create_world("base", seed=13, size=16, theme="arena")
    await host.step_world("base", 240)  # settle
    await host.fork_world("base", "branch")

    baseline = host.compare_worlds("base", "branch")
    assert baseline["identical"] is True

    world = host.load_world("branch")
    movable = next(
        key for key in sorted(world.physics.bodies)
        if not world.physics.bodies[key].is_static
    )
    await host.apply_impulse("branch", movable, (6.0, 0.0, 3.0))
    await host.step_world("branch", 240)
    await host.step_world("base", 240)

    report = host.compare_worlds("base", "branch")
    assert report["identical"] is False
    assert report["bodies_diverged"] >= 1
    assert report["largest_divergences"][0]["position_delta"] > 0.1
    # The fork remembers its origin.
    kinds = {event["kind"] for event in host.load_world("branch").journal}
    assert "forked_from" in kinds


async def test_agent_actions_are_journaled(host):
    await host.create_world("diary", seed=2, size=16, theme="plains")
    await host.spawn_agent("diary")
    await host.agent_command("diary", "agent", "walk", heading=1.0, ticks=60)
    await host.agent_command("diary", "agent", "jump")
    journal = host.load_world("diary").journal
    kinds = [event["kind"] for event in journal]
    assert "agent_spawned" in kinds
    assert kinds.count("agent_action") >= 2


async def test_world_forge_skill_embodiment_flow(host):
    from core.skills.world_forge import WorldForgeSkill

    skill = WorldForgeSkill()
    assert (await skill.execute(
        {"action": "create", "world_id": "quest", "seed": 4, "size": 16,
         "theme": "plains"}, {}))["ok"]
    assert (await skill.execute(
        {"action": "spawn_agent", "world_id": "quest"}, {}))["ok"]
    walked = await skill.execute(
        {"action": "agent", "world_id": "quest", "command": "walk",
         "heading": 0.0, "ticks": 120}, {})
    assert walked["ok"] and walked["position"][0] > 1.0
    looked = await skill.execute(
        {"action": "agent", "world_id": "quest", "command": "look"}, {})
    assert looked["ok"] and len(looked["observations"]) > 0
    forked = await skill.execute(
        {"action": "fork", "world_id": "quest", "new_id": "quest-b"}, {})
    assert forked["ok"]
    compared = await skill.execute(
        {"action": "compare", "world_id": "quest", "other_id": "quest-b"}, {})
    assert compared["ok"] and compared["comparison"]["identical"] is True
