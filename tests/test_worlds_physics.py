"""World engine verification: closed-form dynamics, conservation laws,
determinism, generation reproducibility, and persistent hosting.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

import core.worlds.hosting as hosting
from core.worlds import (
    Body,
    PhysicsError,
    PhysicsWorld,
    generate_world,
)
from core.worlds.generation import WorldBlueprint


def _sphere(
    body_id="ball",
    pos=(0, 0, 10.0),
    vel=(0, 0, 0),
    mass=1.0,
    radius=0.5,
    restitution=0.5,
    friction=0.4,
):
    return Body(
        body_id=body_id,
        shape="sphere",
        position=pos,
        velocity=vel,
        mass=mass,
        radius=radius,
        restitution=restitution,
        friction=friction,
    )


def _plane():
    return Body(
        body_id="ground",
        shape="plane",
        position=(0, 0, 0),
        velocity=(0, 0, 0),
        mass=0.0,
        plane_height=0.0,
        restitution=0.5,
        friction=0.6,
    )


# ── Closed-form dynamics ─────────────────────────────────────────


def test_projectile_matches_discrete_closed_form():
    world = PhysicsWorld(dt=1.0 / 120.0)
    v0 = np.array([3.0, -2.0, 5.0])
    world.add_body(_sphere(pos=(0, 0, 100.0), vel=tuple(v0)))
    steps = 100
    world.step(steps)
    g = np.array([0.0, 0.0, -9.81])
    dt = world.dt
    expected_velocity = v0 + steps * g * dt
    expected_position = (
        np.array([0.0, 0.0, 100.0]) + steps * v0 * dt + g * dt * dt * steps * (steps + 1) / 2.0
    )
    body = world.body("ball")
    np.testing.assert_allclose(body.velocity, expected_velocity, atol=1e-9)
    np.testing.assert_allclose(body.position, expected_position, atol=1e-9)


def test_restitution_is_exact_at_impact():
    e = 0.5
    dt = 1.0 / 120.0
    world = PhysicsWorld(dt=dt)
    world.add_body(_plane())
    world.add_body(_sphere(pos=(0, 0, 0.5001), vel=(0, 0, -1.0), restitution=e))
    world.step(1)  # integrates into penetration, then resolves the impact
    impact_speed = 1.0 + 9.81 * dt
    assert world.body("ball").velocity[2] == pytest.approx(e * impact_speed, abs=1e-9)


def test_bounce_height_follows_e_squared():
    e = 0.6
    drop = 2.0
    world = PhysicsWorld(dt=1.0 / 240.0)
    # Contact restitution is min(a, b): give the plane a higher e so the
    # ball's 0.6 governs the bounce.
    plane = _plane()
    plane.restitution = 0.9
    world.add_body(plane)
    world.add_body(_sphere(pos=(0, 0, drop + 0.5), vel=(0, 0, 0), restitution=e))
    bounced = False
    peak_after_bounce = 0.0
    for _ in range(3000):
        world.step()
        body = world.body("ball")
        if not bounced and body.velocity[2] > 0:
            bounced = True
        if bounced:
            if body.velocity[2] <= 0 and peak_after_bounce == 0.0:
                peak_after_bounce = float(body.position[2]) - 0.5
                break
    assert bounced
    assert peak_after_bounce == pytest.approx(e * e * drop, rel=0.05)


# ── Conservation laws ────────────────────────────────────────────


def test_equal_mass_elastic_collision_exchanges_velocities():
    world = PhysicsWorld(gravity=0.0, dt=1.0 / 120.0)
    world.add_body(_sphere("a", pos=(-2, 0, 0), vel=(4, 0, 0), restitution=1.0))
    world.add_body(_sphere("b", pos=(+2, 0, 0), vel=(0, 0, 0), restitution=1.0))
    world.step(200)
    assert world.body("a").velocity[0] == pytest.approx(0.0, abs=1e-9)
    assert world.body("b").velocity[0] == pytest.approx(4.0, abs=1e-9)


def test_momentum_conserved_through_collisions():
    world = PhysicsWorld(gravity=0.0, dt=1.0 / 120.0)
    world.add_body(_sphere("a", pos=(-2, 0.1, 0), vel=(3, 0, 0), mass=2.0, restitution=0.4))
    world.add_body(_sphere("b", pos=(+2, -0.1, 0), vel=(-1, 0.5, 0), mass=1.0, restitution=0.4))
    before = world.total_momentum().copy()
    world.step(400)
    np.testing.assert_allclose(world.total_momentum(), before, atol=1e-9)


def test_inelastic_collision_never_creates_energy():
    world = PhysicsWorld(gravity=0.0, dt=1.0 / 120.0)
    world.add_body(_sphere("a", pos=(-2, 0, 0), vel=(5, 0, 0), restitution=0.3))
    world.add_body(_sphere("b", pos=(+2, 0, 0), vel=(-5, 0, 0), restitution=0.3))
    energy_before = world.total_kinetic_energy()
    world.step(400)
    assert world.total_kinetic_energy() <= energy_before + 1e-9


# ── Rest, sleep, containment ─────────────────────────────────────


def test_dropped_sphere_comes_to_rest_on_plane():
    world = PhysicsWorld(dt=1.0 / 120.0)
    world.add_body(_plane())
    world.add_body(_sphere(pos=(0, 0, 1.5), restitution=0.3))
    world.step(2400)  # 20 seconds
    body = world.body("ball")
    assert body.sleeping
    assert body.position[2] == pytest.approx(0.5, abs=0.02)
    assert world.total_kinetic_energy() == pytest.approx(0.0, abs=1e-6)


def test_sphere_rests_on_static_box():
    world = PhysicsWorld(dt=1.0 / 120.0)
    world.add_body(
        Body(
            body_id="table",
            shape="box",
            position=(0, 0, 0.5),
            velocity=(0, 0, 0),
            mass=0.0,
            half_extents=(1, 1, 0.5),
            restitution=0.2,
            friction=0.8,
        )
    )
    world.add_body(_sphere(pos=(0, 0, 2.5), restitution=0.2))
    world.step(2400)
    body = world.body("ball")
    assert body.sleeping
    assert body.position[2] == pytest.approx(1.5, abs=0.02)


# ── Determinism ──────────────────────────────────────────────────


def _demo_world() -> PhysicsWorld:
    world = PhysicsWorld(dt=1.0 / 120.0)
    world.add_body(_plane())
    for i in range(6):
        world.add_body(
            _sphere(
                f"ball_{i}",
                pos=(i * 0.4 - 1.0, i * 0.3, 2.0 + i),
                vel=(0.5 * i, -0.2 * i, 0),
                restitution=0.4,
            )
        )
    return world


def test_identical_worlds_produce_identical_digests():
    a, b = _demo_world(), _demo_world()
    a.step(500)
    b.step(500)
    assert a.state_digest() == b.state_digest()


def test_serialization_roundtrip_preserves_trajectory():
    reference = _demo_world()
    reference.step(300)

    split = _demo_world()
    split.step(150)
    resumed = PhysicsWorld.from_dict(json.loads(json.dumps(split.to_dict())))
    resumed.step(150)
    assert resumed.state_digest() == reference.state_digest()


# ── Validation ───────────────────────────────────────────────────


def test_invalid_bodies_rejected():
    with pytest.raises(PhysicsError):
        Body(body_id="x", shape="tetrahedron", position=(0, 0, 0), velocity=(0, 0, 0))
    with pytest.raises(PhysicsError):
        Body(body_id="x", shape="sphere", position=(0, 0), velocity=(0, 0, 0))
    with pytest.raises(PhysicsError):
        Body(body_id="x", shape="sphere", position=(0, 0, 0), velocity=(0, 0, 0), mass=-1)
    with pytest.raises(PhysicsError):
        Body(body_id="x", shape="plane", position=(0, 0, 0), velocity=(0, 0, 0), mass=2.0)
    world = PhysicsWorld()
    world.add_body(_sphere())
    with pytest.raises(PhysicsError):
        world.add_body(_sphere())  # duplicate id


# ── Procedural generation ────────────────────────────────────────


def test_generation_is_seed_deterministic():
    a = generate_world(1234, size=32, theme="highlands")
    b = generate_world(1234, size=32, theme="highlands")
    assert a.digest() == b.digest()
    c = generate_world(1235, size=32, theme="highlands")
    assert c.digest() != a.digest()


def test_generation_respects_bounds_and_themes():
    blueprint = generate_world(7, size=24, theme="arena")
    assert len(blueprint.heightfield) == 24
    for entity in blueprint.entities:
        x, y, _ = entity["position"]
        assert -12.0 <= x <= 12.0 and -12.0 <= y <= 12.0
    with pytest.raises(PhysicsError):
        generate_world(1, size=4)
    with pytest.raises(PhysicsError):
        generate_world(1, theme="volcano")


def test_blueprint_roundtrip_and_realization():
    blueprint = generate_world(99, size=16, theme="plains")
    clone = WorldBlueprint.from_dict(json.loads(json.dumps(blueprint.to_dict())))
    assert clone.digest() == blueprint.digest()

    world = blueprint.to_physics_world()
    assert "ground" in world.bodies
    world.step(120)  # must simulate without error
    assert float(
        np.sum(np.isfinite(np.concatenate([world.bodies[k].position for k in world.bodies])))
    ) == 3 * len(world.bodies)


# ── Persistent hosting ───────────────────────────────────────────


@pytest.fixture
def host(tmp_path):
    yield hosting.reset_world_host_for_tests(tmp_path / "worlds")
    hosting.reset_world_host_for_tests(tmp_path / "unused")


async def test_world_creation_persists_governed_envelope(host, tmp_path):
    summary = await host.create_world("proving-ground", seed=42, size=16, theme="arena")
    assert summary["bodies"] > 1
    path = tmp_path / "worlds" / "proving-ground.json"
    assert path.exists()
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_name"] == "hosted_world"
    assert document["payload"]["blueprint"]["seed"] == 42


async def test_world_survives_restart_with_identical_trajectory(host, tmp_path):
    await host.create_world("alpha", seed=7, size=16, theme="plains")
    straight = await host.step_world("alpha", 300)

    fresh_host = hosting.reset_world_host_for_tests(tmp_path / "worlds2")
    await fresh_host.create_world("alpha", seed=7, size=16, theme="plains")
    await fresh_host.step_world("alpha", 150)
    # Simulate a restart: drop the in-memory cache, reload from disk.
    reloaded_host = hosting.reset_world_host_for_tests(tmp_path / "worlds2")
    resumed = await reloaded_host.step_world("alpha", 150)

    assert resumed["state_digest"] == straight["state_digest"]
    assert resumed["tick"] == straight["tick"] == 300


async def test_journal_records_history(host):
    await host.create_world("memoir", seed=3, size=16, theme="arena")
    await host.step_world("memoir", 240)
    detail = host.inspect("memoir")
    kinds = {event["kind"] for event in detail["recent_events"]}
    assert "stepped" in kinds
    assert detail["journal_entries"] >= 2  # genesis + stepped at minimum


async def test_impulse_moves_bodies_and_is_journaled(host):
    await host.create_world("kicks", seed=11, size=16, theme="arena")
    world = host.load_world("kicks")
    movable = next(
        key for key in sorted(world.physics.bodies) if not world.physics.bodies[key].is_static
    )
    before = world.physics.bodies[movable].velocity.copy()
    await host.apply_impulse("kicks", movable, (5.0, 0.0, 2.0))
    after = world.physics.bodies[movable].velocity
    assert not np.allclose(before, after)
    assert any(event["kind"] == "impulse" for event in world.journal)

    with pytest.raises(PhysicsError):
        await host.apply_impulse("kicks", "ground", (1.0, 0.0, 0.0))


async def test_host_enforces_bounds(host):
    await host.create_world("bounded", seed=1, size=16)
    with pytest.raises(PhysicsError):
        await host.step_world("bounded", 0)
    with pytest.raises(PhysicsError):
        await host.step_world("bounded", hosting.MAX_TICKS_PER_STEP + 1)
    with pytest.raises(PhysicsError):
        await host.create_world("bounded", seed=1)  # duplicate
    with pytest.raises(PhysicsError):
        await host.create_world("Bad Name!", seed=1)
    with pytest.raises(PhysicsError):
        host.load_world("never-made")


async def test_persistence_failure_does_not_publish_partial_world_state(
    host,
    tmp_path,
    monkeypatch,
):
    await host.create_world("transactional", seed=8, size=16, theme="arena")
    world = host.load_world("transactional")
    movable = next(
        key for key in sorted(world.physics.bodies) if not world.physics.bodies[key].is_static
    )
    before_digest = world.physics.state_digest()
    before_journal = list(world.journal)

    class _FailingGateway:
        async def ensure_directory_async(self, *_args, **_kwargs):
            return str(tmp_path / "worlds")

        async def write_json_async(self, *_args, **_kwargs):
            raise OSError("injected durable write failure")

    monkeypatch.setattr(hosting, "get_file_write_gateway", lambda: _FailingGateway())

    with pytest.raises(hosting.WorldPersistenceError, match="committed durably"):
        await host.apply_impulse("transactional", movable, (5.0, 0.0, 0.0))

    assert host.load_world("transactional") is world
    assert world.physics.state_digest() == before_digest
    assert world.journal == before_journal

    reloaded = hosting.WorldHost(tmp_path / "worlds").load_world("transactional")
    assert reloaded.physics.state_digest() == before_digest
    assert reloaded.journal == before_journal


async def test_same_world_mutations_serialize_and_persist_without_lost_updates(host, tmp_path):
    await host.create_world("serialized", seed=14, size=16, theme="plains")

    await asyncio.gather(
        host.step_world("serialized", 40),
        host.step_world("serialized", 60),
        host.step_world("serialized", 25),
    )

    assert host.inspect("serialized")["tick"] == 125
    reloaded = hosting.WorldHost(tmp_path / "worlds").load_world("serialized")
    assert reloaded.physics.tick == 125
    assert [event["ticks"] for event in reloaded.journal if event["kind"] == "stepped"] == [
        40,
        60,
        25,
    ]


async def test_duplicate_world_creation_is_reserved_across_racing_calls(host):
    outcomes = await asyncio.gather(
        host.create_world("one-winner", seed=1, size=16),
        host.create_world("one-winner", seed=2, size=16),
        return_exceptions=True,
    )

    successes = [outcome for outcome in outcomes if isinstance(outcome, dict)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, PhysicsError)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert host.load_world("one-winner").blueprint.seed in {1, 2}


async def test_cancellation_during_durable_write_finishes_commit_before_propagating(
    host,
    tmp_path,
    monkeypatch,
):
    await host.create_world("cancel-safe", seed=4, size=16, theme="plains")
    initial_tick = host.inspect("cancel-safe")["tick"]
    real_gateway = hosting.get_file_write_gateway()
    write_started = asyncio.Event()
    allow_write = asyncio.Event()

    class _DelayedGateway:
        async def ensure_directory_async(self, *args, **kwargs):
            return await real_gateway.ensure_directory_async(*args, **kwargs)

        async def write_json_async(self, *args, **kwargs):
            write_started.set()
            await allow_write.wait()
            await real_gateway.write_json_async(*args, **kwargs)

    monkeypatch.setattr(hosting, "get_file_write_gateway", lambda: _DelayedGateway())
    task = asyncio.create_task(host.step_world("cancel-safe", 10))
    await asyncio.wait_for(write_started.wait(), timeout=2.0)
    task.cancel()
    allow_write.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert host.inspect("cancel-safe")["tick"] == initial_tick + 10
    reloaded = hosting.WorldHost(tmp_path / "worlds").load_world("cancel-safe")
    assert reloaded.physics.tick == initial_tick + 10


# ── Skill facade ─────────────────────────────────────────────────


async def test_world_forge_skill_end_to_end(host):
    from core.skills.world_forge import WorldForgeSkill

    skill = WorldForgeSkill()
    created = await skill.execute(
        {"action": "create", "world_id": "skilltest", "seed": 5, "size": 16, "theme": "arena"}, {}
    )
    assert created["ok"], created
    capabilities = created["simulation_capabilities"]
    assert capabilities["schema"] == "aura.world_forge.capabilities.v1"
    assert capabilities["supported"]["oriented_box_sat_manifolds"] is True
    assert capabilities["integration_surfaces"]["vr_renderer"] == {
        "available": False,
        "reason": "outside_world_forge_backend",
    }
    stepped = await skill.execute({"action": "step", "world_id": "skilltest", "ticks": 120}, {})
    assert stepped["ok"] and stepped["world"]["tick"] == 120
    inspected = await skill.execute({"action": "inspect", "world_id": "skilltest"}, {})
    assert inspected["ok"] and inspected["world"]["state_digest"]
    listed = await skill.execute({"action": "list"}, {})
    assert listed["ok"] and any(world["world_id"] == "skilltest" for world in listed["worlds"])
    bad = await skill.execute({"action": "create", "world_id": "skilltest", "seed": 5}, {})
    assert not bad["ok"]


async def test_world_forge_typed_contract_rejects_shallow_or_silent_inputs(host):
    from core.skills.world_forge import WorldForgeInput, WorldForgeSkill

    schema = WorldForgeInput.model_json_schema()
    assert schema["properties"]["ticks"]["type"] == "integer"
    assert schema["properties"]["impulse"]["anyOf"][0]["type"] == "array"

    skill = WorldForgeSkill()
    missing_identity = await skill.execute({"action": "inspect"}, {})
    unknown = await skill.execute({"action": "list", "surprise": True}, {})
    silently_clamped = await skill.execute(
        {"action": "agent", "world_id": "missing", "command": "walk", "ticks": 7000},
        {},
    )
    nonfinite = await skill.execute(
        {
            "action": "impulse",
            "world_id": "missing",
            "body_id": "ball",
            "impulse": [float("nan"), 0.0, 0.0],
        },
        {},
    )

    assert missing_identity["ok"] is False
    assert unknown["ok"] is False
    assert silently_clamped["ok"] is False
    assert "at most 6000" in silently_clamped["error"]
    assert nonfinite["ok"] is False


def test_world_forge_is_discovered_by_the_skill_catalog():
    from core.skills.discovery import build_skill_catalog

    catalog = build_skill_catalog(try_rust=False)
    assert "world_forge" in {declaration.name for declaration in catalog.accepted}
