"""World authorship: Aura conjures, banishes, sculpts, and chooses.

Every act is bounded, journaled, physically real (colliders change,
gravity changes are felt), and survives restarts.
"""
from __future__ import annotations

import numpy as np
import pytest

import core.worlds.hosting as hosting
from core.worlds import PhysicsError


@pytest.fixture
def host(tmp_path):
    yield hosting.reset_world_host_for_tests(tmp_path / "worlds")
    hosting.reset_world_host_for_tests(tmp_path / "unused")


async def _world(host, world_id="atelier", theme="arena"):
    await host.create_world(world_id, seed=5, size=16, theme=theme)
    return host.load_world(world_id)


async def test_conjured_body_is_physically_real(host):
    world = await _world(host)
    await host.conjure_body("atelier", {
        "body_id": "muse", "shape": "sphere", "at": (2.0, 2.0, 4.0),
        "mass": 1.0, "radius": 0.3,
    })
    z_before = float(world.physics.body("muse").position[2])
    await host.step_world("atelier", 60)
    world = host.load_world("atelier")
    assert float(world.physics.body("muse").position[2]) < z_before  # it falls
    assert any(e["kind"] == "conjured" for e in world.journal)


async def test_conjured_static_body_holds_position(host):
    world = await _world(host)
    await host.conjure_body("atelier", {
        "body_id": "monument", "shape": "box", "at": (0.0, 3.0, 1.0),
        "mass": 0.0, "half_extents": (0.5, 0.5, 1.0),
    })
    await host.step_world("atelier", 120)
    monument = world.physics.body("monument")
    assert monument.is_static
    np.testing.assert_allclose(monument.position, [0.0, 3.0, 1.0], atol=1e-9)


async def test_conjure_bounds_are_enforced(host):
    await _world(host)
    with pytest.raises(PhysicsError):
        await host.conjure_body("atelier", {"body_id": "Bad Name!", "at": (0, 0, 1)})
    with pytest.raises(PhysicsError):
        await host.conjure_body("atelier", {
            "body_id": "giant", "shape": "sphere", "radius": 50.0, "at": (0, 0, 1)})
    with pytest.raises(PhysicsError):
        await host.conjure_body("atelier", {
            "body_id": "lead", "mass": 99999.0, "at": (0, 0, 1)})
    with pytest.raises(PhysicsError):
        await host.conjure_body("atelier", {
            "body_id": "tesseract", "shape": "tesseract", "at": (0, 0, 1)})


async def test_body_cap_is_enforced(host, monkeypatch):
    await _world(host)
    monkeypatch.setattr(hosting, "MAX_BODIES_PER_WORLD", 30)
    world = host.load_world("atelier")
    with pytest.raises(PhysicsError, match="body cap"):
        for index in range(40):
            await host.conjure_body("atelier", {
                "body_id": f"clutter_{index}", "at": (0.0, 0.0, 2.0 + index)})
    assert len(world.physics.bodies) <= 30


async def test_banish_removes_body_and_ghost_impulses(host):
    world = await _world(host)
    await host.conjure_body("atelier", {
        "body_id": "ephemeral", "at": (1.0, 1.0, 0.5), "mass": 1.0, "radius": 0.4})
    await host.step_world("atelier", 120)  # let it land and build warm contacts
    await host.banish_body("atelier", "ephemeral")
    assert "ephemeral" not in world.physics.bodies
    assert not any(
        "ephemeral" in pair for pair in world.physics._contact_cache)
    assert any(e["kind"] == "banished" for e in world.journal)
    with pytest.raises(PhysicsError):
        await host.banish_body("atelier", "never-was")


async def test_agents_cannot_be_banished(host):
    await _world(host)
    await host.spawn_agent("atelier", "aura")
    with pytest.raises(PhysicsError, match="inhabitants"):
        await host.banish_body("atelier", "aura")


async def test_sculpting_changes_the_land_and_its_collisions(host):
    world = await _world(host, theme="plains")
    heights_before = np.asarray(world.blueprint.heightfield)
    await host.sculpt_terrain("atelier", 0.0, 0.0, 4.0, 5.0)
    heights_after = np.asarray(world.blueprint.heightfield)
    center = world.blueprint.size // 2
    assert heights_after[center, center] > heights_before[center, center] + 3.0
    # The hill is a real collider: a ball dropped on it rests high.
    await host.conjure_body("atelier", {
        "body_id": "pebble", "at": (0.0, 0.0, 12.0), "mass": 0.5, "radius": 0.2,
        "rolling_resistance": 0.05})
    await host.step_world("atelier", 2400)
    pebble = world.physics.body("pebble")
    assert float(pebble.position[2]) > 2.0
    assert any(e["kind"] == "sculpted" for e in world.journal)


async def test_gravity_choice_is_felt_exactly(host):
    world = await _world(host)
    await host.set_environment("atelier", gravity=-1.62)  # her moon
    await host.conjure_body("atelier", {
        "body_id": "feather", "at": (5.0, 5.0, 50.0), "mass": 0.1, "radius": 0.1})
    steps = 60
    await host.step_world("atelier", steps)
    feather = world.physics.body("feather")
    dt = world.physics.dt
    assert float(feather.velocity[2]) == pytest.approx(-1.62 * steps * dt, abs=1e-9)
    with pytest.raises(PhysicsError):
        await host.set_environment("atelier", gravity=-99.0)


async def test_structures_stand(host):
    world = await _world(host)
    result = await host.conjure_structure(
        "atelier", kind="tower", at=(4.0, 4.0, 0.0), span=5)
    assert len(result["bodies_created"]) == 5
    await host.step_world("atelier", 600)
    tops = [float(world.physics.body(b).position[2])
            for b in result["bodies_created"]]
    assert tops == sorted(tops) and tops[-1] > 4.0  # still a tower
    with pytest.raises(PhysicsError):
        await host.conjure_structure("atelier", kind="ziggurat", at=(0, 0, 0))


async def test_authorship_survives_restart(host, tmp_path):
    world = await _world(host)
    await host.sculpt_terrain("atelier", 2.0, 2.0, 3.0, 4.0)
    await host.conjure_body("atelier", {
        "body_id": "keepsake", "at": (1.0, 1.0, 6.0), "mass": 1.0, "radius": 0.3})
    await host.step_world("atelier", 240)
    digest = world.physics.state_digest()
    heights = [row[:] for row in world.blueprint.heightfield]

    reloaded_host = hosting.reset_world_host_for_tests(tmp_path / "worlds")
    reloaded = reloaded_host.load_world("atelier")
    assert reloaded.physics.state_digest() == digest
    assert reloaded.blueprint.heightfield == heights
    assert "keepsake" in reloaded.physics.bodies


async def test_world_forge_authorship_actions(host):
    from core.skills.world_forge import WorldForgeSkill

    skill = WorldForgeSkill()
    assert (await skill.execute(
        {"action": "create", "world_id": "studio", "seed": 3, "size": 16,
         "theme": "arena"}, {}))["ok"]
    conjured = await skill.execute(
        {"action": "conjure", "world_id": "studio", "body_id": "orb",
         "shape": "sphere", "at": (1.0, 1.0, 3.0), "mass": 1.0}, {})
    assert conjured["ok"], conjured
    sculpted = await skill.execute(
        {"action": "sculpt", "world_id": "studio", "target": (0.0, 0.0),
         "delta": 3.0}, {})
    assert sculpted["ok"], sculpted
    env = await skill.execute(
        {"action": "environment", "world_id": "studio", "gravity": -3.0}, {})
    assert env["ok"], env
    built = await skill.execute(
        {"action": "structure", "world_id": "studio", "structure": "stairs",
         "at": (2.0, 2.0, 0.0), "span": 3}, {})
    assert built["ok"], built
    banished = await skill.execute(
        {"action": "banish", "world_id": "studio", "body_id": "orb"}, {})
    assert banished["ok"], banished
    # Validation refuses incomplete authorship.
    bad = await skill.execute({"action": "conjure", "world_id": "studio"}, {})
    assert not bad["ok"]
