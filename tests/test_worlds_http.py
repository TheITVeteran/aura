"""World viewer HTTP surface: read access for paired devices, owner-only
mutation, complete render payloads.
"""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

import core.security.device_pairing as dp
import core.worlds.hosting as hosting
from interface import auth
from interface import server

MASTER = "test-master-token"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(auth.config, "api_token", MASTER, raising=False)
    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(dp, "registry_path", lambda: tmp_path / "paired_devices.json")
    dp.reset_device_registry_for_tests(tmp_path / "paired_devices.json")
    hosting.reset_world_host_for_tests(tmp_path / "worlds")
    # No lifespan (would boot the runtime); middleware + routes still run.
    yield TestClient(server.app, backend="asyncio")
    dp.reset_device_registry_for_tests(tmp_path / "unused.json")
    hosting.reset_world_host_for_tests(tmp_path / "unused-worlds")


def _owner_headers():
    return {"Authorization": f"Bearer {MASTER}"}


def _pair_device(client: TestClient) -> None:
    begin = client.post("/api/devices/pair/begin", headers=_owner_headers(), json={})
    assert begin.status_code == 200
    complete = client.post(
        "/api/devices/pair/complete",
        json={"code": begin.json()["code"], "device_name": "viewer phone"},
    )
    assert complete.status_code == 200


async def _make_world():
    await hosting.get_world_host().create_world(
        "vista", seed=8, size=16, theme="arena")


async def test_owner_can_list_render_and_step(client):
    await _make_world()
    listed = client.get("/api/worlds", headers=_owner_headers())
    assert listed.status_code == 200
    assert any(w["world_id"] == "vista" for w in listed.json()["worlds"])

    rendered = client.get("/api/worlds/vista/render", headers=_owner_headers())
    assert rendered.status_code == 200
    body = rendered.json()
    shapes = {b["shape"] for b in body["bodies"]}
    assert "plane" in shapes
    assert all({"position", "orientation", "static"} <= set(b) for b in body["bodies"])

    stepped = client.post("/api/worlds/vista/step", headers=_owner_headers(),
                          json={"ticks": 60})
    assert stepped.status_code == 200
    assert stepped.json()["world"]["tick"] == 60


async def test_paired_device_can_watch_but_not_mutate(client):
    await _make_world()
    _pair_device(client)

    assert client.get("/api/worlds").status_code == 200
    assert client.get("/api/worlds/vista/render").status_code == 200
    page = client.get("/worlds")
    assert page.status_code == 200 and "Aura — Worlds" in page.text

    stepped = client.post("/api/worlds/vista/step", json={"ticks": 60})
    assert stepped.status_code == 403


async def test_unpaired_remote_gets_nothing(client):
    await _make_world()
    assert client.get("/api/worlds").status_code == 401
    assert client.get("/api/worlds/vista/render").status_code == 401
    page = client.get("/worlds", follow_redirects=False)
    assert page.status_code in (307, 401)


async def test_unknown_world_is_404(client):
    _pair_device(client)
    assert client.get("/api/worlds/nowhere/render").status_code == 404
