"""ASGI-level integration proof for the LAN device-pairing surface.

Drives the real FastAPI app (middleware stack, route registration,
cookie issuance) with starlette's TestClient. The client peer is
"testclient", which the auth layer treats as a remote (non-loopback)
host — exactly the phone's position on the LAN.
"""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

import core.security.device_pairing as dp
from interface import auth
from interface import server


MASTER = "test-master-token"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(auth.config, "api_token", MASTER, raising=False)
    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(dp, "registry_path", lambda: tmp_path / "paired_devices.json")
    dp.reset_device_registry_for_tests(tmp_path / "paired_devices.json")
    # Deliberately NOT `with TestClient(...)`: entering the client runs the
    # app lifespan, which boots the full runtime (services, model). Requests
    # without the context manager exercise the complete middleware + route
    # stack, which is exactly the surface under test.
    test_client = TestClient(server.app, backend="asyncio")
    yield test_client
    dp.reset_device_registry_for_tests(tmp_path / "unused.json")


def _pair(client: TestClient) -> dict:
    begin = client.post(
        "/api/devices/pair/begin",
        headers={"Authorization": f"Bearer {MASTER}"},
        json={},
    )
    assert begin.status_code == 200, begin.text
    complete = client.post(
        "/api/devices/pair/complete",
        json={"code": begin.json()["code"], "device_name": "pytest phone"},
    )
    assert complete.status_code == 200, complete.text
    return complete.json()


def test_pair_begin_requires_owner(client):
    response = client.post("/api/devices/pair/begin", json={})
    assert response.status_code in (401, 403)


def test_pairing_page_is_reachable_unauthenticated(client):
    response = client.get("/pair")
    assert response.status_code == 200
    assert "Pair" in response.text


def test_full_pairing_flow_grants_conversation_surface(client):
    issued = _pair(client)
    assert issued["token"].startswith("adt1.")
    # The completion response set a session cookie on the client jar.
    assert auth.DEVICE_SESSION_COOKIE_NAME in client.cookies

    # Static shell assets now load (the phone can render the UI).
    asset = client.get("/static/aura.js")
    assert asset.status_code == 200

    # The chat endpoint is authorized (any non-auth failure is fine here:
    # the cognitive engine is not booted in tests).
    chat = client.post("/api/chat", json={"message": "hello from the phone"})
    assert chat.status_code not in (401, 403)


def test_wrong_code_rejected(client):
    begin = client.post(
        "/api/devices/pair/begin",
        headers={"Authorization": f"Bearer {MASTER}"},
        json={},
    )
    assert begin.status_code == 200
    complete = client.post(
        "/api/devices/pair/complete",
        json={"code": "99999999", "device_name": "intruder"},
    )
    assert complete.status_code == 401


def test_unpaired_remote_root_redirects_to_pair(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/pair"


def test_unpaired_remote_cannot_load_shell_assets(client):
    response = client.get("/static/aura.js")
    assert response.status_code == 401


def test_device_token_rejected_on_control_surface(client):
    _pair(client)
    for path in ("/api/reboot", "/api/system/hot-reload", "/api/skill/execute"):
        response = client.post(path, json={})
        assert response.status_code == 403, path


def test_revocation_cuts_off_device(client):
    issued = _pair(client)
    revoke = client.post(
        "/api/devices/revoke",
        headers={"Authorization": f"Bearer {MASTER}"},
        json={"device_id": issued["device_id"]},
    )
    assert revoke.status_code == 200
    response = client.get("/static/aura.js")
    assert response.status_code == 401
