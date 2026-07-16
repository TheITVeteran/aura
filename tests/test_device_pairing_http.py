"""ASGI-level integration proof for the LAN device-pairing surface.

Drives the real FastAPI app (middleware stack, route registration,
cookie issuance) with starlette's TestClient. The client peer is
"testclient", which the auth layer treats as a remote (non-loopback)
host — exactly the phone's position on the LAN.
"""
from __future__ import annotations

import asyncio
import re
import threading

import pytest
from starlette.testclient import TestClient

import core.security.device_pairing as dp
from interface import auth, server
from interface.routes import chat as chat_routes

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

    bootstrap = client.get("/api/ui/bootstrap")
    assert bootstrap.status_code == 200
    bootstrap_payload = bootstrap.json()
    assert bootstrap_payload["access"]["surface"] == "paired_device"
    assert bootstrap_payload["access"]["conversation_only"] is True
    assert bootstrap_payload["tools"] == []
    assert bootstrap_payload["diagnostics"] == {}
    assert bootstrap_payload["desktop_access"]["overall_status"] == "surface_not_authorized"

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
    for path in (
        "/api/reboot",
        "/api/system/hot-reload",
        "/api/skill/execute",
        "/api/performance/frame",
        "/api/chat/regenerate",
    ):
        response = client.post(path, json={})
        assert response.status_code == 403, path


def test_paired_history_is_scoped_to_its_device_principal(client):
    issued = _pair(client)
    paired_session = f"paired-device:{issued['device_id']}"
    original = list(chat_routes._conversation_log)
    chat_routes._conversation_log[:] = [
        {
            "session_id": "127.0.0.1",
            "role": "user",
            "content": "owner-only history",
        },
        {
            "session_id": paired_session,
            "role": "user",
            "content": "paired history",
        },
    ]
    try:
        sessions = client.get("/api/sessions")
        assert sessions.status_code == 200
        payload = sessions.json()
        messages = payload["current_session"]["messages"]
        assert [message["content"] for message in messages] == ["paired history"]
        assert payload["persisted_sessions"] == []

        bootstrap = client.get("/api/ui/bootstrap")
        recent = bootstrap.json()["conversation"]["recent"]
        assert [message["content"] for message in recent] == ["paired history"]
    finally:
        chat_routes._conversation_log[:] = original


def test_paired_websocket_receives_only_sanitized_conversation_heartbeat(client):
    _pair(client)

    with client.websocket_connect("/ws") as websocket:
        authenticated = websocket.receive_json()
        assert authenticated == {"type": "auth_success", "note": "paired_device"}

        websocket.send_json({"type": "ping"})
        heartbeat = websocket.receive_json()

    assert heartbeat["type"] == "pong"
    assert "required_probes" not in heartbeat
    assert "blockers" not in heartbeat
    assert "runtime_status" not in heartbeat
    assert "model_path" not in heartbeat.get("conversation_lane", {})


def test_paired_websocket_rejects_binary_voice_frames(client, monkeypatch):
    _pair(client)

    def _voice_engine_must_not_run():
        raise AssertionError("paired binary input reached the owner voice engine")

    monkeypatch.setattr(server, "_voice_engine_fn", _voice_engine_must_not_run)
    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json() == {
            "type": "auth_success",
            "note": "paired_device",
        }
        websocket.send_bytes(b"not-authorized-audio")
        denied = websocket.receive_json()
        websocket.send_json({"type": "ping"})
        heartbeat = websocket.receive_json()

    assert denied["status"] == "paired_device_voice_scope_denied"
    assert heartbeat["type"] == "pong"


def test_paired_chat_projects_owner_only_metadata_and_ignores_internal_headers(client):
    _pair(client)

    response = client.post(
        "/api/chat",
        headers={
            "X-Aura-Benchmark": "true",
            "X-Aura-Allow-Legacy-Orchestrator": "true",
        },
        json={"message": "What can you do?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "paired_device_capability_scope"
    assert set(payload) <= {
        "response",
        "response_confidence",
        "status",
        "conversation_lane",
        "turn_id",
        "idempotency_key",
        "delivery_state",
        "delivery_generation",
        "delivery_replayed",
    }
    assert set(payload.get("conversation_lane", {})) <= {
        "active_generation",
        "active_generations",
        "conversation_ready",
        "state",
    }
    serialized = response.text.lower()
    assert re.fullmatch(r"[0-9a-f]{32}", payload["turn_id"])
    assert re.fullmatch(r"server-[0-9a-f]{32}", payload["idempotency_key"])
    assert payload["delivery_state"] == "completed"
    assert payload["delivery_generation"] == 1
    assert payload["delivery_replayed"] is False
    assert "model_path" not in serialized
    assert "live_turn_contract" not in serialized
    assert "required_subsystems" not in serialized


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


def test_revocation_terminates_an_already_authenticated_websocket(client):
    issued = _pair(client)

    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json() == {
            "type": "auth_success",
            "note": "paired_device",
        }
        revoke = client.post(
            "/api/devices/revoke",
            headers={"Authorization": f"Bearer {MASTER}"},
            json={"device_id": issued["device_id"]},
        )
        assert revoke.status_code == 200

        websocket.send_json({"type": "ping"})
        revoked = websocket.receive_json()

    assert revoked["status"] == "paired_device_session_revoked"


def test_revocation_cancels_inflight_websocket_turn_and_overlap_is_rejected(
    client,
    monkeypatch,
):
    issued = _pair(client)
    started = threading.Event()
    cancelled = threading.Event()

    async def _blocking_cognitive_turn(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(
        chat_routes,
        "_run_cognitive_engine_chat_turn",
        _blocking_cognitive_turn,
    )

    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["note"] == "paired_device"
        websocket.send_json({"type": "user_message", "content": "first turn"})
        assert started.wait(timeout=2.0)

        websocket.send_json({"type": "user_message", "content": "overlap"})
        busy = websocket.receive_json()
        assert busy["status"] == "conversation_turn_in_progress"

        revoke = client.post(
            "/api/devices/revoke",
            headers={"Authorization": f"Bearer {MASTER}"},
            json={"device_id": issued["device_id"]},
        )
        assert revoke.status_code == 200
        websocket.send_json({"type": "ping"})
        revoked = websocket.receive_json()

    assert revoked["status"] == "paired_device_session_revoked"
    assert cancelled.wait(timeout=2.0)
