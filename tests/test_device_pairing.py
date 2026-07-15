"""Paired-device LAN embodiment: registry lifecycle + auth scoping.

Covers core/security/device_pairing.py and the interface/auth.py
enforcement that keeps device tokens on the conversation surface and
off the sovereign control surface.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import core.security.device_pairing as dp
from interface import auth


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setattr(dp, "registry_path", lambda: tmp_path / "paired_devices.json")
    reg = dp.reset_device_registry_for_tests(tmp_path / "paired_devices.json")
    yield reg
    dp.reset_device_registry_for_tests(tmp_path / "unused.json")


def _internal_only(monkeypatch, value: bool) -> None:
    monkeypatch.setattr(
        dp.get_config().security, "internal_only_mode", value, raising=False
    )


# ── Registry lifecycle ───────────────────────────────────────────

async def test_pairing_roundtrip_and_verify(registry, monkeypatch):
    _internal_only(monkeypatch, False)
    challenge = registry.begin_pairing("bryan")
    assert len(challenge["code"]) == 8 and challenge["code"].isdigit()

    issued = await registry.complete_pairing(challenge["code"], "Bryan's phone")
    assert issued["token"].startswith("adt1.")

    device = registry.verify_token(issued["token"])
    assert device is not None
    assert device.name == "Bryan's phone"
    assert device.scopes == (dp.SCOPE_CONVERSATION,)
    assert device.principal_id == "bryan"


async def test_token_never_persisted_in_clear(registry, monkeypatch, tmp_path):
    _internal_only(monkeypatch, False)
    challenge = registry.begin_pairing("bryan")
    issued = await registry.complete_pairing(challenge["code"], "phone")

    on_disk = (tmp_path / "paired_devices.json").read_text(encoding="utf-8")
    secret = issued["token"].split(".")[2]
    assert issued["token"] not in on_disk
    assert secret not in on_disk
    document = json.loads(on_disk)
    assert document["payload"]["devices"][0]["token_sha256"]


async def test_registry_persistence_survives_reload(registry, monkeypatch, tmp_path):
    _internal_only(monkeypatch, False)
    issued = await registry.complete_pairing(
        registry.begin_pairing("bryan")["code"], "phone"
    )

    reloaded = dp.DevicePairingRegistry.load(tmp_path / "paired_devices.json")
    restored = reloaded.verify_token(issued["token"])
    assert restored is not None
    assert restored.principal_id == "bryan"


def test_pairing_rejects_missing_relational_principal(registry, monkeypatch):
    _internal_only(monkeypatch, False)
    with pytest.raises(dp.PairingError, match="verified relational principal"):
        registry.begin_pairing("")


async def test_wrong_code_attempts_exhaust_challenge(registry, monkeypatch):
    _internal_only(monkeypatch, False)
    registry.begin_pairing("bryan")
    for _ in range(dp._MAX_ATTEMPTS):
        with pytest.raises(dp.PairingError):
            await registry.complete_pairing("00000000", "intruder")
    # Challenge is now dead even with the right code.
    with pytest.raises(dp.PairingError):
        await registry.complete_pairing("00000000", "intruder")


async def test_expired_code_rejected(registry, monkeypatch):
    _internal_only(monkeypatch, False)
    challenge = registry.begin_pairing("bryan")
    registry._challenge.expires_at = time.time() - 1
    with pytest.raises(dp.PairingError):
        await registry.complete_pairing(challenge["code"], "late")


async def test_revocation_kills_token(registry, monkeypatch):
    _internal_only(monkeypatch, False)
    issued = await registry.complete_pairing(
        registry.begin_pairing("bryan")["code"], "phone"
    )
    assert registry.verify_token(issued["token"]) is not None

    assert await registry.revoke_device(issued["device_id"]) is True
    assert registry.verify_token(issued["token"]) is None


async def test_internal_only_mode_fails_closed(registry, monkeypatch):
    _internal_only(monkeypatch, False)
    issued = await registry.complete_pairing(
        registry.begin_pairing("bryan")["code"], "phone"
    )

    _internal_only(monkeypatch, True)
    with pytest.raises(dp.PairingDisabledError):
        registry.begin_pairing("bryan")
    with pytest.raises(dp.PairingDisabledError):
        await registry.complete_pairing("12345678", "phone")
    # Even already-issued tokens stop verifying.
    assert registry.verify_token(issued["token"]) is None


def test_malformed_tokens_rejected(registry, monkeypatch):
    _internal_only(monkeypatch, False)
    for bad in (None, "", "adt1", "adt1.only-two", "wrong.prefix.secret", "adt1.x.y.z"):
        assert registry.verify_token(bad) is None


async def test_corrupt_registry_file_authorizes_nobody(registry, monkeypatch, tmp_path):
    _internal_only(monkeypatch, False)
    issued = await registry.complete_pairing(
        registry.begin_pairing("bryan")["code"], "phone"
    )
    (tmp_path / "paired_devices.json").write_text("{not json", encoding="utf-8")

    reloaded = dp.DevicePairingRegistry.load(tmp_path / "paired_devices.json")
    assert reloaded.devices == {}
    assert reloaded.verify_token(issued["token"]) is None


# ── HTTP auth scoping ────────────────────────────────────────────

def _remote_request(path: str, token: str | None = None, method: str = "GET",
                    headers: dict | None = None):
    hdrs = dict(headers or {})
    if token:
        hdrs["X-Aura-Device-Token"] = token
    return SimpleNamespace(
        client=SimpleNamespace(host="192.168.1.77"),
        headers=hdrs,
        cookies={},
        method=method,
        url=SimpleNamespace(path=path),
    )


@pytest.fixture
async def paired_token(registry, monkeypatch):
    _internal_only(monkeypatch, False)
    monkeypatch.setattr(auth.config, "api_token", "master-token-value", raising=False)
    issued = await registry.complete_pairing(
        registry.begin_pairing("bryan")["code"], "phone"
    )
    return issued["token"]


async def test_device_token_allows_conversation_surface(paired_token):
    operations = (
        ("/api/chat", "POST"),
        ("/static/aura.js", "GET"),
        ("/api/ui/bootstrap", "GET"),
        ("/api/sessions", "GET"),
    )
    for path, method in operations:
        auth.validate_runtime_security_request(
            _remote_request(path, paired_token, method=method)
        )


async def test_device_token_denied_on_control_surface(paired_token):
    for path in ("/api/skill/execute", "/api/reboot", "/api/system/hot-reload",
                 "/api/privacy/export", "/api/devices", "/api/performance/frame",
                 "/api/chat/regenerate", "/api/sessions"):
        with pytest.raises(HTTPException) as err:
            auth.validate_runtime_security_request(
                _remote_request(path, paired_token, method="POST")
            )
        assert err.value.status_code == 403


async def test_forged_device_token_still_unauthorized(registry, monkeypatch):
    _internal_only(monkeypatch, False)
    monkeypatch.setattr(auth.config, "api_token", "master-token-value", raising=False)
    with pytest.raises(HTTPException) as err:
        auth.validate_runtime_security_request(
            _remote_request("/api/chat", "adt1.deadbeef.forged-secret")
        )
    assert err.value.status_code == 401


async def test_revoked_device_token_unauthorized(registry, monkeypatch, paired_token):
    device_id = paired_token.split(".")[1]
    await registry.revoke_device(device_id)
    with pytest.raises(HTTPException) as err:
        auth.validate_runtime_security_request(_remote_request("/api/chat", paired_token))
    assert err.value.status_code == 401


async def test_device_cookie_authenticates(paired_token):
    request = _remote_request("/api/chat", method="POST")
    request.cookies = {auth.DEVICE_SESSION_COOKIE_NAME: paired_token}
    auth.validate_runtime_security_request(request)
    assert auth.relational_principal_id_for_request(request) == "bryan"


async def test_unbound_legacy_device_cannot_select_a_relational_profile(
    registry,
    monkeypatch,
    paired_token,
):
    device_id = paired_token.split(".")[1]
    registry.devices[device_id].principal_id = ""
    request = _remote_request("/api/chat", paired_token, method="POST")

    assert auth.relational_principal_id_for_request(request) is None


def test_owner_request_uses_configured_identity_not_conversation_state(monkeypatch):
    class _IdentityKernel:
        @staticmethod
        def get_current_identity():
            return {"primary_operator": "  Bryan   Primary  "}

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: (
                _IdentityKernel() if name == "identity_kernel" else default
            )
        ),
    )
    owner_request = _remote_request("/api/chat", method="POST")
    owner_request.client.host = "127.0.0.1"
    owner_request.headers["Host"] = "127.0.0.1:8000"

    assert auth.relational_principal_id_for_request(owner_request) == "bryan primary"


def test_unknown_remote_request_cannot_select_a_relational_profile(monkeypatch):
    monkeypatch.setattr(auth.config, "api_token", "master-token-value", raising=False)

    assert (
        auth.relational_principal_id_for_request(
            _remote_request("/api/chat", method="POST")
        )
        is None
    )


async def test_pairing_public_paths_reachable_without_credentials(registry, monkeypatch):
    _internal_only(monkeypatch, False)
    monkeypatch.setattr(auth.config, "api_token", "master-token-value", raising=False)
    auth.validate_runtime_security_request(_remote_request("/pair"))
    auth.validate_runtime_security_request(
        _remote_request("/api/devices/pair/complete", method="POST")
    )


async def test_internal_only_blocks_devices_at_http_layer(paired_token, monkeypatch):
    monkeypatch.setattr(auth.config.security, "internal_only_mode", True, raising=False)
    with pytest.raises(HTTPException) as err:
        auth.validate_runtime_security_request(_remote_request("/api/chat", paired_token))
    assert err.value.status_code == 403


async def test_same_host_origin_not_treated_as_csrf(paired_token):
    request = _remote_request(
        "/api/chat", paired_token, method="POST",
        headers={"Origin": "http://192.168.1.20:8000", "Host": "192.168.1.20:8000"},
    )
    auth.validate_runtime_security_request(request)


async def test_foreign_origin_still_treated_as_csrf(paired_token):
    request = _remote_request(
        "/api/chat", paired_token, method="POST",
        headers={"Origin": "http://evil.example", "Host": "192.168.1.20:8000"},
    )
    with pytest.raises(HTTPException) as err:
        auth.validate_runtime_security_request(request)
    assert err.value.status_code == 403


def test_device_path_allowlist_is_deny_by_default():
    assert auth.device_path_allowed("/api/chat", "POST")
    assert not auth.device_path_allowed("/api/chat", "GET")
    assert not auth.device_path_allowed("/api/chat/regenerate", "POST")
    assert auth.device_path_allowed("/static/aura.css", "GET")
    assert not auth.device_path_allowed("/static/aura.css", "POST")
    assert auth.device_path_allowed("/ws", "GET")
    assert auth.device_path_allowed("/api/worlds/demo/render", "HEAD")
    assert not auth.device_path_allowed("/api/worlds/demo/step", "POST")
    assert not auth.device_path_allowed("/api/skill/execute", "POST")
    assert not auth.device_path_allowed("/api/devices", "GET")
    assert not auth.device_path_allowed("/memory", "GET")
    assert not auth.device_path_allowed("/rpc/anything", "GET")
    assert not auth.device_path_allowed("/api/settings", "GET")


async def test_access_profile_advertises_paired_capability_boundary(paired_token):
    profile = auth.request_access_profile(
        _remote_request("/api/ui/bootstrap", paired_token)
    )

    assert profile["surface"] == "paired_device"
    assert profile["conversation_only"] is True
    assert profile["capabilities"]["chat"] is True
    assert profile["capabilities"]["world_read"] is True
    assert profile["capabilities"]["performance_telemetry"] is False
    assert profile["capabilities"]["desktop_control"] is False


async def test_explicit_paired_identity_precedes_loopback_owner_in_access_profile(
    paired_token,
):
    request = _remote_request("/api/ui/bootstrap", paired_token)
    request.client.host = "127.0.0.1"

    profile = auth.request_access_profile(request)

    assert profile["surface"] == "paired_device"
    assert profile["capabilities"]["performance_telemetry"] is False
    assert profile["capabilities"]["diagnostics"] is False


async def test_repeated_device_scope_denials_are_rate_limited(
    paired_token, caplog
):
    auth._DEVICE_DENIAL_LOG_STATE.clear()
    request = _remote_request(
        "/api/performance/frame",
        paired_token,
        method="POST",
    )

    for _ in range(3):
        with pytest.raises(HTTPException) as err:
            auth.validate_runtime_security_request(request)
        assert err.value.status_code == 403

    warnings = [
        record
        for record in caplog.records
        if "out-of-scope operation POST /api/performance/frame" in record.message
    ]
    assert len(warnings) == 1


async def test_paired_cognitive_turn_refuses_desktop_action_before_model_use():
    from interface.routes import chat

    trace = {}
    result = await chat._run_cognitive_engine_chat_turn(
        "Open Notes and write the owner's private memory there",
        visible_user_message="Open Notes and write the owner's private memory there",
        require_engine=True,
        conversation_only_surface=True,
        turn_trace=trace,
    )

    assert "require the owner surface" in result
    assert trace["response_path"] == "paired_device_action_scope_denied"
    assert trace["bounded_contract_used"] is True


async def test_paired_cognitive_turn_scopes_owner_diagnostics_before_model_use():
    from interface.routes import chat

    trace = {}
    result = await chat._run_cognitive_engine_chat_turn(
        "Which exact model is loaded and what internal services are failing?",
        require_engine=True,
        conversation_only_surface=True,
        lane={
            "state": "ready",
            "conversation_ready": True,
            "model_path": "/private/owner/model",
            "last_failure_reason": "private failure detail",
        },
        turn_trace=trace,
    )

    assert "owner surface" in result
    assert "/private/owner/model" not in result
    assert trace["response_path"] == "paired_device_runtime_scope"


def test_paired_chat_wire_projection_is_allowlist_based():
    from interface.routes import chat

    projected = chat._paired_chat_response_payload(
        {
            "response": "hello",
            "status": "ok",
            "thought": "private chain",
            "live_turn_contract": {"required_subsystems": {"vault": False}},
            "memory_pressure": {"available_gb": 12.5},
            "conversation_lane": {
                "state": "ready",
                "conversation_ready": True,
                "model_path": "/private/owner/model",
                "last_failure_reason": "private failure detail",
            },
        }
    )

    assert projected == {
        "response": "hello",
        "status": "ok",
        "conversation_lane": {
            "state": "ready",
            "conversation_ready": True,
        },
    }


def test_chat_idempotency_keys_are_namespaced_by_authenticated_session():
    from interface.routes import chat

    owner = chat._chat_idempotency_cache_key("owner", "same-wire-key")
    phone = chat._chat_idempotency_cache_key(
        "paired-device:phone",
        "same-wire-key",
    )

    assert owner != phone
    assert owner == ("owner", "same-wire-key")
    assert phone == ("paired-device:phone", "same-wire-key")
