"""Phone voice lane: owner-granted per-device voice scope + local TLS.

Deny-by-default is preserved (Zenflow's posture): a paired device gets
NO voice at pairing; only an explicit owner grant opens the lane, and
revocation takes effect on the next frame, not the next connection.
"""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

import core.security.device_pairing as dp
from interface import auth, server

MASTER = "test-master-token"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(auth.config, "api_token", MASTER, raising=False)
    monkeypatch.setattr(auth.config.security, "internal_only_mode", False, raising=False)
    monkeypatch.setattr(dp, "registry_path", lambda: tmp_path / "paired_devices.json")
    dp.reset_device_registry_for_tests(tmp_path / "paired_devices.json")
    yield TestClient(server.app, backend="asyncio")
    dp.reset_device_registry_for_tests(tmp_path / "unused.json")


def _pair(client: TestClient) -> dict:
    begin = client.post("/api/devices/pair/begin",
                        headers={"Authorization": f"Bearer {MASTER}"}, json={})
    assert begin.status_code == 200, begin.text
    complete = client.post("/api/devices/pair/complete",
                           json={"code": begin.json()["code"],
                                 "device_name": "voice phone"})
    assert complete.status_code == 200, complete.text
    return complete.json()


# ── registry semantics ───────────────────────────────────────────

async def test_voice_is_never_minted_at_pairing(tmp_path, monkeypatch):
    monkeypatch.setattr(dp.get_config().security, "internal_only_mode",
                        False, raising=False)
    registry = dp.reset_device_registry_for_tests(tmp_path / "d.json")
    issued = await registry.complete_pairing(
        registry.begin_pairing("bryan")["code"], "phone")
    device = registry.verify_token(issued["token"])
    assert dp.SCOPE_VOICE not in device.scopes

    assert await registry.grant_scope(issued["device_id"], dp.SCOPE_VOICE)
    assert dp.SCOPE_VOICE in registry.devices[issued["device_id"]].scopes
    assert await registry.revoke_scope(issued["device_id"], dp.SCOPE_VOICE)
    assert dp.SCOPE_VOICE not in registry.devices[issued["device_id"]].scopes

    with pytest.raises(dp.PairingError):
        await registry.grant_scope(issued["device_id"], "privileged_mutation")
    assert not await registry.grant_scope("no-such-device", dp.SCOPE_VOICE)


# ── HTTP + WS end to end ─────────────────────────────────────────

def test_scope_grant_is_owner_only(client):
    issued = _pair(client)
    denied = client.post("/api/devices/grant-scope",
                         json={"device_id": issued["device_id"], "scope": "voice"})
    assert denied.status_code == 403  # the paired device cannot self-grant

    granted = client.post("/api/devices/grant-scope",
                          headers={"Authorization": f"Bearer {MASTER}"},
                          json={"device_id": issued["device_id"], "scope": "voice"})
    assert granted.status_code == 200

    bogus = client.post("/api/devices/grant-scope",
                        headers={"Authorization": f"Bearer {MASTER}"},
                        json={"device_id": issued["device_id"], "scope": "root"})
    assert bogus.status_code == 400


def test_voice_frames_denied_without_scope_and_flow_with_it(client, monkeypatch):
    issued = _pair(client)
    fed: list[bytes] = []

    class _Engine:
        async def feed_chunk(self, chunk: bytes) -> None:
            fed.append(chunk)

    monkeypatch.setattr(server, "_voice_engine_fn", lambda: _Engine())

    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["type"] == "auth_success"
        websocket.send_bytes(b"pcm-without-scope")
        denied = websocket.receive_json()
        assert denied["status"] == "paired_device_voice_scope_denied"
    assert fed == []

    grant = client.post("/api/devices/grant-scope",
                        headers={"Authorization": f"Bearer {MASTER}"},
                        json={"device_id": issued["device_id"], "scope": "voice"})
    assert grant.status_code == 200

    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["type"] == "auth_success"
        websocket.send_bytes(b"pcm-with-scope")
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"
    assert fed == [b"pcm-with-scope"]

    # Live revocation: mid-session, the very next frame is denied.
    fed.clear()
    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["type"] == "auth_success"
        websocket.send_bytes(b"first-ok")
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"
        revoke = client.post("/api/devices/revoke-scope",
                             headers={"Authorization": f"Bearer {MASTER}"},
                             json={"device_id": issued["device_id"],
                                   "scope": "voice"})
        assert revoke.status_code == 200
        websocket.send_bytes(b"after-revocation")
        denied = websocket.receive_json()
        assert denied["status"] == "paired_device_voice_scope_denied"
    assert fed == [b"first-ok"]


# ── local TLS ────────────────────────────────────────────────────

def test_local_certificate_generation_and_reuse(tmp_path, monkeypatch):
    import core.security.tls_local as tls

    monkeypatch.setattr(tls, "tls_dir", lambda: tmp_path / "tls")
    first = tls.ensure_local_certificate()
    assert first is not None
    cert_path, key_path = first
    assert cert_path.exists() and key_path.exists()
    assert (key_path.stat().st_mode & 0o777) == 0o600

    from cryptography import x509

    certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
    san = certificate.extensions.get_extension_for_class(
        x509.SubjectAlternativeName).value
    assert "127.0.0.1" in {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
    assert "localhost" in san.get_values_for_type(x509.DNSName)

    # Second call reuses, byte-identical.
    certificate_before = cert_path.read_bytes()
    key_before = key_path.read_bytes()
    again = tls.ensure_local_certificate()
    assert again == first
    assert cert_path.read_bytes() == certificate_before
    assert key_path.read_bytes() == key_before


def test_local_certificate_regenerates_mismatched_key_pair(tmp_path, monkeypatch):
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    import core.security.tls_local as tls

    monkeypatch.setattr(tls, "tls_dir", lambda: tmp_path / "tls")
    first = tls.ensure_local_certificate()
    assert first is not None
    cert_path, key_path = first
    certificate_before = cert_path.read_bytes()
    unrelated_key = ec.generate_private_key(ec.SECP256R1())
    key_path.write_bytes(
        unrelated_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )

    regenerated = tls.ensure_local_certificate()

    assert regenerated == first
    assert cert_path.read_bytes() != certificate_before
    certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
    private_key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    assert certificate.public_key().public_numbers() == private_key.public_key().public_numbers()
    assert (key_path.stat().st_mode & 0o777) == 0o600


def test_local_certificate_fails_closed_when_batch_commit_fails(tmp_path, monkeypatch):
    import core.runtime.file_write_gateway as file_write_gateway
    import core.security.tls_local as tls

    class FailingGateway:
        def ensure_directory(self, path, *, source):
            path.mkdir(parents=True, exist_ok=True)

        def write_bytes_batch(self, entries, *, source):
            raise OSError("injected certificate persistence failure")

    monkeypatch.setattr(tls, "tls_dir", lambda: tmp_path / "tls")
    monkeypatch.setattr(
        file_write_gateway,
        "get_file_write_gateway",
        lambda: FailingGateway(),
    )

    assert tls.ensure_local_certificate() is None
    assert not (tmp_path / "tls" / "aura_local.crt").exists()
    assert not (tmp_path / "tls" / "aura_local.key").exists()


def test_tls_enabled_flag(monkeypatch):
    import core.security.tls_local as tls

    monkeypatch.delenv("AURA_ENABLE_TLS", raising=False)
    assert not tls.tls_enabled()
    monkeypatch.setenv("AURA_ENABLE_TLS", "1")
    assert tls.tls_enabled()
