"""Signed connect + capability manifest for paired devices.

A device token is possession-is-authentication, and this module's own
threat model admits the wire is plain HTTP on the home LAN. Anyone who
captures a token can present it from anywhere, as anything.

Adapted from OpenClaw's gateway, which requires clients to sign a
server-chosen `connect.challenge` nonce, binds `platform` and
`deviceFamily` into the signed payload, and pins paired metadata so a
change forces re-pairing.
"""
from __future__ import annotations

import hashlib
import hmac

import pytest

from core.security.device_pairing import (
    SCOPE_CONVERSATION,
    DevicePairingRegistry,
    PairedDevice,
    PairingError,
    _manifest_digest,
)

PHONE_CAPS = ("camera.capture", "location.get", "notify")


@pytest.fixture
def registry(tmp_path):
    return DevicePairingRegistry(path=tmp_path / "devices.json")


def _add_device(registry, *, secret="s3cret", pinned=True, caps=PHONE_CAPS):
    device = PairedDevice(
        device_id="dev1",
        name="phone",
        token_sha256=hashlib.sha256(secret.encode()).hexdigest(),
        scopes=(SCOPE_CONVERSATION,),
        created_at=0.0,
        last_seen=0.0,
        principal_id="bryan",
        capabilities=tuple(caps) if pinned else (),
        platform="ios" if pinned else "",
        device_family="phone" if pinned else "",
    )
    registry.devices[device.device_id] = device
    return device


def _sign(registry, device, nonce, *, platform="ios", family="phone", caps=PHONE_CAPS):
    """What an honest device computes. It knows the secret, so it can
    derive the same digest the server stored."""
    return registry.connect_signature(
        token_sha256=device.token_sha256,
        nonce=nonce,
        device_id=device.device_id,
        platform=platform,
        device_family=family,
        manifest_sha256=_manifest_digest(tuple(sorted(caps))),
    )


# ── The handshake ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_correctly_signed_connect_is_accepted(registry):
    device = _add_device(registry)
    challenge = registry.begin_connect("dev1")
    signature = _sign(registry, device, challenge["nonce"])

    verified = await registry.verify_connect(
        "dev1",
        nonce=challenge["nonce"],
        signature=signature,
        platform="ios",
        device_family="phone",
        capabilities=PHONE_CAPS,
    )
    assert verified.device_id == "dev1"


@pytest.mark.asyncio
async def test_a_nonce_cannot_be_replayed(registry):
    device = _add_device(registry)
    challenge = registry.begin_connect("dev1")
    signature = _sign(registry, device, challenge["nonce"])

    await registry.verify_connect(
        "dev1", nonce=challenge["nonce"], signature=signature,
        platform="ios", device_family="phone", capabilities=PHONE_CAPS,
    )
    with pytest.raises(PairingError, match="No connect challenge"):
        await registry.verify_connect(
            "dev1", nonce=challenge["nonce"], signature=signature,
            platform="ios", device_family="phone", capabilities=PHONE_CAPS,
        )


@pytest.mark.asyncio
async def test_a_stale_nonce_is_refused(registry, monkeypatch):
    import core.security.device_pairing as dp

    device = _add_device(registry)
    challenge = registry.begin_connect("dev1")
    signature = _sign(registry, device, challenge["nonce"])

    monkeypatch.setattr(dp.time, "time", lambda: challenge["expires_at"] + 1.0)
    with pytest.raises(PairingError, match="expired"):
        await registry.verify_connect(
            "dev1", nonce=challenge["nonce"], signature=signature,
            platform="ios", device_family="phone", capabilities=PHONE_CAPS,
        )


@pytest.mark.asyncio
async def test_a_wrong_signature_is_refused(registry):
    _add_device(registry)
    challenge = registry.begin_connect("dev1")
    with pytest.raises(PairingError, match="signature invalid"):
        await registry.verify_connect(
            "dev1", nonce=challenge["nonce"], signature="0" * 64,
            platform="ios", device_family="phone", capabilities=PHONE_CAPS,
        )


@pytest.mark.asyncio
async def test_the_secret_never_has_to_cross_the_wire(registry):
    """The signature is keyed by the stored digest, which the device can
    derive from its own secret. A passive observer on this plain-HTTP LAN
    sees a nonce and an HMAC over it, neither of which is reusable."""
    secret = "s3cret"
    device = _add_device(registry, secret=secret)
    challenge = registry.begin_connect("dev1")

    device_side_key = hashlib.sha256(secret.encode()).hexdigest()
    assert device_side_key == device.token_sha256

    signature = _sign(registry, device, challenge["nonce"])
    assert secret not in signature
    assert challenge["nonce"] not in signature
    await registry.verify_connect(
        "dev1", nonce=challenge["nonce"], signature=signature,
        platform="ios", device_family="phone", capabilities=PHONE_CAPS,
    )


# ── Pinned identity ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_stolen_token_replayed_from_another_device_fails(registry):
    """The attacker has the token digest and a valid nonce, but is not an
    iOS phone. Declaring the truth fails the pin; lying fails the
    signature, because the pinned values are what gets signed."""
    device = _add_device(registry)

    challenge = registry.begin_connect("dev1")
    with pytest.raises(PairingError, match="identity changed"):
        await registry.verify_connect(
            "dev1",
            nonce=challenge["nonce"],
            signature=_sign(registry, device, challenge["nonce"], platform="linux", family="server"),
            platform="linux",
            device_family="server",
            capabilities=PHONE_CAPS,
        )

    challenge = registry.begin_connect("dev1")
    with pytest.raises(PairingError, match="signature invalid"):
        await registry.verify_connect(
            "dev1",
            nonce=challenge["nonce"],
            signature=_sign(registry, device, challenge["nonce"], platform="linux", family="server"),
            platform="ios",
            device_family="phone",
            capabilities=PHONE_CAPS,
        )


@pytest.mark.asyncio
async def test_a_changed_manifest_forces_repairing(registry):
    device = _add_device(registry)
    grabby = PHONE_CAPS + ("screen.record",)
    challenge = registry.begin_connect("dev1")
    with pytest.raises(PairingError, match="capability manifest changed"):
        await registry.verify_connect(
            "dev1",
            nonce=challenge["nonce"],
            signature=_sign(registry, device, challenge["nonce"], caps=grabby),
            platform="ios",
            device_family="phone",
            capabilities=grabby,
        )


@pytest.mark.asyncio
async def test_a_device_paired_before_metadata_pins_on_first_connect(registry, monkeypatch):
    """Devices paired before this existed have nothing pinned. Locking
    them out would punish the owner for our schema change, so the first
    connect pins what it declares — and every connect after is enforced."""
    monkeypatch.setattr(registry, "_persist", _noop_persist)
    monkeypatch.setattr(registry, "_audit", _noop_audit)

    device = _add_device(registry, pinned=False)
    assert device.metadata_pinned is False

    challenge = registry.begin_connect("dev1")
    await registry.verify_connect(
        "dev1",
        nonce=challenge["nonce"],
        signature=_sign(registry, device, challenge["nonce"]),
        platform="ios",
        device_family="phone",
        capabilities=PHONE_CAPS,
    )
    assert device.metadata_pinned is True
    assert device.platform == "ios"

    challenge = registry.begin_connect("dev1")
    with pytest.raises(PairingError, match="identity changed"):
        await registry.verify_connect(
            "dev1",
            nonce=challenge["nonce"],
            signature=_sign(registry, device, challenge["nonce"], platform="linux", family="server"),
            platform="linux",
            device_family="server",
            capabilities=PHONE_CAPS,
        )


@pytest.mark.asyncio
async def test_a_failed_signature_does_not_pin_what_it_claimed(registry):
    """An unauthenticated caller must not be able to write the pin by
    declaring metadata and then failing the signature."""
    _add_device(registry, pinned=False)
    challenge = registry.begin_connect("dev1")
    with pytest.raises(PairingError, match="signature invalid"):
        await registry.verify_connect(
            "dev1", nonce=challenge["nonce"], signature="0" * 64,
            platform="linux", device_family="server", capabilities=("shell.run",),
        )
    assert registry.devices["dev1"].metadata_pinned is False


@pytest.mark.asyncio
async def test_a_revoked_device_cannot_connect(registry):
    device = _add_device(registry)
    device.revoked = True
    with pytest.raises(PairingError, match="Unknown device"):
        registry.begin_connect("dev1")


# ── The manifest as a ceiling ────────────────────────────────────────


def test_a_device_can_only_serve_what_it_declared(registry):
    _add_device(registry)
    assert registry.device_can_serve("dev1", "camera.capture") is True
    assert registry.device_can_serve("dev1", "screen.record") is False
    assert registry.device_can_serve("dev1", "shell.run") is False


def test_a_device_with_no_manifest_serves_nothing(registry):
    """Declaring nothing is not declaring everything."""
    _add_device(registry, pinned=False)
    assert registry.device_can_serve("dev1", "camera.capture") is False


def test_an_unknown_or_revoked_device_serves_nothing(registry):
    device = _add_device(registry)
    assert registry.device_can_serve("nope", "camera.capture") is False
    device.revoked = True
    assert registry.device_can_serve("dev1", "camera.capture") is False


def test_the_manifest_digest_is_a_property_of_the_set_not_the_order():
    assert _manifest_digest(tuple(sorted(("b.x", "a.y")))) == _manifest_digest(
        tuple(sorted(("a.y", "b.x")))
    )
    assert _manifest_digest(("a.y",)) != _manifest_digest(("a.y", "b.x"))


def test_signed_fields_cannot_be_shifted_across_the_separator():
    """Fields are joined with the unit separator, so a device must not be
    able to smuggle one in and move the boundary between two claims."""
    from core.security.device_pairing import _sanitize_identity_field

    assert "\x1f" not in _sanitize_identity_field("ios\x1fphone")


# ── Pairing pins the declaration ─────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_pairing_pins_the_declared_manifest(registry, monkeypatch):
    monkeypatch.setattr(registry, "_persist", _noop_persist)
    monkeypatch.setattr(registry, "_audit", _noop_audit)

    registry.begin_pairing("bryan")
    code = registry._challenge.code
    issued = await registry.complete_pairing(
        code,
        "Bryan's phone",
        platform="iOS",
        device_family="Phone",
        capabilities=["Camera.Capture", "notify", "notify"],
    )

    device = registry.devices[issued["device_id"]]
    assert device.platform == "ios"
    assert device.device_family == "phone"
    # Normalized, deduped, sorted.
    assert device.capabilities == ("camera.capture", "notify")
    assert issued["manifest_sha256"] == device.manifest_sha256


async def _noop_persist(*_a, **_kw):
    return None


async def _noop_audit(*_a, **_kw):
    return None


# ── Persistence ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pinned_metadata_survives_a_reload(registry, tmp_path, monkeypatch):
    """The pin is worthless if it evaporates on restart — the device would
    silently re-pin whatever it declared next."""
    import json

    _add_device(registry)
    snapshot = registry._snapshot_locked()
    path = tmp_path / "devices.json"
    path.write_text(json.dumps({"payload": snapshot}), encoding="utf-8")

    reloaded = DevicePairingRegistry.load(path)
    device = reloaded.devices["dev1"]
    assert device.platform == "ios"
    assert device.device_family == "phone"
    assert device.capabilities == tuple(sorted(PHONE_CAPS))
    assert device.metadata_pinned is True
    assert reloaded.device_can_serve("dev1", "camera.capture") is True
    assert reloaded.device_can_serve("dev1", "screen.record") is False


def test_a_registry_written_before_metadata_still_loads(tmp_path):
    """Older files have no platform/capabilities keys at all."""
    import json

    legacy = {
        "payload": {
            "schema_version": 2,
            "devices": [
                {
                    "device_id": "old1",
                    "name": "old phone",
                    "token_sha256": "a" * 64,
                    "scopes": ["conversation"],
                    "created_at": 1.0,
                    "last_seen": 2.0,
                    "principal_id": "bryan",
                    "revoked": False,
                }
            ],
        }
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    reloaded = DevicePairingRegistry.load(path)
    device = reloaded.devices["old1"]
    assert device.metadata_pinned is False
    assert device.capabilities == ()
