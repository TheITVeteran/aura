"""One global microphone switch, one physical owner, one admitted ingress."""

from __future__ import annotations

import asyncio
import time

from core.voice import microphone_authority as microphone_module
from core.voice.microphone_authority import (
    STALE_MICROPHONE_LEASE_S,
    AudioIngressBroker,
    MicrophoneAuthority,
    MicrophoneDenial,
    MicrophoneLease,
)


def _authority(monkeypatch, *, allowed: bool = True) -> MicrophoneAuthority:
    monkeypatch.setattr(microphone_module, "microphone_allowed", lambda: allowed)
    return MicrophoneAuthority()


def test_owner_switch_is_checked_before_any_capture_owner_is_admitted(monkeypatch):
    authority = _authority(monkeypatch, allowed=False)

    result = authority.acquire(
        "browser",
        principal="owner:local",
        source="browser_duplex",
        mode="focused",
    )

    assert isinstance(result, MicrophoneDenial)
    assert result.reason == "owner_disabled"
    assert authority.state()["lease_active"] is False


def test_host_microphone_has_one_owner(monkeypatch):
    authority = _authority(monkeypatch)
    first = authority.acquire(
        "resident",
        principal="owner:local",
        source="sounddevice",
        mode="ambient",
    )
    second = authority.acquire(
        "other",
        principal="owner:local",
        source="browser_duplex",
        mode="ambient",
    )

    assert isinstance(first, MicrophoneLease)
    assert isinstance(second, MicrophoneDenial)
    assert second.reason == "device_busy"


def test_focused_conversation_preempts_passive_capture(monkeypatch):
    authority = _authority(monkeypatch)
    revoked: list[str] = []
    passive = authority.acquire(
        "resident",
        principal="owner:local",
        source="sounddevice",
        mode="passive",
        revoke_callback=revoked.append,
    )

    focused = authority.acquire(
        "browser",
        principal="owner:local",
        source="browser_duplex",
        mode="focused",
    )

    assert isinstance(passive, MicrophoneLease)
    assert isinstance(focused, MicrophoneLease)
    assert passive.active is False
    assert revoked == ["preempted_by:browser:focused"]
    assert authority.state()["preemptions"] == 1


def test_remote_microphone_does_not_contend_with_host_microphone(monkeypatch):
    authority = _authority(monkeypatch)
    host = authority.acquire(
        "desktop",
        principal="owner:local",
        source="browser_duplex",
        mode="focused",
    )
    remote = authority.acquire(
        "phone",
        principal="paired:device-1",
        source="browser_duplex",
        mode="focused",
    )

    assert isinstance(host, MicrophoneLease)
    assert isinstance(remote, MicrophoneLease)
    assert host.group != remote.group
    assert len(authority.state()["holders"]) == 2


def test_stale_holder_is_reclaimed(monkeypatch):
    authority = _authority(monkeypatch)
    first = authority.acquire(
        "dead",
        principal="owner:local",
        source="sounddevice",
        mode="focused",
    )
    assert isinstance(first, MicrophoneLease)
    first.last_seen = time.monotonic() - STALE_MICROPHONE_LEASE_S - 1

    second = authority.acquire(
        "replacement",
        principal="owner:local",
        source="sounddevice",
        mode="focused",
    )

    assert isinstance(second, MicrophoneLease)
    assert first.revoked_reason == "stale_holder_reclaimed"


def test_global_revocation_invalidates_every_transport(monkeypatch):
    authority = _authority(monkeypatch)
    host = authority.acquire(
        "desktop",
        principal="owner:local",
        source="browser_duplex",
        mode="focused",
    )
    remote = authority.acquire(
        "phone",
        principal="paired:device-1",
        source="browser_duplex",
        mode="focused",
    )

    receipt = authority.revoke_all(reason="runtime_setting_disabled")

    assert receipt["revoked"] == 2
    assert isinstance(host, MicrophoneLease) and not host.active
    assert isinstance(remote, MicrophoneLease) and not remote.active
    assert authority.state()["lease_active"] is False


def test_ingress_broker_rejects_audio_without_current_lease(monkeypatch):
    authority = _authority(monkeypatch)
    broker = AudioIngressBroker(authority)
    lease = authority.acquire(
        "desktop",
        principal="owner:local",
        source="browser_duplex",
        mode="focused",
    )
    assert isinstance(lease, MicrophoneLease)

    assert broker.admit(lease, 640) is True
    authority.release(lease)
    assert broker.admit(lease, 640) is False
    status = broker.state()
    assert status["frames_by_lease"][lease.lease_id] == 1
    assert status["bytes_by_lease"][lease.lease_id] == 640
    assert status["rejected_frames"] == 1


def test_native_voice_engine_never_touches_sounddevice_when_lease_is_denied(
    monkeypatch,
    tmp_path,
):
    from core.senses import voice_engine as voice_module

    opened: list[bool] = []

    class SoundDevice:
        def InputStream(self, **_kwargs):  # noqa: N802 - sounddevice API
            opened.append(True)
            raise AssertionError("sounddevice opened without microphone authority")

    class Authority:
        def acquire(self, *_args, **_kwargs):
            return MicrophoneDenial("device_busy", "focused voice owns it", "wait")

    monkeypatch.setattr(voice_module, "sd", SoundDevice())
    monkeypatch.setattr(voice_module, "get_microphone_authority", lambda: Authority())
    engine = voice_module.SovereignVoiceEngine(data_dir=str(tmp_path / "voice"))
    engine._stt_initialized = True

    assert asyncio.run(engine.start_listening()) is False
    assert opened == []


def test_native_audio_enters_asr_only_through_the_ingress_broker(
    monkeypatch,
    tmp_path,
):
    from core.senses import voice_engine as voice_module

    authority = _authority(monkeypatch)
    broker = AudioIngressBroker(authority)
    monkeypatch.setattr(voice_module, "get_microphone_authority", lambda: authority)
    monkeypatch.setattr(voice_module, "get_audio_ingress_broker", lambda: broker)

    engine = voice_module.SovereignVoiceEngine(data_dir=str(tmp_path / "voice"))
    engine.microphone_enabled = True
    lease = authority.acquire(
        engine._voice_owner_generation,
        principal="owner:local",
        source="sounddevice",
        mode="passive",
    )
    assert isinstance(lease, MicrophoneLease)
    engine._mic_lease = lease
    engine._mic_listening = True

    engine._mic_callback(b"\x01\x02" * 320, 320, None, None)

    assert engine._audio_buffer.get_nowait() == b"\x01\x02" * 320
    assert broker.state()["frames_by_lease"][lease.lease_id] == 1

    authority.release(lease)
    engine._mic_callback(b"\x03\x04" * 320, 320, None, None)
    assert engine._audio_buffer.empty()
