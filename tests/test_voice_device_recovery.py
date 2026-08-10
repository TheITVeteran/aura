"""Voice-device changes preserve one authenticated conversation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from core.voice.duplex.config import VAD_FRAME_SAMPLES
from core.voice.duplex.session import DuplexVoiceSession, SessionState
from core.voice.microphone_authority import MicrophoneLease


class _Mind:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def publish(self, event: str, payload: dict) -> None:
        self.events.append((event, payload))


def test_device_discontinuity_discards_partial_audio_and_is_monotonic() -> None:
    sent: list[dict] = []
    mind = _Mind()

    async def send_json(payload: dict) -> None:
        sent.append(payload)

    async def send_binary(_payload: bytes) -> None:
        return None

    async def exercise() -> None:
        session = DuplexVoiceSession(
            session_id="device-recovery",
            send_json=send_json,
            send_binary=send_binary,
            mind=mind,  # type: ignore[arg-type]
        )
        frame = np.ones(VAD_FRAME_SAMPLES, dtype=np.float32)
        session._splitter.push(frame[:100])
        session._utterance.begin()
        session._utterance.append(frame)
        session._stable_text = "a clipped thought"
        session._state = SessionState.USER_SPEAKING

        await session.handle_command(
            {
                "command": "device_state",
                "state": "recovering",
                "reason": "track_ended",
                "generation": 4,
            }
        )

        assert session.state is SessionState.LISTENING
        assert session._splitter.pending_samples == 0
        assert session._utterance.sample_count == 0
        assert session._stable_text == ""
        assert session.status()["device"]["capture_available"] is False

        await session.handle_command(
            {
                "command": "device_state",
                "state": "active",
                "reason": "stale",
                "generation": 3,
            }
        )
        assert session.status()["device"]["state"] == "recovering"

        await session.handle_command(
            {
                "command": "device_state",
                "state": "active",
                "reason": "recovered:track_ended",
                "generation": 5,
            }
        )
        device = session.status()["device"]
        assert device["state"] == "active"
        assert device["generation"] == 5
        assert device["capture_available"] is True
        assert [event for event, _payload in mind.events] == [
            "voice_device_state",
            "voice_device_state",
        ]
        assert [payload["state"] for payload in sent] == ["listening", "recovering", "active"]

    asyncio.run(exercise())


def test_browser_reopens_only_capture_inside_the_same_authoritative_socket() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "interface/static/voice_mode.js"
    ).read_text(encoding="utf-8")
    recovery = source[source.index("async function recoverCapture") :]
    recovery = recovery[: recovery.index("async function handleMediaDeviceChange")]
    capture = source[source.index("async function startCapture") :]
    capture = capture[: capture.index("async function stopCapture")]

    assert "sendDeviceState('recovering'" in recovery
    assert "startCapture({ reason: `recovered:${reason}` })" in recovery
    assert "connect()" not in recovery
    assert "state.ws.readyState !== WebSocket.OPEN" in recovery
    assert "track.addEventListener('ended'" in capture
    assert "track.addEventListener('mute'" in capture
    assert "stream.getTracks().forEach((track) => track.stop())" in capture
    assert "navigator.mediaDevices.addEventListener('devicechange'" in source


def test_server_status_exposes_capture_generation_not_just_socket_state() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "core/voice/duplex/session.py"
    ).read_text(encoding="utf-8")
    assert '"device": {' in source
    assert '"capture_available": self._capture_available' in source
    assert "generation < self._device_generation" in source


def test_native_monitor_detects_a_dead_stream_without_waiting_for_a_callback(
    monkeypatch,
    tmp_path,
) -> None:
    from core.senses import voice_engine as voice_module

    async def exercise() -> None:
        engine = voice_module.SovereignVoiceEngine(data_dir=str(tmp_path / "voice"))
        lease = MicrophoneLease(
            lease_id="native-device",
            holder="resident",
            principal="owner:local",
            source="sounddevice",
            group="host_microphone",
            mode="passive",
            session_id="",
            generation=1,
            acquired_at=0.0,
            last_seen=0.0,
            preemptible=True,
        )
        engine._mic_listening = True
        engine._mic_lease = lease
        reasons: list[str] = []
        monkeypatch.setattr(
            voice_module,
            "get_microphone_authority",
            lambda: type("Authority", (), {"validate": lambda _self, _lease: (True, "active")})(),
        )
        monkeypatch.setattr(engine, "_schedule_microphone_recovery", reasons.append)

        await engine._monitor_microphone_stream(
            lease,
            type("Stream", (), {"active": False})(),
        )

        assert reasons == ["stream_inactive"]

    asyncio.run(exercise())


def test_native_recovery_retries_without_reloading_voice_or_mind_state(
    monkeypatch,
    tmp_path,
) -> None:
    from core.senses import voice_engine as voice_module

    async def exercise() -> None:
        engine = voice_module.SovereignVoiceEngine(data_dir=str(tmp_path / "voice"))
        engine._mic_capture_requested = True
        engine.microphone_enabled = True
        outcomes = iter((False, True))
        starts = 0
        stops: list[bool] = []

        async def start() -> bool:
            nonlocal starts
            starts += 1
            return next(outcomes)

        def stop(*, preserve_request: bool = False) -> bool:
            stops.append(preserve_request)
            return True

        monkeypatch.setattr(engine, "start_listening", start)
        monkeypatch.setattr(engine, "stop_listening", stop)
        monkeypatch.setattr(engine, "_voice_closing", lambda: False)

        recovered = await engine._recover_microphone("device_removed")

        assert recovered is True
        assert starts == 2
        assert stops == [True]
        assert engine._mic_device_state == "active"
        assert engine._mic_device_reason == "recovered:device_removed"

    asyncio.run(exercise())
