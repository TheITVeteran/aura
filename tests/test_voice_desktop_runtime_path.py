from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_voice_transcript_callbacks_fan_out_without_clobbering(tmp_path) -> None:
    from core.senses.voice_engine import SovereignVoiceEngine

    engine = SovereignVoiceEngine(data_dir=str(tmp_path))
    calls: list[tuple[str, str]] = []

    async def full_cognitive_path(text: str) -> None:
        calls.append(("full_cognitive_path", text))

    def wake_word_listener(text: str) -> None:
        calls.append(("wake_word_listener", text))

    engine.on_transcript(full_cognitive_path, key="sovereign_ears")
    engine.on_transcript(wake_word_listener, key="continuous_perception")

    await engine._handle_transcript("Hey Aura, can you look this up?")

    assert calls == [
        ("full_cognitive_path", "Hey Aura, can you look this up?"),
        ("wake_word_listener", "Hey Aura, can you look this up?"),
    ]


@pytest.mark.asyncio
async def test_microphone_privacy_enable_starts_live_listener(monkeypatch) -> None:
    from interface.routes import privacy

    class Voice:
        microphone_enabled = False
        speaking_enabled = False
        _mic_listening = False

        async def start_listening(self) -> bool:
            self._mic_listening = True
            return True

        def stop_listening(self) -> None:
            self._mic_listening = False

    voice = Voice()
    original = privacy.get_voice_engine_fn()
    privacy.set_voice_engine_fn(lambda: voice)
    try:
        result = await privacy.api_privacy_microphone(
            privacy.PrivacyPayload(enabled=True),
            None,
        )
    finally:
        privacy.set_voice_engine_fn(original)

    assert result["ok"] is True
    assert result["enabled"] is True
    assert result["microphone_enabled"] is True
    assert result["speaking_enabled"] is True
    assert result["listening"] is True
    assert result["listening_started"] is True


@pytest.mark.asyncio
async def test_microphone_privacy_enable_fails_closed_when_listener_will_not_start(monkeypatch) -> None:
    from interface.routes import privacy

    class Voice:
        microphone_enabled = False
        speaking_enabled = False
        _mic_listening = False

        async def start_listening(self) -> bool:
            return False

        def stop_listening(self) -> None:
            self._mic_listening = False

    voice = Voice()
    original = privacy.get_voice_engine_fn()
    privacy.set_voice_engine_fn(lambda: voice)
    try:
        result = await privacy.api_privacy_microphone(
            privacy.PrivacyPayload(enabled=True),
            None,
        )
    finally:
        privacy.set_voice_engine_fn(original)

    assert result["ok"] is False
    assert result["enabled"] is False
    assert result["microphone_enabled"] is False
    assert result["speaking_enabled"] is False
    assert result["listening"] is False
    assert result["error"] == "microphone_start_failed"


def test_bootstrap_voice_summary_reports_real_listener_state() -> None:
    from interface.routes import privacy
    from interface.routes import system

    class State:
        name = "LISTENING"

    class Voice:
        microphone_enabled = True
        speaking_enabled = True
        _mic_listening = True
        state = State()

        def get_status(self) -> dict[str, object]:
            return {
                "auto_listen": True,
                "server_capture": True,
                "stt": "Whisper (Direct)",
                "tts": "pyttsx3 (Native)",
            }

    original = privacy.get_voice_engine_fn()
    privacy.set_voice_engine_fn(lambda: Voice())
    try:
        summary = system._collect_voice_summary()
    finally:
        privacy.set_voice_engine_fn(original)

    assert summary["available"] is True
    assert summary["microphone_enabled"] is True
    assert summary["speaking_enabled"] is True
    assert summary["listening"] is True
    assert summary["auto_listen"] is True
    assert summary["server_capture"] is True
    assert summary["state"] == "listening"
