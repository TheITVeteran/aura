from __future__ import annotations

from types import SimpleNamespace

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


@pytest.mark.asyncio
async def test_microphone_privacy_enable_fails_closed_on_device_error(monkeypatch) -> None:
    from interface.routes import privacy

    class Voice:
        microphone_enabled = False
        speaking_enabled = False
        _mic_listening = False

        def __init__(self):
            self.start_calls = 0

        async def start_listening(self) -> bool:
            self.start_calls += 1
            raise OSError("input device unavailable")

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

    assert voice.start_calls == 1
    assert result["ok"] is False
    assert result["enabled"] is False
    assert result["microphone_enabled"] is False
    assert result["speaking_enabled"] is False
    assert result["listening"] is False
    assert result["error"].startswith("OSError:")


def test_bootstrap_voice_summary_reports_real_listener_state() -> None:
    from interface.routes import privacy, system

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
                "capture_available": True,
                "stt_available": True,
                "stt_initialized": True,
                "capture_backend": "sounddevice",
                "stt_backend": "faster_whisper",
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
    assert summary["capture_available"] is True
    assert summary["stt_available"] is True
    assert summary["stt_initialized"] is True
    assert summary["capture_backend"] == "sounddevice"
    assert summary["stt_backend"] == "faster_whisper"
    assert summary["state"] == "listening"


def test_voice_engine_status_reports_missing_capture_backend(monkeypatch, tmp_path) -> None:
    import core.senses.voice_engine as voice_module

    monkeypatch.setattr(voice_module, "sd", None)
    engine = voice_module.SovereignVoiceEngine(data_dir=str(tmp_path))

    status = engine.get_status()

    assert status["server_capture"] is False
    assert status["capture_available"] is False
    assert status["capture_backend"] == "unavailable"


@pytest.mark.asyncio
async def test_voice_local_playback_uses_gateways(monkeypatch, tmp_path) -> None:
    import core.senses.voice_engine as voice_module

    writes: list[tuple[str, bytes, str]] = []
    spawns: list[tuple[tuple[str, ...], str]] = []

    class FileGateway:
        def write_bytes(self, path, payload, *, source):
            writes.append((str(path), payload, source))
            path.write_bytes(payload)

    class Proc:
        def __init__(self):
            self.done = False
            self.terminated = False

        def poll(self):
            return 0 if self.done else None

        def wait(self, timeout=None):
            self.done = True
            return 0

        def terminate(self):
            self.terminated = True
            self.done = True

    class SubprocessGateway:
        def spawn(self, argv, *, source, **_):
            spawns.append((tuple(argv), source))
            return Proc()

    monkeypatch.setattr(voice_module, "get_file_write_gateway", lambda: FileGateway())
    monkeypatch.setattr(voice_module, "get_subprocess_gateway", lambda: SubprocessGateway())

    engine = voice_module.SovereignVoiceEngine(data_dir=str(tmp_path))

    await engine._play_locally(b"RIFF-audio")

    assert writes == [
        (
            str(tmp_path / "tts_play_cache.wav"),
            b"RIFF-audio",
            "core.senses.voice_engine.play_locally",
        )
    ]
    assert spawns == [
        (
            ("afplay", str(tmp_path / "tts_play_cache.wav")),
            "core.senses.voice_engine.play_locally",
        )
    ]


def test_sensory_capabilities_require_capture_and_stt() -> None:
    from core.senses.sensory_registry import SensoryCapabilityFlags

    assert SensoryCapabilityFlags.from_boot_status(
        {"sounddevice": True, "faster_whisper": True}
    ).hearing_enabled is True
    assert SensoryCapabilityFlags.from_boot_status(
        {"sounddevice": True, "faster_whisper": False}
    ).hearing_enabled is False
    assert SensoryCapabilityFlags.from_boot_status(
        {"sounddevice": False, "faster_whisper": True}
    ).hearing_enabled is False


@pytest.mark.asyncio
async def test_voice_bridge_prefers_cognitive_engine_over_orchestrator(monkeypatch) -> None:
    from core.container import ServiceContainer
    from core.voice.voice_bridge import VoiceConversationBridge

    calls: list[dict[str, object]] = []

    class CognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            calls.append(
                {
                    "engine": "cognitive",
                    "objective": objective,
                    "context": dict(context or {}),
                    "origin": origin,
                    "foreground_request": kwargs.get("foreground_request"),
                }
            )
            return SimpleNamespace(content="Voice reply from CognitiveEngine.", mode=mode)

    class Orchestrator:
        async def process_user_input(self, *_args, **_kwargs):
            calls.append({"engine": "orchestrator"})
            raise AssertionError("voice should prefer CognitiveEngine when it is available")

    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(lambda name, default=None: CognitiveEngine() if name == "cognitive_engine" else default),
    )

    bridge = VoiceConversationBridge(Orchestrator(), None)
    response = await bridge.process_voice_input("Hey Aura, are you there?")

    assert response == "Voice reply from CognitiveEngine."
    assert calls == [
        {
            "engine": "cognitive",
            "objective": "are you there?",
            "context": {
                "route": "voice_desktop",
                "source": "voice",
                "origin": "voice",
                "foreground_request": True,
                "user_facing": True,
            },
            "origin": "voice",
            "foreground_request": True,
        }
    ]


@pytest.mark.asyncio
async def test_voice_bridge_executes_spoken_desktop_objective_after_cognition(monkeypatch) -> None:
    from core.container import ServiceContainer
    from core.voice.voice_bridge import VoiceConversationBridge

    calls: list[dict[str, object]] = []

    class CognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            calls.append({"engine": "cognitive", "objective": objective})
            return SimpleNamespace(
                content="Timestamped Aura summary from the cognitive engine.",
                mode=mode,
            )

    class CapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append(
                {
                    "engine": "capability",
                    "skill_name": skill_name,
                    "params": dict(params),
                    "context": dict(context or {}),
                }
            )
            return {
                "ok": True,
                "summary": "Desktop task completed 4/4 governed computer-use steps.",
                "steps_requested": 4,
                "steps_completed": 4,
            }

    def get_service(name, default=None):
        if name == "cognitive_engine":
            return CognitiveEngine()
        if name == "capability_engine":
            return CapabilityEngine()
        return default

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(get_service))

    bridge = VoiceConversationBridge(SimpleNamespace(), None)
    response = await bridge.process_voice_input(
        "Hey Aura, can you open Notes, write a timestamped summary, and save it as a PDF in a folder?"
    )

    assert response == "Desktop task completed 4/4 governed computer-use steps."
    assert calls[0] == {
        "engine": "cognitive",
        "objective": "can you open Notes, write a timestamped summary, and save it as a PDF in a folder?",
    }
    assert calls[1]["engine"] == "capability"
    assert calls[1]["skill_name"] == "desktop_task"
    assert calls[1]["params"] == {
        "objective": "can you open Notes, write a timestamped summary, and save it as a PDF in a folder?",
        "steps": [],
    }
    assert calls[1]["context"]["route"] == "voice.desktop_objective"
    assert calls[1]["context"]["origin"] == "voice"
    assert calls[1]["context"]["foreground_request"] is True
    assert calls[1]["context"]["desktop_task_document_body"] == "Timestamped Aura summary from the cognitive engine."


@pytest.mark.asyncio
async def test_voice_bridge_routes_generic_browser_document_objective(monkeypatch) -> None:
    from core.container import ServiceContainer
    from core.voice.voice_bridge import VoiceConversationBridge

    calls: list[dict[str, object]] = []

    class CognitiveEngine:
        async def think(self, objective, context=None, mode=None, origin=None, **kwargs):
            calls.append(
                {
                    "engine": "cognitive",
                    "objective": objective,
                    "context": dict(context or {}),
                    "origin": origin,
                    "foreground_request": kwargs.get("foreground_request"),
                }
            )
            return SimpleNamespace(
                content=(
                    "I will open the requested browser/document surface, draft the essay body, "
                    "and keep the work inside governed desktop control."
                ),
                mode=mode,
            )

    class CapabilityEngine:
        async def execute(self, skill_name, params, context=None):
            calls.append(
                {
                    "engine": "capability",
                    "skill_name": skill_name,
                    "params": dict(params),
                    "context": dict(context or {}),
                }
            )
            return {
                "ok": True,
                "summary": "Desktop task completed 3/3 governed computer-use steps.",
                "steps_requested": 3,
                "steps_completed": 3,
            }

    def get_service(name, default=None):
        if name == "cognitive_engine":
            return CognitiveEngine()
        if name == "capability_engine":
            return CapabilityEngine()
        return default

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(get_service))

    bridge = VoiceConversationBridge(SimpleNamespace(), None)
    response = await bridge.process_voice_input(
        "Hey Aura, open a tab for Google Docs and start typing a coherent essay about climate adaptation."
    )

    assert response == "Desktop task completed 3/3 governed computer-use steps."
    assert calls[0]["engine"] == "cognitive"
    assert calls[0]["objective"] == "open a tab for Google Docs and start typing a coherent essay about climate adaptation."
    assert calls[1]["engine"] == "capability"
    assert calls[1]["skill_name"] == "desktop_task"
    assert calls[1]["params"] == {
        "objective": "open a tab for Google Docs and start typing a coherent essay about climate adaptation.",
        "steps": [],
    }
    assert calls[1]["context"]["route"] == "voice.desktop_objective"
    assert calls[1]["context"]["desktop_task_document_body"].startswith("I will open the requested")


@pytest.mark.asyncio
async def test_voice_bridge_reports_capability_lookup_failure_without_legacy_claim(monkeypatch) -> None:
    from core.container import ServiceContainer
    from core.voice.voice_bridge import VoiceConversationBridge

    class CognitiveEngine:
        async def think(self, *_args, **_kwargs):
            return SimpleNamespace(content="Cognitive plan body.")

    def get_service(name, default=None):
        if name == "cognitive_engine":
            return CognitiveEngine()
        if name == "capability_engine":
            raise RuntimeError("container locked")
        return default

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(get_service))

    bridge = VoiceConversationBridge(SimpleNamespace(), None)
    response = await bridge.process_voice_input("Hey Aura, open Notes and save a PDF.")

    assert "governed desktop control" in response
    assert "capability_engine_unavailable" in response
    assert "did not complete" in response


@pytest.mark.asyncio
async def test_spoken_desktop_objective_requires_cognitive_engine(monkeypatch) -> None:
    from core.container import ServiceContainer
    from core.voice.voice_bridge import VoiceConversationBridge

    legacy_calls: list[str] = []

    class Orchestrator:
        async def process_user_input(self, *_args, **_kwargs):
            legacy_calls.append("legacy")
            return "legacy claimed it opened Notes"

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(lambda _name, default=None: default))

    bridge = VoiceConversationBridge(Orchestrator(), None)
    response = await bridge.process_voice_input("Hey Aura, open Notes and save a PDF.")

    assert "required CognitiveEngine" in response
    assert "legacy voice fallback" in response
    assert legacy_calls == []
