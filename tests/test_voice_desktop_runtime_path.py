from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest


def test_voice_stt_automatic_initialization_is_local_only(monkeypatch, tmp_path) -> None:
    import core.senses.voice_engine as voice_module

    constructor_calls: list[tuple[str, dict]] = []

    class FakeWhisperModel:
        def __init__(self, model_name: str, **kwargs) -> None:
            constructor_calls.append((model_name, dict(kwargs)))

    monkeypatch.delenv("AURA_STT_ALLOW_MODEL_DOWNLOAD", raising=False)
    monkeypatch.setattr(voice_module, "_get_whisper_model_class", lambda: FakeWhisperModel)
    monkeypatch.setattr(voice_module, "_runtime_shutdown_requested", lambda: False)
    engine = voice_module.SovereignVoiceEngine(data_dir=str(tmp_path))
    monkeypatch.setattr(engine, "_pulse_hypha", lambda *_args, **_kwargs: None)

    initialized = engine.ensure_stt()

    assert initialized is True
    assert constructor_calls == [
        (
            "base",
            {
                "device": "cpu",
                "compute_type": "int8",
                "local_files_only": True,
            },
        )
    ]
    assert engine.stt_model is not None
    assert engine.get_status()["stt_load_state"] == "ready"
    assert engine.get_status()["stt_local_files_only"] is True


@pytest.mark.asyncio
async def test_cancelled_stt_waiter_cannot_publish_model_after_voice_shutdown(
    monkeypatch,
    tmp_path,
) -> None:
    import core.senses.voice_engine as voice_module

    constructor_started = threading.Event()
    release_constructor = threading.Event()

    class BlockingWhisperModel:
        def __init__(self, _model_name: str, **_kwargs) -> None:
            constructor_started.set()
            release_constructor.wait(1.0)

    monkeypatch.delenv("AURA_STT_ALLOW_MODEL_DOWNLOAD", raising=False)
    monkeypatch.setattr(voice_module, "_get_whisper_model_class", lambda: BlockingWhisperModel)
    monkeypatch.setattr(voice_module, "_runtime_shutdown_requested", lambda: False)
    engine = voice_module.SovereignVoiceEngine(data_dir=str(tmp_path))
    monkeypatch.setattr(engine, "_pulse_hypha", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "_signal_mycelium", lambda *_args, **_kwargs: None)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(engine.ensure_stt_async(), timeout=0.02)
    assert await asyncio.to_thread(constructor_started.wait, 0.2)
    shared_task = engine._stt_init_task
    assert shared_task is not None and not shared_task.done()

    engine.on_stop()
    release_constructor.set()
    assert await asyncio.wait_for(asyncio.shield(shared_task), timeout=0.5) is False

    assert engine.stt_model is None
    assert engine._stt_initialized is False
    assert engine.get_status()["stt_load_state"] == "stopping"
    assert engine.get_status()["closing"] is True


@pytest.mark.asyncio
async def test_sovereign_ears_reports_listener_start_failure() -> None:
    from core.senses.ears import SovereignEars

    callbacks: list[str] = []

    class VoiceEngine:
        def on_transcript(self, _callback, *, key: str) -> None:
            callbacks.append(key)

        async def start_listening(self) -> bool:
            return False

    ears = object.__new__(SovereignEars)
    ears.capabilities = SimpleNamespace(hearing_enabled=True)
    ears._engine = VoiceEngine()

    started = await ears.start_listening(lambda _text: None)

    assert started is False
    assert callbacks == ["sovereign_ears"]


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
async def test_ambient_transcript_candidate_cannot_enter_full_cognitive_path(tmp_path) -> None:
    from core.senses.voice_engine import SovereignVoiceEngine

    engine = SovereignVoiceEngine(data_dir=str(tmp_path))
    calls: list[tuple[str, str]] = []

    async def full_cognitive_path(text: str) -> None:
        calls.append(("full_cognitive_path", text))

    def perceptual_listener(text: str) -> None:
        calls.append(("perception", text))

    engine.on_transcript(full_cognitive_path, key="sovereign_ears")
    engine.on_transcript(
        perceptual_listener,
        key="continuous_perception",
        candidate_safe=True,
    )

    await engine._handle_transcript(
        "ambient television dialogue without a wake word",
        authorized_command=False,
    )

    assert calls == [
        ("perception", "ambient television dialogue without a wake word")
    ]


def test_unauthorized_transcript_candidate_routes_to_sensory_gate(tmp_path, monkeypatch) -> None:
    from core.senses.voice_engine import SovereignVoiceEngine

    monkeypatch.delenv("AURA_VOICE_DIRECT_EVENTBUS", raising=False)
    monkeypatch.delenv("AURA_VOICE_ALWAYS_DIRECT_TO_CHAT", raising=False)

    engine = SovereignVoiceEngine(data_dir=str(tmp_path))
    signals: list[tuple[str, str, dict]] = []

    def capture_signal(source: str, target: str, payload: dict) -> None:
        signals.append((source, target, dict(payload)))

    monkeypatch.setattr(engine, "_signal_mycelium", capture_signal)
    monkeypatch.setattr(engine, "_pulse_hypha", lambda *_args, **_kwargs: None)

    engine._dispatch_transcript(
        "background video says something unrelated",
        source_assessment={"source": "device_media", "response_authorized": False},
    )

    assert signals
    assert signals[-1][0:2] == ("voice_engine", "sensory_gate")
    assert signals[-1][2]["event"] == "transcript_candidate"
    assert signals[-1][2]["authorized_command"] is False
    assert signals[-1][2]["conversation_context_eligible"] is False


def test_audio_attention_distinguishes_media_direct_address_and_nearby_speech() -> None:
    from core.senses.audio_attention import classify_audio_attention

    media = classify_audio_attention(
        "The documentary continues with a long historical explanation about migration patterns.",
        rms_db=-27.0,
        transcript_confidence=-0.30,
        duration_s=12.0,
        active_app="Google Chrome - YouTube",
    )
    addressed = classify_audio_attention(
        "Hey Aura, can you help me understand this?",
        rms_db=-18.0,
        transcript_confidence=-0.12,
        duration_s=3.0,
        active_app="Google Chrome",
    )
    nearby = classify_audio_attention(
        "That is an interesting idea about the design.",
        rms_db=-18.0,
        transcript_confidence=-0.20,
        duration_s=3.0,
        active_app="Finder",
    )

    assert media.source == "device_media"
    assert media.response_authorized is False
    assert media.attention_mode in {"ignore", "observe"}
    assert addressed.source == "direct_address"
    assert addressed.addressed_to_aura is True
    assert addressed.attention_mode == "conversation_candidate"
    assert addressed.response_authorized is False
    assert nearby.source == "nearby_person"
    assert nearby.response_authorized is False


def test_audio_attention_uses_fresh_camera_speaker_evidence() -> None:
    import time

    from core.senses.audio_attention import classify_audio_attention

    visible_speaker = classify_audio_attention(
        "That connection between memory and imagination is interesting.",
        rms_db=-19.0,
        transcript_confidence=-0.18,
        duration_s=4.0,
        active_app="Finder",
        visual_context={
            "updated_at": time.time(),
            "face_present": True,
            "face_count": 1,
            "speaking_likelihood": 0.72,
            "attention_available": 0.82,
        },
    )
    stale_camera = classify_audio_attention(
        "That connection between memory and imagination is interesting.",
        rms_db=-27.0,
        transcript_confidence=-0.18,
        duration_s=4.0,
        active_app="Finder",
        visual_context={
            "updated_at": time.time() - 30.0,
            "face_present": True,
            "speaking_likelihood": 0.95,
        },
    )

    assert visible_speaker.source == "nearby_visible_speaker"
    assert "lower_face_motion" in visible_speaker.reasons
    assert visible_speaker.response_authorized is False
    assert stale_camera.source != "nearby_visible_speaker"


def test_explicit_voice_capture_authorizes_response_without_source_guessing() -> None:
    from core.senses.audio_attention import classify_audio_attention

    assessment = classify_audio_attention(
        "Draft a short note from this dictation.",
        rms_db=-20.0,
        transcript_confidence=-0.20,
        duration_s=3.0,
        active_app="Notes",
        explicit_command=True,
    )

    assert assessment.source == "direct_user"
    assert assessment.response_authorized is True
    assert assessment.attention_mode == "conversation_candidate"


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


def test_voice_threshold_signal_only_on_change_or_heartbeat(monkeypatch, tmp_path) -> None:
    import core.senses.voice_engine as voice_module

    engine = voice_module.SovereignVoiceEngine(data_dir=str(tmp_path))
    calls: list[tuple[str, str, dict]] = []
    now = [1000.0]
    homeostasis = SimpleNamespace(
        get_modifiers=lambda: SimpleNamespace(overall_vitality=0.7)
    )

    monkeypatch.setattr(engine, "_get_homeostasis", lambda: homeostasis)
    monkeypatch.setattr(
        engine,
        "_signal_mycelium",
        lambda source, target, payload: calls.append((source, target, dict(payload))),
    )
    monkeypatch.setattr(voice_module.time, "time", lambda: now[0])

    engine._get_sensory_thresholds()
    engine._get_sensory_thresholds()
    now[0] += 31.0
    engine._get_sensory_thresholds()

    assert [call[2]["event"] for call in calls] == ["threshold_shift", "threshold_shift"]


@pytest.mark.asyncio
async def test_voice_local_playback_uses_gateways(monkeypatch, tmp_path) -> None:
    import core.senses.voice_engine as voice_module

    writes: list[tuple[str, bytes, str]] = []
    spawns: list[tuple[tuple[str, ...], str]] = []

    class FileGateway:
        def write_bytes(self, path, payload, *, source):
            writes.append((str(path), payload, source))
            path.write_bytes(payload)

        # Async lane delegators: production code now calls *_async; fakes
        # must mirror the gateway surface or every governed write breaks.
        async def write_bytes_async(self, *args, **kwargs):
            return self.write_bytes(*args, **kwargs)

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
            calls.append(
                {
                    "engine": "cognitive",
                    "objective": objective,
                    "context": dict(context or {}),
                    "mode": mode,
                }
            )
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
    assert calls[0]["engine"] == "cognitive"
    assert calls[0]["objective"] == (
        "can you open Notes, write a timestamped summary, and save it as a PDF in a folder?"
    )
    assert calls[0]["context"]["desktop_execution_contract"] is True
    assert calls[0]["context"]["allow_heuristic_desktop_plan"] is True
    assert calls[0]["context"]["desktop_task_allowed_actions"]
    assert calls[0]["context"]["desktop_task_planning_schema"]["steps"]
    assert calls[0]["mode"].name == "SLOW"
    assert calls[1]["engine"] == "capability"
    assert calls[1]["skill_name"] == "desktop_task"
    assert calls[1]["params"] == {
        "objective": "can you open Notes, write a timestamped summary, and save it as a PDF in a folder?",
        "steps": [],
        "disable_outer_skill_retry": True,
        "desktop_execution_contract": True,
        "allow_heuristic_desktop_plan": True,
    }
    assert calls[1]["context"]["route"] == "voice.desktop_objective"
    assert calls[1]["context"]["origin"] == "voice"
    assert calls[1]["context"]["foreground_request"] is True
    assert calls[1]["context"]["desktop_execution_contract"] is True
    assert calls[1]["context"]["allow_heuristic_desktop_plan"] is True
    assert calls[1]["context"]["verification_required"] is True
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
    assert calls[0]["context"]["desktop_execution_contract"] is True
    assert calls[0]["context"]["allow_heuristic_desktop_plan"] is True
    assert calls[0]["context"]["max_tokens"] == 1024
    assert calls[1]["engine"] == "capability"
    assert calls[1]["skill_name"] == "desktop_task"
    assert calls[1]["params"] == {
        "objective": "open a tab for Google Docs and start typing a coherent essay about climate adaptation.",
        "steps": [],
        "disable_outer_skill_retry": True,
        "desktop_execution_contract": True,
        "allow_heuristic_desktop_plan": True,
    }
    assert calls[1]["context"]["route"] == "voice.desktop_objective"
    assert calls[1]["context"]["allow_heuristic_desktop_plan"] is True
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
