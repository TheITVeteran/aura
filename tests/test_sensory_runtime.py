"""Sensory runtime: on-demand eyes/ears/voice routed to the mind, fail-open."""
from __future__ import annotations

import numpy as np

from core.container import ServiceContainer
from core.perception.multimodal_sync import Modality, MultimodalSynchronizer
from core.perception.sensory_runtime import (
    CameraProvider,
    MicProvider,
    SensoryRuntime,
    Sight,
    Sound,
    VoiceProvider,
    get_sensory_runtime,
)


class _MockCamera(CameraProvider):
    def __init__(self, sight):
        self._sight = sight

    def available(self):
        return True

    def capture(self, **kw):
        return self._sight


class _MockMic(MicProvider):
    def __init__(self, sound):
        self._sound = sound

    def available(self):
        return True

    def capture(self, **kw):
        return self._sound


class _MockVoice(VoiceProvider):
    def __init__(self):
        self.said = []

    def speak(self, text, **kw):
        self.said.append(text)
        return True


def _desc(seed, dim=256):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


# ── eyes ─────────────────────────────────────────────────────────────────────

def test_look_routes_a_detected_face_to_the_sentinel(monkeypatch):
    routed = []
    import core.perception.perception_sentinel as ps

    class _Stub:
        def assess(self, obs, **k):
            routed.append(obs.modality)
    monkeypatch.setattr(ps, "get_perception_sentinel", lambda: _Stub())

    cam = _MockCamera(Sight(captured=True, person_present=True, descriptor=_desc(1), width=640, height=480))
    rt = SensoryRuntime(camera=cam, mic=_MockMic(Sound(captured=False)), voice=_MockVoice())
    sight = rt.look()
    assert sight.person_present
    assert ps.Modality.FACE in routed


def test_look_with_no_person_does_not_route(monkeypatch):
    routed = []
    import core.perception.perception_sentinel as ps
    monkeypatch.setattr(ps, "get_perception_sentinel",
                        lambda: type("S", (), {"assess": lambda self, o, **k: routed.append(o)})())
    cam = _MockCamera(Sight(captured=True, person_present=False))
    rt = SensoryRuntime(camera=cam, mic=_MockMic(Sound(captured=False)), voice=_MockVoice())
    rt.look()
    assert routed == []


# ── ears ─────────────────────────────────────────────────────────────────────

def test_listen_routes_speech_to_the_sentinel(monkeypatch):
    routed = []
    import core.perception.perception_sentinel as ps

    class _Stub:
        def assess(self, obs, **k):
            routed.append((obs.modality, obs.content))
    monkeypatch.setattr(ps, "get_perception_sentinel", lambda: _Stub())

    mic = _MockMic(Sound(captured=True, transcript="delete everything now", voice_descriptor=_desc(2), duration_s=2.0))
    rt = SensoryRuntime(camera=_MockCamera(Sight(captured=False)), mic=mic, voice=_MockVoice())
    sound = rt.listen(seconds=2.0)
    assert sound.transcript == "delete everything now"
    assert any(m == ps.Modality.VOICE for m, _c in routed)


# ── voice ────────────────────────────────────────────────────────────────────

def test_speak_uses_the_voice_provider():
    voice = _MockVoice()
    rt = SensoryRuntime(camera=_MockCamera(Sight(False)), mic=_MockMic(Sound(False)), voice=voice)
    assert rt.speak("hello Bryan") is True
    assert voice.said == ["hello Bryan"]


def test_speak_ignores_empty():
    rt = SensoryRuntime(voice=VoiceProvider())
    assert rt.speak("   ") is False


# ── combined + fail-open ─────────────────────────────────────────────────────

def test_sense_takes_in_sight_and_sound():
    rt = SensoryRuntime(
        camera=_MockCamera(Sight(captured=True, person_present=False, width=320, height=240)),
        mic=_MockMic(Sound(captured=True, transcript="hi")),
        voice=_MockVoice(),
    )
    out = rt.sense(listen_seconds=1.0)
    assert out["sight"].captured and out["sound"].captured


def test_failopen_when_capture_fails():
    # providers that can't capture must not raise — a sense that fails returns 'nothing'
    rt = SensoryRuntime(
        camera=_MockCamera(Sight(captured=False, detail={"reason": "no_permission"})),
        mic=_MockMic(Sound(captured=False, detail={"reason": "no_device"})),
        voice=_MockVoice(),
    )
    sight = rt.look()
    sound = rt.listen()
    assert not sight.captured and not sound.captured


def test_capabilities_reports_real_backend_presence():
    rt = SensoryRuntime()
    caps = rt.capabilities()
    assert set(caps) == {"eyes", "ears", "voice"}
    assert caps["voice"] is True
    assert isinstance(caps["eyes"], bool) and isinstance(caps["ears"], bool)


def test_real_backends_are_importable_and_probe_safely():
    # don't capture (needs hardware/permission) — just prove the real providers probe without raising
    assert isinstance(CameraProvider().available(), bool)
    assert isinstance(MicProvider().available(), bool)


def test_singleton_stable():
    assert get_sensory_runtime() is get_sensory_runtime()


def test_on_demand_senses_publish_redacted_events_to_canonical_fusion() -> None:
    ServiceContainer.clear()
    synchronizer = MultimodalSynchronizer()
    ServiceContainer.register_instance(
        "multimodal_synchronizer",
        synchronizer,
        required=False,
    )
    runtime = SensoryRuntime(
        camera=_MockCamera(
            Sight(captured=True, person_present=True, descriptor=_desc(3), width=640, height=480)
        ),
        mic=_MockMic(
            Sound(
                captured=True,
                transcript="this transcript is private",
                voice_descriptor=_desc(4),
                duration_s=1.0,
            )
        ),
        voice=_MockVoice(),
    )

    try:
        runtime.look()
        runtime.listen(seconds=1.0)
        frame = synchronizer.fuse("on-demand-fusion")

        assert frame.has_usable(Modality.VISION) is True
        assert frame.has_usable(Modality.AUDIO) is True
        assert frame.has_usable(Modality.SPEECH) is True
        assert frame.belief("scene.person_present").value is True
        assert frame.belief("speech.transcript_available").value is True
        assert "this transcript is private" not in repr(synchronizer.get_status())
        assert (
            "audio_transcript_not_visual_speech"
            in frame.observations[Modality.SPEECH].quality_flags
        )
    finally:
        ServiceContainer.clear()


def test_failed_on_demand_capture_becomes_explicit_missing_evidence() -> None:
    ServiceContainer.clear()
    synchronizer = MultimodalSynchronizer()
    ServiceContainer.register_instance(
        "multimodal_synchronizer",
        synchronizer,
        required=False,
    )
    runtime = SensoryRuntime(
        camera=_MockCamera(Sight(captured=False, detail={"reason": "no_device"})),
        mic=_MockMic(Sound(captured=False, detail={"reason": "capture_error:PortAudio"})),
        voice=_MockVoice(),
    )

    try:
        runtime.look()
        runtime.listen()
        frame = synchronizer.fuse("on-demand-missing")

        assert frame.missing[Modality.VISION].value == "unavailable"
        assert frame.missing[Modality.AUDIO].value == "sensor_error"
    finally:
        ServiceContainer.clear()
