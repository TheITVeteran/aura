from __future__ import annotations

import asyncio
import base64

import pytest

from core.container import ServiceContainer
from core.perception.multimodal_sync import Modality, MultimodalSynchronizer
from core.perception.sensory_integration import (
    HearingSystem,
    SensoryModality,
    SensorySystem,
    VisionSystem,
)


@pytest.mark.asyncio
async def test_in_memory_camera_data_reaches_primary_vision_lane() -> None:
    captured: dict[str, object] = {}

    class Brain:
        async def think(self, prompt, **kwargs):
            captured["prompt"] = prompt
            captured["images"] = kwargs.get("images")
            return "A person is standing beside a whiteboard."

    ServiceContainer.clear()
    ServiceContainer.register_instance("cognitive_engine", Brain(), required=False)
    encoded = base64.b64encode(b"jpeg-bytes").decode("ascii")

    try:
        result = await VisionSystem().analyze({"type": "image", "data": encoded})

        assert result["scene_description"] == "A person is standing beside a whiteboard."
        assert captured["images"] == [encoded]
    finally:
        ServiceContainer.clear()


@pytest.mark.asyncio
async def test_sensory_memory_drops_raw_base64_and_publishes_redacted_fusion() -> None:
    raw_image = base64.b64encode(b"private-camera-frame" * 100).decode("ascii")

    class Vision:
        async def capture(self, **_kwargs):
            return {"type": "image", "data": raw_image, "timestamp": 1_700_000_000.0}

        async def analyze(self, _capture):
            return {
                "scene_description": "Bryan at a private whiteboard",
                "faces_detected": 1,
                "objects_detected": ["whiteboard"],
            }

    ServiceContainer.clear()
    synchronizer = MultimodalSynchronizer()
    ServiceContainer.register_instance(
        "multimodal_synchronizer",
        synchronizer,
        required=False,
    )
    system = SensorySystem()
    system.vision = Vision()

    try:
        result = await system.perceive(SensoryModality.VISION)
        stored = system.get_recent_perceptions(count=1)[0]
        frame = synchronizer.fuse("sensory-system-vision")

        assert result["data"]["data"] == raw_image
        assert "data" not in stored["data"]
        assert stored["data"]["raw_retained"] is False
        assert stored["data"]["encoded_chars"] == len(raw_image)
        assert stored["privacy"]["raw_media_retained"] is False
        assert frame.has_usable(Modality.VISION) is True
        assert frame.belief("scene.face_count").value == 1
        assert raw_image not in repr(system.sensory_memory)
        assert "Bryan at a private whiteboard" not in repr(synchronizer.get_status())
    finally:
        ServiceContainer.clear()


@pytest.mark.asyncio
async def test_same_modality_captures_are_serialized_and_memory_is_copy_out() -> None:
    class Vision:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        async def capture(self, **_kwargs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            return {"type": "image", "data": "cHJpdmF0ZQ=="}

        async def analyze(self, _capture):
            await asyncio.sleep(0.01)
            self.active -= 1
            return {"scene_description": "bounded", "faces_detected": 0}

    system = SensorySystem()
    vision = Vision()
    system.vision = vision

    await asyncio.gather(
        system.perceive(SensoryModality.VISION),
        system.perceive(SensoryModality.VISION),
    )

    assert vision.max_active == 1
    exported = system.get_recent_perceptions(count=2)
    exported[0]["modality"] = "mutated"
    assert system.get_recent_perceptions(count=2)[0]["modality"] == "vision"


@pytest.mark.asyncio
async def test_audio_and_text_publish_without_raw_content_in_fusion() -> None:
    class Hearing:
        async def listen(self, **_kwargs):
            return {"type": "audio", "path": "/private/recording.wav", "duration": 1.5}

        async def transcribe(self, _audio):
            return {"text": "private spoken instruction", "confidence": 0.91}

    ServiceContainer.clear()
    synchronizer = MultimodalSynchronizer()
    ServiceContainer.register_instance(
        "multimodal_synchronizer",
        synchronizer,
        required=False,
    )
    system = SensorySystem()
    system.hearing = Hearing()

    try:
        await system.perceive(SensoryModality.HEARING)
        await system.perceive(SensoryModality.TEXT, text="private typed instruction")
        frame = synchronizer.fuse("sensory-system-audio-text")

        assert frame.has_usable(Modality.AUDIO) is True
        assert frame.has_usable(Modality.SPEECH) is True
        assert frame.has_usable(Modality.TEXT) is True
        assert frame.belief("speech.transcript_available").value is True
        status_text = repr(synchronizer.get_status())
        assert "private spoken instruction" not in status_text
        assert "private typed instruction" not in status_text
    finally:
        ServiceContainer.clear()


@pytest.mark.asyncio
async def test_hearing_transcription_uses_canonical_local_stt(monkeypatch) -> None:
    class Segment:
        def __init__(self, text: str) -> None:
            self.text = text

    class Info:
        language = "en"
        language_probability = 0.93

    class Model:
        def transcribe(self, path, *, beam_size):
            assert path == "/tmp/aura-audio.wav"
            assert beam_size == 5
            return [Segment(" Aura hears"), Segment(" locally. ")], Info()

    monkeypatch.setattr(
        "core.senses.voice_socket_logic.get_whisper_model",
        lambda _name: Model(),
    )

    result = await HearingSystem().transcribe({"path": "/tmp/aura-audio.wav"})

    assert result["text"] == "Aura hears locally."
    assert result["language"] == "en"
    assert result["confidence"] is None
    assert result["language_confidence"] == 0.93
    assert result["backend"] == "canonical_whisper"
    assert result["disposition"] == "transcribed"
    assert "error" not in result


@pytest.mark.asyncio
async def test_hearing_transcription_fails_truthfully_without_local_stt(monkeypatch) -> None:
    monkeypatch.setattr(
        "core.senses.voice_socket_logic.get_whisper_model",
        lambda _name: None,
    )

    result = await HearingSystem().transcribe({"path": "/tmp/aura-audio.wav"})

    assert result["text"] == ""
    assert result["error"] == "canonical_stt_unavailable"
    assert result["disposition"] == "unavailable"
    assert result["backend"] == "canonical_whisper"
