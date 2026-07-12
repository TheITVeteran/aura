from __future__ import annotations

import asyncio
from dataclasses import replace

import numpy as np
import pytest

from core.container import ServiceContainer
from core.perception.multimodal_sync import MissingReason, Modality, MultimodalSynchronizer
from core.perception.visual_speech import (
    AudioActivitySample,
    BackendPrediction,
    VisualSpeechConsent,
    VisualSpeechEngine,
    VisualSpeechEvidence,
    VisualSpeechStatus,
)

NOW = 1_700_000_000.0


def _consent(**changes) -> VisualSpeechConsent:
    values = {
        "consent_id": "consent-visual-1",
        "subject_id": "operator-bryan",
        "purpose": "consented visual-only speech recognition test",
        "issued_at": NOW - 60.0,
        "expires_at": NOW + 600.0,
        "allow_visual_speech": True,
        "allow_audio_alignment": False,
        "allow_raw_retention": False,
    }
    values.update(changes)
    return VisualSpeechConsent(**values)


def _evidence(*, brightness: float = 100.0, competing: float = 0.1) -> VisualSpeechEvidence:
    rng = np.random.default_rng(7)
    frame_count = 40
    crops = rng.integers(0, 255, size=(frame_count, 96, 96, 3), dtype=np.uint8)
    activity = tuple(float(value) for value in rng.random(frame_count))
    timestamps = tuple(index / 25.0 for index in range(frame_count))
    return VisualSpeechEvidence(
        source_digest="video-sha256-test",
        mouth_crops=crops,
        timestamps_s=timestamps,
        mouth_activity=activity,
        source_fps=30.0,
        sampled_fps=25.0,
        duration_s=1.56,
        decoded_frames=40,
        mouth_frames=40,
        face_detection_coverage=1.0,
        mouth_landmark_coverage=1.0,
        mean_brightness=brightness,
        mean_blur_variance=120.0,
        mean_mouth_motion=6.0,
        competing_face_ratio=competing,
        ambiguous_face_frames=0,
        track_switches=0,
        speaker_track_id="visual-track-1",
        source_audio_present=True,
        source_audio_presence_known=True,
        extractor="unit-native-vision",
        quality_flags=("native_face_landmarks",),
    )


class Extractor:
    def __init__(self, evidence: VisualSpeechEvidence) -> None:
        self.evidence = evidence
        self.calls = 0

    def extract(self, _video_path, _policy):
        self.calls += 1
        return self.evidence


class Backend:
    def __init__(
        self,
        *,
        transcript: str = "read the visual sentence",
        confidence: float | None = 0.95,
        calibrated: bool = True,
        available: bool = True,
    ) -> None:
        self.transcript = transcript
        self.confidence = confidence
        self.calibrated = calibrated
        self.is_available = available
        self.calls = 0
        self.saw_nonzero = False

    def available(self) -> tuple[bool, str]:
        return self.is_available, "ready" if self.is_available else "checkpoint_missing"

    async def infer(self, mouth_crops, *, fps):
        self.calls += 1
        self.saw_nonzero = bool(np.any(mouth_crops))
        assert fps == 25.0
        return BackendPrediction(
            transcript=self.transcript,
            confidence=self.confidence,
            calibrated=self.calibrated,
            backend="unit-visual-only",
            model_id="unit-vsr-1",
        )


@pytest.mark.asyncio
async def test_expired_consent_denies_before_video_or_model_access(tmp_path) -> None:
    evidence = _evidence()
    extractor = Extractor(evidence)
    backend = Backend()
    sync = MultimodalSynchronizer()
    ServiceContainer.clear()
    ServiceContainer.register_instance("multimodal_synchronizer", sync, required=False)
    source = tmp_path / "private.mp4"
    source.write_bytes(b"video")
    engine = VisualSpeechEngine(
        extractor=extractor,
        backend=backend,
        wall_clock=lambda: NOW,
    )

    try:
        result = await engine.transcribe_video(
            source,
            consent=_consent(expires_at=NOW - 1.0),
        )
        frame = sync.fuse("expired-consent")

        assert result.status is VisualSpeechStatus.DENIED
        assert result.reason == "consent_expired"
        assert extractor.calls == 0
        assert backend.calls == 0
        assert frame.missing[Modality.SPEECH] is MissingReason.PERMISSION_DENIED
    finally:
        ServiceContainer.clear()


@pytest.mark.asyncio
async def test_audio_alignment_requires_separate_consent(tmp_path) -> None:
    evidence = _evidence()
    extractor = Extractor(evidence)
    backend = Backend()
    source = tmp_path / "private.mp4"
    source.write_bytes(b"video")
    engine = VisualSpeechEngine(
        extractor=extractor,
        backend=backend,
        wall_clock=lambda: NOW,
    )

    result = await engine.transcribe_video(
        source,
        consent=_consent(),
        audio_activity=[AudioActivitySample(0.0, 0.5)] * 5,
    )

    assert result.status is VisualSpeechStatus.DENIED
    assert result.reason == "audio_alignment_not_consented"
    assert extractor.calls == 0
    assert backend.calls == 0


@pytest.mark.asyncio
async def test_poor_light_and_speaker_ambiguity_abstain_before_decoder(tmp_path) -> None:
    evidence = _evidence(brightness=5.0, competing=0.9)
    extractor = Extractor(evidence)
    backend = Backend()
    source = tmp_path / "dark.mp4"
    source.write_bytes(b"video")
    engine = VisualSpeechEngine(
        extractor=extractor,
        backend=backend,
        wall_clock=lambda: NOW,
    )

    result = await engine.transcribe_video(source, consent=_consent())

    assert result.status is VisualSpeechStatus.ABSTAINED
    assert "poor_lighting" in result.reason
    assert "speaker_face_ambiguous" in result.reason
    assert backend.calls == 0
    assert np.count_nonzero(evidence.mouth_crops) == 0


@pytest.mark.asyncio
async def test_calibrated_visual_only_result_is_actionable_and_causal(tmp_path) -> None:
    evidence = _evidence()
    extractor = Extractor(evidence)
    backend = Backend()
    source = tmp_path / "speaker.mp4"
    source.write_bytes(b"video")
    sync = MultimodalSynchronizer()
    ServiceContainer.clear()
    ServiceContainer.register_instance("multimodal_synchronizer", sync, required=False)
    engine = VisualSpeechEngine(
        extractor=extractor,
        backend=backend,
        wall_clock=lambda: NOW,
    )
    audio = [
        AudioActivitySample(timestamp, activity)
        for timestamp, activity in zip(
            evidence.timestamps_s,
            evidence.mouth_activity,
            strict=True,
        )
    ]

    try:
        result = await engine.transcribe_video(
            source,
            consent=_consent(allow_audio_alignment=True),
            audio_activity=audio,
        )
        frame = sync.fuse("visual-speech-success")

        assert result.status is VisualSpeechStatus.TRANSCRIBED
        assert result.transcript == "read the visual sentence"
        assert result.actionable is True
        assert result.calibrated is True
        assert result.alignment.passed is True
        assert result.speaker_association == "single_visible_track_not_identity_verified"
        assert backend.saw_nonzero is True
        assert np.count_nonzero(evidence.mouth_crops) == 0
        assert frame.has_usable(Modality.SPEECH) is True
        assert frame.belief("visual_speech.video_only").value is True
        assert frame.belief("visual_speech.actionable").value is True
        event = frame.observations[Modality.SPEECH]
        assert "visual_only_lip_reading" in event.quality_flags
        assert "read the visual sentence" not in repr(sync.get_status())
    finally:
        ServiceContainer.clear()


@pytest.mark.asyncio
async def test_uncalibrated_decoder_output_remains_non_actionable_candidate(tmp_path) -> None:
    evidence = _evidence()
    backend = Backend(confidence=None, calibrated=False)
    source = tmp_path / "speaker.mp4"
    source.write_bytes(b"video")
    sync = MultimodalSynchronizer()
    ServiceContainer.clear()
    ServiceContainer.register_instance("multimodal_synchronizer", sync, required=False)
    engine = VisualSpeechEngine(
        extractor=Extractor(evidence),
        backend=backend,
        wall_clock=lambda: NOW,
    )

    try:
        result = await engine.transcribe_video(source, consent=_consent())
        frame = sync.fuse("uncalibrated-visual-speech")

        assert result.status is VisualSpeechStatus.CANDIDATE
        assert result.transcript == "read the visual sentence"
        assert result.actionable is False
        assert result.calibrated is False
        assert result.confidence <= 0.49
        assert result.reason == "uncalibrated_visual_only_candidate"
        assert frame.missing[Modality.SPEECH] is MissingReason.UNCALIBRATED
        assert frame.has_usable(Modality.SPEECH) is False
        assert frame.belief("visual_speech.actionable") is None
    finally:
        ServiceContainer.clear()


@pytest.mark.asyncio
async def test_decoder_unavailable_is_explicit_and_zeroes_ephemeral_crops(tmp_path) -> None:
    evidence = _evidence()
    backend = Backend(available=False)
    source = tmp_path / "speaker.mp4"
    source.write_bytes(b"video")
    engine = VisualSpeechEngine(
        extractor=Extractor(evidence),
        backend=backend,
        wall_clock=lambda: NOW,
    )

    result = await engine.transcribe_video(source, consent=_consent())

    assert result.status is VisualSpeechStatus.UNAVAILABLE
    assert result.reason == "decoder_unavailable:checkpoint_missing"
    assert backend.calls == 0
    assert np.count_nonzero(evidence.mouth_crops) == 0


@pytest.mark.asyncio
async def test_cancellation_zeroes_ephemeral_mouth_crops(tmp_path) -> None:
    evidence = _evidence()
    started = asyncio.Event()

    class BlockingBackend(Backend):
        async def infer(self, mouth_crops, *, fps):
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    source = tmp_path / "speaker.mp4"
    source.write_bytes(b"video")
    engine = VisualSpeechEngine(
        extractor=Extractor(evidence),
        backend=BlockingBackend(),
        wall_clock=lambda: NOW,
    )
    task = asyncio.create_task(engine.transcribe_video(source, consent=_consent()))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert np.count_nonzero(evidence.mouth_crops) == 0


def test_raw_retention_consent_is_rejected_by_contract() -> None:
    with pytest.raises(ValueError, match="raw visual-speech retention is not supported"):
        _consent(allow_raw_retention=True)


def test_evidence_rejects_unsorted_timestamps_and_invalid_crop_shape() -> None:
    evidence = _evidence()
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(evidence, timestamps_s=tuple(reversed(evidence.timestamps_s)))
    with pytest.raises(ValueError, match="outside decoder bounds"):
        replace(evidence, mouth_crops=np.zeros((40, 16, 16, 3), dtype=np.uint8))


@pytest.mark.asyncio
async def test_service_lifecycle_registers_and_reports_without_transcript_leak(tmp_path) -> None:
    evidence = _evidence()
    backend = Backend(transcript="private visible sentence")
    source = tmp_path / "speaker.mp4"
    source.write_bytes(b"video")
    engine = VisualSpeechEngine(
        extractor=Extractor(evidence),
        backend=backend,
        wall_clock=lambda: NOW,
    )
    ServiceContainer.clear()

    try:
        await engine.start()
        result = await engine.transcribe_video(source, consent=_consent())
        status = engine.get_status()

        assert ServiceContainer.get("visual_speech") is engine
        assert result.transcript == "private visible sentence"
        assert status["started"] is True
        assert status["requests"] == 1
        assert status["status_counts"] == {"transcribed": 1}
        assert status["latest"]["transcript_available"] is True
        assert len(status["latest"]["transcript_digest"]) == 24
        assert "private visible sentence" not in repr(status)
    finally:
        await engine.stop()
        ServiceContainer.clear()
