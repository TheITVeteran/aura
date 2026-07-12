from __future__ import annotations

import sys

import numpy as np
import pytest

from core.perception.visual_speech import VisualSpeechPolicy
from core.perception.visual_speech_tracking import (
    MacOSVisionMouthDetector,
    MouthDetection,
    NativeVisualSpeechExtractor,
    _interpolate_crops,
    _iou,
)


def _write_video(path, *, frames: int = 40, fps: float = 25.0) -> None:
    import cv2

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        fps,
        (160, 120),
    )
    assert writer.isOpened()
    try:
        for index in range(frames):
            value = min(250, 20 + index * 5)
            frame = np.full((120, 160, 3), value, dtype=np.uint8)
            cv2.rectangle(frame, (50 + index % 4, 70), (110 + index % 4, 100), (255, 255, 255), -1)
            writer.write(frame)
    finally:
        writer.release()


class FakeLandmarkDetector:
    name = "fake_native_landmarks"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame_rgb, previous_bbox):
        index = self.calls
        self.calls += 1
        bbox = (0.25, 0.2, 0.5, 0.6)
        if index >= 20:
            bbox = (0.8, 0.2, 0.18, 0.6)
        if index % 5 == 0:
            crop = None
            landmarks = False
        else:
            crop = np.ascontiguousarray(frame_rgb[20:116, 32:128], dtype=np.uint8)
            landmarks = True
        return MouthDetection(
            face_count=2 if index in (10, 11) else 1,
            selected_bbox=bbox,
            crop=crop,
            landmarks_present=landmarks,
            competing_face_ratio=0.8 if index in (10, 11) else 0.1,
            detector=self.name,
        )


def test_native_extractor_preserves_timeline_and_interpolates_missing_mouth_frames(
    tmp_path,
) -> None:
    source = tmp_path / "visual-only.mp4"
    _write_video(source)
    detector = FakeLandmarkDetector()
    extractor = NativeVisualSpeechExtractor(detector=detector)

    evidence = extractor.extract(source, VisualSpeechPolicy())

    assert evidence.decoded_frames == 40
    assert evidence.mouth_frames == 40
    assert evidence.mouth_crops.shape == (40, 96, 96, 3)
    assert evidence.face_detection_coverage == 1.0
    assert evidence.mouth_landmark_coverage == pytest.approx(0.8)
    assert evidence.ambiguous_face_frames == 2
    assert evidence.competing_face_ratio == 0.8
    assert evidence.track_switches == 1
    assert evidence.source_audio_present is False
    assert evidence.source_audio_presence_known is (sys.platform == "darwin")
    assert evidence.extractor == "fake_native_landmarks"
    assert "video_stream_only_decoded" in evidence.quality_flags
    assert "raw_frames_not_retained" in evidence.quality_flags
    assert evidence.mean_brightness > 20.0
    assert evidence.mean_mouth_motion > 0.0
    assert len(evidence.source_digest) == 64
    assert str(source) not in repr(evidence)


def test_crop_interpolation_fills_edges_and_internal_gaps() -> None:
    left = np.full((96, 96, 3), 10, dtype=np.uint8)
    right = np.full((96, 96, 3), 30, dtype=np.uint8)

    result = _interpolate_crops([None, left, None, right, None])

    assert result.shape == (5, 96, 96, 3)
    assert int(result[0, 0, 0, 0]) == 10
    assert int(result[1, 0, 0, 0]) == 10
    assert int(result[2, 0, 0, 0]) == 20
    assert int(result[3, 0, 0, 0]) == 30
    assert int(result[4, 0, 0, 0]) == 30


def test_crop_interpolation_returns_bounded_empty_tensor_without_landmarks() -> None:
    result = _interpolate_crops([None, None, None])

    assert result.shape == (0, 96, 96, 3)
    assert result.dtype == np.uint8


def test_iou_handles_overlap_and_disjoint_tracks() -> None:
    assert _iou((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)) == 1.0
    assert _iou((0.0, 0.0, 0.2, 0.2), (0.8, 0.8, 0.2, 0.2)) == 0.0
    assert 0.0 < _iou((0.0, 0.0, 0.5, 0.5), (0.25, 0.25, 0.5, 0.5)) < 1.0


def test_native_detector_constructor_matches_platform_contract() -> None:
    if sys.platform == "darwin":
        detector = MacOSVisionMouthDetector()
        assert detector.name == "macos_vision_face_landmarks"
    else:
        with pytest.raises(RuntimeError, match="available only on Darwin"):
            MacOSVisionMouthDetector()
