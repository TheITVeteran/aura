"""Visual speech pipeline: synthetic-frame verification.

No camera, no model downloads, no cv2 dependence in-process — the face
detector is injected, and the motion/band-pass/calibration logic runs
on synthetic articulating mouths with known ground truth.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.senses.visual_speech import (
    MouthRegion,
    VisualSpeechPipeline,
    mouth_roi_from_face,
)

FRAME_H, FRAME_W = 120, 160
FACE = (40, 20, 80, 80)  # x, y, w, h


def _fixed_detector(gray):
    return FACE


def _frame_with_mouth(openness: float) -> np.ndarray:
    """Synthetic face: light skin block with a dark mouth whose vertical
    aperture follows ``openness`` in [0, 1]."""
    frame = np.full((FRAME_H, FRAME_W), 180, dtype=np.uint8)
    region = mouth_roi_from_face(FACE)
    center_y = region.y + region.height // 2
    half_open = max(1, int((region.height // 2 - 2) * openness))
    frame[
        center_y - half_open: center_y + half_open,
        region.x + 4: region.x + region.width - 4,
    ] = 30
    return frame


def _run_sequence(pipeline: VisualSpeechPipeline, openness_series) -> list:
    return [
        pipeline.process_frame(_frame_with_mouth(o), at=i / pipeline.fps)
        for i, o in enumerate(openness_series)
    ]


def _talking_series(frames: int, fps: float, syllable_hz: float = 4.0):
    return [
        0.5 + 0.5 * math.sin(2 * math.pi * syllable_hz * i / fps)
        for i in range(frames)
    ]


# ── Core discrimination: talking vs still ────────────────────────

def test_articulating_mouth_is_detected_as_speech():
    pipeline = VisualSpeechPipeline(fps=15.0, face_detector=_fixed_detector)
    observations = _run_sequence(pipeline, _talking_series(90, 15.0))
    tail = observations[45:]
    assert any(obs.speaking for obs in tail)
    assert max(obs.speaking_probability for obs in tail) > 0.9


def test_static_mouth_is_not_speech():
    pipeline = VisualSpeechPipeline(fps=15.0, face_detector=_fixed_detector)
    observations = _run_sequence(pipeline, [0.4] * 90)
    assert all(not obs.speaking for obs in observations)
    assert all(obs.speaking_probability < 0.2 for obs in observations[10:])


def test_slow_drift_outside_syllabic_band_is_rejected():
    pipeline = VisualSpeechPipeline(fps=15.0, face_detector=_fixed_detector)
    # 0.2 Hz slow drift — mouth moves, but far below articulation rate.
    series = [0.5 + 0.4 * math.sin(2 * math.pi * 0.2 * i / 15.0) for i in range(90)]
    observations = _run_sequence(pipeline, series)
    assert all(not obs.speaking for obs in observations)


def test_hysteresis_prevents_flicker():
    pipeline = VisualSpeechPipeline(fps=15.0, face_detector=_fixed_detector)
    talking = _talking_series(60, 15.0)
    still = [0.5] * 60
    observations = _run_sequence(pipeline, talking + still)
    # Once ON, the state must not flap frame-to-frame: transitions are rare.
    states = [obs.speaking for obs in observations]
    transitions = sum(1 for a, b in zip(states, states[1:]) if a != b)
    assert transitions <= 3
    # And it must eventually release after silence.
    assert states[-1] is False


# ── Face handling ────────────────────────────────────────────────

def test_no_face_yields_honest_absence():
    pipeline = VisualSpeechPipeline(fps=15.0, face_detector=lambda gray: None)
    obs = pipeline.process_frame(_frame_with_mouth(0.5), at=0.0)
    assert not obs.face_present
    assert obs.speaking_probability == 0.0
    assert obs.transcript is None


def test_mouth_roi_geometry():
    region = mouth_roi_from_face((40, 20, 80, 80))
    assert region == MouthRegion(x=60, y=73, width=40, height=26)
    clamped = MouthRegion(x=-5, y=1000, width=999, height=999).clamp(120, 160)
    assert clamped.x >= 0 and clamped.y <= 118
    assert clamped.x + clamped.width <= 160
    assert clamped.y + clamped.height <= 120


def test_bgr_frames_accepted():
    pipeline = VisualSpeechPipeline(fps=15.0, face_detector=_fixed_detector)
    gray = _frame_with_mouth(0.5)
    bgr = np.stack([gray, gray, gray], axis=-1)
    obs = pipeline.process_frame(bgr, at=0.0)
    assert obs.face_present


# ── Viseme features ──────────────────────────────────────────────

def test_viseme_openness_tracks_mouth_aperture():
    pipeline = VisualSpeechPipeline(fps=15.0, face_detector=_fixed_detector)
    closed = pipeline.process_frame(_frame_with_mouth(0.05), at=0.0)
    open_wide = pipeline.process_frame(_frame_with_mouth(0.95), at=0.1)
    assert open_wide.viseme_features[0] > closed.viseme_features[0]
    assert open_wide.viseme_features[1] > closed.viseme_features[1]
    for value in open_wide.viseme_features:
        assert 0.0 <= value <= 1.0


# ── Honesty contract ─────────────────────────────────────────────

def test_transcript_absent_without_model_and_source_says_why():
    pipeline = VisualSpeechPipeline(fps=15.0, face_detector=_fixed_detector)
    obs = pipeline.process_frame(_frame_with_mouth(0.5), at=0.0)
    assert obs.transcript is None
    assert obs.transcript_source == "unavailable_no_vsr_model"
    assert not pipeline.vsr_model_attached


def test_vsr_seam_refuses_missing_or_wrong_files(tmp_path):
    pipeline = VisualSpeechPipeline(fps=15.0, face_detector=_fixed_detector)
    result = pipeline.attach_vsr_model(tmp_path / "nope.onnx")
    assert not result["ok"]
    bogus = tmp_path / "model.txt"
    bogus.write_text("not a model")
    assert not pipeline.attach_vsr_model(bogus)["ok"]
    garbage = tmp_path / "model.onnx"
    garbage.write_bytes(b"garbage-not-onnx")
    assert not pipeline.attach_vsr_model(garbage)["ok"]
    assert not pipeline.vsr_model_attached


def test_observation_serializes():
    pipeline = VisualSpeechPipeline(fps=15.0, face_detector=_fixed_detector)
    obs = pipeline.process_frame(_frame_with_mouth(0.5), at=1.0)
    payload = obs.to_dict()
    assert payload["face_present"] is True
    assert "speaking_probability" in payload
    assert payload["transcript"] is None


def test_invalid_construction_rejected():
    with pytest.raises(ValueError):
        VisualSpeechPipeline(fps=0.0, face_detector=_fixed_detector)
    pipeline = VisualSpeechPipeline(fps=15.0, face_detector=_fixed_detector)
    with pytest.raises(ValueError):
        pipeline.process_frame(np.zeros((2, 2, 2, 2), dtype=np.uint8))
