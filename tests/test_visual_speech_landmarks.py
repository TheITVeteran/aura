"""Landmark-grade visual speech (core/senses/visual_speech.py):
mediapipe lip geometry with injectable trackers — no camera, no model
downloads in tests.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.senses.visual_speech import (
    VisualSpeechPipeline,
    lip_metrics_from_points,
    mouth_roi_from_face,
)

FACE = (40, 20, 80, 80)
FRAME_H, FRAME_W = 120, 160


def _fixed_detector(gray):
    return FACE


def _frame_with_mouth(openness: float) -> np.ndarray:
    frame = np.full((FRAME_H, FRAME_W), 180, dtype=np.uint8)
    region = mouth_roi_from_face(FACE)
    center_y = region.y + region.height // 2
    half_open = max(1, int((region.height // 2 - 2) * openness))
    frame[center_y - half_open: center_y + half_open,
          region.x + 4: region.x + region.width - 4] = 30
    return frame


# ── Landmark-grade lip tracking (mediapipe path) ─────────────────

class _SyntheticLipTracker:
    """Injectable tracker: replays a scripted aperture series."""

    def __init__(self, apertures):
        self._apertures = list(apertures)
        self._index = 0

    def lip_metrics(self, frame):
        if self._index >= len(self._apertures):
            return None
        aperture = self._apertures[self._index]
        self._index += 1
        if aperture is None:
            return None
        return {"aperture": aperture, "width": 0.42}


def _aperture_series(frames, hz, fps=15.0, amplitude=0.04, base=0.05):
    return [base + amplitude * (0.5 + 0.5 * math.sin(2 * math.pi * hz * i / fps))
            for i in range(frames)]


def test_landmark_articulation_detected_as_speech():
    tracker = _SyntheticLipTracker(_aperture_series(90, hz=4.0))
    pipeline = VisualSpeechPipeline(
        fps=15.0, face_detector=lambda g: None, lip_tracker=tracker)
    observations = [
        pipeline.process_frame(np.zeros((10, 10), dtype=np.uint8), at=i / 15.0)
        for i in range(90)
    ]
    tail = observations[45:]
    assert any(obs.speaking for obs in tail)
    assert max(obs.speaking_probability for obs in tail) > 0.9
    # Viseme features carry real geometry.
    assert 0.0 < tail[-1].viseme_features[0] <= 1.0
    assert 0.0 < tail[-1].viseme_features[1] <= 1.0


def test_landmark_static_mouth_is_not_speech():
    tracker = _SyntheticLipTracker([0.05] * 90)
    pipeline = VisualSpeechPipeline(
        fps=15.0, face_detector=lambda g: None, lip_tracker=tracker)
    observations = [
        pipeline.process_frame(np.zeros((10, 10), dtype=np.uint8), at=i / 15.0)
        for i in range(90)
    ]
    assert all(not obs.speaking for obs in observations)


def test_landmark_loss_falls_back_to_detector_path():
    tracker = _SyntheticLipTracker([0.05, 0.08, None, None])
    pipeline = VisualSpeechPipeline(
        fps=15.0, face_detector=_fixed_detector, lip_tracker=tracker)
    first = pipeline.process_frame(_frame_with_mouth(0.5), at=0.0)
    assert first.mouth_region is None  # landmark path served it
    pipeline.process_frame(_frame_with_mouth(0.5), at=0.1)
    fallback = pipeline.process_frame(_frame_with_mouth(0.5), at=0.2)
    assert fallback.face_present  # detector path took over seamlessly
    assert fallback.mouth_region is not None


def test_lip_metrics_geometry_is_scale_invariant():
    from core.senses.visual_speech import lip_metrics_from_points

    def points(scale):
        return {
            13: (0.5 * scale, 0.60 * scale), 14: (0.5 * scale, 0.66 * scale),
            61: (0.40 * scale, 0.63 * scale), 291: (0.60 * scale, 0.63 * scale),
            10: (0.5 * scale, 0.20 * scale), 152: (0.5 * scale, 0.90 * scale),
        }

    small, large = lip_metrics_from_points(points(1.0)), lip_metrics_from_points(points(7.0))
    assert small["aperture"] == pytest.approx(large["aperture"], abs=1e-12)
    assert small["width"] == pytest.approx(large["width"], abs=1e-12)
    assert lip_metrics_from_points({13: (0, 0)}) is None
