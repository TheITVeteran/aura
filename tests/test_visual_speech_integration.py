"""Integration: the calibrated visual-speech pipeline drives the LIVE
interaction-signals → audio-attention path.

The chain under test:
  InteractionSignalsEngine._visual_speech_observation (calibrated
  pipeline on the engine's own face box)
  → VisionSignalState.speaking_likelihood
  → get_status()["vision"]  (what voice_engine reads)
  → classify_audio_attention(visual_context=...)  (response authority)
"""
from __future__ import annotations

import math
import time

import numpy as np

from core.senses.audio_attention import classify_audio_attention
from core.senses.interaction_signals import InteractionSignalsEngine
from core.senses.visual_speech import mouth_roi_from_face

FACE = (40, 20, 80, 80)
FRAME_H, FRAME_W = 120, 160


def _frame(openness: float) -> np.ndarray:
    frame = np.full((FRAME_H, FRAME_W), 180, dtype=np.uint8)
    region = mouth_roi_from_face(FACE)
    center_y = region.y + region.height // 2
    half_open = max(1, int((region.height // 2 - 2) * openness))
    frame[center_y - half_open: center_y + half_open,
          region.x + 4: region.x + region.width - 4] = 30
    return frame


def _feed_articulation(engine: InteractionSignalsEngine, frames: int, hz: float):
    last = None
    for i in range(frames):
        openness = 0.5 + 0.5 * math.sin(2 * math.pi * hz * i / 10.0)
        last = engine._visual_speech_observation(_frame(openness), FACE)
    return last


def test_engine_visual_speech_channel_detects_articulation():
    engine = InteractionSignalsEngine()
    observation = _feed_articulation(engine, 80, hz=3.0)
    assert observation.face_present
    assert observation.speaking_probability > 0.9
    assert observation.speaking is True


def test_engine_visual_speech_channel_rejects_static_face():
    engine = InteractionSignalsEngine()
    last = None
    for _ in range(80):
        last = engine._visual_speech_observation(_frame(0.4), FACE)
    assert last.speaking_probability < 0.2
    assert last.speaking is False


def test_vision_state_carries_calibrated_probability_to_status():
    engine = InteractionSignalsEngine()
    observation = _feed_articulation(engine, 80, hz=3.0)
    payload = {
        "updated_at": time.time(),
        "face_present": True,
        "speaking_likelihood": observation.speaking_probability,
        "speaking_active": observation.speaking,
        "mouth_motion_score": observation.motion_energy,
    }
    engine._vision = engine._update_vision_state(payload)
    vision = engine.get_status()["vision"]
    assert vision["speaking_likelihood"] > 0.9
    assert vision["speaking_active"] is True


def test_audio_attention_authorizes_with_visible_speaker():
    """End of the chain: a fresh visible speaker flips the audio-attention
    outcome for the same ambiguous utterance."""
    visual_speaking = {
        "updated_at": time.time(),
        "face_present": True,
        "speaking_likelihood": 0.95,
    }
    visual_absent = {"updated_at": time.time() - 3600.0, "face_present": False}

    kwargs = dict(
        rms_db=-30.0,           # quiet, far-field
        transcript_confidence=-0.8,
        duration_s=2.0,
    )
    # No wake word: the visual channel is the discriminating evidence.
    with_speaker = classify_audio_attention(
        "so what do you think about that", visual_context=visual_speaking, **kwargs)
    without_speaker = classify_audio_attention(
        "so what do you think about that", visual_context=visual_absent, **kwargs)

    assert with_speaker.source == "nearby_visible_speaker"
    assert with_speaker.attention_score > without_speaker.attention_score
    assert "fresh_visible_face" in with_speaker.reasons
    assert without_speaker.source != "nearby_visible_speaker"
