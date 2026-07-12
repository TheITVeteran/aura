"""core/senses/visual_speech.py
────────────────────────────
Visual speech perception (the lip-reading channel), honestly scoped.

What this genuinely does today:
- tracks a mouth region from camera frames (face detection + geometric
  mouth ROI, no model downloads);
- measures syllabic-band mouth-motion energy and converts it into a
  calibrated speaking-probability with hysteresis — real visual
  speech-activity detection, suitable for "Bryan is talking to me even
  though the mic missed it" and audio-visual disambiguation;
- extracts coarse viseme features (mouth openness / width / motion)
  per frame — features, not words.

What it deliberately does NOT claim: word-level lip reading. That
requires a trained visual-speech-recognition model. The pipeline has a
governed ONNX seam (``attach_vsr_model``) that accepts a user-supplied
model file; until one is attached, ``transcript`` stays ``None`` and
confidence reporting says why. No silent fabrication.

cv2 policy: imports honor the main-process camera policy via
core.media.safe_imports — in the main runtime process the detector
falls back to injected detectors (the sidecar owns real capture).
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Senses.VisualSpeech")

_VISUAL_SPEECH_ERRORS = (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError)

# Syllable-rate band for human speech articulation, in Hz.
_SPEECH_BAND_LOW = 1.5
_SPEECH_BAND_HIGH = 8.0
_DEFAULT_FPS = 15.0
_ACTIVITY_WINDOW_FRAMES = 30
# Logistic calibration for motion-energy → speaking probability.
_CALIBRATION_MIDPOINT = 0.035
_CALIBRATION_STEEPNESS = 120.0
_SPEAK_ON_THRESHOLD = 0.65
_SPEAK_OFF_THRESHOLD = 0.35


@dataclass
class MouthRegion:
    x: int
    y: int
    width: int
    height: int

    def clamp(self, frame_height: int, frame_width: int) -> "MouthRegion":
        x = max(0, min(self.x, frame_width - 2))
        y = max(0, min(self.y, frame_height - 2))
        return MouthRegion(
            x=x,
            y=y,
            width=max(2, min(self.width, frame_width - x)),
            height=max(2, min(self.height, frame_height - y)),
        )


@dataclass
class VisualSpeechObservation:
    at: float
    face_present: bool
    mouth_region: MouthRegion | None
    motion_energy: float
    speaking_probability: float
    speaking: bool
    viseme_features: list[float] = field(default_factory=list)
    transcript: str | None = None
    transcript_source: str = "unavailable_no_vsr_model"

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "face_present": self.face_present,
            "motion_energy": round(self.motion_energy, 6),
            "speaking_probability": round(self.speaking_probability, 4),
            "speaking": self.speaking,
            "viseme_features": [round(f, 5) for f in self.viseme_features],
            "transcript": self.transcript,
            "transcript_source": self.transcript_source,
        }


def _default_face_detector() -> Callable[[np.ndarray], tuple[int, int, int, int] | None] | None:
    """Haar face detector from cv2's bundled cascades (no downloads).
    Returns None when cv2 is policy-blocked or unavailable."""
    try:
        from core.media.safe_imports import cv2_main_process_blocked

        if cv2_main_process_blocked():
            return None
        import cv2

        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if cascade.empty():
            return None

        def detect(gray: np.ndarray) -> tuple[int, int, int, int] | None:
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                return None
            # Largest face wins, deterministically.
            x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
            return int(x), int(y), int(w), int(h)

        return detect
    except _VISUAL_SPEECH_ERRORS as exc:
        record_degradation("senses.visual_speech.detector", exc)
        return None


def mouth_roi_from_face(face: tuple[int, int, int, int]) -> MouthRegion:
    """Geometric mouth region: lower third of the face, central half width."""
    x, y, w, h = face
    return MouthRegion(
        x=x + w // 4,
        y=y + (2 * h) // 3,
        width=w // 2,
        height=h // 3,
    )


class VisualSpeechPipeline:
    """Frame-in, observation-out visual speech perception."""

    def __init__(
        self,
        *,
        fps: float = _DEFAULT_FPS,
        face_detector: Callable[[np.ndarray], tuple[int, int, int, int] | None] | None = None,
    ):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = float(fps)
        self._face_detector = face_detector if face_detector is not None else _default_face_detector()
        self._previous_mouth: np.ndarray | None = None
        self._energy_window: deque[float] = deque(maxlen=_ACTIVITY_WINDOW_FRAMES)
        self._speaking = False
        self._vsr_session = None
        self._vsr_path: str | None = None
        self.observations = 0

    @property
    def detector_available(self) -> bool:
        return self._face_detector is not None

    @property
    def vsr_model_attached(self) -> bool:
        return self._vsr_session is not None

    # ── model seam (no downloads; user-supplied file only) ─────

    def attach_vsr_model(self, model_path: str | Path) -> dict[str, Any]:
        """Attach a user-supplied ONNX visual-speech-recognition model.
        Until this succeeds, transcripts remain None by design."""
        path = Path(model_path)
        if not path.exists() or path.suffix.lower() != ".onnx":
            return {
                "ok": False,
                "error": f"VSR model must be an existing .onnx file (got {path})",
            }
        try:
            import onnxruntime

            self._vsr_session = onnxruntime.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            self._vsr_path = str(path)
            logger.info("Visual speech: attached VSR model %s", path.name)
            return {"ok": True, "model": path.name}
        # onnxruntime raises pybind11 exception types that subclass
        # Exception directly (InvalidProtobuf, Fail, …) — a malformed
        # model file must degrade cleanly, never crash perception.
        except Exception as exc:  # noqa: BLE001
            record_degradation("senses.visual_speech.vsr_attach", exc)
            self._vsr_session = None
            return {"ok": False, "error": f"Could not load VSR model: {exc}"}

    # ── per-frame processing ───────────────────────────────────

    def process_frame(self, frame: np.ndarray, *, at: float | None = None) -> VisualSpeechObservation:
        """Consume one frame (grayscale or BGR uint8) and update state."""
        at = time.time() if at is None else float(at)
        gray = self._to_gray(frame)
        face = self._face_detector(gray) if self._face_detector else None
        if face is None:
            self._previous_mouth = None
            self._energy_window.append(0.0)
            self._speaking = False
            self.observations += 1
            return VisualSpeechObservation(
                at=at, face_present=False, mouth_region=None,
                motion_energy=0.0, speaking_probability=0.0, speaking=False,
            )

        region = mouth_roi_from_face(face).clamp(*gray.shape[:2])
        mouth = gray[
            region.y: region.y + region.height,
            region.x: region.x + region.width,
        ].astype(np.float64) / 255.0

        energy = self._motion_energy(mouth)
        self._energy_window.append(energy)
        band_energy = self._band_limited_energy()
        probability = 1.0 / (1.0 + math.exp(
            -_CALIBRATION_STEEPNESS * (band_energy - _CALIBRATION_MIDPOINT)
        ))
        # Hysteresis: flip on above the high mark, off below the low mark.
        if not self._speaking and probability >= _SPEAK_ON_THRESHOLD:
            self._speaking = True
        elif self._speaking and probability <= _SPEAK_OFF_THRESHOLD:
            self._speaking = False

        self.observations += 1
        return VisualSpeechObservation(
            at=at,
            face_present=True,
            mouth_region=region,
            motion_energy=energy,
            speaking_probability=probability,
            speaking=self._speaking,
            viseme_features=self._viseme_features(mouth),
            transcript=None,
            transcript_source=(
                "vsr_model_attached_but_decoding_not_wired"
                if self._vsr_session is not None
                else "unavailable_no_vsr_model"
            ),
        )

    # ── internals ──────────────────────────────────────────────

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 3:
            # BGR → luma without requiring cv2 in this process.
            arr = (
                0.114 * arr[..., 0] + 0.587 * arr[..., 1] + 0.299 * arr[..., 2]
            )
        if arr.ndim != 2:
            raise ValueError("frame must be HxW or HxWx3")
        return arr.astype(np.uint8)

    def _motion_energy(self, mouth: np.ndarray) -> float:
        previous = self._previous_mouth
        if previous is None or previous.shape != mouth.shape:
            self._previous_mouth = mouth
            return 0.0
        energy = float(np.mean(np.abs(mouth - previous)))
        self._previous_mouth = mouth
        return energy

    def _band_limited_energy(self) -> float:
        """Mean spectral energy of the mouth-motion signal restricted to
        the syllabic band — steady drift and camera noise fall outside."""
        window = np.array(self._energy_window, dtype=np.float64)
        if window.size < 8:
            return 0.0
        window = window - np.mean(window)
        spectrum = np.abs(np.fft.rfft(window)) / window.size
        freqs = np.fft.rfftfreq(window.size, d=1.0 / self.fps)
        band = (freqs >= _SPEECH_BAND_LOW) & (freqs <= _SPEECH_BAND_HIGH)
        if not np.any(band):
            return float(np.mean(spectrum[1:])) if spectrum.size > 1 else 0.0
        return float(np.sum(spectrum[band]))

    @staticmethod
    def _viseme_features(mouth: np.ndarray) -> list[float]:
        """Coarse geometric articulation features in [0, 1]:
        openness (dark-pixel ratio), vertical aperture profile, contrast."""
        if mouth.size == 0:
            return [0.0, 0.0, 0.0]
        darkness = mouth < max(0.15, float(np.percentile(mouth, 20)))
        openness = float(np.mean(darkness))
        # Vertical aperture: the tallest dark column fraction — how far
        # open the mouth is, independent of its width.
        aperture_profile = np.mean(darkness, axis=0)
        aperture = float(np.max(aperture_profile)) if aperture_profile.size else 0.0
        contrast = float(np.clip(np.std(mouth) * 4.0, 0.0, 1.0))
        return [round(openness, 5), round(aperture, 5), round(contrast, 5)]
