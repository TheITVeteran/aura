"""core/voice/duplex/vad_gate.py — Frame-level voice activity with hysteresis.

Silero VAD (ONNX, ~1.8 MB) costs 0.073 ms per 32 ms frame on this host —
about 0.2% of one core — so it can run on every frame of every session
without competing with the resident 32B for anything.

The gate itself is a hysteresis state machine, not a threshold. A bare
threshold flutters across the boundary on fricatives and breaks one
sentence into four utterances.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.voice.duplex.config import VAD_FRAME_SAMPLES, CAPTURE_RATE, VadConfig

logger = logging.getLogger("Aura.Voice.Vad")


class SpeechEvent(Enum):
    """What changed on this frame."""

    NONE = "none"
    ONSET = "onset"          # speech began
    CONTINUING = "continuing"  # still speaking
    PAUSE = "pause"          # fell below threshold, utterance still open
    RESUMED = "resumed"      # speech came back before the endpoint fired


@dataclass(slots=True)
class VadFrame:
    probability: float
    is_speech: bool
    event: SpeechEvent
    speech_ms: float   # how long the current utterance has been open
    silence_ms: float  # how long since the last speech frame (0 while speaking)


class _SileroBackend:
    """Silero VAD behind a lock.

    The ONNX session carries recurrent state across frames, so concurrent
    calls from two sessions would interleave each other's hidden state and
    produce nonsense. One backend instance per voice session; the lock is
    belt-and-braces for the shared-model case.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._lock = threading.Lock()
        self._torch: Any = None
        self._available = False
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from silero_vad import load_silero_vad

            self._torch = torch
            self._model = load_silero_vad(onnx=True)
            self._available = True
            logger.info("Silero VAD loaded (onnx)")
        except (ImportError, OSError, RuntimeError, AttributeError, ValueError) as exc:
            record_degradation(
                "voice_duplex.vad",
                exc,
                action="fell back to RMS energy gate; endpointing accuracy reduced",
            )
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def probability(self, frame: np.ndarray) -> float:
        if not self._available:
            raise RuntimeError("silero_unavailable")
        with self._lock:
            tensor = self._torch.from_numpy(np.ascontiguousarray(frame, dtype=np.float32))
            return float(self._model(tensor, CAPTURE_RATE).item())

    def reset(self) -> None:
        if not self._available:
            return
        with self._lock:
            try:
                self._model.reset_states()
            except (AttributeError, RuntimeError) as exc:
                logger.debug("VAD reset ignored: %s", exc)


class _EnergyBackend:
    """RMS fallback so the lane degrades instead of dying.

    Calibrates its own noise floor from the first ~1 s of audio, because a
    fixed threshold is wrong on every microphone. Materially worse than
    Silero on non-stationary noise — this is a degraded mode and the
    degradation record says so.
    """

    def __init__(self) -> None:
        self._floor = 0.004
        self._observed = 0

    @property
    def available(self) -> bool:
        return True

    def probability(self, frame: np.ndarray) -> float:
        level = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))) if frame.size else 0.0)
        if self._observed < 32:
            self._observed += 1
            self._floor = max(0.0015, 0.85 * self._floor + 0.15 * level)
            return 0.0
        # Map "how far above the noise floor" onto a pseudo-probability so
        # the same thresholds work for both backends.
        ratio = level / max(self._floor, 1e-6)
        if ratio <= 2.0:
            return 0.0
        return float(min(1.0, (ratio - 2.0) / 6.0))

    def reset(self) -> None:
        return


class VadGate:
    """Per-session speech-activity state machine."""

    def __init__(self, config: VadConfig | None = None) -> None:
        self._config = config or VadConfig()
        backend: Any = _SileroBackend()
        if not backend.available:
            backend = _EnergyBackend()
            self.degraded = True
        else:
            self.degraded = False
        self._backend = backend

        self._in_speech = False
        self._onset_run = 0
        self._speech_frames = 0
        self._silence_frames = 0
        self._last_probability = 0.0

    @property
    def backend_name(self) -> str:
        return "energy_rms" if self.degraded else "silero_onnx"

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def last_probability(self) -> float:
        return self._last_probability

    def process(self, frame: np.ndarray) -> VadFrame:
        """Advance the state machine by one 32 ms frame."""
        if frame.size != VAD_FRAME_SAMPLES:
            # Silero's graph is shape-specialised; a wrong-sized frame is a
            # caller bug, and silently padding it would corrupt the recurrent
            # state for every subsequent frame.
            raise ValueError(
                f"vad frame must be {VAD_FRAME_SAMPLES} samples, got {frame.size}"
            )
        try:
            prob = self._backend.probability(frame)
        except (RuntimeError, ValueError, AttributeError) as exc:
            record_degradation(
                "voice_duplex.vad",
                exc,
                action="treated frame as silence",
                severity="warning",
            )
            prob = 0.0
        self._last_probability = prob

        cfg = self._config
        event = SpeechEvent.NONE

        if not self._in_speech:
            if prob >= cfg.speech_threshold:
                self._onset_run += 1
                if self._onset_run >= cfg.onset_frames:
                    self._in_speech = True
                    # Credit the frames that proved the onset, so speech_ms
                    # reflects when speech actually started.
                    self._speech_frames = self._onset_run
                    self._silence_frames = 0
                    self._onset_run = 0
                    event = SpeechEvent.ONSET
            else:
                self._onset_run = 0
        else:
            if prob >= cfg.silence_threshold:
                resumed = self._silence_frames > 0
                self._silence_frames = 0
                self._speech_frames += 1
                event = SpeechEvent.RESUMED if resumed else SpeechEvent.CONTINUING
            else:
                self._silence_frames += 1
                event = SpeechEvent.PAUSE

        return VadFrame(
            probability=prob,
            is_speech=self._in_speech and self._silence_frames == 0,
            event=event,
            speech_ms=self._speech_frames * (VAD_FRAME_SAMPLES / CAPTURE_RATE * 1000.0),
            silence_ms=self._silence_frames * (VAD_FRAME_SAMPLES / CAPTURE_RATE * 1000.0),
        )

    def close_utterance(self) -> None:
        """Called when the endpointer decides the turn is over."""
        self._in_speech = False
        self._onset_run = 0
        self._speech_frames = 0
        self._silence_frames = 0

    def reset(self) -> None:
        self.close_utterance()
        self._backend.reset()
