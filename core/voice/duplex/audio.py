"""core/voice/duplex/audio.py — Format conversion and framing for the duplex lane.

Deliberately dependency-light: numpy only. This code runs on every 32 ms
frame of every voice session, so it must not allocate more than it has to.
"""
from __future__ import annotations

import numpy as np

INT16_SCALE = 32768.0


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Little-endian int16 PCM bytes -> float32 in [-1, 1).

    An odd trailing byte means a torn frame from the socket; drop it rather
    than letting numpy raise, because a single dropped sample is inaudible
    and a raised exception kills the capture lane.
    """
    if not data:
        return np.zeros(0, dtype=np.float32)
    if len(data) % 2:
        data = data[:-1]
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / INT16_SCALE


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """float32 -> little-endian int16 PCM bytes, hard-clipped.

    Clipping rather than normalising: normalising a chunk changes its gain
    relative to its neighbours, which is audible as pumping between the
    chunks of a single sentence.
    """
    if samples.size == 0:
        return b""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * (INT16_SCALE - 1)).astype("<i2").tobytes()


def rms(samples: np.ndarray) -> float:
    """Root-mean-square level, used for the UI's amplitude meter."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


class FrameSplitter:
    """Accumulates an arbitrary byte stream into fixed-size float32 frames.

    The browser sends whatever the AudioWorklet hands it; Silero needs
    exactly 512 samples. This absorbs that mismatch and keeps the remainder
    across calls so no sample is ever dropped or double-counted.
    """

    __slots__ = ("_frame_samples", "_buffer")

    def __init__(self, frame_samples: int) -> None:
        self._frame_samples = int(frame_samples)
        self._buffer = np.zeros(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        """Add samples; return every whole frame now available."""
        if samples.size:
            self._buffer = (
                samples.astype(np.float32, copy=False)
                if self._buffer.size == 0
                else np.concatenate((self._buffer, samples))
            )
        frames: list[np.ndarray] = []
        n = self._frame_samples
        while self._buffer.size >= n:
            frames.append(self._buffer[:n])
            self._buffer = self._buffer[n:]
        return frames

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)

    @property
    def pending_samples(self) -> int:
        return int(self._buffer.size)


class UtteranceBuffer:
    """Rolling audio for the utterance currently being spoken.

    Keeps a bounded amount of pre-onset audio ("preroll") because VAD always
    detects speech a frame or two after it truly starts, and Whisper needs
    that leading consonant to get the first word right.
    """

    __slots__ = ("_chunks", "_samples", "_max_samples", "_preroll", "_preroll_max")

    def __init__(self, max_seconds: float, sample_rate: int, preroll_ms: float = 320.0) -> None:
        self._chunks: list[np.ndarray] = []
        self._samples = 0
        self._max_samples = int(max_seconds * sample_rate)
        self._preroll: list[np.ndarray] = []
        self._preroll_max = int(preroll_ms / 1000.0 * sample_rate)

    def observe_silence(self, frame: np.ndarray) -> None:
        """Hold non-speech frames in case they turn out to precede speech."""
        self._preroll.append(frame)
        total = sum(c.size for c in self._preroll)
        while total > self._preroll_max and len(self._preroll) > 1:
            total -= self._preroll.pop(0).size

    def begin(self) -> None:
        """Open an utterance, seeded with the retained preroll."""
        self._chunks = list(self._preroll)
        self._samples = sum(c.size for c in self._chunks)
        self._preroll.clear()

    def append(self, frame: np.ndarray) -> None:
        if self._samples >= self._max_samples:
            return
        self._chunks.append(frame)
        self._samples += frame.size

    def audio(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks)

    @property
    def duration_s(self) -> float:
        return self._samples / float(CAPTURE_RATE_FALLBACK)

    @property
    def sample_count(self) -> int:
        return self._samples

    def clear(self) -> None:
        self._chunks = []
        self._samples = 0
        self._preroll.clear()


# Kept separate from config to avoid a circular import on this hot path.
CAPTURE_RATE_FALLBACK = 16_000
