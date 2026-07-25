"""core/voice/duplex/streaming_asr.py — Incremental transcription.

Whisper is not a streaming model: it decodes a window and can revise any
word in it. Showing raw output live produces text that visibly rewrites
itself, and endpointing on it is worse than useless.

The fix is LocalAgreement-2 (Liu et al.): decode the growing buffer
repeatedly and treat as *stable* only the word prefix on which the last two
independent decodes agree. Whisper is free to revise the tail; the prefix
does not move. Live captions render the stable prefix solidly and the tail
faintly, and endpointing reasons only over the stable part.

Measured on this host: partial decode (small.en) ~72 ms, final decode
(large-v3-turbo) ~195 ms.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.voice.duplex.config import AsrConfig, CAPTURE_RATE

logger = logging.getLogger("Aura.Voice.Asr")

# Whisper's canonical outputs for "the mic was on but nobody spoke". It
# produces these confidently on silence, so they must never become a turn.
_HALLUCINATION_PATTERNS = (
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subscribe to",
    "you you you",
    "[blank_audio]",
    "[ silence ]",
    "(silence)",
    "♪",
)


def _normalise_words(text: str) -> list[str]:
    """Word list for prefix comparison.

    Case and punctuation are stripped for *matching* only — Whisper's
    capitalisation and commas legitimately change as context grows, and
    treating that as disagreement would keep the stable prefix empty.
    """
    return [w for w in re.split(r"\s+", text.strip()) if w]


def _match_key(word: str) -> str:
    return re.sub(r"[^\w']", "", word).lower()


def _common_prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if _match_key(x) != _match_key(y):
            break
        n += 1
    return n


def looks_hallucinated(text: str) -> bool:
    """True when the text is Whisper's silence-filler rather than speech."""
    low = text.strip().lower()
    if not low:
        return True
    if len(low) <= 2 and low not in ("hi", "no", "ok", "ye"):
        return True
    return any(p in low for p in _HALLUCINATION_PATTERNS)


@dataclass(slots=True)
class Transcript:
    """One incremental result."""

    stable: str = ""
    tentative: str = ""
    is_final: bool = False
    decode_ms: float = 0.0
    audio_s: float = 0.0

    @property
    def full(self) -> str:
        return " ".join(p for p in (self.stable.strip(), self.tentative.strip()) if p)


class _WhisperBackend:
    """mlx-whisper if available, faster-whisper otherwise.

    Model handles are cached per repo id because construction cost is
    dominated by weight load and Metal kernel compilation (measured 13–35 s
    cold), which must never happen inside a live turn.
    """

    _cache: dict[str, Any] = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: AsrConfig) -> None:
        self._config = config
        self._impl = ""
        self._mlx: Any = None
        try:
            import mlx_whisper

            self._mlx = mlx_whisper
            self._impl = "mlx"
        except (ImportError, OSError, RuntimeError) as exc:
            record_degradation(
                "voice_duplex.asr",
                exc,
                action="mlx-whisper unavailable; trying faster-whisper",
                severity="warning",
            )
            self._impl = ""

    @property
    def available(self) -> bool:
        return bool(self._impl) or self._faster_whisper_available()

    @staticmethod
    def _faster_whisper_available() -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except (ImportError, OSError) as _exc:
            logger.debug("faster-whisper unavailable: %s", _exc)
            return False

    def _faster_model(self, repo: str) -> Any:
        with self._cache_lock:
            key = f"fw::{repo}"
            model = self._cache.get(key)
            if model is None:
                from faster_whisper import WhisperModel

                # Repo ids carry an mlx-community prefix that faster-whisper
                # does not understand; fall back to a size it does.
                size = "small.en" if "small" in repo else "large-v3"
                model = WhisperModel(size, device="cpu", compute_type="int8")
                self._cache[key] = model
            return model

    def transcribe(self, audio: np.ndarray, repo: str) -> str:
        """Blocking decode. Always called on a worker thread."""
        if self._impl == "mlx":
            result = self._mlx.transcribe(
                audio,
                path_or_hf_repo=repo,
                language=self._config.language,
                fp16=True,
                condition_on_previous_text=False,
            )
            return str(result.get("text", "") or "")
        model = self._faster_model(repo)
        segments, _info = model.transcribe(
            audio, beam_size=1, language=self._config.language
        )
        return "".join(seg.text for seg in segments)

    def warm(self, repo: str) -> None:
        """Force weight load + kernel compile outside the latency path."""
        silence = np.zeros(CAPTURE_RATE, dtype=np.float32)
        try:
            self.transcribe(silence, repo)
        except (RuntimeError, ValueError, OSError, AttributeError) as exc:
            record_degradation(
                "voice_duplex.asr",
                exc,
                action=f"warmup decode failed for {repo}; first live decode will pay the cost",
                severity="warning",
            )


class StreamingAsr:
    """Growing-buffer incremental decoder with a stable prefix.

    Decoding runs on a thread executor: mlx-whisper releases the GIL during
    Metal work, but the Python-side pre/post-processing does not, and
    blocking the event loop here would stall audio intake for the whole
    session.
    """

    def __init__(self, config: AsrConfig | None = None) -> None:
        self._config = config or AsrConfig()
        self._backend = _WhisperBackend(self._config)
        self._prev_words: list[str] = []
        self._stable_words: list[str] = []
        self._tentative_words: list[str] = []
        self._last_partial_at = 0.0
        self._decode_lock = asyncio.Lock()
        self._warmed = False

    @property
    def available(self) -> bool:
        return self._backend.available

    async def warm_up(self) -> None:
        """Pay the cold-start cost before the user says anything."""
        if self._warmed:
            return
        self._warmed = True
        loop = asyncio.get_running_loop()
        for repo in (self._config.partial_model, self._config.final_model):
            await loop.run_in_executor(None, self._backend.warm, repo)
        logger.info("ASR warm: partial=%s final=%s", self._config.partial_model, self._config.final_model)

    def reset(self) -> None:
        self._prev_words = []
        self._stable_words = []
        self._tentative_words = []
        self._last_partial_at = 0.0

    def due_for_partial(self, now: float, audio_s: float) -> bool:
        """Rate-limit partials; decoding faster than this buys nothing."""
        if audio_s * 1000.0 < self._config.min_decode_ms:
            return False
        return (now - self._last_partial_at) * 1000.0 >= self._config.partial_interval_ms

    async def partial(self, audio: np.ndarray) -> Transcript | None:
        """Decode the buffer and fold the result into the stable prefix."""
        if not self.available or audio.size == 0:
            return None
        if self._decode_lock.locked():
            # A decode is already in flight. Skipping is correct: the next
            # one sees a longer buffer and supersedes this one anyway.
            return None
        async with self._decode_lock:
            self._last_partial_at = time.monotonic()
            started = time.perf_counter()
            try:
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(
                    None, self._backend.transcribe, audio, self._config.partial_model
                )
            except (RuntimeError, ValueError, OSError, AttributeError, MemoryError) as exc:
                record_degradation(
                    "voice_duplex.asr",
                    exc,
                    action="dropped one partial decode; endpointing continues on VAD alone",
                    severity="warning",
                )
                return None
            decode_ms = (time.perf_counter() - started) * 1000.0

        words = _normalise_words(text)
        # LocalAgreement-2: stability is agreement between consecutive
        # independent decodes, never a single decode's own confidence.
        agreed = _common_prefix_len(self._prev_words, words)
        if agreed > len(self._stable_words):
            self._stable_words = words[:agreed]
        self._prev_words = words
        self._tentative_words = words[len(self._stable_words):]

        return Transcript(
            stable=" ".join(self._stable_words),
            tentative=" ".join(self._tentative_words),
            is_final=False,
            decode_ms=decode_ms,
            audio_s=audio.size / float(CAPTURE_RATE),
        )

    async def finalize(self, audio: np.ndarray) -> Transcript:
        """One accurate decode of the complete utterance.

        This is the text her mind reasons over, so it runs on the large
        model even though it costs ~195 ms — accuracy here is worth more
        than the latency, and the filler lane covers the gap.
        """
        if audio.size == 0:
            return Transcript(is_final=True)
        started = time.perf_counter()
        text = ""
        try:
            async with self._decode_lock:
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(
                    None, self._backend.transcribe, audio, self._config.final_model
                )
        except (RuntimeError, ValueError, OSError, AttributeError, MemoryError) as exc:
            record_degradation(
                "voice_duplex.asr",
                exc,
                action="final decode failed; falling back to the stable partial prefix",
            )
            # The stable prefix is real transcribed speech, not a guess, so
            # using it is honest — but it may be missing the tail.
            text = " ".join(self._stable_words)

        cleaned = text.strip()
        if looks_hallucinated(cleaned):
            logger.info("Discarded hallucinated final transcript: %r", cleaned[:60])
            cleaned = ""

        return Transcript(
            stable=cleaned,
            tentative="",
            is_final=True,
            decode_ms=(time.perf_counter() - started) * 1000.0,
            audio_s=audio.size / float(CAPTURE_RATE),
        )
