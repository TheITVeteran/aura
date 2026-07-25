"""core/voice/duplex/tts_stream.py — Streaming synthesis with instant cancel.

Two properties matter more than anything else here.

**Time to first audio.** Perceived latency is when she *starts* speaking,
not when she finishes. So the first clause is cut short (clause_chunker),
synthesised alone, and pushed out while the rest is still being generated.
Measured on this host: Kokoro-82M runs 6–8.6x realtime, ~190 ms for a short
clause, which is the floor this lane can hit.

**Cancellation.** Barge-in is worthless if it takes a second to take
effect. Every synthesis job checks a cancellation token between chunks, and
the session flushes the client's playback buffer independently — so audio
stops even if a chunk is mid-flight through the ONNX graph.

Kokoro is ONNX on CPU, which matters on this host: the resident 32B holds
~20 GB of GPU memory, and a TTS engine competing for Metal would show up as
jitter in her actual thinking. This one does not touch the GPU at all.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.voice.duplex.config import OUTPUT_RATE, TtsConfig
from core.voice.duplex.prosody import ProsodySpec

logger = logging.getLogger("Aura.Voice.Tts")


class CancellationToken:
    """One-shot cancel flag, checked between and inside synthesis stages."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()


@dataclass(slots=True)
class SynthesisResult:
    samples: np.ndarray          # float32 mono @ OUTPUT_RATE
    sample_rate: int
    text: str
    synth_ms: float
    engine: str

    @property
    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate or OUTPUT_RATE)


class _KokoroEngine:
    """Kokoro-82M via ONNX Runtime. Primary engine."""

    name = "kokoro"

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._kokoro: Any = None
        self._lock = threading.Lock()
        self._rate = 24_000
        self._available = False
        self._voices: frozenset[str] = frozenset()

    def load(self) -> bool:
        model = Path(self._config.model_path)
        voices = Path(self._config.voices_path)
        if not model.is_file() or not voices.is_file():
            logger.warning(
                "Kokoro assets missing (model=%s voices=%s); run tools/fetch_voice_models.py",
                model.exists(),
                voices.exists(),
            )
            return False
        try:
            from kokoro_onnx import Kokoro

            self._kokoro = Kokoro(str(model), str(voices))
            try:
                self._voices = frozenset(self._kokoro.get_voices())
            except (AttributeError, RuntimeError, TypeError) as exc:
                logger.debug("Kokoro voice enumeration unavailable: %s", exc)
                self._voices = frozenset()
            self._available = True
            logger.info("Kokoro TTS loaded (%d voices)", len(self._voices))
            return True
        except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
            record_degradation(
                "voice_duplex.tts",
                exc,
                action="Kokoro unavailable; falling back to Piper",
            )
            return False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def voices(self) -> frozenset[str]:
        return self._voices

    def resolve_voice(self, requested: str) -> str:
        if not self._voices or requested in self._voices:
            return requested
        logger.warning("Voice %r not in Kokoro pack; using %r", requested, self._config.voice)
        return self._config.voice if self._config.voice in self._voices else sorted(self._voices)[0]

    def synthesize(self, text: str, spec: ProsodySpec) -> tuple[np.ndarray, int]:
        # The ONNX session is not re-entrant; concurrent create() calls
        # corrupt each other's output buffers.
        with self._lock:
            samples, rate = self._kokoro.create(
                text,
                voice=self.resolve_voice(spec.voice),
                speed=float(spec.speed),
                lang="en-us",
            )
        return np.asarray(samples, dtype=np.float32), int(rate)


class _PiperEngine:
    """Piper fallback. Robotic next to Kokoro but very fast and dependency-light."""

    name = "piper"

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._voice: Any = None
        self._lock = threading.Lock()
        self._rate = 22_050
        self._available = False

    def load(self) -> bool:
        try:
            from piper import PiperVoice

            root = Path.home() / ".aura/live-source/data/voice_models/piper_voices"
            models = sorted(root.glob("*.onnx")) if root.is_dir() else []
            if not models:
                return False
            self._voice = PiperVoice.load(str(models[0]))
            self._rate = int(getattr(getattr(self._voice, "config", None), "sample_rate", 22_050))
            self._available = True
            logger.info("Piper TTS loaded: %s", models[0].name)
            return True
        except (ImportError, OSError, RuntimeError, ValueError, AttributeError, IndexError) as exc:
            record_degradation(
                "voice_duplex.tts",
                exc,
                action="Piper unavailable; falling back to system speech",
                severity="warning",
            )
            return False

    @property
    def available(self) -> bool:
        return self._available

    def synthesize(self, text: str, spec: ProsodySpec) -> tuple[np.ndarray, int]:
        with self._lock:
            chunks: list[np.ndarray] = []
            for audio in self._voice.synthesize(text):
                raw = getattr(audio, "audio_int16_bytes", None)
                if raw is None:
                    raw = bytes(audio)
                chunks.append(np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
        if not chunks:
            return np.zeros(0, dtype=np.float32), self._rate
        return np.concatenate(chunks), self._rate


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Rate-convert to the lane's output rate.

    Uses polyphase resampling when scipy is present because linear
    interpolation aliases audibly on speech; falls back to interpolation
    rather than failing, since a slightly harsh voice beats no voice.
    """
    if src_rate == dst_rate or samples.size == 0:
        return samples
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(src_rate), int(dst_rate))
        return resample_poly(samples, dst_rate // g, src_rate // g).astype(np.float32)
    except (ImportError, ValueError, RuntimeError) as exc:
        record_degradation(
            "voice_duplex.tts",
            exc,
            action="resampled with linear interpolation",
            severity="debug",
        )
        ratio = dst_rate / float(src_rate)
        n = int(round(samples.size * ratio))
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        idx = np.linspace(0, samples.size - 1, n, dtype=np.float64)
        return np.interp(idx, np.arange(samples.size), samples).astype(np.float32)


@dataclass(slots=True)
class _EngineState:
    kokoro: _KokoroEngine | None = None
    piper: _PiperEngine | None = None
    say_available: bool = False
    loaded: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class StreamingTts:
    """Chunked synthesis with pipelining and cancellation."""

    def __init__(self, config: TtsConfig | None = None) -> None:
        self._config = config or TtsConfig()
        self._state = _EngineState()
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, self._config.workers),
            thread_name_prefix="aura-tts",
        )

    async def ensure_loaded(self) -> bool:
        """Load engines once, off the event loop."""
        async with self._state.lock:
            if self._state.loaded:
                return self.available
            loop = asyncio.get_running_loop()

            kokoro = _KokoroEngine(self._config)
            if await loop.run_in_executor(self._pool, kokoro.load):
                self._state.kokoro = kokoro

            if self._state.kokoro is None:
                piper = _PiperEngine(self._config)
                if await loop.run_in_executor(self._pool, piper.load):
                    self._state.piper = piper

            self._state.say_available = bool(shutil.which("say"))
            self._state.loaded = True

            if not self.available:
                logger.error("No TTS engine available — the voice lane cannot speak")
            return self.available

    @property
    def available(self) -> bool:
        return bool(
            self._state.kokoro
            or self._state.piper
            or self._state.say_available
        )

    def available_voices(self) -> list[str]:
        """Voices the loaded engine can actually produce."""
        kokoro = self._state.kokoro
        if kokoro is not None:
            return sorted(kokoro.voices)
        return []

    @property
    def engine_name(self) -> str:
        if self._state.kokoro:
            return "kokoro"
        if self._state.piper:
            return "piper"
        if self._state.say_available:
            return "macos_say"
        return "none"

    async def warm_up(self, spec: ProsodySpec) -> None:
        """Run one throwaway synthesis so the first real one is not the cold one.

        Measured cold-start on this host is ~635 ms versus ~190 ms warm —
        the difference between a natural opening and an awkward one.
        """
        if not await self.ensure_loaded():
            return
        try:
            await self.synthesize("Okay.", spec, CancellationToken())
        except (RuntimeError, ValueError, OSError) as exc:
            record_degradation(
                "voice_duplex.tts",
                exc,
                action="warmup synthesis failed; first utterance pays cold-start cost",
                severity="warning",
            )

    async def synthesize(
        self,
        text: str,
        spec: ProsodySpec,
        token: CancellationToken,
    ) -> SynthesisResult | None:
        """Synthesise one chunk. Returns None if cancelled or empty."""
        text = (text or "").strip()
        if not text or token.cancelled:
            return None
        if not await self.ensure_loaded():
            return None

        started = time.perf_counter()
        loop = asyncio.get_running_loop()

        for engine in (self._state.kokoro, self._state.piper):
            if engine is None:
                continue
            try:
                samples, rate = await loop.run_in_executor(
                    self._pool, engine.synthesize, text, spec
                )
            except (RuntimeError, ValueError, OSError, AttributeError, MemoryError) as exc:
                record_degradation(
                    "voice_duplex.tts",
                    exc,
                    action=f"{engine.name} synthesis failed; trying next engine",
                    severity="warning",
                )
                continue

            if token.cancelled:
                # Interrupted while the graph was running. Discard rather
                # than play stale audio over the user.
                return None

            samples = _resample(samples, rate, OUTPUT_RATE)
            if spec.gain != 1.0:
                samples = samples * float(spec.gain)
            if spec.trailing_pause_ms > 0:
                pad = int(OUTPUT_RATE * spec.trailing_pause_ms / 1000.0)
                if pad > 0:
                    samples = np.concatenate((samples, np.zeros(pad, dtype=np.float32)))

            return SynthesisResult(
                samples=samples.astype(np.float32, copy=False),
                sample_rate=OUTPUT_RATE,
                text=text,
                synth_ms=(time.perf_counter() - started) * 1000.0,
                engine=engine.name,
            )

        return None

    async def stream(
        self,
        chunks: AsyncIterator[str],
        spec: ProsodySpec,
        token: CancellationToken,
    ) -> AsyncIterator[SynthesisResult]:
        """Synthesise an async stream of text chunks, one chunk ahead.

        Pipelining matters: synthesising chunk N+1 while chunk N is playing
        keeps the audio gapless. Without it there is an audible seam at every
        clause boundary, which is exactly the artefact that makes streaming
        TTS sound synthetic.
        """
        pending: asyncio.Task[SynthesisResult | None] | None = None

        async def _synth(text: str) -> SynthesisResult | None:
            return await self.synthesize(text, spec, token)

        try:
            async for chunk in chunks:
                if token.cancelled:
                    break
                text = (chunk or "").strip()
                if not text:
                    continue
                task = asyncio.ensure_future(_synth(text))
                if pending is not None:
                    result = await pending
                    if token.cancelled:
                        task.cancel()
                        break
                    if result is not None:
                        yield result
                pending = task

            if pending is not None and not token.cancelled:
                result = await pending
                pending = None
                if result is not None and not token.cancelled:
                    yield result
        finally:
            if pending is not None and not pending.done():
                pending.cancel()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
