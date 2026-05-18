"""Audio Service (The Ears).

Runs microphone voice-activity detection in the background and writes the
latest audio event to ``sensory_audio.json`` for the sensorimotor stack.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import scipy.io.wavfile as wav
except ImportError:
    wav = None

logger = logging.getLogger("Aura.AudioService")

_BASE_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_OUTPUT = _BASE_DIR / "sensory_audio.json"
_AUDIO_RECOVERABLE_ERRORS = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    OSError,
    ImportError,
    LookupError,
    TimeoutError,
)

AudioRecorder = Callable[[int, int], np.ndarray]
WavWriter = Callable[[Path, int, np.ndarray], None]
Transcriber = Callable[[Path], str | Awaitable[str]]
Sleeper = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class AudioServiceConfig:
    sample_rate: int = 44_100
    chunk_seconds: float = 5.0
    voice_threshold: float = 0.05
    error_backoff_seconds: float = 2.0
    max_consecutive_errors: int = 5
    output_path: Path = _DEFAULT_OUTPUT


def _validated_config(config: AudioServiceConfig | None) -> AudioServiceConfig:
    cfg = config or AudioServiceConfig()
    errors: list[str] = []
    if cfg.sample_rate <= 0:
        errors.append("sample_rate must be positive")
    if cfg.chunk_seconds <= 0:
        errors.append("chunk_seconds must be positive")
    if cfg.voice_threshold < 0:
        errors.append("voice_threshold must be non-negative")
    if cfg.error_backoff_seconds < 0:
        errors.append("error_backoff_seconds must be non-negative")
    if cfg.max_consecutive_errors <= 0:
        errors.append("max_consecutive_errors must be positive")
    if errors:
        raise ValueError("; ".join(errors))
    return cfg


def _default_recorder(frame_count: int, sample_rate: int) -> np.ndarray:
    if sd is None:
        raise RuntimeError("sounddevice is not installed")
    recording = sd.rec(frame_count, samplerate=sample_rate, channels=1)
    sd.wait()
    return np.asarray(recording, dtype=np.float32)


def _default_wav_writer(path: Path, sample_rate: int, recording: np.ndarray) -> None:
    if wav is None:
        raise RuntimeError("scipy wav writer is not installed")
    wav.write(str(path), sample_rate, recording)


def _openai_transcriber_from_env() -> Transcriber | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError as exc:
        record_degradation("audio_service", exc)
        logger.info("OpenAI package unavailable; audio will be recorded without cloud transcription.")
        return None

    client = OpenAI(api_key=api_key)

    def _transcribe(path: Path) -> str:
        with path.open("rb") as audio_file:
            return str(
                client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                ).text
            )

    return _transcribe


def _audio_snapshot(status: str, *, transcript: str = "", rms: float = 0.0, detail: str = "") -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "type": "audio",
        "status": status,
        "transcript": transcript,
        "rms": float(rms),
        "detail": detail,
    }


def _save_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(snapshot, sort_keys=True, indent=2))


async def _maybe_transcribe(transcriber: Transcriber | None, audio_path: Path) -> tuple[str, str]:
    if transcriber is None:
        return "", "audio_detected_transcription_unavailable"
    result = transcriber(audio_path)
    if inspect.isawaitable(result):
        result = await result
    text = str(result or "").strip()
    return text, "active" if text else "audio_detected_empty_transcript"


async def run_audio_loop(
    *,
    config: AudioServiceConfig | None = None,
    stop_event: asyncio.Event | None = None,
    max_iterations: int | None = None,
    recorder: AudioRecorder | None = None,
    wav_writer: WavWriter | None = None,
    transcriber: Transcriber | None = None,
    sleep: Sleeper = asyncio.sleep,
) -> None:
    """Run the microphone loop until stopped.

    Dependency injection keeps tests fast and lets the production service use
    real microphone, WAV, and transcription providers without global state.
    """
    try:
        cfg = _validated_config(config)
    except ValueError as exc:
        output_path = config.output_path if config is not None else _DEFAULT_OUTPUT
        _save_snapshot(output_path, _audio_snapshot("audio_configuration_invalid", detail=str(exc)))
        logger.error("Audio service configuration invalid: %s", exc)
        return

    stopper = stop_event or asyncio.Event()
    recorder = recorder or _default_recorder
    wav_writer = wav_writer or _default_wav_writer
    transcriber = transcriber if transcriber is not None else _openai_transcriber_from_env()

    if sd is None and recorder is _default_recorder:
        _save_snapshot(cfg.output_path, _audio_snapshot("audio_device_unavailable", detail="sounddevice unavailable"))
        logger.warning("Audio capture unavailable: sounddevice is not installed.")
        return
    if wav is None and wav_writer is _default_wav_writer:
        _save_snapshot(cfg.output_path, _audio_snapshot("audio_writer_unavailable", detail="scipy wav writer unavailable"))
        logger.warning("Audio capture unavailable: scipy wav writer is not installed.")
        return

    logger.info("Audio service starting: sample_rate=%s chunk_seconds=%.2f", cfg.sample_rate, cfg.chunk_seconds)
    iterations = 0
    consecutive_errors = 0
    frame_count = max(1, int(cfg.sample_rate * cfg.chunk_seconds))
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)

    while not stopper.is_set():
        if max_iterations is not None and iterations >= max_iterations:
            return
        iterations += 1

        try:
            recording = await asyncio.to_thread(recorder, frame_count, cfg.sample_rate)
            rms = float(np.sqrt(np.mean(np.asarray(recording, dtype=np.float32) ** 2)))
            if rms <= cfg.voice_threshold:
                _save_snapshot(cfg.output_path, _audio_snapshot("listening", rms=rms))
                consecutive_errors = 0
                continue

            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                prefix="aura_audio_",
                dir=str(cfg.output_path.parent),
                delete=False,
            ) as tmp:
                audio_path = Path(tmp.name)

            try:
                await asyncio.to_thread(wav_writer, audio_path, cfg.sample_rate, recording)
                transcript, status = await _maybe_transcribe(transcriber, audio_path)
            finally:
                try:
                    await asyncio.to_thread(audio_path.unlink, missing_ok=True)
                except OSError as exc:
                    record_degradation("audio_service", exc)
                    logger.debug("Temporary audio cleanup failed for %s: %s", audio_path, exc)

            _save_snapshot(cfg.output_path, _audio_snapshot(status, transcript=transcript, rms=rms))
            consecutive_errors = 0
        except _AUDIO_RECOVERABLE_ERRORS as exc:
            consecutive_errors += 1
            record_degradation("audio_service", exc)
            logger.warning("Audio loop error %d/%d: %s", consecutive_errors, cfg.max_consecutive_errors, exc)
            if consecutive_errors >= cfg.max_consecutive_errors:
                _save_snapshot(
                    cfg.output_path,
                    _audio_snapshot("audio_error_limit_reached", detail=str(exc)),
                )
                return
            await sleep(cfg.error_backoff_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(run_audio_loop())
    except KeyboardInterrupt:
        logger.info("Audio service stopping.")
