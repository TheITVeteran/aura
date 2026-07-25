"""core/voice/duplex/config.py — Tunables for the full-duplex voice lane.

Every number here is a latency or naturalness budget that was measured on
this host, not guessed. The comments record what each one buys so a future
change is an informed trade rather than a nudge.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ── Audio format ──────────────────────────────────────────────────────────
# 16 kHz mono is what Silero and Whisper both want; resampling once at the
# browser edge is cheaper than resampling every frame server-side.
CAPTURE_RATE = 16_000
# Kokoro emits 24 kHz. We keep it — downsampling would throw away the top
# octave that makes the voice sound like a person instead of a phone.
OUTPUT_RATE = 24_000

# Silero's ONNX graph is compiled for exactly 512 samples at 16 kHz (32 ms).
VAD_FRAME_SAMPLES = 512
VAD_FRAME_MS = VAD_FRAME_SAMPLES / CAPTURE_RATE * 1000.0


@dataclass(slots=True)
class VadConfig:
    """Hysteresis around Silero's frame probability.

    Two thresholds, not one: a high bar to *start* believing there is speech
    and a lower bar to keep believing it. A single threshold chatters on
    every fricative and turns one sentence into four utterances.
    """

    speech_threshold: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_VAD_ON", 0.55)
    )
    silence_threshold: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_VAD_OFF", 0.35)
    )
    # Consecutive speech frames before onset is real. 2 frames = 64 ms, which
    # rejects key clicks and lip smacks without eating the first phoneme.
    onset_frames: int = field(
        default_factory=lambda: _env_int("AURA_VOICE_VAD_ONSET_FRAMES", 2)
    )
    # Speech shorter than this is a noise burst, not a turn.
    min_utterance_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_MIN_UTTERANCE_MS", 220.0)
    )


@dataclass(slots=True)
class AsrConfig:
    """Two models on purpose.

    Partials only steer endpointing, backchannels and live captions, so they
    run on the small model (~72 ms/window measured). The final transcript is
    what her mind actually reasons over, so it runs once on the accurate
    model (~195 ms measured). Paying 195 ms on every 500 ms partial would
    contend with the resident 32B for the GPU for no accuracy that matters.
    """

    partial_model: str = field(
        default_factory=lambda: os.environ.get(
            "AURA_VOICE_ASR_PARTIAL", "mlx-community/whisper-small.en-mlx"
        )
    )
    final_model: str = field(
        default_factory=lambda: os.environ.get(
            "AURA_VOICE_ASR_FINAL", "mlx-community/whisper-large-v3-turbo"
        )
    )
    # How often a partial decode runs while the user is speaking.
    partial_interval_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_PARTIAL_INTERVAL_MS", 480.0)
    )
    # Don't bother decoding less than this — Whisper hallucinates on scraps.
    min_decode_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_MIN_DECODE_MS", 700.0)
    )
    # Hard ceiling on a single utterance's audio buffer (memory + decode cost).
    max_utterance_s: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_MAX_UTTERANCE_S", 60.0)
    )
    language: str = "en"


@dataclass(slots=True)
class EndpointConfig:
    """Adaptive turn-end detection — the thing that makes it feel human.

    A fixed silence timeout is the single biggest tell of a robot: too short
    and it interrupts you the moment you pause to think, too long and every
    reply feels sluggish. So the wait is a function of whether what you just
    said *sounds finished*.
    """

    # Sounds complete: falling intonation, terminal punctuation, full clause.
    complete_silence_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_EP_COMPLETE_MS", 340.0)
    )
    # No strong signal either way.
    neutral_silence_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_EP_NEUTRAL_MS", 620.0)
    )
    # Ends on a conjunction/preposition — they are mid-thought, wait.
    incomplete_silence_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_EP_INCOMPLETE_MS", 1100.0)
    )
    # Ends on an audible filler ("uh…", "so…") — they are actively thinking.
    thinking_silence_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_EP_THINKING_MS", 1500.0)
    )
    # Absolute ceiling; a turn always ends eventually.
    max_silence_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_EP_MAX_MS", 2000.0)
    )


@dataclass(slots=True)
class BackchannelConfig:
    """"mhm" while they talk.

    Overlapping acknowledgement is what separates a listener from a recorder.
    It is also the easiest thing in this whole system to overdo — an agent
    that says "mhm" every two seconds is worse than one that says nothing.
    """

    enabled: bool = field(
        default_factory=lambda: _env_flag("AURA_VOICE_BACKCHANNEL", True)
    )
    # They must have held the floor this long before acknowledgement is apt.
    min_floor_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BC_MIN_FLOOR_MS", 3200.0)
    )
    # Never twice inside this window.
    cooldown_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BC_COOLDOWN_MS", 5000.0)
    )
    # A micro-pause is a prosodic boundary — the natural slot for "mhm".
    # Below the floor it is just inter-word silence; above it, an endpoint.
    pause_min_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BC_PAUSE_MIN_MS", 140.0)
    )
    pause_max_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BC_PAUSE_MAX_MS", 420.0)
    )
    # Even at a perfect slot, stay quiet sometimes. Humans do.
    fire_probability: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BC_PROBABILITY", 0.55)
    )
    # Backchannels ride under the user's voice, so they are quiet.
    gain: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BC_GAIN", 0.34)
    )


@dataclass(slots=True)
class FillerConfig:
    """Thinking sounds, tied to the real reason she is slow.

    These are not decoration. Aura's mind is a governed 32B pipeline; some
    turns genuinely take seconds. A filler is an honest status report in the
    register of speech — and critically it asserts nothing about the answer,
    so it can never be wrong about content.
    """

    enabled: bool = field(default_factory=lambda: _env_flag("AURA_VOICE_FILLERS", True))
    # Under this, silence reads as "instant" and a filler would be noise.
    first_delay_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_FILLER_FIRST_MS", 380.0)
    )
    # A short breath/"uh" has covered the gap; now say something contentful
    # about what is actually taking the time.
    second_delay_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_FILLER_SECOND_MS", 1900.0)
    )
    # Long enough that silence would read as a crash.
    third_delay_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_FILLER_THIRD_MS", 6500.0)
    )


@dataclass(slots=True)
class TtsConfig:
    """Kokoro-82M primary, Piper fallback, macOS ``say`` last.

    Measured on this host: Kokoro loads in 0.66 s and runs 6–8.6x realtime,
    ~190 ms for a short clause. That is the time-to-first-audio floor.
    """

    voice: str = field(
        default_factory=lambda: os.environ.get("AURA_VOICE_KOKORO_VOICE", "af_heart")
    )
    speed: float = field(default_factory=lambda: _env_float("AURA_VOICE_SPEED", 1.0))
    model_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "AURA_VOICE_KOKORO_MODEL",
                str(Path.home() / ".aura/live-source/data/voice_models/kokoro/kokoro-v1.0.onnx"),
            )
        )
    )
    voices_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "AURA_VOICE_KOKORO_VOICES",
                str(Path.home() / ".aura/live-source/data/voice_models/kokoro/voices-v1.0.bin"),
            )
        )
    )
    # First spoken chunk is deliberately tiny: time-to-first-audio dominates
    # perceived latency, and a 3-word opener sounds like a person starting to
    # answer. Later chunks grow because longer context = better prosody.
    first_chunk_max_chars: int = field(
        default_factory=lambda: _env_int("AURA_VOICE_FIRST_CHUNK_CHARS", 48)
    )
    chunk_max_chars: int = field(
        default_factory=lambda: _env_int("AURA_VOICE_CHUNK_CHARS", 180)
    )
    # Synthesis threads. Kokoro is CPU/ONNX so this does not touch the GPU
    # the resident 32B is holding.
    workers: int = field(default_factory=lambda: _env_int("AURA_VOICE_TTS_WORKERS", 2))

    # Zero-shot voice cloning from a reference clip (XTTS-v2). Off unless a
    # clip is configured, because it costs seconds of latency per reply —
    # see _ClonedVoiceEngine for the trade.
    clone_reference: str = field(
        default_factory=lambda: os.environ.get("AURA_VOICE_CLONE_REFERENCE", "")
    )
    prefer_clone: bool = field(
        default_factory=lambda: _env_flag("AURA_VOICE_PREFER_CLONE", False)
    )


@dataclass(slots=True)
class BargeInConfig:
    """Interruption handling.

    Two things matter. First, stop fast — anything over ~200 ms and the user
    talks over themselves. Second, and more subtly: remember only what she
    *actually said out loud*. If her memory holds a paragraph the user never
    heard, every later reference to it is a hallucination from the user's
    point of view.
    """

    enabled: bool = field(default_factory=lambda: _env_flag("AURA_VOICE_BARGE_IN", True))
    # Sustained speech probability over this window cancels playback. Short
    # enough to feel instant, long enough to ignore a cough or a door.
    trigger_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BARGE_TRIGGER_MS", 128.0)
    )
    # Barge-in needs more confidence than ordinary onset: a false positive
    # cuts her off mid-word, which is worse than a slightly late stop.
    threshold: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BARGE_THRESHOLD", 0.72)
    )
    # Grace period after she starts speaking. Without it, the tail of the
    # user's own question (or residual echo) instantly kills the reply.
    grace_ms: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_BARGE_GRACE_MS", 320.0)
    )


@dataclass(slots=True)
class DuplexConfig:
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    backchannel: BackchannelConfig = field(default_factory=BackchannelConfig)
    filler: FillerConfig = field(default_factory=FillerConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)

    # How long a cognition turn may run before the lane gives up and says so.
    cognition_timeout_s: float = field(
        default_factory=lambda: _env_float("AURA_VOICE_COGNITION_TIMEOUT_S", 120.0)
    )

    @classmethod
    def load(cls) -> DuplexConfig:
        return cls()
