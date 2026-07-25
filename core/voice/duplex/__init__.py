"""core/voice/duplex — Full-duplex conversational voice.

The pre-existing voice path (core/senses/voice_engine.py) is half-duplex:
listen, stop, transcribe, think, speak, repeat. That is a walkie-talkie.
Real conversation overlaps — you acknowledge while the other person is
still talking, you start before they have quite finished, and either of you
can cut in.

This package implements that overlap:

    vad_gate       frame-level speech activity with hysteresis
    streaming_asr  incremental transcription with a stable prefix
    endpointing    adaptive turn-end detection driven by syntax, not a timer
    backchannel    "mhm" at prosodic boundaries while the user still holds the floor
    fillers        thinking sounds keyed to what her mind is actually doing
    clause_chunker text cut for minimum time-to-first-audio
    prosody        her substrate state compiled into how the voice moves
    tts_stream     streaming synthesis with sub-frame cancellation
    session        the duplex state machine tying it together
    mind_bridge    cognition, event bus, and governance wiring

Nothing here replaces the existing engine; the legacy lane stays intact and
is still what the wake-word path uses.
"""
from __future__ import annotations

__all__ = [
    "DuplexConfig",
    "DuplexVoiceSession",
    "SessionEvent",
    "SessionState",
]


def __getattr__(name: str):
    # Lazy so that importing the package does not drag in numpy, onnx and
    # the ASR stack on a runtime that never opens a voice session.
    if name == "DuplexConfig":
        from core.voice.duplex.config import DuplexConfig

        return DuplexConfig
    if name in ("DuplexVoiceSession", "SessionEvent", "SessionState"):
        from core.voice.duplex import session as _session

        return getattr(_session, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
