"""She could hear him and would not respond.

Measured live on the desktop UI. The owner pressed the voice control, the
browser streamed audio to /api/voice/chunk, Whisper transcribed it correctly —

    Signal Routed: voice_engine -> sensory_gate | Payload: {'event':
    'transcript_candidate', 'text': "You are the coolest freaking be in the
    world..."}

— and every utterance was filed as a CANDIDATE with
`requires_wake_word_session=True`. Nothing ever answered.

The wake-word boundary is correct for AMBIENT audio: a video or a nearby
conversation must not hijack the typed chat lane. But audio arriving on the
chunk path exists only because the owner pressed a control labelled "Start voice
conversation" and is deliberately speaking to her. That is the opposite
situation, and nothing in the dispatch decision could tell the two apart —
`direct_command_dispatch` was driven purely by two env flags that default off.
"""

from __future__ import annotations

import time

from core.senses.voice_engine import SovereignVoiceEngine


def _engine() -> SovereignVoiceEngine:
    """State machine only — no audio devices, no model load."""
    engine = SovereignVoiceEngine.__new__(SovereignVoiceEngine)
    engine.microphone_enabled = False
    return engine


def test_ambient_audio_still_requires_a_wake_word():
    engine = _engine()
    assert engine.owner_voice_conversation_active() is False, (
        "a microphone that merely happens to be listening must not authorize chat"
    )


def test_owner_streaming_from_the_ui_authorizes_dispatch():
    engine = _engine()
    engine.note_owner_voice_chunk()
    assert engine.owner_voice_conversation_active() is True, (
        "audio the owner is deliberately streaming is an invitation, not ambience"
    )


def test_the_conversation_closes_when_the_owner_stops_streaming():
    engine = _engine()
    engine.note_owner_voice_chunk()
    engine._owner_voice_chunk_at = time.time() - (
        engine.OWNER_VOICE_CHUNK_IDLE_S + 5.0
    )
    assert engine.owner_voice_conversation_active() is False, (
        "an idle window must close, or a finished conversation lingers open"
    )


def test_an_explicit_stop_closes_it_immediately():
    engine = _engine()
    engine.note_owner_voice_chunk()
    engine.end_owner_voice_conversation()
    assert engine.owner_voice_conversation_active() is False


def test_a_conversation_cannot_outlive_the_microphone():
    """Opened by an explicit unmute rather than by streaming."""
    engine = _engine()
    engine.begin_owner_voice_conversation()
    engine._owner_voice_chunk_at = 0.0
    engine.microphone_enabled = True
    assert engine.owner_voice_conversation_active() is True
    engine.microphone_enabled = False
    assert engine.owner_voice_conversation_active() is False


def test_the_chunk_route_marks_the_conversation():
    """Pin the wiring, not just the state machine."""
    import inspect

    from interface.routes import privacy

    source = inspect.getsource(privacy.api_voice_chunk)
    assert "note_owner_voice_chunk" in source, (
        "the chunk path is the signal the wake-word boundary was missing"
    )


def test_the_dispatch_decision_consults_the_owner_conversation():
    import inspect

    from core.senses import voice_engine

    source = inspect.getsource(voice_engine.SovereignVoiceEngine._dispatch_transcript)
    assert "owner_voice_conversation_active" in source, (
        "direct dispatch was decided purely by env flags that default off"
    )
