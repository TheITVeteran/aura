"""Wake-word matching, tested against what a transcriber actually emits.

LIVE DEFECT, 2026-08-10: "when i talk to my computer nothing happens. no
response or anything." Microphone, voice-activity gate and transcriber were all
working; the wake pattern was written against idealised strings and missed
every real one. The strings below are copied from the live log, not invented —
that is the whole point of the file.
"""
from __future__ import annotations

import pytest

from core.voice.wake_word import WAKE_PATTERN


def _command_after_wake(transcript: str) -> str | None:
    """The command portion, extracted exactly as WakeWordDetector does."""
    match = WAKE_PATTERN.search(transcript)
    if not match:
        return None
    return transcript[match.end():].lstrip(" ,.!?;:-—").strip()


# Captured verbatim from ~/.aura/logs on the day this was found.
CAPTURED_FROM_LIVE_ASR = [
    ("Hey, Aura, can you turn on your camera?", "can you turn on your camera?"),
    ("Hey, Laura, can you hear me right now?", "can you hear me right now?"),
    ("Hey, Orrick, can you hear me?", "can you hear me?"),
    ("Hey, Aura, can you respond in chat?", "can you respond in chat?"),
]


@pytest.mark.parametrize("transcript,expected_command", CAPTURED_FROM_LIVE_ASR)
def test_real_transcriber_output_wakes_her(transcript, expected_command):
    assert _command_after_wake(transcript) == expected_command


def test_punctuation_between_greeting_and_name_does_not_block_the_wake():
    """`\\s+` could not cross the comma Whisper puts in "Hey, Aura"."""
    for transcript in ("hey, aura", "Hey. Aura", "hey - aura", "Hey — Aura"):
        assert WAKE_PATTERN.search(transcript), transcript


def test_name_variants_a_transcriber_produces_still_wake_her():
    """ASR writes her name as it sounds.

    "Laura" and "Orrick" were both observed live within one afternoon, which
    is why the name is matched by shape rather than by a list of spellings —
    a list loses to the next spelling the transcriber invents.
    """
    for name in ("aura", "Aura", "Laura", "Aurora", "Lora", "Ora", "Orrick",
                 "Dora", "Nora", "Aurah"):
        assert WAKE_PATTERN.search(f"Hey, {name}, what time is it?"), name


def test_every_greeting_form_works_with_punctuation():
    for greeting in ("Hey", "Hi", "Hello", "Ok", "Okay", "Yo"):
        assert WAKE_PATTERN.search(f"{greeting}, Aura, open my notes"), greeting


@pytest.mark.parametrize(
    "transcript",
    [
        "hey there, how are you",
        "the aurora borealis was visible last night",
        "okay so lets move on to the next thing",
        "hi how are you doing today",
        "laura called me yesterday about the invoice",
        "can you hear me",
        "Sorry, or can you hear me right now? Can you respond?",
    ],
)
def test_ordinary_speech_does_not_open_a_command_session(transcript):
    """A name variant must not fire without a greeting immediately before it."""
    assert WAKE_PATTERN.search(transcript) is None


def test_single_utterance_keeps_its_command():
    """"Hey Aura, open my notes" arrives as one chunk; the command must survive."""
    assert _command_after_wake("Hey Aura, open my notes") == "open my notes"
    assert _command_after_wake("Hello, Aura. Minimize this window.") == (
        "Minimize this window."
    )


def test_bare_wake_leaves_no_command():
    assert _command_after_wake("Hey Aura") == ""


def test_ambient_speech_may_be_answered_without_a_wake_phrase(monkeypatch):
    """OWNER REPORT 2026-08-10: "talking normally doesnt work."

    Requiring "Hey Aura" before every sentence made an open microphone behave
    like a closed one. The wake phrase was never the real protection — it is a
    password anyone's television can say. Audio provenance is.
    """
    from core.senses import voice_engine as ve

    engine = ve.SovereignVoiceEngine.__new__(ve.SovereignVoiceEngine)

    monkeypatch.setattr(ve, "_env_flag", lambda name, default=False: False)
    monkeypatch.setattr(
        "core.voice.audio_provenance.attribute_wake_audio",
        lambda evidence: {
            "owner_attributed": True,
            "owner_attribution_reason": "no_competing_audio_source",
        },
    )
    assert engine._ambient_speech_is_addressed_to_her(
        {"source": "nearby_person"}
    ) is True


def test_ambient_speech_is_refused_while_something_else_is_making_sound(monkeypatch):
    """The guard that stops a video — and her own TTS — from talking to her."""
    from core.senses import voice_engine as ve

    engine = ve.SovereignVoiceEngine.__new__(ve.SovereignVoiceEngine)

    monkeypatch.setattr(ve, "_env_flag", lambda name, default=False: False)
    monkeypatch.setattr(
        "core.voice.audio_provenance.attribute_wake_audio",
        lambda evidence: {
            "owner_attributed": False,
            "owner_attribution_reason": "unverified_speaker_while_host_audio_playing:afplay",
        },
    )
    assert engine._ambient_speech_is_addressed_to_her(
        {"source": "nearby_person"}
    ) is False


def test_wake_phrase_boundary_can_be_restored_for_a_shared_room(monkeypatch):
    from core.senses import voice_engine as ve

    engine = ve.SovereignVoiceEngine.__new__(ve.SovereignVoiceEngine)
    monkeypatch.setattr(
        ve, "_env_flag",
        lambda name, default=False: name == "AURA_VOICE_REQUIRE_WAKE_PHRASE",
    )
    assert engine._ambient_speech_is_addressed_to_her({}) is False


@pytest.mark.parametrize(
    "source",
    ["device_media", "ambient_speech", "unknown_speech", ""],
)
def test_ambient_answering_refuses_audio_that_is_not_a_person_at_the_machine(
    monkeypatch, source
):
    """A documentary must not be able to hold a conversation with her.

    "ambient_speech" and "unknown_speech" are refused for the same reason as
    media: both mean the evidence was too thin to say who spoke, and "cannot
    tell" must not become "answer it".
    """
    from core.senses import voice_engine as ve

    engine = ve.SovereignVoiceEngine.__new__(ve.SovereignVoiceEngine)
    engine._last_audio_source_assessment = {}

    monkeypatch.setattr(ve, "_env_flag", lambda name, default=False: False)
    monkeypatch.setattr(
        "core.voice.audio_provenance.attribute_wake_audio",
        lambda evidence: {"owner_attributed": True, "owner_attribution_reason": "x"},
    )

    assert engine._ambient_speech_is_addressed_to_her({"source": source}) is False
