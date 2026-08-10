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
]


@pytest.mark.parametrize("transcript,expected_command", CAPTURED_FROM_LIVE_ASR)
def test_real_transcriber_output_wakes_her(transcript, expected_command):
    assert _command_after_wake(transcript) == expected_command


def test_punctuation_between_greeting_and_name_does_not_block_the_wake():
    """`\\s+` could not cross the comma Whisper puts in "Hey, Aura"."""
    for transcript in ("hey, aura", "Hey. Aura", "hey - aura", "Hey — Aura"):
        assert WAKE_PATTERN.search(transcript), transcript


def test_name_variants_a_transcriber_produces_still_wake_her():
    """ASR writes her name as it sounds. "Laura" was observed live."""
    for name in ("aura", "Aura", "Laura", "Aurora", "Lora", "Ora"):
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
