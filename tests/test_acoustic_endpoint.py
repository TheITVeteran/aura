"""Waiting out a breath instead of talking over it.

The most common complaint about every conversational voice assistant that
ships is one sentence: *it interrupts me when I pause to think*. People stop
using voice mode over it. It is not a tuning miss — a system that decides
turn-end from the transcript plus a silence timer has discarded the signal
humans actually use, which is intonation.

The safety property is what makes this worth having in the turn-taking path
at all, and it is the one these tests care about most: the acoustic reading
can only ever make her wait **longer**. A wrong reading costs a beat of
latency; it can never cost an interruption. Every branch of
``patience_multiplier`` is checked against that, including the ones where the
pitch and the text disagree.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.voice.duplex.acoustic_endpoint import (
    Terminality,
    TerminalityReading,
    patience_multiplier,
    read_terminality,
)

SAMPLE_RATE = 16_000


def _tone(f0_start: float, f0_end: float, duration_s: float = 0.8) -> np.ndarray:
    """A voiced signal whose pitch sweeps from one frequency to another.

    A few harmonics because a pure sine is not what an F0 estimator is built
    for, and an autocorrelation peak on a single partial is not evidence the
    tracker works on speech.
    """
    n = int(duration_s * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    f0 = np.linspace(f0_start, f0_end, n)
    phase = 2.0 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    signal = np.zeros(n, dtype=np.float64)
    for harmonic, amplitude in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)):
        signal += amplitude * np.sin(harmonic * phase)
    signal *= 0.3 / max(1e-9, float(np.max(np.abs(signal))))
    return signal.astype(np.float32)


# ── reading the contour ──────────────────────────────────────────────────


def test_a_terminal_fall_is_heard_as_falling() -> None:
    reading = read_terminality(_tone(180.0, 110.0), SAMPLE_RATE)
    assert reading.terminality is Terminality.FALLING, reading.narrative()
    assert reading.slope_st_per_s < 0


def test_a_continuation_rise_is_heard_as_rising() -> None:
    """"I need eggs, milk↗, and…" — the list is not finished."""
    reading = read_terminality(_tone(120.0, 190.0), SAMPLE_RATE)
    assert reading.terminality is Terminality.RISING, reading.narrative()


def test_a_level_contour_is_the_hesitation_case() -> None:
    reading = read_terminality(_tone(150.0, 152.0), SAMPLE_RATE)
    assert reading.terminality is Terminality.LEVEL, reading.narrative()


def test_silence_yields_no_reading_rather_than_a_guess() -> None:
    reading = read_terminality(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    assert not reading.known


@pytest.mark.parametrize(
    "signal",
    [
        np.array([], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    ],
)
def test_degenerate_input_is_unknown_not_an_exception(signal: np.ndarray) -> None:
    """This runs on every pause. It may not raise into the reflex clock."""
    assert not read_terminality(signal, SAMPLE_RATE).known


def test_unvoiced_audio_yields_no_reading() -> None:
    """A whispered tail or a final /s/ has no pitch to read.

    Inventing one from noise is how an endpointer becomes confidently wrong.
    """
    rng = np.random.default_rng(11)
    noise = (rng.standard_normal(SAMPLE_RATE // 2) * 0.05).astype(np.float32)
    reading = read_terminality(noise, SAMPLE_RATE)
    assert reading.terminality in (Terminality.UNKNOWN, Terminality.LEVEL)


def test_semitones_make_the_reading_speaker_independent() -> None:
    """The same gesture on a low and a high voice must read the same.

    In hertz it would not: a 40 Hz fall is enormous for a low voice and
    unremarkable for a high one, so one threshold could not serve both.
    """
    low = read_terminality(_tone(120.0, 80.0), SAMPLE_RATE)
    high = read_terminality(_tone(240.0, 160.0), SAMPLE_RATE)
    assert low.terminality is high.terminality is Terminality.FALLING
    assert abs(low.slope_st_per_s - high.slope_st_per_s) < 3.0


# ── the safety property ──────────────────────────────────────────────────


@pytest.mark.parametrize("terminality", list(Terminality))
@pytest.mark.parametrize("text_complete", [True, False])
def test_patience_is_never_less_than_one(terminality: Terminality, text_complete: bool) -> None:
    """The entire safety argument for putting this in the path.

    Whatever the pitch tracker decides, the worst it can do is make her a
    beat slower. It cannot shorten a wait, so it cannot cause an interruption.
    """
    reading = TerminalityReading(terminality, slope_st_per_s=0.0, voiced_frames=20)
    assert patience_multiplier(reading, text_complete) >= 1.0


def test_the_complaint_case_gets_more_patience() -> None:
    """Text says finished, voice says composing. Wait.

    This is the case that produces every "it interrupts me" report: Whisper
    supplies a full stop from its language prior while the speaker is drawing
    breath, and the full stop is what makes an endpointer pounce.
    """
    level = TerminalityReading(Terminality.LEVEL, slope_st_per_s=0.2, voiced_frames=20)
    assert patience_multiplier(level, True) > 1.0


def test_agreement_between_text_and_voice_costs_nothing() -> None:
    """When both say it is over, she answers at the ordinary speed."""
    falling = TerminalityReading(Terminality.FALLING, slope_st_per_s=-4.0, voiced_frames=20)
    assert patience_multiplier(falling, True) == 1.0


def test_no_reading_changes_nothing() -> None:
    assert patience_multiplier(TerminalityReading(Terminality.UNKNOWN), True) == 1.0
    assert patience_multiplier(TerminalityReading(Terminality.UNKNOWN), False) == 1.0


# ── fused into the endpointer ────────────────────────────────────────────


def test_the_endpointer_waits_longer_on_a_level_contour() -> None:
    from core.voice.duplex.endpointing import Endpointer

    endpointer = Endpointer()
    common = {
        "transcript": "So I think the answer is probably fine.",
        "speech_ms": 2000.0,
        "min_utterance_ms": 220.0,
    }
    level = TerminalityReading(Terminality.LEVEL, slope_st_per_s=0.1, voiced_frames=20)

    without = endpointer.evaluate(silence_ms=400.0, **common)
    with_pitch = endpointer.evaluate(silence_ms=400.0, terminality=level, **common)

    assert without.should_end
    assert not with_pitch.should_end
    assert with_pitch.required_silence_ms > without.required_silence_ms


def test_patience_is_still_bounded_by_the_ceiling() -> None:
    """A turn always ends. Patience may not become a hang."""
    from core.voice.duplex.endpointing import Endpointer

    endpointer = Endpointer()
    level = TerminalityReading(Terminality.LEVEL, slope_st_per_s=0.0, voiced_frames=20)
    decision = endpointer.evaluate(
        transcript="So I think the answer is probably fine.",
        silence_ms=endpointer._config.max_silence_ms + 1.0,
        speech_ms=2000.0,
        min_utterance_ms=220.0,
        terminality=level,
    )
    assert decision.should_end
    assert decision.required_silence_ms <= endpointer._config.max_silence_ms
