"""core/voice/duplex/acoustic_endpoint.py — the half of "am I finished" that is not words.

The most common complaint about every conversational voice assistant that
ships is the same sentence, in the same words, over and over: *it interrupts
me when I pause to think*. People stop using voice mode over it. The reason
is structural rather than a tuning miss — a system that decides turn-end from
the transcript plus a silence timer has thrown away the signal humans
actually use, which is intonation. You know a sentence is over before the
silence starts, because the pitch fell.

``endpointing.classify`` reads the text and is good at it. But text alone
cannot separate the two cases that matter most, and they are the two that
produce every complaint:

    "So I think the answer is"        — obviously unfinished, text catches it
    "So I think the answer is... um"  — obviously unfinished, text catches it
    "So I think the answer is"        — said with a *level* pitch and a breath,
                                       meaning "hold on, I'm composing"
    "So I think the answer is"        — said with a *falling* pitch, meaning
                                       the speaker garbled it and stopped

Identical transcripts, opposite intentions, and only the audio distinguishes
them. Worse, Whisper punctuates from a language-model prior rather than from
what it heard, so it will cheerfully write a full stop onto a speaker who is
mid-breath — and a full stop is exactly what makes an endpointer pounce.

The current best open turn-detection models are audio-native for this reason.
This module takes the same insight and pays for it in numpy rather than in a
second neural model: the F0 estimator this package already runs for
paralinguistics gives a pitch track, and the shape of that track over the
final voiced stretch is the reading. No download, no GPU contention with the
resident 32B, and it works with no network.

**What the numbers rest on.** The thresholds below come from the descriptive
phonetics of English intonation — declarative terminal falls run to a few
semitones over the final syllable, continuation rises are smaller and go the
other way, and level terminals are the hesitation case. They are literature-
shaped priors, not measurements taken on this host, and they are named and
tunable so that measuring them later changes a constant rather than a design.
What is *not* a guess is the direction of the asymmetry: this module can only
ever make Aura wait longer than the text alone would, never shorter. A wrong
reading therefore costs a little latency, and cannot cost an interruption.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from core.voice.duplex.paralinguistics import estimate_f0

logger = logging.getLogger("Aura.Voice.AcousticEndpoint")

# The final stretch of speech that carries the terminal contour. Shorter than
# this and there is not enough voiced material to fit a trend; much longer and
# the fit is dominated by the middle of the sentence rather than its end.
TERMINAL_WINDOW_S = 0.75

# Voiced frames required before a contour is reported at all. Unvoiced tails
# (a final /s/, a whispered word) legitimately have no pitch, and inventing
# one from three frames is how an endpointer becomes confidently wrong.
MIN_VOICED_FRAMES = 5

# Semitones per second. A declarative ending falls; the fall is large and
# fast enough that a threshold well clear of jitter still catches it.
FALLING_ST_PER_S = -2.5

# A rise this size at the end of a clause is either a question or — far more
# often in dictation — a continuation rise: "I need eggs, milk↗, and…".
RISING_ST_PER_S = 1.5

# Below this the contour is flat. Level terminals are the hesitation case:
# the speaker has not committed to ending, and this is precisely the moment
# every shipped assistant talks over its user.
LEVEL_BAND_ST_PER_S = 1.0


class Terminality(Enum):
    """What the pitch at the end of the utterance says about finishing."""

    FALLING = "falling"  # committed to ending
    RISING = "rising"  # a question, or a list still going
    LEVEL = "level"  # composing; the hesitation case
    UNKNOWN = "unknown"  # not enough voiced audio to say


@dataclass(slots=True)
class TerminalityReading:
    terminality: Terminality
    slope_st_per_s: float = 0.0
    voiced_frames: int = 0

    @property
    def known(self) -> bool:
        return self.terminality is not Terminality.UNKNOWN

    def narrative(self) -> str:
        if not self.known:
            return "no usable pitch at the end of the utterance"
        return (
            f"pitch {self.terminality.value} at "
            f"{self.slope_st_per_s:+.1f} semitones/s over the last "
            f"{self.voiced_frames} voiced frames"
        )


def read_terminality(
    signal: np.ndarray,
    sample_rate: int,
    *,
    window_s: float = TERMINAL_WINDOW_S,
) -> TerminalityReading:
    """Fit the pitch trend over the final stretch of an utterance.

    The fit is in semitones rather than hertz on purpose: pitch is perceived
    logarithmically, so a 20 Hz fall is a large gesture for a low voice and a
    shrug for a high one. Semitones make the same number mean the same thing
    for every speaker, which is what lets one threshold serve all of them.
    """
    if signal is None or getattr(signal, "size", 0) == 0 or sample_rate <= 0:
        return TerminalityReading(Terminality.UNKNOWN)

    tail_samples = int(max(0.1, float(window_s)) * sample_rate)
    tail = np.asarray(signal[-tail_samples:], dtype=np.float32)
    if tail.size < sample_rate // 20:
        return TerminalityReading(Terminality.UNKNOWN)

    try:
        f0 = estimate_f0(tail, sample_rate)
    except (ValueError, TypeError, ZeroDivisionError, FloatingPointError) as exc:
        logger.debug("terminality: f0 estimation failed (%s)", exc)
        return TerminalityReading(Terminality.UNKNOWN)

    f0 = np.asarray(f0, dtype=np.float64)
    voiced_mask = f0 > 0
    voiced_count = int(voiced_mask.sum())
    if voiced_count < MIN_VOICED_FRAMES:
        return TerminalityReading(Terminality.UNKNOWN, voiced_frames=voiced_count)

    # Frame times for the voiced frames only. estimate_f0 returns one value
    # per hop, so position in the array *is* time.
    hop_s = float(tail.size) / sample_rate / max(1, f0.size)
    times = np.arange(f0.size, dtype=np.float64) * hop_s
    voiced_times = times[voiced_mask]
    voiced_f0 = f0[voiced_mask]

    # Hertz to semitones relative to the median of this stretch. The
    # reference cancels out of the slope, so any positive reference works;
    # the median keeps the numbers small and well-conditioned.
    reference = float(np.median(voiced_f0))
    if reference <= 0:
        return TerminalityReading(Terminality.UNKNOWN, voiced_frames=voiced_count)
    semitones = 12.0 * np.log2(np.maximum(voiced_f0, 1e-6) / reference)

    span = float(voiced_times[-1] - voiced_times[0])
    if span <= 1e-3:
        return TerminalityReading(Terminality.UNKNOWN, voiced_frames=voiced_count)

    # Least squares on a straight line. Robust enough here: octave errors are
    # already suppressed by the estimator's search range, and the alternative
    # (first-to-last difference) is dominated by whichever single frame
    # happened to land last.
    slope = float(np.polyfit(voiced_times, semitones, 1)[0])

    if slope <= FALLING_ST_PER_S:
        terminality = Terminality.FALLING
    elif slope >= RISING_ST_PER_S:
        terminality = Terminality.RISING
    elif abs(slope) <= LEVEL_BAND_ST_PER_S:
        terminality = Terminality.LEVEL
    else:
        # Between the level band and a threshold: real movement, not enough
        # to call. Treating this as LEVEL keeps the module's one-way
        # guarantee — it can only add patience.
        terminality = Terminality.LEVEL

    return TerminalityReading(
        terminality=terminality,
        slope_st_per_s=slope,
        voiced_frames=voiced_count,
    )


def patience_multiplier(reading: TerminalityReading, text_says_complete: bool) -> float:
    """How much longer to wait than the text alone would suggest.

    Never less than 1.0. That is the whole safety argument for putting a
    heuristic pitch tracker in the turn-taking path: the worst a wrong reading
    can do is make her a beat slower, and the failure it prevents is the one
    that makes people stop using voice mode.

    The interesting case is the first branch. The text looks finished — often
    because Whisper supplied a full stop from its language prior — but the
    voice did not fall. That is a person mid-thought, and it is exactly where
    every shipped assistant cuts in.
    """
    if not reading.known:
        return 1.0

    if text_says_complete:
        if reading.terminality is Terminality.FALLING:
            return 1.0  # text and voice agree it is over
        if reading.terminality is Terminality.LEVEL:
            return 1.8  # composing: the complaint case
        if reading.terminality is Terminality.RISING:
            return 1.5  # a question she can answer, or a list still running
        return 1.0

    # The text already thinks this is unfinished; a falling contour is the
    # only thing that argues otherwise, and it does not argue strongly enough
    # to shorten anything.
    if reading.terminality is Terminality.FALLING:
        return 1.0
    return 1.3


__all__ = [
    "FALLING_ST_PER_S",
    "LEVEL_BAND_ST_PER_S",
    "MIN_VOICED_FRAMES",
    "RISING_ST_PER_S",
    "TERMINAL_WINDOW_S",
    "Terminality",
    "TerminalityReading",
    "patience_multiplier",
    "read_terminality",
]
