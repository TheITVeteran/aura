"""core/voice/duplex/style.py — "Can you talk a bit slower?"

In a text chat, a request about delivery is just another message. Spoken
aloud it is a live control input, and the thing that makes a voice agent
feel responsive is that it obeys *immediately* — on the very next sentence,
not after a round trip through the model, and then keeps obeying.

So delivery requests are detected on the transcript and applied directly to
the prosody spec. The request still reaches her mind as a normal turn, so
she can acknowledge it in her own words; this layer only guarantees that the
acknowledgement is itself delivered at the requested rate.

Adjustments are relative and accumulate, the way they do with a person:
"slower" twice is slower than once, and each is clamped to a range that
still sounds like speech.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("Aura.Voice.Style")

# Ordered: the first pattern that matches wins, so more specific phrasings
# must precede the general ones.
_RATE_PATTERNS: tuple[tuple[re.Pattern[str], float, str], ...] = (
    (re.compile(r"\b(much|way|lot)\s+(slower|more slowly)\b"), -0.18, "much slower"),
    (re.compile(r"\b(much|way|lot)\s+faster\b"), 0.18, "much faster"),
    (re.compile(r"\b(slow(er)?\s+down|more slowly|slower|take your time)\b"), -0.10, "slower"),
    (re.compile(r"\b(speed\s+up|faster|quicker|hurry)\b"), 0.10, "faster"),
    (re.compile(r"\b(normal|regular|usual)\s+(speed|pace|rate)\b"), 0.0, "reset rate"),
)

_GAIN_PATTERNS: tuple[tuple[re.Pattern[str], float, str], ...] = (
    (re.compile(r"\b(whisper|much quieter|way quieter)\b"), -0.30, "much quieter"),
    (re.compile(r"\b(quieter|softer|lower your voice|turn it down|too loud)\b"), -0.15, "quieter"),
    (re.compile(r"\b(louder|speak up|turn it up|can'?t hear you)\b"), 0.15, "louder"),
)

# Only treat these as commands when the utterance is *about* delivery.
# Without this, "we should speed up the release" retunes her voice.
_DIRECTED = re.compile(
    r"\b(you|your voice|talk|speak|say|saying|voice|tone)\b|^\s*(slower|faster|quieter|louder)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class StyleAdjustment:
    """Persistent, user-requested deviation from her compiled prosody."""

    rate_delta: float = 0.0
    gain_delta: float = 0.0

    def clamp(self) -> None:
        # Outside roughly ±25% the voice stops sounding like a person
        # speaking differently and starts sounding like broken playback.
        self.rate_delta = max(-0.28, min(0.28, self.rate_delta))
        self.gain_delta = max(-0.45, min(0.35, self.gain_delta))

    @property
    def active(self) -> bool:
        return abs(self.rate_delta) > 1e-6 or abs(self.gain_delta) > 1e-6


class StyleController:
    """Detects and accumulates spoken delivery requests."""

    def __init__(self) -> None:
        self._adjustment = StyleAdjustment()

    @property
    def adjustment(self) -> StyleAdjustment:
        return self._adjustment

    def observe(self, transcript: str) -> str:
        """Apply any delivery request in ``transcript``.

        Returns a short human description of what changed, or "" if the
        utterance was not about delivery. The caller surfaces that to the UI
        so the change is visible rather than mysterious.
        """
        text = (transcript or "").strip().lower()
        if not text or not _DIRECTED.search(text):
            return ""

        changed: list[str] = []

        for pattern, delta, label in _RATE_PATTERNS:
            if pattern.search(text):
                if delta == 0.0:
                    self._adjustment.rate_delta = 0.0
                else:
                    self._adjustment.rate_delta += delta
                changed.append(label)
                break

        for pattern, delta, label in _GAIN_PATTERNS:
            if pattern.search(text):
                self._adjustment.gain_delta += delta
                changed.append(label)
                break

        if not changed:
            return ""

        self._adjustment.clamp()
        logger.info(
            "Voice style: %s (rate %+.2f, gain %+.2f)",
            ", ".join(changed),
            self._adjustment.rate_delta,
            self._adjustment.gain_delta,
        )
        return ", ".join(changed)

    def reset(self) -> None:
        self._adjustment = StyleAdjustment()
