"""core/voice/duplex/prosody.py — Her state, in how the voice moves.

core/voice/speech_profile.py already compiles Aura's substrate — affect,
neurochemistry, homeostatic drives, personality — into a SpeechProfile that
governs *what words* she chooses. This module continues that same signal
into *how those words sound*, so a tired Aura is audibly slower and a
delighted one audibly quicker.

Without this, an emotionally-grounded system still speaks in the flat
register of a train announcement, and every bit of interior state the
substrate computed is thrown away at the last inch.

Deliberately conservative ranges. Speech rate outside roughly 0.85–1.15x
stops reading as mood and starts reading as a broken audio device.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Voice.Prosody")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe(obj: Any, attr: str, default: float) -> float:
    try:
        raw = getattr(obj, attr, default)
        val = float(raw)
        # A NaN here silently poisons the speed parameter and Kokoro emits
        # garbage, so treat it as absent rather than propagating it.
        if val != val:
            return default
        return val
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class ProsodySpec:
    """Concrete synthesis parameters for one utterance."""

    voice: str
    speed: float = 1.0
    gain: float = 1.0
    # Extra silence after the chunk, in ms. Pauses carry as much affect as
    # rate does — a considered answer has air around it.
    trailing_pause_ms: float = 0.0

    def scaled(self, *, gain: float | None = None, speed: float | None = None) -> ProsodySpec:
        return ProsodySpec(
            voice=self.voice,
            speed=speed if speed is not None else self.speed,
            gain=gain if gain is not None else self.gain,
            trailing_pause_ms=self.trailing_pause_ms,
        )


class ProsodyCompiler:
    """SpeechProfile -> ProsodySpec."""

    def __init__(self, *, base_voice: str, base_speed: float = 1.0) -> None:
        self._base_voice = base_voice
        self._base_speed = base_speed

    def compile(self, profile: Any | None) -> ProsodySpec:
        """Derive synthesis parameters from her compiled speech profile.

        Returns the neutral baseline when no profile is available — voice
        must never fail closed on a missing substrate reading, because
        silence is a worse failure than flat affect.
        """
        if profile is None:
            return ProsodySpec(voice=self._base_voice, speed=self._base_speed)

        try:
            energy = _clamp(_safe(profile, "energy", 0.5), 0.0, 1.0)
            warmth = _clamp(_safe(profile, "warmth", 0.5), 0.0, 1.0)
            directness = _clamp(_safe(profile, "directness", 0.7), 0.0, 1.0)
            playfulness = _clamp(_safe(profile, "playfulness", 0.3), 0.0, 1.0)

            # Energy is the dominant driver of rate: lethargic 0.92x to
            # electric 1.12x, centred on the configured base.
            speed = self._base_speed * (0.92 + 0.20 * energy)
            # Bluntness clips along slightly; hedging slows down.
            speed *= 0.98 + 0.04 * directness
            speed = _clamp(speed, 0.84, 1.16)

            # Warm speech sits a touch fuller; cool speech a touch back.
            gain = _clamp(0.94 + 0.10 * warmth, 0.9, 1.05)

            # Low energy leaves more air between clauses. Playfulness closes
            # the gap — quick wit does not pause to admire itself.
            pause = 40.0 + 130.0 * (1.0 - energy) - 60.0 * playfulness
            pause = _clamp(pause, 0.0, 220.0)

            return ProsodySpec(
                voice=self._base_voice,
                speed=round(speed, 3),
                gain=round(gain, 3),
                trailing_pause_ms=round(pause, 1),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "voice_duplex.prosody",
                exc,
                action="spoke with neutral baseline prosody",
                severity="warning",
            )
            return ProsodySpec(voice=self._base_voice, speed=self._base_speed)


def live_speech_profile() -> Any | None:
    """Fetch her current compiled SpeechProfile, if the substrate is up.

    Best-effort by design: the voice lane must work on a bare runtime with
    no substrate, so absence is normal rather than an error.
    """
    try:
        from core.voice.substrate_voice_engine import get_substrate_voice_engine

        engine = get_substrate_voice_engine()
        if engine is None:
            return None
        profile = engine.get_current_profile()
        if profile is not None:
            return profile
        # No profile compiled yet this conversation — compile one now so the
        # first spoken turn is already in her voice, not the default.
        return engine.compile_profile()
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "voice_duplex.prosody",
            exc,
            action="used baseline prosody; substrate speech profile unavailable",
            severity="debug",
        )
        return None
