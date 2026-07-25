"""core/voice/duplex/overlap.py — Not every interruption is an interruption.

The first version of this lane treated *any* user speech over her output as a
barge-in and stopped her dead. That is a real defect, and a common one: on a
phone call people constantly say "mhm", "yeah", "right" and laugh while the
other person is talking, and none of it means stop. Punishing a listener for
listening is worse than not supporting overlap at all.

The problem is that at the moment overlap starts, the two cases are
genuinely indistinguishable — "yeah" and "yeah, but wait, that's wrong" begin
identically. Any classifier that must decide immediately will be wrong
constantly in one direction or the other.

So don't decide immediately. Do what people do: **duck, then decide.**

    0 ms    overlap detected
    ~130 ms drop her volume — the right response to *both* cases, and it is
            what a person does the instant someone else starts talking
    ~450 ms decide. Still going, or loud, or long? Real barge-in: stop, and
            truncate her memory to what was actually heard. Short and
            backchannel-shaped? Restore volume and carry on mid-sentence.

Ducking is instant, so responsiveness never depends on the decision being
fast. That removes the tradeoff entirely: the user always hears an immediate
reaction, and the *irreversible* action waits for evidence.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("Aura.Voice.Overlap")


class OverlapVerdict(Enum):
    PENDING = "pending"          # not enough evidence yet; stay ducked
    BACKCHANNEL = "backchannel"  # they are listening — keep talking
    BARGE_IN = "barge_in"        # they want the floor — stop


# What acknowledgement actually sounds like. Kept deliberately tight: a false
# "backchannel" means talking over someone who wanted to interrupt, which is
# the more annoying error of the two.
_BACKCHANNEL_TOKENS = frozenset(
    """
    mhm mm mmm mmhm hm hmm uhhuh uhuh huh yeah yea yep yup yes ok okay
    right sure exactly totally true nice wow oh ah aha haha hah ha
    gotcha cool word real damn
    """.split()
)

_LAUGHTER = re.compile(r"^(ha|heh|hah|haha|hehe|lol)+$", re.IGNORECASE)


def looks_like_backchannel(text: str) -> bool:
    """Is this transcript pure acknowledgement?

    Up to three tokens, every one of them an acknowledgement. "yeah yeah
    yeah" is backchannel; "yeah but no" is not, because "but" is not in the
    set and carries the actual meaning.
    """
    cleaned = re.sub(r"[^\w\s']", " ", (text or "").lower()).split()
    if not cleaned or len(cleaned) > 3:
        return False
    return all(
        token in _BACKCHANNEL_TOKENS or _LAUGHTER.match(token) for token in cleaned
    )


@dataclass(slots=True)
class OverlapConfig:
    # When her volume drops. Fast enough to feel like a reaction.
    duck_after_ms: float = 130.0
    # How far down. Audible enough to signal "I heard you start", not so far
    # that a backchannel leaves a hole in her sentence.
    duck_gain: float = 0.32
    # When the irreversible call is made.
    decide_after_ms: float = 460.0
    # Overlap longer than this is a barge-in whatever it sounded like — no
    # acknowledgement runs this long.
    certain_barge_in_ms: float = 900.0
    # Sustained loudness relative to the speaker's own norm that means they
    # are talking *over* rather than *under* her.
    loud_z: float = 1.6


@dataclass(slots=True)
class OverlapState:
    active: bool = False
    elapsed_ms: float = 0.0
    speech_ms: float = 0.0
    silence_ms: float = 0.0
    ducked: bool = False
    peak_energy: float = 0.0
    verdict: OverlapVerdict = OverlapVerdict.PENDING


class OverlapArbiter:
    """Decides what a stretch of user speech over her output means."""

    def __init__(self, config: OverlapConfig | None = None) -> None:
        self._config = config or OverlapConfig()
        self._state = OverlapState()

    @property
    def state(self) -> OverlapState:
        return self._state

    @property
    def active(self) -> bool:
        return self._state.active

    def reset(self) -> None:
        self._state = OverlapState()

    def begin(self) -> None:
        self._state = OverlapState(active=True)

    def observe(
        self,
        *,
        frame_ms: float,
        is_speech: bool,
        energy: float,
    ) -> OverlapVerdict:
        """Advance by one frame. Returns the current verdict."""
        state = self._state
        if not state.active:
            return OverlapVerdict.PENDING

        cfg = self._config
        state.elapsed_ms += frame_ms
        if is_speech:
            state.speech_ms += frame_ms
            state.silence_ms = 0.0
            state.peak_energy = max(state.peak_energy, energy)
        else:
            state.silence_ms += frame_ms

        # Nobody acknowledges for a second straight.
        if state.speech_ms >= cfg.certain_barge_in_ms:
            state.verdict = OverlapVerdict.BARGE_IN
            return state.verdict

        if state.elapsed_ms < cfg.decide_after_ms:
            return OverlapVerdict.PENDING

        # Decision time. Still holding the floor -> they want it.
        if state.silence_ms < 120.0:
            state.verdict = OverlapVerdict.BARGE_IN
        else:
            # They started and stopped inside half a second. That is the
            # shape of "mhm". Transcript confirmation, when the caller has
            # one, is applied in resolve().
            state.verdict = OverlapVerdict.BACKCHANNEL
        return state.verdict

    def should_duck(self) -> bool:
        """True once, at the point her volume should drop."""
        state = self._state
        if not state.active or state.ducked:
            return False
        if state.elapsed_ms >= self._config.duck_after_ms:
            state.ducked = True
            return True
        return False

    def resolve(self, transcript: str = "") -> OverlapVerdict:
        """Final call, refined by a transcript of the overlap if available.

        The transcript is the strongest signal but the slowest, so it only
        ever *corrects* a timing-based verdict rather than gating it.
        """
        state = self._state
        verdict = state.verdict

        if transcript:
            if looks_like_backchannel(transcript):
                # Short acknowledgement even if the timing looked marginal.
                verdict = OverlapVerdict.BACKCHANNEL
            elif len(transcript.split()) >= 3:
                # Real words means they said something, not just noise —
                # they were taking the floor.
                verdict = OverlapVerdict.BARGE_IN

        logger.info(
            "Overlap %s after %.0fms (speech %.0fms, transcript %r)",
            verdict.value,
            state.elapsed_ms,
            state.speech_ms,
            transcript[:40],
        )
        return verdict

    @property
    def duck_gain(self) -> float:
        return self._config.duck_gain
