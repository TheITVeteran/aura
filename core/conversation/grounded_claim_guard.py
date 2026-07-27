"""Claims the runtime can measure are checked against the measurement.

Telling a model what time it is is a prior, and a prior loses to a fluent
sentence — silently. Measured live 2026-07-27, all of these from a runtime with
no window, no camera and no light sensor:

    at 00:30  "The sun's up but I'm not sure it will be warm today — there are
               clouds gathering in the east."
    at 01:40  "my clock says it's 06:15 and the ambient light sensors report
               very low illumination values with a cool spectrum"

Worth recording precisely, because the prompt-side fix does work: by 10:52 the
same question got "It's 10:52 AM on a Monday", which was exactly right, and the
dispatch log confirmed the grounding block reaching the model. So this guard is
not a replacement for that. It is the causal half — the part that does not
depend on the model choosing to read what it was given.

The runtime takes the reading it can actually take, and a claim contradicting
that reading does not get spoken. Two kinds:

* **the clock** — the system clock owns the time and the part of day;
* **instruments that do not exist** — "ambient light sensors report..." is not
  a wrong value, it is a claim to an organ. There is no reading to compare
  against, so the claim is removed rather than corrected.

Daylight itself is deliberately not policed: at 10:52 "the sun is up" follows
from the clock, and a guard that fights correct inferences is worse than no
guard. Sky and weather detail is, because nothing here can see it.

Every repair is recorded, so a persistent gap between what she says and what is
true stays visible rather than being quietly patched over.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from core.runtime.errors import record_degradation

_RECOVERABLE = (RuntimeError, AttributeError, TypeError, ValueError, OSError, ImportError)

# How far a stated time may drift before it is wrong rather than approximate.
_CLOCK_TOLERANCE_MINUTES = 25


@dataclass(frozen=True)
class GroundedReply:
    """A reply with its measurable claims reconciled against measurements."""

    text: str
    corrections: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.corrections)


def _part_of_day(hour: int) -> str:
    if hour < 5:
        return "the middle of the night"
    if hour < 8:
        return "early morning"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "late evening"


_PART_OF_DAY_WORDS: dict[str, tuple[int, int]] = {
    "middle of the night": (0, 5),
    "the dead of night": (0, 5),
    "early morning": (5, 8),
    "morning": (5, 12),
    "midday": (11, 14),
    "noon": (11, 14),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "late evening": (21, 24),
    "night": (21, 5),
}

# "it's 10:52 AM", "the time is 6:15", "my clock says 06:15"
_STATED_TIME_RE = re.compile(
    r"(?P<lead>\b(?:it(?:'s| is)|the time is|my clock (?:says|reads)|right now it(?:'s| is))\s+"
    r"(?:about\s+|around\s+|roughly\s+)?)"
    r"(?P<time>\d{1,2}:\d{2}(?:\s*(?:am|pm|AM|PM))?)",
    re.IGNORECASE,
)

# A part-of-day word asserted as the present, not as a topic.
_STATED_PART_RE = re.compile(
    r"\b(?:it(?:'s| is)|right now it(?:'s| is)|we(?:'re| are) in the)\s+"
    r"(?:currently\s+|still\s+)?"
    r"(?P<part>the middle of the night|the dead of night|late evening|early morning|"
    r"afternoon|evening|morning|midday|noon|night)\b",
    re.IGNORECASE,
)

# Perception with no organ behind it. Each pattern matches a whole clause so the
# repair removes a sentence fragment rather than leaving a dangling one.
_PHANTOM_PERCEPTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?:^|(?<=[.!?;,—-]\s))[^.!?]*\b(?:ambient|light)\s+sensors?\b[^.!?]*[.!?]?",
            re.IGNORECASE,
        ),
        "claimed a light sensor reading",
    ),
    (
        # Weather and sky detail, which no reading available to this process
        # can support. Daylight itself is deliberately NOT here: at 10:52 "the
        # sun is up" follows from the clock, and stripping a correct inference
        # would be the same error in the other direction — a guard that fights
        # right answers is worse than no guard.
        re.compile(
            r"(?:^|(?<=[.!?;,—-]\s))[^.!?]*\b(?:the light outside|"
            r"it(?:'s| is)\s+(?:sunny|raining|snowing|overcast|cloudy)|"
            r"clouds? (?:are )?gathering|the sky (?:is|looks)|"
            r"i can see (?:the|it|outside))\b[^.!?]*[.!?]?",
            re.IGNORECASE,
        ),
        "described weather it has no reading for",
    ),
)


def _real_time_text(stamp: datetime, sample: str) -> str:
    """Match the format she used, so the repair reads like the sentence."""
    if re.search(r"(?i)\b(?:am|pm)\b", sample):
        rendered = stamp.strftime("%I:%M %p").lstrip("0")
        return rendered
    return stamp.strftime("%H:%M")


def _minutes_apart(stated: str, stamp: datetime) -> int | None:
    match = re.match(r"(\d{1,2}):(\d{2})\s*(am|pm)?", stated.strip(), re.IGNORECASE)
    if not match:
        return None
    try:
        hour = int(match.group(1))
        minute = int(match.group(2))
    except (TypeError, ValueError):
        return None
    meridiem = (match.group(3) or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    stated_minutes = hour * 60 + minute
    actual_minutes = stamp.hour * 60 + stamp.minute
    delta = abs(stated_minutes - actual_minutes)
    # A clock is circular: 23:55 and 00:05 are ten minutes apart.
    return min(delta, 1440 - delta)


def _part_matches_clock(part: str, hour: int) -> bool:
    # "the middle of the night" and "middle of the night" are the same claim;
    # keying on only one of them silently disabled the check for the other.
    key = re.sub(r"^the\s+", "", part.strip().lower())
    window = _PART_OF_DAY_WORDS.get(key)
    if window is None:
        return True
    start, end = window
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end  # wraps midnight


def verify_grounded_claims(reply: str, *, now: datetime | None = None) -> GroundedReply:
    """Reconcile a user-facing reply with what this runtime can measure.

    Conservative on purpose. It corrects a stated clock time, corrects a stated
    part of day, and removes claims to instruments that do not exist. It leaves
    a time that is being discussed rather than asserted, it leaves a correct
    statement alone, and it never invents a claim that was not already there.
    """
    text = str(reply or "")
    if not text.strip():
        return GroundedReply(text=text)
    try:
        stamp = now or datetime.now().astimezone()
    except _RECOVERABLE:
        return GroundedReply(text=text)

    corrections: list[str] = []

    def _fix_time(match: re.Match[str]) -> str:
        stated = match.group("time")
        apart = _minutes_apart(stated, stamp)
        if apart is None or apart <= _CLOCK_TOLERANCE_MINUTES:
            return match.group(0)
        corrected = _real_time_text(stamp, stated)
        corrections.append(f"stated time {stated} -> {corrected} (off by {apart} min)")
        return f"{match.group('lead')}{corrected}"

    text = _STATED_TIME_RE.sub(_fix_time, text)

    def _fix_part(match: re.Match[str]) -> str:
        part = match.group("part")
        if _part_matches_clock(part, stamp.hour):
            return match.group(0)
        truth = _part_of_day(stamp.hour)
        corrections.append(f"stated part of day {part!r} -> {truth!r}")
        return match.group(0).replace(part, truth)

    text = _STATED_PART_RE.sub(_fix_part, text)

    for pattern, reason in _PHANTOM_PERCEPTION_PATTERNS:
        repaired, count = pattern.subn("", text)
        if count:
            corrections.append(reason)
            text = repaired

    if corrections:
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\s+([.,;!?])", r"\1", text)
        text = re.sub(r"(?m)^\s+", "", text).strip()
        try:
            record_degradation(
                "grounded_claim_guard",
                RuntimeError("; ".join(corrections)[:300]),
                severity="warning",
                action="reconciled a user-facing claim against a real runtime reading",
            )
        except _RECOVERABLE:
            pass

    return GroundedReply(text=text, corrections=tuple(corrections))


__all__ = ["GroundedReply", "verify_grounded_claims"]
