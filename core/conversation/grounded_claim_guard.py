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


# Describing an artifact that was never made.
#
# Live 2026-07-27: the 2048 build failed and she said so plainly. One turn
# later, asked whether it was playable, she described Minesweeper — "you click
# cells to reveal numbers; if you hit a mine, the game shows you which squares
# had mines and ends." There was no file. The receipts were in the prompt by
# then and lost to a fluent paragraph, which is what a prompt block does when
# it competes with plausibility.
#
# So the ledger decides, not the instruction. If the most recent attempt to
# build something DID NOT SUCCEED, a reply narrating how that artifact behaves
# is contradicted by a reading the runtime can actually take.
_ARTIFACT_BEHAVIOUR_RE = re.compile(
    r"\b(?:when you (?:run|open|start|launch|play)|"
    r"you(?:'ll| will) see|the (?:game|app|program|window|board) (?:opens|starts|shows|pops)|"
    r"to play(?: it)?,|"
    r"once (?:it(?:'s| is) )?(?:running|open|installed))\b",
    re.IGNORECASE,
)


# Claiming an action finished when nothing finished it.
#
# Live 2026-07-27: "I've found a beautiful image of a blue whale and set it as
# your desktop background. Enjoy!" The wallpaper never changed. The capability
# exists and is reachable — os_settings.set_wallpaper through computer_use's
# system_control — so this was not a missing feature. It was a completed-action
# claim with no completed action behind it.
#
# That is the same shape as describing a game she never built, and it deserves
# the same treatment: the ledger decides. A sentence in the past tense about
# having done something is checkable against whether anything was done.
#
# Deliberately narrow. Only first-person past-tense claims about consequential
# acts count — "I set", "I saved", "I created". Intent ("I'll set it"),
# capability ("I can set it") and refusal ("I couldn't set it") are all
# untouched, because none of them asserts that the world changed.
# The verb rarely sits next to the pronoun. "I've found a beautiful image and
# set it as your desktop background" is one first-person sentence with the
# action six words from the "I've", so the two halves are matched separately.
_FIRST_PERSON_RE = re.compile(r"\b(?:I|I'(?:ve|m|ll)|we|we'(?:ve|ll))\b", re.IGNORECASE)
_ACTION_VERB_RE = re.compile(
    r"\b(?:set|saved|created|written|wrote|made|placed|put|changed|updated|"
    r"exported|downloaded|installed|deleted|moved|renamed|sent|posted|added|"
    r"generated|built|opened|launched)\b",
    re.IGNORECASE,
)
_COMPLETED_STATE_RE = re.compile(
    r"\b(?:it(?:'s| is)|that(?:'s| is))\s+(?:now\s+)?"
    r"(?:set|saved|created|done|written|exported|installed|ready|there|on your)\b",
    re.IGNORECASE,
)


def _claims_an_action_completed(text: str) -> bool:
    """A sentence asserting the world already changed, not that it might."""
    if _NOT_A_COMPLETION_RE.search(text):
        return False
    if _COMPLETED_STATE_RE.search(text):
        return True
    return bool(_FIRST_PERSON_RE.search(text) and _ACTION_VERB_RE.search(text))
# Words that make a sentence about the future or the hypothetical rather than
# the past. A claim is only a claim when it says the thing already happened.
_NOT_A_COMPLETION_RE = re.compile(
    r"\b(?:I(?:'ll| will| can| could| would| should| might)|going to|about to|"
    r"trying to|couldn(?:'t| not)|didn(?:'t| not)|wasn(?:'t| not)|"
    r"unable to|failed to|not going to)\b",
    re.IGNORECASE,
)


def _last_action_succeeded() -> tuple[bool, str]:
    """Did the most recent consequential attempt succeed? From the ledger.

    Returns (claim_is_unsupported, what_was_attempted).
    """
    # THIS turn's receipts outrank any older ledger entry.
    #
    # The guard read only the intention loop, so a turn that genuinely ran a
    # tool — the desktop search lane collects web evidence directly, outside
    # the loop — had no supporting entry here, and a true sentence was
    # rewritten into a confession: "I said that as though it were done, and it
    # isn't." Two records of what happened, disagreeing, and the one without
    # the evidence won. A receipt from this turn is the closer record.
    try:
        from core.conversation.surface_disposition import turn_tool_receipts

        if any(bool(receipt.get("ok")) for receipt in turn_tool_receipts()):
            return False, ""
    except _RECOVERABLE:
        pass

    try:
        from core.agency.intention_loop import get_intention_loop

        completed = list(
            getattr(get_intention_loop(), "_completed_intentions", None) or []
        )
    except _RECOVERABLE:
        return False, ""
    if not completed:
        # No ledger entries at all is not evidence of failure — the loop may
        # not be running. Silence is not a refutation.
        return False, ""
    for record in reversed(completed):
        actions = list(getattr(record, "actions_taken", None) or [])
        if not actions:
            continue
        succeeded = all(bool(getattr(action, "success", False)) for action in actions)
        return (not succeeded), str(getattr(record, "intention", ""))[:80]
    return False, ""


def _last_build_failed() -> tuple[bool, str]:
    """Did the most recent build-shaped attempt fail? From the ledger."""
    try:
        from core.agency.intention_loop import get_intention_loop

        completed = list(
            getattr(get_intention_loop(), "_completed_intentions", None) or []
        )
    except _RECOVERABLE:
        return False, ""
    build_words = ("build", "reconstruct", "create", "write", "make", "generate", "place")
    for record in reversed(completed):
        intention = str(getattr(record, "intention", "") or "").lower()
        if not any(word in intention for word in build_words):
            continue
        actions = list(getattr(record, "actions_taken", None) or [])
        succeeded = bool(actions) and all(
            bool(getattr(action, "success", False)) for action in actions
        )
        return (not succeeded), str(getattr(record, "intention", ""))[:80]
    return False, ""



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

    if _claims_an_action_completed(text):
        unsupported, attempted = _last_action_succeeded()
        if unsupported:
            corrections.append(
                f"claimed an action completed after {attempted or 'the attempt'} did not"
            )
            text = (
                "I said that as though it were done, and it isn't — the action "
                "didn't go through. I'd rather tell you that than let you find out."
            )
            return GroundedReply(text=text, corrections=tuple(corrections))

    if _ARTIFACT_BEHAVIOUR_RE.search(text):
        failed, what = _last_build_failed()
        if failed:
            corrections.append(
                f"described how an artifact behaves after {what or 'the build'} did not succeed"
            )
            text = (
                "I didn't actually get that built, so I can't tell you how it "
                "behaves — I'd only be describing something that isn't there."
            )

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
