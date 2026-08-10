"""Noticing that she has stopped getting anywhere.

Two narrow guards already existed. `environment/action_gateway` suppresses an
action that has failed twice with the same context, and `perception/action_gateway`
blocks one that recently failed with high surprise. Both catch the same single
shape — *repeated failure* — inside a single gateway, and both answer only by
vetoing the next action.

Four other shapes were invisible:

* the same action **succeeding** identically forever (`ls` in a loop returns
  0 and looks healthy every time),
* **oscillation** — edit, test, edit, test, edit, test, with neither ever
  changing anything,
* **monologue** — consecutive messages with no tool call and no new input,
* repeated **context-window overflow**, which is a memory-management failure
  wearing a model error's clothes.

None of these are failures at the step level. Every individual step succeeds.
That is exactly why nothing caught them: a per-action gate cannot see a pattern
that only exists across steps.

**The false positive this is designed around.** The prior art (OpenHands, MIT)
shipped loop detection and then had to fix it, because it killed agents that
were legitimately polling a long-running process — a build, a test run, a
deploy. Waiting *is* repeating, and it is also correct. So `kind="wait"` steps
are excluded from repeat detection outright, and any step carrying a
``progress_marker`` that has changed is not a repeat however identical the rest
of it looks. Learning the lesson from someone else's incident is the entire
point of reading their code; shipping their bug first would waste it.

**Recovery is a ladder, not a switch.** Their other fix was replacing a hard
error state — from which the agent could not be talked back — with a graceful
transition. A detector that can only halt turns a recoverable rut into a dead
session, so the verdict names the pattern and the evidence, and the remedy
escalates: nudge, then force a change of strategy, then ask the human.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable, Sequence

logger = logging.getLogger("Aura.StuckDetector")

__all__ = [
    "StuckPattern",
    "AgentStep",
    "StuckVerdict",
    "Remedy",
    "StuckDetector",
]

#: Volatile substrings stripped before comparison. A tool result that differs
#: only by a timestamp or an object id is the same result.
_VOLATILE = re.compile(
    r"0x[0-9a-fA-F]{6,}"                      # object ids
    r"|\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*"  # ISO timestamps
    r"|\b\d{10,}\b"                           # epoch-ish integers
    r"|\bpid[= ]\d+",                          # pids
)


class StuckPattern(StrEnum):
    REPEATED_ACTION_OBSERVATION = "repeated_action_observation"
    REPEATED_ACTION_ERROR = "repeated_action_error"
    MONOLOGUE = "monologue"
    OSCILLATION = "oscillation"
    REPEATED_CONTEXT_OVERFLOW = "repeated_context_overflow"


class Remedy(StrEnum):
    """Escalating responses. A detector that can only halt kills a session."""

    NONE = "none"
    NUDGE = "nudge"                    # tell her she is repeating; let her adjust
    FORCE_NEW_STRATEGY = "force_new_strategy"  # constrain away the repeated move
    ASK_HUMAN = "ask_human"            # surface it; she is not getting out alone


@dataclass(frozen=True)
class AgentStep:
    """One action and what came back.

    ``kind`` distinguishes a tool call from a message from a deliberate wait.
    ``progress_marker`` is any caller-supplied token that legitimately advances
    while the action stays identical — a build's line count, a queue depth, a
    poll's elapsed time. Its whole job is to keep honest waiting from reading
    as a loop.
    """

    action: str
    arguments: str = ""
    observation: str = ""
    is_error: bool = False
    kind: str = "tool"
    progress_marker: str | None = None

    @property
    def is_wait(self) -> bool:
        return self.kind == "wait"

    def signature(self) -> tuple[str, str]:
        """What the agent *did*, ignoring volatile detail."""
        return (self.action.strip(), _normalize(self.arguments))

    def outcome(self) -> str:
        """What came *back*, ignoring volatile detail."""
        return _normalize(self.observation)

    def fingerprint(self) -> tuple[str, str, str, str | None]:
        """Everything that decides whether two steps are 'the same'.

        The progress marker is part of the fingerprint on purpose: two polls of
        a running build are only the same step if the build has not moved.
        """
        action, arguments = self.signature()
        return (action, arguments, self.outcome(), self.progress_marker)


def _normalize(text: str) -> str:
    return _VOLATILE.sub("<v>", (text or "").strip())


@dataclass(frozen=True)
class StuckVerdict:
    """The finding, with the evidence attached.

    A bare "you are stuck" cannot be acted on — not by the model, not by a
    human reading a log. The pattern says what shape the rut is and the
    evidence says which steps proved it.
    """

    stuck: bool = False
    pattern: StuckPattern | None = None
    remedy: Remedy = Remedy.NONE
    evidence: tuple[str, ...] = ()
    detail: str = ""

    def __bool__(self) -> bool:
        return self.stuck

    def describe(self) -> str:
        if not self.stuck:
            return "not stuck"
        return f"{self.pattern}: {self.detail}"


class StuckDetector:
    """Rule-based detection of a run that has stopped going anywhere.

    Deliberately not a model call. This runs on every step, it has to be
    trustworthy while the thing it watches is misbehaving, and a detector that
    needs the cortex cannot report that the cortex is the problem.
    """

    def __init__(
        self,
        *,
        window: int = 24,
        repeat_threshold: int = 4,
        error_threshold: int = 3,
        monologue_threshold: int = 3,
        oscillation_cycles: int = 3,
    ) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        for name, value in (
            ("repeat_threshold", repeat_threshold),
            ("error_threshold", error_threshold),
            ("monologue_threshold", monologue_threshold),
            ("oscillation_cycles", oscillation_cycles),
        ):
            if value < 2:
                raise ValueError(f"{name} must be at least 2 to describe a repetition")
        self.window = window
        self.repeat_threshold = repeat_threshold
        self.error_threshold = error_threshold
        self.monologue_threshold = monologue_threshold
        self.oscillation_cycles = oscillation_cycles
        #: How many times a rut has already been called on this run, so the
        #: remedy can escalate rather than repeating a nudge that did not work.
        self._interventions = 0

    # -- public ------------------------------------------------------------

    def check(self, steps: Sequence[AgentStep]) -> StuckVerdict:
        """Look at the recent history and decide whether she is going in circles."""
        recent = list(steps)[-self.window:]
        if len(recent) < 2:
            return StuckVerdict()

        for detect in (
            self._context_overflow,
            self._repeated_error,
            self._repeated_observation,
            self._oscillation,
            self._monologue,
        ):
            verdict = detect(recent)
            if verdict.stuck:
                self._interventions += 1
                escalated = StuckVerdict(
                    stuck=True,
                    pattern=verdict.pattern,
                    remedy=self._remedy_for(verdict.pattern),
                    evidence=verdict.evidence,
                    detail=verdict.detail,
                )
                logger.info("stuck: %s", escalated.describe())
                return escalated

        return StuckVerdict()

    def reset(self) -> None:
        """Forget prior interventions. Call when real progress resumes."""
        self._interventions = 0

    @property
    def interventions(self) -> int:
        return self._interventions

    def _remedy_for(self, pattern: StuckPattern | None) -> Remedy:
        # Context overflow is never something she can talk her way out of: the
        # window is full and the next call fails the same way.
        if pattern is StuckPattern.REPEATED_CONTEXT_OVERFLOW:
            return Remedy.FORCE_NEW_STRATEGY if self._interventions < 2 else Remedy.ASK_HUMAN
        # One nudge, then escalate. Repeating a nudge that already failed to
        # change anything is itself a loop, which would be a poor look for the
        # loop detector. _interventions is >= 1 here: check() increments before
        # asking.
        if self._interventions == 1:
            return Remedy.NUDGE
        if self._interventions == 2:
            return Remedy.FORCE_NEW_STRATEGY
        return Remedy.ASK_HUMAN

    # -- patterns ----------------------------------------------------------

    @staticmethod
    def _actionable(steps: Iterable[AgentStep]) -> list[AgentStep]:
        """Steps that can meaningfully repeat.

        Waits are dropped here, once, rather than special-cased in each
        detector — polling a build is repetition and is also correct, and that
        distinction should live in one place.
        """
        return [s for s in steps if not s.is_wait]

    def _repeated_observation(self, steps: Sequence[AgentStep]) -> StuckVerdict:
        """The same action returning the same thing, over and over."""
        candidates = [s for s in self._actionable(steps) if not s.is_error]
        if len(candidates) < self.repeat_threshold:
            return StuckVerdict()

        counts = Counter(s.fingerprint() for s in candidates)
        fingerprint, count = counts.most_common(1)[0]
        if count < self.repeat_threshold:
            return StuckVerdict()

        return StuckVerdict(
            stuck=True,
            pattern=StuckPattern.REPEATED_ACTION_OBSERVATION,
            evidence=(fingerprint[0],),
            detail=(
                f"{fingerprint[0]!r} ran {count} times with an identical result "
                "and nothing changed"
            ),
        )

    def _repeated_error(self, steps: Sequence[AgentStep]) -> StuckVerdict:
        """The same action failing the same way."""
        candidates = [s for s in self._actionable(steps) if s.is_error]
        if len(candidates) < self.error_threshold:
            return StuckVerdict()

        counts = Counter(s.fingerprint() for s in candidates)
        fingerprint, count = counts.most_common(1)[0]
        if count < self.error_threshold:
            return StuckVerdict()

        return StuckVerdict(
            stuck=True,
            pattern=StuckPattern.REPEATED_ACTION_ERROR,
            evidence=(fingerprint[0],),
            detail=(
                f"{fingerprint[0]!r} failed {count} times with the same error; "
                "retrying it again will fail the same way"
            ),
        )

    def _oscillation(self, steps: Sequence[AgentStep]) -> StuckVerdict:
        """A, B, A, B — two moves that undo or ignore each other."""
        candidates = self._actionable(steps)
        needed = self.oscillation_cycles * 2
        if len(candidates) < needed:
            return StuckVerdict()

        tail = candidates[-needed:]
        evens = {s.fingerprint() for s in tail[0::2]}
        odds = {s.fingerprint() for s in tail[1::2]}
        if len(evens) != 1 or len(odds) != 1 or evens == odds:
            return StuckVerdict()

        first, second = next(iter(evens)), next(iter(odds))
        return StuckVerdict(
            stuck=True,
            pattern=StuckPattern.OSCILLATION,
            evidence=(first[0], second[0]),
            detail=(
                f"alternating between {first[0]!r} and {second[0]!r} for "
                f"{self.oscillation_cycles} cycles without either changing anything"
            ),
        )

    def _monologue(self, steps: Sequence[AgentStep]) -> StuckVerdict:
        """Consecutive messages with no action and no new input."""
        trailing = 0
        for step in reversed(steps):
            if step.kind != "message":
                break
            trailing += 1
        if trailing < self.monologue_threshold:
            return StuckVerdict()

        return StuckVerdict(
            stuck=True,
            pattern=StuckPattern.MONOLOGUE,
            evidence=tuple(s.action for s in steps[-trailing:]),
            detail=(
                f"{trailing} consecutive messages with no tool call and no new "
                "input — talking rather than working"
            ),
        )

    def _context_overflow(self, steps: Sequence[AgentStep]) -> StuckVerdict:
        """Repeated context-window errors.

        Checked first, and separately from ordinary repeated errors, because
        the remedy is different in kind: no amount of rephrasing helps, the
        window has to be made smaller.
        """
        overflows = [
            s for s in steps
            if s.is_error and "context" in s.observation.lower()
            and any(w in s.observation.lower() for w in ("window", "length", "token"))
        ]
        if len(overflows) < 2:
            return StuckVerdict()

        return StuckVerdict(
            stuck=True,
            pattern=StuckPattern.REPEATED_CONTEXT_OVERFLOW,
            evidence=tuple(dict.fromkeys(s.action for s in overflows)),
            detail=(
                f"{len(overflows)} context-window errors; the window must be "
                "condensed, not the request retried"
            ),
        )
