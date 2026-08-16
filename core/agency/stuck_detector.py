"""Loop detection over actions and observations, reported as a no-change impasse.

What Aura had
-------------
One repetition guard, in ``core/orchestrator/mixins/message_handling.py``. It
compares the *text* of consecutive replies, and when three come back more than
eighty percent alike it appends a paragraph to the prompt telling the model it
must try a different approach.

That guard has two problems and the second is the serious one.

It watches the wrong channel. An agent stuck in a tool loop — reading the same
file, running the same failing command, alternating between two calls forever —
produces different prose every turn while doing exactly the same thing. Nothing
was watching what it *did*.

And its response is an instruction. Whether the loop breaks depends on whether
the model complies with a paragraph, which is the same faculty that produced the
loop. A detector whose only actuator is a request to the thing that is stuck is
not a control.

So this watches actions and observations, and its output is a typed verdict the
caller must handle, not text appended to a context window.

Why it reports an impasse
-------------------------
OpenHands calls this a stuck detector and enumerates repeated action-observation
cycles, repeated action-error cycles, monologue, and alternating patterns. Soar
already named the same condition thirty years earlier and named it better: a
:attr:`~core.cognition.impasse.ImpasseType.NO_CHANGE` impasse is "something was
chosen and applying it changed nothing". That is the definition of being stuck.

Making it an impasse is not a relabelling. It puts loop detection into machinery
that already exists here: :class:`~core.cognition.impasse.ImpasseLearner` counts
impasses by type, so the loop rate becomes a reportable diagnostic beside every
other kind of deadlock, and a resolution that gets a loop unstuck can be chunked
and reused the next time the same situation arises. A bespoke counter would have
had none of that.

The fifth pattern
-----------------
:attr:`StuckPattern.NO_PROGRESS` has no OpenHands equivalent and is the one worth
having. The four repetition patterns all ask whether the *agent* is repeating
itself. This one asks whether the *world* is moving: several different actions,
each observed to leave the state exactly as it was. An agent trying a new thing
every turn and changing nothing looks busy under every other check here, and it
is the more expensive failure because nothing about it looks wrong.

Thresholds
----------
None of the counts below are tuned. Each is the smallest number that separates
the failure from the legitimate behaviour it resembles, and each says which
behaviour that is where it is defined.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.cognition.impasse import Impasse, ImpasseType, situation_signature

__all__ = [
    "StuckPattern",
    "AgentStep",
    "StuckVerdict",
    "StuckDetector",
    "digest_of",
]

#: Repeats of an identical action-and-outcome before it counts as a loop.
#:
#: Three occurrences, because three is the first count containing *two*
#: consecutive repeats of the same transition. Two occurrences is one retry, and
#: retrying once after a transient failure is correct behaviour that a detector
#: must not punish.
_REPEAT_THRESHOLD = 3

#: Steps of an A-B-A-B alternation before it counts as a loop.
#:
#: Four, which is two complete cycles. Three steps of A-B-A is an ordinary
#: sequence — read, edit, read — and calling it a loop would fire on the most
#: common shape in normal tool use.
_ALTERNATION_THRESHOLD = 4

#: Consecutive agent turns with no action at all.
#:
#: Three, for the same reason as the repeat threshold: one turn of thinking
#: aloud before acting is normal, two is a plan, three with nothing done is a
#: monologue.
_MONOLOGUE_THRESHOLD = 3

#: Steps over which the observation never changes, with more than one distinct
#: action attempted, before the world is called unmoved.
#:
#: Three again, and the added condition that the actions differ is what makes it
#: a different finding from plain repetition: the agent varied its behaviour and
#: the state still did not move.
_NO_PROGRESS_THRESHOLD = 3

#: Most recent steps considered. A window rather than the whole history because
#: an agent that looped, recovered, and moved on is not stuck now, and a
#: detector with unbounded memory would keep reporting a resolved loop.
DEFAULT_WINDOW = 20


class StuckPattern(StrEnum):
    """What kind of not-getting-anywhere this is."""

    #: The same action producing the same observation, repeatedly.
    REPEATED_ACTION_OBSERVATION = "repeated_action_observation"
    #: The same action failing the same way, repeatedly.
    REPEATED_ACTION_ERROR = "repeated_action_error"
    #: Consecutive turns that took no action at all.
    MONOLOGUE = "monologue"
    #: Two actions alternating without progress.
    ALTERNATING = "alternating"
    #: Different actions, unchanged world.
    NO_PROGRESS = "no_progress"


def digest_of(value: Any) -> str:
    """A stable short digest for arguments or an observation.

    Sorted keys, and a repr fallback for anything JSON cannot hold, so two calls
    that differ only in dict ordering are recognised as the same call. Without
    the sort, ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` are different
    actions and no repetition is ever detected.
    """
    try:
        payload = json.dumps(value, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        payload = repr(value)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()


@dataclass(frozen=True)
class AgentStep:
    """One thing the agent did, and what came back.

    ``observation`` is digested rather than stored. The detector only ever asks
    whether two observations are the same, and keeping the payloads would mean a
    loop-detection window holding twenty tool outputs — including whatever they
    contained — for the lifetime of the loop.
    """

    action: str
    arguments_digest: str = ""
    observation_digest: str = ""
    failed: bool = False
    error_kind: str = ""

    @staticmethod
    def of(
        action: str,
        *,
        arguments: Any = None,
        observation: Any = None,
        failed: bool = False,
        error_kind: str = "",
    ) -> "AgentStep":
        return AgentStep(
            action=str(action),
            arguments_digest=digest_of(arguments),
            observation_digest=digest_of(observation),
            failed=bool(failed),
            error_kind=str(error_kind),
        )

    @property
    def call_key(self) -> str:
        """Identity of the call: what was done, with what."""
        return f"{self.action}#{self.arguments_digest}"

    @property
    def cycle_key(self) -> str:
        """Identity of the whole transition: the call and what it produced."""
        return f"{self.call_key}->{self.observation_digest}"

    @property
    def error_key(self) -> str:
        return f"{self.call_key}!{self.error_kind}"


@dataclass(frozen=True)
class StuckVerdict:
    """A detected loop, the evidence for it, and the impasse it constitutes."""

    pattern: StuckPattern
    detail: str
    repetitions: int
    actions: tuple[str, ...]
    impasse: Impasse

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern.value,
            "detail": self.detail,
            "repetitions": self.repetitions,
            "actions": list(self.actions),
            "impasse": self.impasse.type.value,
            "signature": self.impasse.signature,
        }


@dataclass
class StuckDetector:
    """Watches a stream of steps and says when it has stopped going anywhere.

    One detector per agent loop. :meth:`reset` is called when a new instruction
    arrives, because a user turn is what makes prior repetition irrelevant — the
    agent may legitimately be asked to do the same thing again.
    """

    scope: str = "agent_loop"
    window: int = DEFAULT_WINDOW
    _steps: deque[AgentStep] = field(default_factory=deque)
    _idle_turns: int = 0
    _reported: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.window < _ALTERNATION_THRESHOLD:
            raise ValueError(
                f"window must hold at least {_ALTERNATION_THRESHOLD} steps to detect alternation"
            )
        self._steps = deque(self._steps, maxlen=self.window)

    # -- recording -------------------------------------------------------

    def reset(self) -> None:
        """Forget the window. Called when the user says something new."""
        self._steps.clear()
        self._idle_turns = 0
        self._reported.clear()

    def observe(self, step: AgentStep) -> None:
        self._steps.append(step)
        self._idle_turns = 0

    def observe_idle_turn(self) -> None:
        """Record a turn that produced words and no action."""
        self._idle_turns += 1

    @property
    def steps(self) -> tuple[AgentStep, ...]:
        return tuple(self._steps)

    # -- detection -------------------------------------------------------

    def assess(self, *, context: Mapping[str, Any] | None = None) -> StuckVerdict | None:
        """The strongest loop currently visible, or None.

        Order matters. A repeated failure is checked before a repeated success
        because it is the more actionable finding: the caller can surface the
        error. Alternation is checked last because a two-cycle alternation is
        the weakest evidence here and would otherwise mask a straightforward
        repetition inside it.
        """
        ctx = dict(context or {})
        ctx.setdefault("scope", self.scope)

        for check in (
            self._check_repeated_error,
            self._check_repeated_cycle,
            self._check_no_progress,
            self._check_monologue,
            self._check_alternation,
        ):
            verdict = check(ctx)
            if verdict is not None:
                return verdict
        return None

    def assess_once(self, *, context: Mapping[str, Any] | None = None) -> StuckVerdict | None:
        """:meth:`assess`, but each distinct loop is reported only once.

        A caller acting on a verdict does not necessarily clear the window, so
        the same loop would be re-reported on every subsequent step and a single
        stuck episode would look like a dozen. The signature is what identifies
        an episode, so a genuinely new loop still reports.
        """
        verdict = self.assess(context=context)
        if verdict is None:
            return None
        key = f"{verdict.pattern.value}:{verdict.impasse.signature}"
        if key in self._reported:
            return None
        self._reported.add(key)
        return verdict

    def record_to_learner(self, verdict: StuckVerdict) -> None:
        """File the verdict with the process-wide impasse learner.

        Separate from detection so that a caller can assess without recording —
        a probe, a dry run, a test — and so the import stays out of the hot path
        for callers that only want the verdict.
        """
        from core.cognition.impasse import get_impasse_learner

        get_impasse_learner().record_impasse(verdict.impasse)

    # -- individual patterns ---------------------------------------------

    def _impasse(
        self, ctx: Mapping[str, Any], pattern: StuckPattern, actions: Sequence[str], detail: str
    ) -> Impasse:
        return Impasse(
            type=ImpasseType.NO_CHANGE,
            signature=situation_signature({**ctx, "pattern": pattern.value}, actions),
            candidates=tuple(sorted(set(actions))),
            detail=detail,
        )

    def _trailing_run(self, key: "callable[[AgentStep], str]") -> tuple[str, int]:
        """Length of the identical run ending at the most recent step."""
        if not self._steps:
            return "", 0
        steps = list(self._steps)
        target = key(steps[-1])
        count = 0
        for step in reversed(steps):
            if key(step) != target:
                break
            count += 1
        return target, count

    def _check_repeated_cycle(self, ctx: Mapping[str, Any]) -> StuckVerdict | None:
        _key, count = self._trailing_run(lambda s: s.cycle_key)
        if count < _REPEAT_THRESHOLD:
            return None
        last = self._steps[-1]
        detail = (
            f"{last.action} ran {count} times with identical arguments and returned "
            "an identical observation every time"
        )
        return StuckVerdict(
            pattern=StuckPattern.REPEATED_ACTION_OBSERVATION,
            detail=detail,
            repetitions=count,
            actions=(last.action,),
            impasse=self._impasse(
                ctx, StuckPattern.REPEATED_ACTION_OBSERVATION, [last.action], detail
            ),
        )

    def _check_repeated_error(self, ctx: Mapping[str, Any]) -> StuckVerdict | None:
        if not self._steps or not self._steps[-1].failed:
            return None
        steps = list(self._steps)
        target = steps[-1].error_key
        count = 0
        for step in reversed(steps):
            if not step.failed or step.error_key != target:
                break
            count += 1
        if count < _REPEAT_THRESHOLD:
            return None
        last = steps[-1]
        detail = (
            f"{last.action} failed {count} times in a row with the same arguments and the "
            f"same error ({last.error_kind or 'unclassified'})"
        )
        return StuckVerdict(
            pattern=StuckPattern.REPEATED_ACTION_ERROR,
            detail=detail,
            repetitions=count,
            actions=(last.action,),
            impasse=self._impasse(ctx, StuckPattern.REPEATED_ACTION_ERROR, [last.action], detail),
        )

    def _check_no_progress(self, ctx: Mapping[str, Any]) -> StuckVerdict | None:
        if len(self._steps) < _NO_PROGRESS_THRESHOLD:
            return None
        recent = list(self._steps)[-_NO_PROGRESS_THRESHOLD:]
        observations = {s.observation_digest for s in recent}
        if len(observations) != 1:
            return None
        calls = {s.call_key for s in recent}
        if len(calls) < 2:
            # Identical calls are plain repetition and the earlier check owns
            # that finding. This one is specifically about varied effort.
            return None
        actions = tuple(dict.fromkeys(s.action for s in recent))
        detail = (
            f"{len(calls)} different calls across {len(recent)} steps and the observed "
            "state did not change once"
        )
        return StuckVerdict(
            pattern=StuckPattern.NO_PROGRESS,
            detail=detail,
            repetitions=len(recent),
            actions=actions,
            impasse=self._impasse(ctx, StuckPattern.NO_PROGRESS, actions, detail),
        )

    def _check_monologue(self, ctx: Mapping[str, Any]) -> StuckVerdict | None:
        if self._idle_turns < _MONOLOGUE_THRESHOLD:
            return None
        detail = f"{self._idle_turns} consecutive turns produced no action"
        return StuckVerdict(
            pattern=StuckPattern.MONOLOGUE,
            detail=detail,
            repetitions=self._idle_turns,
            actions=(),
            impasse=self._impasse(ctx, StuckPattern.MONOLOGUE, ["<no action>"], detail),
        )

    def _check_alternation(self, ctx: Mapping[str, Any]) -> StuckVerdict | None:
        if len(self._steps) < _ALTERNATION_THRESHOLD:
            return None
        recent = list(self._steps)[-_ALTERNATION_THRESHOLD:]
        keys = [s.cycle_key for s in recent]
        if keys[0] == keys[1]:
            return None
        if keys[0::2] != [keys[0]] * len(keys[0::2]):
            return None
        if keys[1::2] != [keys[1]] * len(keys[1::2]):
            return None
        actions = tuple(dict.fromkeys(s.action for s in recent))
        detail = (
            f"{' and '.join(actions)} alternated for {len(recent)} steps, each returning "
            "what it returned the time before"
        )
        return StuckVerdict(
            pattern=StuckPattern.ALTERNATING,
            detail=detail,
            repetitions=len(recent) // 2,
            actions=actions,
            impasse=self._impasse(ctx, StuckPattern.ALTERNATING, actions, detail),
        )

    # -- reporting -------------------------------------------------------

    def report(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "window": self.window,
            "steps_held": len(self._steps),
            "idle_turns": self._idle_turns,
            "episodes_reported": len(self._reported),
        }


def steps_from(records: Iterable[Mapping[str, Any]]) -> list[AgentStep]:
    """Build steps from loosely-shaped call records.

    A convenience for callers holding dicts of tool calls rather than typed
    events. Records missing an action name are dropped rather than given a
    placeholder, because a run of placeholder-named steps would look exactly
    like a repetition loop.
    """
    steps: list[AgentStep] = []
    for record in records:
        name = str(record.get("action") or record.get("name") or record.get("tool") or "").strip()
        if not name:
            continue
        steps.append(
            AgentStep.of(
                name,
                arguments=record.get("arguments") or record.get("args") or record.get("params"),
                observation=record.get("observation")
                if "observation" in record
                else record.get("result") or record.get("output"),
                failed=bool(record.get("failed") or record.get("error")),
                error_kind=str(record.get("error_kind") or record.get("error") or "")[:80],
            )
        )
    return steps
