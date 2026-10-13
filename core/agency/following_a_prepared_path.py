"""Keep planning work while checking its remaining path against each new observation.

Preparation can search far ahead. Feedback only replays a retained path,
without expanding alternative actions. A single owner supplies a current
state, a model identity covering rules, goal and controlled body, and a
receipt after each complete action. Visual state and model validity remain
the caller's responsibility. Predicted arrival never declares an observed win.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any

from core.agency.solving_for_a_required_state import RequiredStatePlan
from core.verify import invariant


@dataclass(frozen=True)
class PreparedInput:
    action: Hashable
    observed_at: float
    chosen_at: float
    expires_at: float
    transitions: int


@dataclass(frozen=True)
class StartedInput:
    offered: PreparedInput
    began: float
    until: float
    complete_by: float


class FollowingPreparedPath:
    """Offer one input only after the remaining path still reaches its required state.

    An action means the full transition supplied to the planner, including
    its duration. A dispatch acknowledgement alone cannot finish that action.
    The executor checks expiry before acting and supplies its completion
    receipt. The next observation must be newer than that completion.
    """

    def __init__(self, plan: RequiredStatePlan, model_identity: Hashable):
        if (plan.reached is not True or model_identity is None or len(plan.actions) > 1024
                or len(plan.states) != len(plan.actions) + 1):
            raise ValueError("feedback needs a prepared path to the required state")
        self.plan: RequiredStatePlan | None = plan
        self.model_identity = model_identity
        self.cursor = 0
        self.status = "needs_observation"
        self._observed_at = -math.inf
        self._finished_at = -math.inf
        self._offered: PreparedInput | None = None
        self._started: StartedInput | None = None

    def feedback(self, state: Any, *, model_identity: Hashable, observed_at: float,
                 now: float, max_age: float, advance: Callable[[Any, Hashable], Any],
                 outcome: Callable[[Any], bool | None], budget: int = 1024) -> PreparedInput | None:
        """Replay only a bounded suffix from fresh state; changed constraints can refuse it."""
        if self.plan is None:
            return None
        if model_identity != self.model_identity:
            self._discard()
            return None
        if self._offered is not None:
            self.status = "waiting_for_input"
            return None
        if (not all(math.isfinite(v) for v in (observed_at, now, max_age)) or max_age <= 0
                or not 0 <= now - observed_at <= max_age or observed_at <= self._observed_at
                or observed_at <= self._finished_at):
            self.status = "needs_observation"
            return None
        self._observed_at = observed_at
        verdict = outcome(state)
        if type(verdict) is not bool:
            self._discard()
            return None
        if verdict is True:
            self._discard()
            self.status = "observed_goal"
            return None
        remaining = self.plan.actions[self.cursor:]
        if not remaining:
            self._discard()
            return None
        if type(budget) is not int or not 1 <= budget <= 1024 or len(remaining) > budget:
            self.status = "needs_preparation"
            return None
        for transitions, action in enumerate(remaining, 1):
            state = advance(state, action)
            verdict = outcome(state)
            if type(verdict) is not bool:
                self._discard()
                return None
            if verdict is True:
                expiry = observed_at + max_age
                if not math.isfinite(expiry):
                    self.status = "needs_observation"
                    return None
                self._offered = PreparedInput(remaining[0], observed_at, now, expiry, transitions)
                self.status = "predicted_path"
                return self._offered
        self._discard()
        return None

    def is_current(self, offered: PreparedInput | None, at: float) -> bool:
        """The executor checks the same offer and its observation deadline before acting."""
        return (self._offered is not None and offered is self._offered and math.isfinite(at)
                and offered.chosen_at <= at <= offered.expires_at)

    def finished(self, offered: PreparedInput, *, at: float, succeeded: bool) -> bool:
        """Consume exactly one action on its own timely completion receipt."""
        if self._offered is None or offered is not self._offered:
            return False
        started = self._started
        timely = (math.isfinite(at) and started.until <= at <= started.complete_by
                  if started is not None else self.is_current(offered, at))
        if succeeded is not True or not timely:
            self._discard()
            return False
        self.cursor += 1
        self._finished_at = at
        self._offered = None
        self._started = None
        self.status = "needs_observation"
        return True

    def started(self, offered: PreparedInput, *, at: float, duration: float, slack: float = 0.) -> StartedInput | None:
        """A fresh dispatch acknowledgement begins a separate physical hold window."""
        if self._offered is None or offered is not self._offered or self._started is not None:
            return None
        if (not self.is_current(offered, at) or any(isinstance(v, bool) for v in (at, duration, slack))
                or not all(math.isfinite(v) for v in (duration, slack)) or duration <= 0 or slack < 0
                or not math.isfinite(at + duration + slack) or at + duration <= at):
            self._discard()
            return None
        self._started = StartedInput(offered, at, at + duration, at + duration + slack)
        self.status = "input_started"
        return self._started

    def invalidate(self) -> None:
        """An executor that could not complete its action retires the remaining path."""
        self._discard()

    def _discard(self) -> None:
        self.plan = None
        self._offered = None
        self._started = None
        self.status = "needs_preparation"


@invariant("agency.prepared_paths_need_current_feedback_and_input_receipts", scope="agency",
           owner="core/agency/following_a_prepared_path.py", observational=False)
def _prepared_path_invariant() -> tuple:
    plan = RequiredStatePlan((1, 1), (0, 1, 2), True, 2, True)
    follower = FollowingPreparedPath(plan, "same rules and body")
    options = dict(model_identity="same rules and body", advance=lambda s, a: s + a,
                   outcome=lambda s: s == 2, max_age=.5)
    offered = follower.feedback(0, observed_at=1., now=1.1, **options)
    assert offered is not None and follower.cursor == 0
    assert follower.finished(offered, at=1.2, succeeded=True) and follower.cursor == 1
    assert not follower.finished(offered, at=1.3, succeeded=True)
    assert follower.feedback(1, observed_at=1.1, now=1.3, **options) is None
    assert follower.feedback(1, observed_at=1.4, now=1.4, **options) is not None
    return ()
