"""Choose inputs by solving a known transition against an explicit outcome.

Source can supply a transition instead of making an agent discover it by
trial. The same search also accepts a measured transition. The caller owns
state observation, model validity, constraints, and the meaning of success.
This module returns predicted plans; it never declares an observed win.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any

from core.verify import invariant


@dataclass(frozen=True)
class RequiredStatePlan:
    actions: tuple[Hashable, ...]
    states: tuple[Any, ...]
    reached: bool
    transitions: int
    complete: bool


@dataclass(frozen=True)
class _Candidate:
    state: Any
    actions: tuple[Hashable, ...]
    states: tuple[Any, ...]
    cost: float


def solve_for_a_required_state(state: Any, actions: Sequence[Hashable], *,
                               advance: Callable[[Any, Hashable], Any],
                               outcome: Callable[[Any], bool | None],
                               cost: Callable[[Any], float],
                               identity: Callable[[Any], Hashable],
                               steps: int, width: int = 32, budget: int = 4096) -> RequiredStatePlan | None:
    """Search finite input paths, preserving hard failures before their recovery.

    ``outcome`` returns True at an achieved state, False while a path remains
    viable, and None at a failure or unavailable state. ``cost`` ranks only
    viable paths; a low cost can never override failure. ``identity`` must
    retain every state variable that can change a future transition.
    Exhaustion yields an explicitly incomplete prediction, not success.
    """
    if (not actions or any(type(v) is not int or v < 1 for v in (steps, width, budget))
            or steps > 1024 or width > 512 or budget > 100000 or len(actions) > 32):
        return None
    initial = outcome(state)
    if type(initial) is not bool:
        return None
    if initial is True:
        return RequiredStatePlan((), (state,), True, 0, True)
    initial_cost = float(cost(state))
    if not math.isfinite(initial_cost):
        return None
    frontier = [_Candidate(state, (), (state,), initial_cost)]
    count, pruned = 0, False
    best = None
    for _ in range(steps):
        next_states = {}
        for parent in frontier:
            for action in actions:
                if count >= budget:
                    return _plan(best, count, False)
                count += 1
                following = advance(parent.state, action)
                verdict = outcome(following)
                if type(verdict) is not bool:
                    continue
                trail = parent.actions + (action,)
                path = parent.states + (following,)
                if verdict is True:
                    return RequiredStatePlan(trail, path, True, count, not pruned)
                rank = float(cost(following))
                if not math.isfinite(rank):
                    continue
                candidate = _Candidate(following, trail, path, rank)
                name = identity(following)
                previous = next_states.get(name)
                if previous is None or candidate.cost < previous.cost:
                    next_states[name] = candidate
                if best is None or candidate.cost < best.cost:
                    best = candidate
        if not next_states:
            return _plan(best, count, not pruned)
        ordered = sorted(next_states.values(), key=lambda candidate: candidate.cost)
        pruned |= len(ordered) > width
        frontier = ordered[:width]
    return _plan(best, count, not pruned)


def _plan(candidate: _Candidate | None, count: int, complete: bool) -> RequiredStatePlan | None:
    if candidate is None:
        return None
    return RequiredStatePlan(candidate.actions, candidate.states, False, count, complete)


@invariant("agency.a_required_state_cannot_be_reached_after_failure", scope="agency",
           owner="core/agency/solving_for_a_required_state.py", observational=False)
def _required_state_invariant() -> tuple:
    transition = {0: {"direct": -1, "around": 1}, 1: {"direct": 2, "around": 1},
                  -1: {"direct": 2, "around": 2}, 2: {"direct": 2, "around": 2}}
    plan = solve_for_a_required_state(0, ("direct", "around"), advance=lambda s, a: transition[s][a],
                                     outcome=lambda s: None if s == -1 else s == 2,
                                     cost=lambda s: abs(2 - s), identity=lambda s: s, steps=3)
    assert plan is not None and plan.reached and plan.actions == ("around", "direct")
    assert -1 not in plan.states
    return ()
