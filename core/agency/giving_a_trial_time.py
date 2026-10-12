"""Keep a control trial intact for the time it needs to answer.

Offline 2026-10-12 a turning trial was let go in the next picture: planning
resumed before any settled motion could show what the turn did. A trial asks
for one control for a measured time. A changed input or observation identity
ends it, and its time is bounded even when no answer comes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.verify import invariant


@dataclass(frozen=True)
class ControlTrial:
    """One delivered control and the observation interval belonging to it."""

    key: str
    began: float
    until: float
    epoch: int

    def holds(self, held: str, epoch: int, at: float) -> bool:
        """Whether this input still owns the current observation, within its finite time."""
        return (self.key == held and self.epoch == epoch
                and all(math.isfinite(v) for v in (self.began, self.until, at))
                and self.began <= at < self.until)


async def try_a_facing_control(hands: Any, run: Any, hers: Any, choosing: Any, at: float, *,
                              hold: Any, seconds: float, limit: int) -> bool:
    """Hold the requested trial between sightings of where the body faces, then try the next one."""
    trial = run.facing_trial
    if run.faces is None or choosing.mine is None or hers.follows_pointer:
        run.facing_trial = None
        return False
    epoch = hers.identification.epoch
    if trial is not None and run.held_trying and trial.holds(run.held, epoch, at):
        return True
    run.facing_trial = None
    if run.facing_trials >= limit or at - run.tried_at < seconds:
        return False
    key = run.faces.trial(run.keys)
    if key is None:
        return False
    run.facing_trials += 1
    await hold(hands, run, hers, key, at, trying=True)
    run.tried_at = run.held_since
    run.facing_trial = ControlTrial(key, run.held_since, max(at, run.held_since) + seconds, epoch)
    return True


@invariant("agency.control_trials_keep_their_measurement_time", scope="agency",
           owner="core/agency/giving_a_trial_time.py", observational=False)
def _control_trial_time_invariant() -> tuple:
    trial = ControlTrial("a", 1.0, 1.5, 3)
    assert trial.holds("a", 3, 1.25)
    assert not trial.holds("b", 3, 1.25)
    assert not trial.holds("a", 4, 1.25)
    assert not trial.holds("a", 3, 1.5)
    assert not ControlTrial("a", 1, math.inf, 3).holds("a", 3, 1.25)
    return ()
