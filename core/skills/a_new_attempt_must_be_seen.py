"""A fresh attempt needs observed activity before its ending can be counted."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from core.verify import invariant


@dataclass
class AttemptObservation:
    """A page reading uses wall-clock captures; reset delivery uses that clock too."""

    after_at: float
    activity_at: float | None = None

    def observe(self, at: Any, *, now: float, active: bool, ended: bool) -> None:
        if (not isinstance(at, (int, float)) or isinstance(at, bool) or not math.isfinite(at)
                or not math.isfinite(now) or not math.isfinite(self.after_at)
                or not self.after_at <= at <= now or not active or ended):
            return
        if self.activity_at is None:
            self.activity_at = at


def note_attempt_activity(reflexes: Any, observation: dict) -> None:
    """An old capture or another reading of an ending cannot establish new activity."""
    import time

    from core.language.how_a_game_ended import (
        says_a_round_is_over,
        says_play_is_ongoing,
        what_says_the_game_is_over,
    )

    receipt = getattr(reflexes, 'keep', {}).get('attempt_observation')
    if not isinstance(receipt, AttemptObservation) or observation.get('on_her_page') is not True:
        return
    words = str(observation.get('text') or '')
    ended = bool(says_a_round_is_over(words) or what_says_the_game_is_over(words))
    receipt.observe(observation.get('capture_at'), now=time.time(),
                    active=bool(getattr(reflexes, 'play_in_sight', False) or says_play_is_ongoing(words)), ended=ended)


def attempt_activity_was_seen(reflexes: Any) -> bool | None:
    """Completed activity records belong to this hand-over, never earlier runs' stretches."""
    receipt = getattr(reflexes, 'keep', {}).get('attempt_observation')
    if not isinstance(receipt, AttemptObservation):
        records = getattr(reflexes, 'stretches', ())
        return True if any((r.get('pictures') or 0) > 0 or (r.get('turns') or 0) > 0 for r in records) else None
    return receipt.activity_at is not None or any(
        (record.get('observations') or 0) > 0 and (record.get('responsive_pictures') or 0) > 0
        for record in getattr(reflexes, 'stretches', ())
    )


@invariant('agency.an_ending_after_reset_needs_new_activity', scope='agency',
           owner='core/skills/a_new_attempt_must_be_seen.py', observational=False)
def _new_attempt_invariant() -> tuple:
    from types import SimpleNamespace

    receipt = AttemptObservation(10.)
    reflexes = SimpleNamespace(keep={'attempt_observation': receipt}, stretches=[])
    for at, active, ended in ((9., True, False), (11., True, False), (10.1, False, False), (10.2, True, True)):
        receipt.observe(at, now=10.5, active=active, ended=ended)
        assert not attempt_activity_was_seen(reflexes)
    receipt.observe(10.3, now=10.5, active=True, ended=False)
    assert attempt_activity_was_seen(reflexes) and receipt.activity_at == 10.3
    return ()
