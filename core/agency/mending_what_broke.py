"""Mending what broke: a machine she built is tested, and where it stopped is the join to mend.

The player in a video of a trap-building game put a device at the end of the last one's arrow, turned it toward the
cage, added the next, tested the whole, and mended the link that broke: they watched where the action stopped and went
to that part, not back to the start. The machine builders that turned a failed test into the right edit were the ones
told where their machine had got to (BesiegeField, 2025); the others rebuilt from nothing.

Here a test of what was made (an act whose words try it, core/cognition/reading_the_rules.py TRIES_IT) is watched as it
plays out (core/perception/watching_it_happen.py where_it_came_to_rest), and the part she placed nearest to where it
came to rest is the join that broke. That is said once, given to her plan as why it is made again
(core/cognition/a_plan_to_an_end.py), and made the part to select and turn (core/agency/aiming_what_was_placed.py)
before anything new is added. Nothing here knows what the machine is.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from typing import Any

__all__ = ["a_click_to_mend", "heard_the_test", "the_joint_that_broke"]

#: How far from where a test came to rest a placed part may be, in its own sizes, and still be where it stopped.
WITHIN_ITS_SIZES = 2.0
_AT = re.compile(r"(\d+(?:\.\d+)?)% across, (\d+(?:\.\d+)?)% down")


def the_joint_that_broke(came_to_rest_at: tuple[float, float] | None,
                         placed: Sequence[tuple[float, float, float, float]]) -> tuple[float, float, float] | None:
    """The middle of the placed part nearest where a test came to rest, and its size, where it is near enough to be
    where it stopped; None where nothing placed is."""
    if came_to_rest_at is None or not placed:
        return None
    x, y = came_to_rest_at
    best, gap = None, math.inf
    for left, top, wide, tall in placed:
        middle = (left + wide / 2, top + tall / 2)
        apart = math.hypot(middle[0] - x, middle[1] - y)
        if apart <= WITHIN_ITS_SIZES * max(wide, tall) and apart < gap:
            best, gap = (*middle, max(wide, tall)), apart
    return best


def _placed(can_do: Any) -> list[tuple[float, float, float, float]]:
    from core.runtime.skill_contract import PredicateState

    states = getattr(can_do, "placement_states", {}) or {}
    return [tuple(receipt.effect_bbox) for attempt_id, receipt in (getattr(can_do, "placement_receipts", {}) or {}).items()
            if states.get(attempt_id) is PredicateState.SATISFIED and getattr(receipt, "effect_bbox", None)]


def heard_the_test(guide: Any, can_do: Any, observation: dict[str, Any], last_act: str,
                   tell: Callable[[str], Any] | None = None) -> tuple[float, float, float] | None:
    """After a test of what she built: the join that broke, kept on the guide as the part to mend and given to her plan;
    None where the last act was no test or nothing placed was where it stopped."""
    from core.agency.what_i_can_do_here import what_is_clicked
    from core.cognition.reading_the_rules import tries_what_was_made

    if guide is None or not tries_what_was_made(what_is_clicked(last_act) or ""):
        return None
    joint = the_joint_that_broke(observation.get("came_to_rest_at"), _placed(can_do))
    if joint is None or joint == getattr(guide, "broke_at", None):
        return joint
    guide.broke_at = joint
    can_do.turned_since_placed = 0
    said = f"the test got as far as the part at {joint[0]:.0%} across and {joint[1]:.0%} down, and stopped there"
    from core.cognition.a_plan_to_an_end import plan_again

    plan_again(guide, said)
    if tell is not None:
        tell(f"{said[0].upper()}{said[1:]}: that's the join to mend.")
    return joint


def a_click_to_mend(guide: Any, offered: Sequence[str]) -> str:
    """The offered click on the part at the join that broke, to select it before turning it; "" where there is none."""
    from core.agency.what_i_can_do_here import what_is_clicked

    joint = getattr(guide, "broke_at", None)
    if joint is None:
        return ""
    best, gap = "", math.inf
    x0, y0, size = joint
    for move in offered:
        found = _AT.search(what_is_clicked(move) or "")
        if found is None:
            continue
        x, y = float(found.group(1)) / 100, float(found.group(2)) / 100
        apart = math.hypot(x - x0, y - y0)
        if apart < gap:
            best, gap = move, apart
    return best if gap <= size else ""
