"""Losing visual continuity does not establish a different way to control a world."""
from __future__ import annotations

import asyncio
import time
from typing import Any

from core.agency.how_a_game_works import AIMS_FROM_A_BODY, FOLLOWS, STEERED, how_it_works
from core.verify import invariant


def body_control_is_still_expected(how: Any, *, timing_pays: bool) -> bool:
    """Keep established body control while missing pixels call for reacquisition.

    Offline 2026-10-13 a rotating body broke into new patches. Failure to
    identify it was treated as timing-only play, leaving controls neutral
    for nearly fifteen seconds. Timing needs its own role or payoff evidence.
    """
    if timing_pays or how is None:
        return False
    if how.me.known():
        return how.me.value in (STEERED, AIMS_FROM_A_BODY, FOLLOWS)
    return bool(how.body_keys_from_code)


async def press_in_time(hands: Any, run: Any, moves: Any, at: float, say: Any) -> bool:
    """Time a press where timing has evidence, while missing body control is reacquired."""
    from core.agency.playing_as_it_happens import PRESSED_FOR_S, RESPONSE_S, _say

    if body_control_is_still_expected(getattr(run, 'how', None), timing_pays=run.timing.pays()):
        return False
    ahead = run.control_timing.ahead(at, time.monotonic()) or 0.
    if not run.control_timing.deliveries:
        ahead += RESPONSE_S
    if run.building_up and not run.timing.pays():
        from core.agency.building_up import drop_now

        if not drop_now(moves.things.values(), ahead):
            return True
    elif not run.timing.press_now(moves.things.values(), at, ahead):
        return run.timing.has_something_to_time(moves.things.values())
    run.timing.credit(run.gains, run.losses)
    if not run.keys:
        _say(run, say, "Nothing here follows the mouse; I'm timing each click to where the moving thing is, and keeping "
             "to the places where a click has paid.", at, once="timing")
        began = time.monotonic()
        await hands.click(0.5, 0.5)
        run.control_timing.delivered(at, began, time.monotonic())
        run.last_click, run.clicks = at, getattr(run, 'clicks', 0) + 1
    else:
        key = run.keys[0]
        _say(run, say, f"Nothing here moves when I press {key}; I'm timing each press to where the moving thing is, "
             "and keeping to the places where a press has paid.", at, once="timing")
        began = time.monotonic()
        await hands.down(key)
        run.control_timing.delivered(at, began, time.monotonic())
        run.input_key_downs[key] += 1
        await asyncio.sleep(PRESSED_FOR_S)
        await hands.up(key)
    run.timing.pressed(moves.things.values(), at, ahead)
    return True


@invariant('agency.missing_body_observations_keep_the_control_role', scope='agency',
           owner='core/agency/a_body_still_expected_when_it_is_unseen.py', observational=False)
def _expected_control_invariant() -> tuple:
    how = how_it_works('Use the arrow keys to guide your craft onto the platform.')
    answer = how.me
    assert body_control_is_still_expected(how, timing_pays=False)
    assert how.me is answer and how.me.value == STEERED
    assert not body_control_is_still_expected(how, timing_pays=True)
    assert not body_control_is_still_expected(how_it_works(''), timing_pays=False)
    return ()
