"""A showing is done again whatever it looks like and whatever does it again (tests/showings_support.py).

The same way of playing, core/agency/doing_again_what_was_shown.py:follow, against one family of screens drawn many
ways: arrows set round a middle, a row, a grid; places lit brighter, darker or in another colour; done again by the
arrow keys, by number keys in an order nothing on the screen says (learned from what each press lights,
core/agency/what_a_press_lights.py), or by clicks; with the pressed place lit back or not; and from a watch begun after
the showing was over.
"""
from __future__ import annotations

import asyncio

import pytest

from core.agency.doing_again_what_was_shown import follow
from core.agency.what_a_press_lights import WhatAPressLights
from tests.showings_support import AShowing

pytestmark = pytest.mark.unit


def _played(screen: AShowing, *, rounds: int = 8, clicks: bool = False, before: float = 0.0) -> AShowing:
    screen.t -= before

    async def run() -> int:
        if before:
            # Looked at only once the first showing is over.
            while screen.t < 0:
                await screen.look()
        return await follow(screen.look, screen, WhatAPressLights(), keys=screen.keys_a_player_is_told(), clicks=clicks,
                            rounds_at_most=rounds, watch_s=8.0)

    asyncio.run(run())
    return screen


@pytest.mark.parametrize("lit", ["brighter", "darker", "recolour"])
def test_a_pad_of_arrows_lit_in_turn_is_done_again_by_the_arrow_keys(lit):
    screen = _played(AShowing(layout="pad", lit=lit, answered_by="arrows"))
    assert screen.score >= 6 and screen.lives == 3, (screen.score, screen.lives, screen.spell)


@pytest.mark.parametrize("layout", ["row", "grid"])
def test_places_lit_in_turn_are_clicked_where_the_place_is_worked_by_the_pointer(layout):
    screen = _played(AShowing(layout=layout, answered_by="clicks"), clicks=True)
    assert screen.score >= 6 and screen.lives == 3, (screen.score, screen.lives, screen.spell)


def test_number_keys_in_an_order_nothing_shows_are_learned_from_what_each_press_lights():
    """Squares in a row, done again by keys 1 to 4 in a hidden order: a press of a key not yet known is tried where she
    knows none, and the place it lights is what that key is for from then on."""
    screen = _played(AShowing(layout="row", answered_by="numbers", seed=3), rounds=14)
    assert screen.score >= 4, (screen.score, screen.lives, screen.keys)
    assert screen.lives >= 0


def test_a_showing_missed_is_had_again_by_a_press():
    """Looked at only after the first showing was over: a press has it shown again, and it is watched from its start."""
    screen = _played(AShowing(layout="pad", answered_by="arrows"), before=1.5)
    assert screen.score >= 5, (screen.score, screen.lives, screen.spell)
    assert screen.lives >= 2


def test_with_nothing_lit_back_the_arrows_still_say_their_keys():
    screen = _played(AShowing(layout="pad", answered_by="arrows", echo=False))
    assert screen.score >= 6 and screen.lives == 3, (screen.score, screen.lives)


def test_the_words_changing_are_no_place_in_the_order():
    screen = _played(AShowing(layout="pad", answered_by="arrows", words=True, lit="brighter"))
    assert screen.lives == 3, (screen.score, screen.lives)
