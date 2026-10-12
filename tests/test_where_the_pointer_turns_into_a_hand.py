"""Where the pointer turns into a hand is a place that answers a click, found by moving the mouse over the screen."""
from __future__ import annotations

import pytest

from core.perception.where_the_pointer_turns import ACROSS, DOWN, places_the_pointer_turns

pytestmark = pytest.mark.unit


def test_neighbouring_places_where_the_pointer_is_a_hand_are_one_place_at_their_middle():
    doorway = {(0, 3), (0, 4), (0, 5)}                    # the left edge of a room, three cells tall
    drawer = {(7, 6), (8, 6)}
    places = places_the_pointer_turns(doorway | drawer)
    assert len(places) == 2
    assert places[0] == pytest.approx((0.5 / ACROSS, 4.5 / DOWN))
    assert places[1] == pytest.approx((8.0 / ACROSS, 6.5 / DOWN))


def test_a_screen_that_is_a_hand_everywhere_or_nowhere_says_nothing_of_where():
    everywhere = {(c, r) for c in range(ACROSS) for r in range(DOWN)}
    assert places_the_pointer_turns(everywhere) == [] and places_the_pointer_turns(set()) == []


def test_a_screen_of_play_in_a_place_played_in_time_is_neither_swept_nor_waited_on_to_rest():
    """Offline 2026-10-11 a food fight's ten-second round was over before play began, sixteen seconds after it showed,
    spent sweeping the pointer over it and waiting for it to stand still after a click."""
    from types import SimpleNamespace

    from core.skills import screen_pursuit_looking as looking
    from core.skills.screen_pursuit_as_it_happens import AS_IT_HAPPENS, a_screen_of_play_in_time

    fight = SimpleNamespace(words=["TAKE AIM WITH YOUR MOUSE AND CLICK TO THROW. DODGE FOOD BY PRESSING THE SPACE BAR."],
                            keep={}, way_on_shown=False, under_her_hand=False)
    assert a_screen_of_play_in_time(fight, ["MENU", "YOU", "JOHNNY"])
    assert not a_screen_of_play_in_time(fight, ["HOW TO PLAY", "START"])
    calm = SimpleNamespace(words=["Choose a page and drag stickers onto it."], keep={}, way_on_shown=False)
    assert not a_screen_of_play_in_time(calm, ["MENU"])
    token = AS_IT_HAPPENS.set(fight)
    try:
        assert looking._how_long_to_wait() == looking.GOING_ON_ANSWERS_S
    finally:
        AS_IT_HAPPENS.reset(token)


def test_a_screen_of_play_in_time_is_played_the_moment_it_is_seen():
    """Offline 2026-10-11 the pursuit clicked a food fight's corner to see what it did, two seconds of a ten-second
    round, before play began on the next look."""
    import asyncio
    import math
    import time

    from core.skills.screen_pursuit_as_it_happens import AS_IT_HAPPENS, looked_at_as_it_happens

    class _Reflexes:
        words = ["TAKE AIM WITH YOUR MOUSE AND CLICK TO THROW. DODGE FOOD BY PRESSING THE SPACE BAR."]
        keep: dict = {}
        way_on_shown = False
        played_at = 0.0
        ends_at = math.inf
        moves = 0

        async def while_it_moves(self):
            self.moves += 1
            if self.moves >= 2:      # the first, before the look, still had the menu in mind
                self.played_at = time.monotonic()

        def read(self, seen):
            pass

        async def read_a_legend(self):
            pass

        def run_is_over(self, seen):
            return False

        def goes_on_playing(self, seen, before):
            return False

    looks = []

    async def look():
        looks.append(1)
        return {"ok": True, "text": "YOU JOHNNY MENU"}

    reflexes = _Reflexes()
    token = AS_IT_HAPPENS.set(reflexes)
    try:
        asyncio.run(looked_at_as_it_happens(look))
    finally:
        AS_IT_HAPPENS.reset(token)
    assert reflexes.moves == 2 and len(looks) == 2


def test_a_place_whose_words_say_how_it_is_played_has_no_board_or_well_looked_for():
    """Offline 2026-10-11, timed: a food fight's play began four and a half seconds late, spent looking for a board and
    for a piece coming down a well."""
    from types import SimpleNamespace

    from core.skills.screen_pursuit_as_it_happens import _played_neither_by_turns_nor_by_pieces

    fight = SimpleNamespace(words=["TAKE AIM WITH YOUR MOUSE AND CLICK TO THROW. DODGE FOOD BY PRESSING THE SPACE BAR."],
                            keep={})
    assert _played_neither_by_turns_nor_by_pieces(fight)
    board = SimpleNamespace(words=["Jump your pieces over your opponent's pieces to capture them."], keep={})
    assert not _played_neither_by_turns_nor_by_pieces(board)
    tiles = SimpleNamespace(words=["Rotate the falling blocks to complete lines."], keep={})
    assert not _played_neither_by_turns_nor_by_pieces(tiles)


def test_things_side_by_side_with_a_little_between_are_told_apart_by_a_finer_sweep():
    """Offline 2026-10-11 a corridor of doors, a row of six at a time, was found as one place a row, and the middle of
    each row was clicked over and over."""
    import asyncio
    from types import SimpleNamespace

    from core.perception.where_the_pointer_turns import where_the_pointer_turns

    doors = [(0.04 + 0.16 * i, 0.12 + 0.33 * j, 0.11, 0.2) for i in range(6) for j in range(3)]

    class _Page:
        viewport_size = {"width": 1280, "height": 768}

        def __init__(self):
            self.at = (0.0, 0.0)
            self.mouse = SimpleNamespace(move=self._move)

        async def _move(self, x, y):
            self.at = (x / 1280, y / 768)

        async def evaluate(self, _script, _where):
            x, y = self.at
            return "pointer" if any(a <= x <= a + w and b <= y <= b + h for a, b, w, h in doors) else "auto"

    places = asyncio.run(where_the_pointer_turns(_Page()))
    found = {(round(x * 6 - 0.5), round((y - 0.12) * 3)) for x, y in places}
    assert len(places) >= 15 and len(found) >= 15, places


def test_a_thing_for_making_has_no_board_looked_for_on_it():
    """Offline 2026-10-12 an alien maker's save screen was taken for three in a row, and three minutes went on searching
    moves on it, none made."""
    from types import SimpleNamespace

    from core.cognition.a_guide_to_a_place import SCREEN, THE_GUIDE, Guide
    from core.skills.screen_pursuit_as_it_happens import _played_neither_by_turns_nor_by_pieces

    maker = Guide(place="an alien maker")
    maker.take_in(SCREEN, "Create, customize and name your own alien hero!")
    token = THE_GUIDE.set(maker)
    try:
        assert _played_neither_by_turns_nor_by_pieces(SimpleNamespace(words=["SAVE TO THE OMNITRIX"], keep={}))
    finally:
        THE_GUIDE.reset(token)
