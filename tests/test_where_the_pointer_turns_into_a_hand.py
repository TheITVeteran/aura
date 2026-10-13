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

        async def look(self):
            return None      # a fresh reading before play that sees nothing new: play goes on as it was going

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


def test_what_the_pointer_turns_into_a_hand_over_is_offered_though_nothing_else_made_it_clickable():
    """Offline 2026-10-12 a title's "start", small letters the pointer turned into a hand over, was read and never
    offered, and its title was played with keys for four minutes."""
    from core.agency.what_i_can_do_here import a_click_on
    from core.skills.screen_pursuit_looking import _what_answers_the_pointer

    observation = {"layout": [{"text": "start", "center_x": 0.75, "center_y": 0.83, "answers_the_pointer": True},
                              {"text": "Candy Machine", "center_x": 0.4, "center_y": 0.5}]}
    offered = _what_answers_the_pointer(observation, ("v", "space"))
    assert a_click_on("start") in offered and a_click_on("Candy Machine") not in offered
    assert _what_answers_the_pointer(observation, (a_click_on("start"),)).count(a_click_on("start")) == 1


def test_a_screen_taken_for_play_that_stands_the_same_a_while_is_looked_over_with_the_pointer(monkeypatch):
    """Offline 2026-10-12 an instructions screen whose backdrop moved, read once without its way on, was taken for play,
    and she pressed one key on it for three and a half minutes, the key having done nothing every time."""
    from types import SimpleNamespace

    import core.skills.screen_pursuit_as_it_happens as reflexes_module
    from core.skills.screen_pursuit_as_it_happens import (
        STUCK_ON_PLAY_S,
        stuck_on_one_screen_of_play,
    )

    now = [100.0]
    monkeypatch.setattr(reflexes_module.time, "monotonic", lambda: now[0])
    reflexes = SimpleNamespace(play_screen=None)
    words = "Find the lost kite in the garden. Use your mouse to look about."
    assert not stuck_on_one_screen_of_play(reflexes, words)
    now[0] += STUCK_ON_PLAY_S / 2
    assert not stuck_on_one_screen_of_play(reflexes, words)
    now[0] += STUCK_ON_PLAY_S / 2 + 1.0
    assert stuck_on_one_screen_of_play(reflexes, words)
    assert not stuck_on_one_screen_of_play(reflexes, "LEVEL 2 Score 40 Lives 3 Time 59")   # another screen begins again
    assert not stuck_on_one_screen_of_play(None, words)


def test_a_place_found_half_a_cell_beside_a_word_is_that_word():
    """A place found on a grid is known no more closely than a cell of it. Offline 2026-10-12 a carved "back" was found
    beside its word, taken for an unnamed thing that stands out, and clicked as a way on; of two words within a cell,
    the nearer is the one."""
    from core.perception.where_the_pointer_turns import ACROSS, the_thing_there

    back = {"text": "back", "x": 0.66, "y": 0.70, "width": 0.07, "height": 0.05}
    on = {"text": "on", "x": 0.78, "y": 0.70, "width": 0.07, "height": 0.05}
    assert the_thing_there((0.745, 0.72), [back, on]) is back            # 0.015 beyond "back", 0.035 short of "on"
    assert the_thing_there((0.77, 0.72), [back, on]) is on
    assert the_thing_there((0.73 + 1.5 / ACROSS, 0.72), [back]) is None  # a cell and a half off: something else
    assert the_thing_there((0.5, 0.5), [{"text": "middle", "center_x": 0.52, "center_y": 0.5}])["text"] == "middle"


def test_a_page_asked_the_whole_sweep_at_once_finds_the_same_places_as_place_by_place():
    """One move and one question to the browser a place took most of a second a screen; asked in one go, a few
    thousandths. What is found must not change, and the mouse is left where the last move left it."""
    import asyncio
    from types import SimpleNamespace

    from core.perception import where_the_pointer_turns as sweep

    doors = [(0.1, 0.2, 0.15, 0.25), (0.6, 0.55, 0.2, 0.3)]

    def shown(x, y):
        return "pointer" if any(a <= x / 1280 <= a + w and b <= y / 768 <= b + h for a, b, w, h in doors) else "auto"

    class _Page:
        viewport_size = {"width": 1280, "height": 768}

        def __init__(self, at_once: bool):
            self.at_once, self.at, self.asked = at_once, (0.0, 0.0), 0
            self.mouse = SimpleNamespace(move=self._move)

        async def _move(self, x, y):
            self.at = (x, y)

        async def evaluate(self, script, where):
            self.asked += 1
            if script == sweep._SWEPT_IN_THE_PAGE:
                if not self.at_once:
                    raise RuntimeError("this page is asked place by place")
                return [shown(x, y) for x, y in where[0]]
            return shown(*self.at)

    at_once, by_place = _Page(True), _Page(False)
    found = asyncio.run(sweep.where_the_pointer_turns(at_once))
    assert found == asyncio.run(sweep.where_the_pointer_turns(by_place)) and len(found) == 2
    assert at_once.asked < 10 < by_place.asked
    assert at_once.at == by_place.at
