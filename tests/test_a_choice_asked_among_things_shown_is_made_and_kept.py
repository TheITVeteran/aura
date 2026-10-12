"""A screen that asks her to choose among things it lays out has one taken, and what she took is hers afterwards.

Nothing here is a golf game's ball screen or a cartoon's character screen. A row of tokens, one drawing in several
paints, beside a figure made of odd shapes; two names side by side under two pictures. Which she takes, and that she
knows it again on the field, is what is held to.
"""
from __future__ import annotations

import asyncio
import math
import time

import numpy as np
import pytest

from core.agency.choosing_among_what_is_shown import the_one_to_take, what_is_offered, what_she_took
from core.agency.playing_by_shots import _places_to_send_from
from core.perception.things_that_stand_out import alike_in_colour, things_that_stand_out

pytestmark = pytest.mark.unit

WIDE, TALL = 640, 400
PAINTS = [(245, 205, 60), (70, 110, 235), (240, 90, 30), (150, 150, 150), (60, 200, 90)]
ROW = [(0.30 + 0.1 * i, 0.82 + 0.02 * (i % 2)) for i in range(len(PAINTS))]


def _token(picture: np.ndarray, at: tuple[float, float], half: int, paint: tuple[int, int, int]) -> None:
    """One drawing in any paint: a disc with a shine on its upper left and a shadow under it."""
    yy, xx = np.mgrid[0:TALL, 0:WIDE]
    cx, cy = at[0] * WIDE, at[1] * TALL
    picture[(yy - cy - half * 0.9) ** 2 * 6 + (xx - cx - 3) ** 2 <= half * half] = (40, 90, 40)
    picture[(yy - cy) ** 2 + (xx - cx) ** 2 <= half * half] = paint
    shine = (yy - cy + half * 0.4) ** 2 + (xx - cx + half * 0.4) ** 2 <= (half * 0.4) ** 2
    picture[shine] = tuple(min(255, int(v * 0.4 + 255 * 0.6)) for v in paint)


def _a_screen_of_tokens() -> np.ndarray:
    picture = np.full((TALL, WIDE, 3), (90, 170, 80), np.uint8)
    picture[:150] = (150, 210, 230)
    # A figure beside them, made of shapes none of which is another: an eye, a mouth, a button, a stripe.
    picture[60:200, 60:160] = (240, 220, 120)
    yy, xx = np.mgrid[0:TALL, 0:WIDE]
    picture[(yy - 90) ** 2 + (xx - 90) ** 2 <= 49] = (255, 255, 255)
    picture[(yy - 92) ** 2 * 4 + (xx - 128) ** 2 <= 60] = (20, 20, 20)
    picture[130:136, 80:140] = (200, 30, 30)
    picture[160:172, 100:112] = (40, 60, 200)
    for at, paint in zip(ROW, PAINTS, strict=True):
        _token(picture, at, 12, paint)
    return picture


def test_of_tokens_laid_out_she_takes_one_of_the_row_and_the_one_with_most_colour():
    picture = _a_screen_of_tokens()
    offered = what_is_offered(picture, [{"text": "Choose a token", "center_x": 0.6, "center_y": 0.3, "height": 0.08},
                                        {"text": "Pick the one you like!", "center_x": 0.5, "center_y": 0.6, "height": 0.05}])
    assert len(offered) >= 4, offered
    assert all(any(math.dist(one.place, at) < 0.03 for at in ROW) for one in offered), offered
    taken = the_one_to_take(offered)
    assert math.dist(taken.place, ROW[2]) < 0.03, taken              # the orange one: the most colour in it
    assert what_she_took(taken) == "the orange one"
    # One tried to no end is not taken again.
    again = the_one_to_take(offered, [taken])
    assert again is not None and math.dist(again.place, taken.place) > 0.03


def test_of_names_side_by_side_she_takes_the_first_and_presses_what_it_names():
    words = [{"text": "CHOOSE YOUR CHARACTER", "center_x": 0.5, "center_y": 0.1, "height": 0.09},
             {"text": "Mandy", "center_x": 0.35, "center_y": 0.62, "height": 0.07},
             {"text": "Billy", "center_x": 0.66, "center_y": 0.61, "height": 0.07},
             {"text": "NEXT", "center_x": 0.9, "center_y": 0.9, "height": 0.05}]
    offered = what_is_offered(np.full((TALL, WIDE, 3), 200, np.uint8), words)
    assert [one.label for one in offered] == ["Mandy", "Billy"], offered
    taken = the_one_to_take(offered)
    assert taken.label == "Mandy" and taken.place[1] < 0.62 and abs(taken.place[0] - 0.35) < 0.01


def test_a_name_with_a_word_that_goes_on_in_it_is_still_a_name_and_the_ways_out_are_not_choices():
    # Offline 2026-10-12: "1 Player" beside "2 Players", with "back" and "next" under them.
    words = [{"text": "Please Select Number Of Players", "center_x": 0.5, "center_y": 0.27, "height": 0.05},
             {"text": "1 Player", "center_x": 0.43, "center_y": 0.57, "height": 0.06},
             {"text": "2 Players", "center_x": 0.64, "center_y": 0.57, "height": 0.06},
             {"text": "back", "center_x": 0.69, "center_y": 0.73, "height": 0.06},
             {"text": "next", "center_x": 0.82, "center_y": 0.73, "height": 0.06}]
    offered = what_is_offered(_a_screen_of_tokens(), words)
    assert [one.label for one in offered] == ["1 Player", "2 Players"], offered
    assert the_one_to_take(offered).label == "1 Player"


def test_a_screen_that_lays_nothing_out_offers_nothing():
    plain = np.full((TALL, WIDE, 3), (90, 170, 80), np.uint8)
    assert what_is_offered(plain, [{"text": "Choose wisely", "center_x": 0.5, "center_y": 0.5, "height": 0.08}]) == []
    assert the_one_to_take([]) is None


def test_the_shine_on_a_token_does_not_change_what_paint_it_is_read_as():
    picture = _a_screen_of_tokens()
    orange = max((thing for thing in things_that_stand_out(picture) if math.dist(thing.place, ROW[2]) < 0.03),
                 key=lambda thing: thing.wide)                       # the token, not the shine on it
    flat = (250, 62, 12)                                             # the same paint drawn flat and small, elsewhere
    pale = (250, 170, 155)                                           # another thing of the same hue, washed out
    assert alike_in_colour(orange.colour, flat) > alike_in_colour(orange.colour, pale), orange.colour


def test_the_thing_painted_like_what_she_chose_is_the_first_place_tried_on_the_field():
    field = np.full((TALL, WIDE, 3), (214, 190, 110), np.uint8)
    yy, xx = np.mgrid[0:TALL, 0:WIDE]
    spots = {(0.2, 0.7): (250, 62, 12), (0.6, 0.4): (250, 170, 155), (0.7, 0.7): (140, 210, 205), (0.4, 0.3): (250, 250, 70)}
    for at, paint in spots.items():
        picture_half = 5 if paint == (250, 62, 12) else 9
        field[(yy - at[1] * TALL) ** 2 + (xx - at[0] * WIDE) ** 2 <= picture_half ** 2] = paint
    taken = the_one_to_take(what_is_offered(_a_screen_of_tokens(), []))
    places = _places_to_send_from(field, {}, chosen=taken.colour)
    assert math.dist(places[0], (0.2, 0.7)) < 0.02, places[:3]


def test_on_her_page_a_screen_that_asks_her_to_choose_has_one_taken_and_kept(monkeypatch):
    import core.skills.screen_pursuit_as_it_happens as reflexes

    picture = _a_screen_of_tokens()
    clicked: list[tuple[float, float]] = []
    said: list[str] = []

    class _OnHerPage(reflexes.PlayingAsItHappens):
        async def look(self):
            return picture, time.monotonic()

        async def click(self, x, y):
            clicked.append((x, y))

    monkeypatch.setattr(reflexes, "_said_while_playing", said.append)
    monkeypatch.setattr(reflexes, "AFTER_A_CHOICE_S", 0.0)
    on_it = {"words": [{"text": "Choose a token", "center_x": 0.6, "center_y": 0.3, "height": 0.08}]}
    monkeypatch.setattr("core.perception.what_the_pixels_show.recognize_text", lambda _picture: on_it["words"])
    playing = _OnHerPage(page=None, band=(0, 0, 1, 1), goal="play it", ends_at=time.monotonic() + 8)
    playing.words = ["Choose a token Pick the one you like!"]
    assert asyncio.run(reflexes._chose_what_it_asks(playing))
    assert len(clicked) == 1 and math.dist(clicked[0], ROW[2]) < 0.03, clicked
    assert said == ["It asks me to choose: I'm taking the orange one."]
    assert alike_in_colour(tuple(playing.keep["chose"]["colour"]), (250, 62, 12)) >= 0.5
    # The screen after it, read before its words are: nothing on it asks, and nothing is taken from it.
    on_it["words"] = [{"text": "ROUND ONE", "center_x": 0.5, "center_y": 0.3, "height": 0.08}]
    assert not asyncio.run(reflexes._chose_what_it_asks(playing)) and len(clicked) == 1
    # A screen that asks for nothing is left alone.
    playing.words = ["HOW TO PLAY Use the arrow keys to move"]
    assert not asyncio.run(reflexes._chose_what_it_asks(playing)) and len(clicked) == 1


def test_of_how_hard_it_is_to_be_the_easiest_is_taken():
    """Offline 2026-10-12 a parking game asked to choose a difficulty; a person out to win takes the beginner's road."""
    from core.agency.choosing_among_what_is_shown import Offered, the_one_to_take

    offered = [Offered((0.3, 0.5), None, "ADVANCED"), Offered((0.5, 0.5), None, "INTERMEDIATE"),
               Offered((0.7, 0.5), None, "BEGINNER")]
    assert the_one_to_take(offered).label == "BEGINNER"
    assert the_one_to_take([Offered((0.3, 0.5), None, "Tokyo"), Offered((0.6, 0.5), None, "Paris")]).label == "Tokyo"
