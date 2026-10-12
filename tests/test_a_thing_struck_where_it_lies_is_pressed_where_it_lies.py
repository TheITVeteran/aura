"""A thing of hers, struck where it lies: found among others by the marker beside her name, and pressed where it rests.

Nothing here is a golf course. A field with several balls on it, one of them the player's; a plate that writes the
player's name with a paler picture of their ball beside it; a dark mouth. A press on her ball, pulled and let go, sends
it the other way, and it stays where it stops. Presses anywhere else do nothing. Which ball is hers, which way a pull
sends it and how far, she is not told.
"""
from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

from core.agency.playing_by_shots import _places_to_send_from, hers_is_in_sight, play_by_shots
from core.perception.the_marker_by_a_name import how_like_the_marker, the_marker_by
from core.perception.things_that_stand_out import alike_in_colour, things_that_stand_out

pytestmark = pytest.mark.unit

WIDE, TALL = 480, 360
GROUND = (214, 190, 110)
HERS = (250, 60, 12)
PALER = (242, 122, 90)                      # the same paint as drawn on the plate: lighter, less of it
OTHERS = {(0.55, 0.62): (250, 170, 156), (0.42, 0.3): (250, 250, 70), (0.7, 0.75): (140, 210, 205),
          (0.6, 0.25): (150, 110, 180)}


def _disc(picture: np.ndarray, at: tuple[float, float], half: int, colour: tuple[int, int, int]) -> None:
    yy, xx = np.mgrid[0:TALL, 0:WIDE]
    picture[(yy - at[1] * TALL) ** 2 + (xx - at[0] * WIDE) ** 2 <= half * half] = colour


class _Field:
    """Several balls on a field; the one painted like the picture by the player's name answers to a press and a pull."""

    def __init__(self, ball=(0.18, 0.72), mouth=(0.78, 0.4), carries=9.0, name="Kes",
                 then=()) -> None:
        self.ball: list[float] | None = list(ball)
        self.velocity = [0.0, 0.0]
        self.mouth, self.carries, self.name = mouth, carries, name
        #: The fields after this one, each a ball and a mouth, and when the next is laid out.
        self.then = list(then)
        self.next_at: float | None = None
        self.between_s = 1.5
        #: Where the card between two fields draws her marker, her name beside it.
        self.card_marker_at = (24 / WIDE, 21 / TALL)
        self.now = 0.0
        self.pressed_at: tuple[float, float] | None = None
        self.pointer = (0.0, 0.0)
        self.strokes = 0
        self.holed = 0
        self.pressed: list[tuple[float, float]] = []
        self.lay_when_pressed: list[tuple[float, float] | None] = []

    async def look(self):
        self.now += 1 / 30
        if self.ball is not None and any(self.velocity):
            self.ball[0] = min(0.97, max(0.03, self.ball[0] + self.velocity[0] / 30))
            self.ball[1] = min(0.97, max(0.12, self.ball[1] + self.velocity[1] / 30))
            self.velocity = [v * 0.9 for v in self.velocity]
            slow = math.hypot(*self.velocity) < 0.25
            if slow and math.dist(self.ball, self.mouth) < 0.035:
                self.ball, self.velocity, self.holed = None, [0.0, 0.0], self.holed + 1
                self.next_at = self.now + self.between_s if self.then else None
            elif math.hypot(*self.velocity) < 0.01:
                self.velocity = [0.0, 0.0]
        if self.next_at is not None and self.now >= self.next_at:
            (ball, self.mouth), self.next_at = self.then.pop(0), None
            self.ball = list(ball)
        if self.next_at is not None:
            # Between two fields: a card of scores, her name and her marker on it, and nothing to send.
            card = np.full((TALL, WIDE, 3), (60, 120, 60), np.uint8)
            left, top = round(self.card_marker_at[0] * WIDE) - 14, round(self.card_marker_at[1] * TALL) - 13
            card[top:top + 26, left:left + 140] = (250, 250, 205)
            _disc(card, self.card_marker_at, 10, PALER)
            return card, self.now
        picture = np.full((TALL, WIDE, 3), GROUND, np.uint8)
        picture[:40] = (120, 190, 220)                                           # a sky over the field
        picture[8:34, 10:150] = (250, 250, 205)                                  # the plate her name is on
        _disc(picture, (24 / WIDE, 21 / TALL), 10, PALER)
        _disc(picture, self.mouth, 11, (16, 16, 16))
        for at, colour in OTHERS.items():
            _disc(picture, at, 9, colour)
        if self.ball is not None:
            _disc(picture, tuple(self.ball), 5, HERS)
        return picture, self.now

    def words(self, _picture):
        if self.next_at is not None:
            across, down = self.card_marker_at
            return [{"text": self.name, "x": across + 20 / WIDE, "y": down - 7 / TALL, "width": 40 / WIDE, "height": 14 / TALL}]
        return [{"text": self.name, "x": 44 / WIDE, "y": 14 / TALL, "width": 40 / WIDE, "height": 14 / TALL},
                {"text": f"HOLED {self.holed}", "x": 0.8, "y": 0.03, "width": 0.12, "height": 0.04,
                 "center_x": 0.86, "center_y": 0.05}]

    async def press(self, x, y):
        self.pressed.append((x, y))
        self.lay_when_pressed.append(tuple(self.ball) if self.ball is not None else None)
        on_it = self.ball is not None and math.dist((x, y), self.ball) < 0.025 and not any(self.velocity)
        self.pressed_at = (x, y) if on_it else None
        self.pointer = (x, y)

    async def point(self, x, y):
        self.pointer = (x, y)

    async def release(self):
        if self.pressed_at is not None:
            pull = (self.pointer[0] - self.pressed_at[0], self.pointer[1] - self.pressed_at[1])
            far = math.hypot(*pull)
            if far > 0.005:
                reach = min(far, 0.14) / far                                     # no pull counts for more than so much
                self.velocity = [-self.carries * pull[0] * reach, -self.carries * pull[1] * reach]
                self.strokes += 1
        self.pressed_at = None


def _a_first_picture(field: _Field) -> np.ndarray:
    return asyncio.run(field.look())[0]


def test_a_ball_as_wide_as_the_old_window_is_among_the_things_that_stand_out():
    # Offline 2026-10-11: a ball about eleven pixels wide, looked for through an eleven-pixel window, was the window's
    # own middle value and so was the ground; she pressed sixty other things and never it.
    field = _Field()
    things = things_that_stand_out(_a_first_picture(field))
    assert any(math.dist(thing.place, field.ball) < 0.02 for thing in things), things
    for at in OTHERS:
        assert any(math.dist(thing.place, at) < 0.02 for thing in things), (at, things)


def test_an_edge_and_a_bar_do_not_stand_out_on_every_side():
    picture = np.full((200, 300, 3), (200, 180, 120), np.uint8)
    picture[:, 150:] = (40, 60, 160)                      # an edge down the middle
    picture[96:104, 20:130] = (250, 250, 250)             # a bar
    assert things_that_stand_out(picture) == []


def test_the_marker_by_her_name_is_its_paint_and_not_the_sky_behind_the_plate():
    field = _Field()
    picture = _a_first_picture(field)
    marker = the_marker_by("Kes", field.words(picture), picture)
    assert marker and all(alike_in_colour(colour, PALER) >= 0.5 for colour, _share in marker.colours), marker
    assert marker.holds((24 / WIDE, 21 / TALL)) and not marker.holds(tuple(field.ball))
    assert how_like_the_marker(HERS, marker) > max(how_like_the_marker(colour, marker) for colour in OTHERS.values())
    assert not the_marker_by("Someone Else", field.words(picture), picture)


def test_the_thing_painted_like_her_marker_is_the_first_place_tried():
    field = _Field()
    picture = _a_first_picture(field)
    marker = the_marker_by("Kes", field.words(picture), picture)
    places = _places_to_send_from(picture, {}, marker=marker)
    assert math.dist(places[0], field.ball) < 0.02, places[:4]
    # With no marker to go by it is still among them, and every place is tried once before any is tried twice.
    assert any(math.dist(place, field.ball) < 0.02 for place in _places_to_send_from(picture, {}))


def test_she_strikes_her_own_ball_follows_it_to_where_it_rests_and_puts_it_in_the_mouth():
    field = _Field()
    keep: dict = {}
    result = asyncio.run(play_by_shots(field.look, field, seconds=40.0, keep=keep, read_words=field.words,
                                       aim_at=lambda _picture, _hers: field.mouth, ways=("pull",), her_name="Kes"))
    assert field.holed == 1, (result, field.strokes, field.ball)
    assert field.strokes <= 3, (field.strokes, result)
    assert result["ended"] == "what she sent got there", result
    # Nothing was pressed once it was in: not the picture of it by her name, not the other balls.
    assert len(field.pressed) == field.strokes, field.pressed
    # Every press after the first that sent it was on the ball where it then lay, never on the place it had left.
    sent = next(i for i, lay in enumerate(field.lay_when_pressed) if lay is not None and math.dist(field.pressed[i], lay) < 0.025)
    assert sent <= 1, (sent, field.pressed[:4])
    on_it = [math.dist(press, lay) < 0.025 for press, lay in zip(field.pressed[sent:], field.lay_when_pressed[sent:], strict=True)
             if lay is not None]
    assert all(on_it), list(zip(field.pressed, field.lay_when_pressed, strict=True))


def test_with_no_name_given_she_still_finds_which_ball_answers_by_trying_each_once():
    field = _Field()
    keep: dict = {}
    asyncio.run(play_by_shots(field.look, field, seconds=40.0, keep=keep, read_words=field.words,
                              aim_at=lambda _picture, _hers: field.mouth, ways=("pull",)))
    assert field.strokes >= 1 and keep.get("sends_what") is not None
    assert alike_in_colour(tuple(keep["sends_what"]), HERS) >= 0.7, keep.get("sends_what")
    # No place was pressed twice before the ball was first struck.
    first = next(i for i, lay in enumerate(field.lay_when_pressed) if lay is not None and math.dist(field.pressed[i], lay) < 0.025)
    before = field.pressed[:first]
    assert all(math.dist(a, b) > 0.02 for i, a in enumerate(before) for b in before[i + 1:]), before


def test_on_the_next_field_she_presses_her_own_ball_at_once_and_nothing_else():
    """Offline 2026-10-12: a golf ball holed on the first hole, and on the second, its tee in plain sight, a minute of
    clicking shapes to see what they do. What she sends is one thing, known by its paint on whatever screen it is."""
    field = _Field(then=[((0.8, 0.8), (0.3, 0.3))])
    keep: dict = {}

    async def a_stretch():
        return await play_by_shots(field.look, field, seconds=40.0, keep=keep, read_words=field.words,
                                   aim_at=lambda _picture, _hers: field.mouth, ways=("pull",), her_name="Kes")

    # The card between two fields stays up longer than she waits for another ball: the stretch ends with the ball in.
    field.between_s = 12.0
    first = asyncio.run(a_stretch())
    assert first["ended"] == "what she sent got there" and field.holed == 1, first
    assert keep.get("lies_where_struck") and keep.get("sends_from") is None
    # The card: her marker is on it, and nothing of hers to send.
    card = asyncio.run(field.look())[0]
    assert not hers_is_in_sight(card, keep, "Kes", field.words)
    while field.ball is None:
        asyncio.run(field.look())
    picture = asyncio.run(field.look())[0]
    assert hers_is_in_sight(picture, keep, "Kes", field.words)
    pressed_before, strokes_before = len(field.pressed), field.strokes
    second = asyncio.run(a_stretch())
    assert field.holed == 2, (second, field.strokes)
    assert field.strokes - strokes_before <= 4, (field.strokes - strokes_before, second)
    on_it = [math.dist(press, lay) < 0.025 for press, lay in zip(field.pressed[pressed_before:],
                                                                 field.lay_when_pressed[pressed_before:], strict=True)]
    assert on_it and all(on_it), list(zip(field.pressed[pressed_before:], field.lay_when_pressed[pressed_before:], strict=True))


def test_a_ball_lying_beside_the_mouth_has_not_got_there():
    from core.agency.how_hard_and_which_way import Setting, Shot

    beside = Shot(Setting(0.1, 0.0), (0.76, 0.41), aimed_at=(0.78, 0.4), lies_in_sight=True)
    gone_in = Shot(Setting(0.1, 0.0), (0.76, 0.41), aimed_at=(0.78, 0.4))
    assert not beside.got_there and gone_in.got_there


def test_the_mouth_is_told_from_the_dark_mat_her_ball_starts_on():
    """Offline 2026-10-12: on a second hole the tee was a dark mat as dark as the hole, and with two dark things she
    aimed at neither and putted at random."""
    from core.cognition.what_a_pastime_is_played_with import pastime_named_in, where_it_goes

    picture = np.full((TALL, WIDE, 3), (120, 170, 220), np.uint8)
    yy, xx = np.mgrid[0:TALL, 0:WIDE]
    picture[(yy - 0.3 * TALL) ** 2 * 4 + (xx - 0.75 * WIDE) ** 2 <= 16 ** 2] = (10, 10, 10)        # the mouth, aslant
    picture[250:268, 70:110] = (30, 30, 30)                                                       # a dark mat
    golf = pastime_named_in("mini golf")
    assert where_it_goes(golf, picture, apart_from=(90 / WIDE, 259 / TALL)) == pytest.approx((0.75, 0.3), abs=0.02)
    # And with nothing said of where her ball is, the filled ellipse is still the mouth, not the oblong.
    assert where_it_goes(golf, picture) == pytest.approx((0.75, 0.3), abs=0.02)


def test_her_marker_on_a_card_of_scores_is_not_her_ball_come_to_rest():
    """Offline 2026-10-12: the ball dropped, a card of scores came up with her name and the little ball beside it, and
    that was taken for her ball lying somewhere new, and pressed."""
    field = _Field(then=[((0.8, 0.8), (0.3, 0.3))])
    field.between_s = 12.0
    field.card_marker_at = (0.1, 0.86)
    keep: dict = {}
    result = asyncio.run(play_by_shots(field.look, field, seconds=40.0, keep=keep, read_words=field.words,
                                       aim_at=lambda _picture, _hers: field.mouth, ways=("pull",), her_name="Kes"))
    assert field.holed == 1 and result["ended"] == "what she sent got there", result
    assert len(field.pressed) == field.strokes, field.pressed


def test_what_she_sends_is_kept_by_its_paint_and_size_for_the_next_time():
    from core.agency.what_she_keeps_of_a_game import kept_from, to_keep

    held = to_keep({"by_shots": {"sends_from": None, "sends_what": (250, 60, 12), "sends_wide": 0.0187, "sends_by": "pull",
                                 "lies_where_struck": True, "shots": {}}})
    again = kept_from(held)["by_shots"]
    assert again["sends_what"] == (250, 60, 12) and again["lies_where_struck"] and again["sends_by"] == "pull"
    assert again["sends_from"] is None and again["sends_wide"] == pytest.approx(0.0187)


def test_a_thing_of_her_paint_and_another_size_is_not_her_thing():
    """Offline 2026-10-12: the red veins on an eyeball were her ball's colour, and she pressed them for a minute."""
    from core.agency.playing_by_shots import _painted_like
    from core.perception.things_that_stand_out import StandsOut

    hers = StandsOut(0.2, 0.7, 0.0187, (250, 60, 12), 130.0)
    veins = StandsOut(0.6, 0.4, 0.0094, (240, 70, 30), 50.0)
    bigger = StandsOut(0.4, 0.4, 0.0469, (250, 60, 12), 90.0)
    assert _painted_like([hers, veins, bigger], (250, 60, 12), 0.0187) == [hers]
    assert len(_painted_like([hers, veins, bigger], (250, 60, 12))) == 3
