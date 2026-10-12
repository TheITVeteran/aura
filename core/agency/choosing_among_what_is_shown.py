"""Choosing among things shown: a screen that stops and asks her to take one of what it lays out.

"Choose a ball", "pick your character", "select a level": the screen offers
several things of a kind, side by side, and nothing goes on until one is taken.
A person does not press keys there to see what they do. They see what is on
offer (the row of balls, the two faces with a name under each), take one, and
remember which. What they took is theirs from then on: the ball they picked is
the ball they look for on the course, and the face they picked is who they are
when the play begins.

What is on offer is told two ways, by how such screens are drawn. Things
named: short labels side by side at one height, each the name of what is drawn
with it. Things laid out alike: several small things of about one size in a
band of the picture, each standing out from the ground.

Which one she takes is a thing she can give a reason for. Of things told apart
only by their paint she takes the one with the most colour in it, because a
thing of strong colour is the easiest to pick out again among others; of things
named she takes the first.

Nothing here knows a game.
"""
from __future__ import annotations

import colorsys
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Offered", "the_one_to_take", "what_is_offered", "what_she_took"]

#: The fewest things laid out alike that are a choice; how wide the patch about a thing is that its shape is read
#: from, in widths of the thing; the grid it is read on; and how alike two shapes are that are one thing drawn twice
#: (the mean of their products, each made of mean nought and spread one: 1 is the same, 0 is unrelated).
LAID_OUT_AT_LEAST = 3
#: How far apart, in widths of one of them, two things laid out together may be.
A_STEP_ALONG = 4.0
PATCH_ABOUT_IT = 1.6
LOOK_GRID = 8
THE_SAME_SHADING = 0.6

#: The most words a label has that names a choice, how alike in height such labels are, and how far apart in height
#: on the picture they may sit and still be side by side.
A_NAME_HAS_AT_MOST = 3
ALIKE_IN_HEIGHT = 1.5
SIDE_BY_SIDE_WITHIN = 0.08

#: How far above a name, in heights of its letters, the thing it names is pressed.
ABOVE_ITS_NAME = 1.6

_LETTERS = re.compile(r"[A-Za-z]{2,}")

#: Labels that leave a screen for somewhere else, as against what is chosen on it.
_GOES_ELSEWHERE = re.compile(r"^\W*(?:back|menu|main menu|quit|exit|help|options|settings|instructions|how to play|"
                             r"credits|more games|sound|music|skip|cancel|close)\W*$", re.I)

#: The narrowest a thing laid out to be taken is, as a share of the picture's width: drawn to be pressed.
PRESSABLE_FROM = 0.02


@dataclass(frozen=True)
class Offered:
    """One of the things a screen offers: where to press to take it (shares of the picture), its paint where it is told
    by paint, and its name where it has one."""

    place: tuple[float, float]
    colour: tuple[int, int, int] | None = None
    label: str = ""


def what_is_offered(picture: Any, words: Sequence[dict[str, Any]], asked: str = "") -> list[Offered]:
    """The things ``picture`` offers to choose among, left to right, or nothing where it lays out no choice: the names
    side by side where there are any, else the things laid out alike."""
    named = _named_side_by_side(words, asked)
    return named if len(named) >= 2 else _laid_out_alike(picture)


def _named_side_by_side(words: Sequence[dict[str, Any]], asked: str) -> list[Offered]:
    """Short labels at one height, of one size: each is the name of a thing offered, pressed just above its name.

    Not the labels that go on or go elsewhere ("next", "back", "menu"): those are how the screen is left, not what is
    chosen on it. A name is not one of those for having such a word in it: offline 2026-10-12 "1 Player" beside
    "2 Players" read as a way on for its "play", one name was left, and she took a leaf off the scenery instead.
    """
    from core.language.a_way_on import asks_to_choose, confirms

    asking = " ".join(str(asked or "").lower().split())
    labels = []
    for region in words:
        said = " ".join(str(region.get("text") or "").split())
        if not _LETTERS.search(said) or len(said.split()) > A_NAME_HAS_AT_MOST:
            continue
        if confirms(said) or _GOES_ELSEWHERE.match(said) or asks_to_choose(said) or said.lower() == asking:
            continue
        labels.append((said, float(region.get("center_x", 0.5)), float(region.get("center_y", 0.5)),
                       float(region.get("height", 0.0))))
    best: list[tuple[str, float, float, float]] = []
    for label in labels:
        beside = [other for other in labels
                  if abs(other[2] - label[2]) <= SIDE_BY_SIDE_WITHIN and label[3] > 0 and other[3] > 0
                  and max(other[3], label[3]) <= ALIKE_IN_HEIGHT * min(other[3], label[3])]
        if len(beside) > len(best):
            best = beside
    if len(best) < 2:
        return []
    return [Offered(place=(across, max(0.0, down - ABOVE_ITS_NAME * tall)), label=said)
            for said, across, down, tall in sorted(best, key=lambda one: one[1])]


def _laid_out_alike(picture: Any) -> list[Offered]:
    """The largest set of small things on the picture that are one thing drawn several times: the same size and the
    same shape against the ground, whatever their paint. Nine balls in nine colours are that; the eyes, teeth and buttons of a figure
    drawn beside them are not, each being its own shape."""
    import numpy as np

    from core.perception.things_that_stand_out import things_that_stand_out

    pixels = np.asarray(picture)
    things = [thing for thing in things_that_stand_out(pixels) if thing.wide >= PRESSABLE_FROM]
    best: list[Any] = []
    for thing in things:
        its_look = _the_look(pixels, thing.place, thing.wide)
        if its_look is None:
            continue
        alike = _laid_out_with(thing, _each_once(
            [other for other in things if other is thing or _alike(its_look, _the_look(pixels, other.place, thing.wide))],
            thing.wide))
        # Of several such sets, the one that covers most: what is laid out to be taken is drawn large enough to press,
        # and the shadows under five tokens are five things alike as well.
        if len(alike) >= LAID_OUT_AT_LEAST and _covers(alike) > _covers(best):
            best = alike
    if len(best) < LAID_OUT_AT_LEAST:
        return []
    return [Offered(place=thing.place, colour=thing.colour) for thing in sorted(best, key=lambda one: one.x)]


def _laid_out_with(thing: Any, alike: list[Any]) -> list[Any]:
    """Those of ``alike`` laid out with ``thing``: reached from it by steps of a few of its widths, one to the next.
    A round eye in a figure across the picture is the shape of a ball in a row, and is not in the row."""
    together, left = [thing], [other for other in alike if other is not thing]
    grew = True
    while grew:
        near = [other for other in left
                if any(math.dist(other.place, one.place) <= A_STEP_ALONG * thing.wide for one in together)]
        grew = bool(near)
        together += near
        left = [other for other in left if other not in near]
    return together


def _covers(things: Sequence[Any]) -> float:
    """How much of the picture a set of things covers, as squares of their widths."""
    return sum(thing.wide * thing.wide for thing in things)


def _each_once(alike: list[Any], wide: float) -> list[Any]:
    """Things found alike, each once: of two findings within one thing's width, the wider, which is the whole of it
    and not the shine on it."""
    kept: list[Any] = []
    for thing in sorted(alike, key=lambda one: -one.wide):
        if all(math.dist(thing.place, other.place) > wide for other in kept):
            kept.append(thing)
    return kept


def _the_look(pixels: Any, place: tuple[float, float], wide: float) -> Any:
    """The shape a thing makes against its ground, whatever its paint: on a grid of eight by eight over the patch
    about it, a little wider than itself, how unlike the ground each cell is. The ground is what the patch's rim is
    mostly painted. Lighter or darker does not come into it: a yellow token is lighter than the grass it lies on and a
    blue one darker, and they are one drawing. None where the patch runs off the picture, or is all ground."""
    import numpy as np

    from core.perception.picture_arithmetic import shrink

    tall, across = pixels.shape[:2]
    half = max(4, round(PATCH_ABOUT_IT * wide * across / 2))
    column, row = round(place[0] * across), round(place[1] * tall)
    if column - half < 0 or row - half < 0 or column + half > across or row + half > tall:
        return None
    patch = pixels[row - half:row + half, column - half:column + half, :3].astype(np.int16)
    rim = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
    unlike = np.abs(patch - np.median(rim, axis=0).astype(np.int16)).max(axis=2).astype(np.uint8)
    cells = shrink(np.ascontiguousarray(unlike), LOOK_GRID, LOOK_GRID).astype(np.float32)
    spread = float(cells.std())
    return (cells - float(cells.mean())) / spread if spread > 1.0 else None


def _alike(a: Any, b: Any) -> bool:
    """Whether two looks are one thing drawn twice: they stand off their grounds in the same places."""
    return a is not None and b is not None and float((a * b).mean()) >= THE_SAME_SHADING


def _how_much_colour(colour: tuple[int, int, int]) -> float:
    _hue, saturation, value = colorsys.rgb_to_hsv(*(max(0, min(255, int(v))) / 255.0 for v in colour[:3]))
    return saturation * value


def the_one_to_take(offered: Sequence[Offered], not_these: Sequence[Offered] = ()) -> Offered | None:
    """Which of the things offered she takes, leaving out any already tried to no end: of things told by their paint,
    the one with the most colour in it; of things named, the easiest where they are how hard it is to be (a person out
    to win takes the beginner's road first), else the first."""
    left = [one for one in offered if all(math.dist(one.place, tried.place) > 0.02 for tried in not_these)]
    if not left:
        return None
    if all(one.colour is not None for one in left):
        return max(left, key=lambda one: _how_much_colour(one.colour))
    easiest = [one for one in left if _EASIEST.search(one.label or "")]
    return easiest[0] if easiest else left[0]


#: The names of the easiest of the ways a place offers to be played.
_EASIEST = re.compile(r"^\W*(?:beginner|easy|easiest|novice|rookie|junior|kids?|casual|simple|level 1|1)\W*$", re.I)


def what_she_took(taken: Offered) -> str:
    """What she took, in words: its name, or its colour."""
    from core.agency.naming_what_she_sees import colour_name

    if taken.label:
        return taken.label
    return f"the {colour_name(taken.colour)} one" if taken.colour is not None else "one of them"
