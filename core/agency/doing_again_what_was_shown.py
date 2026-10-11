"""Doing again what was shown: a sequence read or watched, then done in the same order.

A screen that shows the way and waits for it to be followed asks for one thing:
do again what it showed, in its order. It shows it one of two ways. It draws
it: a row of arrows, the keys to press in turn ("follow the arrow keys in the
book"). Or it plays it: places that light up one after another (a dance's
steps, a lock's buttons, a tutorial that flashes what to click), and then
waits. Either way the order is the thing.

Doing it again is four things. Knowing when the screen is showing and when it
waits. Cutting the showing into its items: a place that changes from how it
usually looks and goes back, one after another, watched until nothing more
lights. Knowing which of her controls does each item again
(core/agency/what_a_press_lights.py): the control seen to light that place when
pressed, else the key an arrow's shape names (core/perception/keys_drawn_on_screen.py),
else a click on the place where the place is worked by the pointer. And doing
them at a person's pace, watching what each press lights, which teaches the
first. A showing missed is had again with a press made to have it shown again,
which most such screens answer by showing it once more.

Nothing here knows what is being shown or why: a dance, a spell, a code, a
lesson in which buttons to press.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("Aura.DoingAgainWhatWasShown")

__all__ = ["arrows_in", "asks_to_follow", "do_again", "follow", "lit_in_turn", "shown_in_turn", "Shown"]

#: Words that ask for what was shown to be done again.
_FOLLOW = re.compile(
    r"\b(?:follow|repeat|copy|match|mimic|do the same|same order|in order|memori[sz]e|remember the|watch and|"
    r"simon says|press the (?:arrow|keys?) (?:shown|that appear|as they appear))\b", re.I)

#: Arrow characters a reading of the screen may give for drawn arrows.
_GLYPHS = {"←": "left", "→": "right", "↑": "up", "↓": "down", "<": "left", ">": "right", "^": "up", "v": "down"}

#: How much brighter than usual a place must be to be lit, in grey levels.
LIT_BY = 40.0

#: The time between two acts of doing it again, in seconds: a person's pace, and a screen's time to answer.
BETWEEN_S = 0.35


def asks_to_follow(text: str) -> bool:
    """Whether a screen's words ask for what it shows to be done again."""
    return bool(_FOLLOW.search(" ".join(str(text or "").split())))


@dataclass
class Shown:
    """What was shown, in order: keys to press (``keys``) or places to click (``places``, shares of the picture), and
    where watched, each place as it lit (``lit``), for the act that does it again to be chosen by."""

    keys: list[str] = field(default_factory=list)
    places: list[tuple[float, float]] = field(default_factory=list)
    lit: list[Any] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.keys or self.places or self.lit)


def _the_way_it_points(mask: np.ndarray) -> str | None:
    """The way an arrow-shaped mask points (core/perception/keys_drawn_on_screen.py), or None where it is no arrow."""
    from core.perception.keys_drawn_on_screen import which_way_it_points

    return which_way_it_points(mask) or None


def arrows_in(picture: Any, regions: list[dict[str, Any]] | None = None) -> list[str]:
    """The arrows drawn on a still picture in a row (left to right) or a column (top to bottom), as the keys they name.

    Read from the writing first where it gives arrow characters; else from the shapes that stand out, each
    read by its own outline. Arrows set round a middle, one each way, are a pad of keys and no order to follow: a book
    of four arrows lit one after another was read as "down, down" and pressed, a life lost each time (offline
    2026-10-11).
    """
    for region in sorted(regions or [], key=lambda r: (round(float(r.get("center_y", 0)) * 10), float(r.get("center_x", 0)))):
        glyphs = [_GLYPHS[ch] for ch in str(region.get("text") or "") if ch in _GLYPHS and ch not in "v^<>"]
        if len(glyphs) >= 2:
            return glyphs
    from core.perception.picture_arithmetic import pieces

    pixels = np.asarray(picture, dtype=np.float32)
    grey = pixels.mean(axis=2) if pixels.ndim == 3 else pixels
    background = float(np.median(grey))
    standing_out = np.abs(grey - background) > LIT_BY
    count, labels, stats, _centres = pieces(standing_out.astype(np.uint8))
    tall, wide = grey.shape
    # A letter is no arrow: the Y of "YOUR TURN" and its T taper as an arrow's head does, and were read as two down
    # arrows to press (offline 2026-10-11). A shape inside a run of writing the reading found is left out.
    writing = [(float(r.get("x", 0)) * wide, float(r.get("y", 0)) * tall,
                (float(r.get("x", 0)) + float(r.get("width", 0))) * wide, (float(r.get("y", 0)) + float(r.get("height", 0))) * tall)
               for r in regions or [] if str(r.get("text") or "").strip()]
    found = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < 20 or w * h > 0.2 * tall * wide or min(w, h) < 4:
            continue
        if any(left <= x + w / 2 <= right and top <= y + h / 2 <= bottom for left, top, right, bottom in writing):
            continue
        way = _the_way_it_points(labels[y:y + h, x:x + w] == label)
        if way:
            found.append((x + w / 2, y + h / 2, max(w, h), way))
    if len(found) < 2:
        return [way for *_where, way in found]
    # Arrows of a row are drawn alike: a shape much larger or smaller than the rest is something else (a line down a
    # book's middle that touches two arrows is read as one long shape), and the row's tolerance is a usual arrow's size.
    size = float(np.median([one[2] for one in found]))
    if any(not 0.6 * size <= one[2] <= 1.6 * size for one in found):
        return []
    if max(one[1] for one in found) - min(one[1] for one in found) <= size:
        return [way for *_where, way in sorted(found)]
    if max(one[0] for one in found) - min(one[0] for one in found) <= size:
        return [way for *_where, way in sorted(found, key=lambda one: one[1])]
    return []


#: The longest side a watched picture is worked at: an arrow's taper still reads at this size (a third of a game's
#: width lost it), and ten seconds of a showing at thirty pictures a second is kept in about thirty-five megabytes.
WATCHED_SIDE = 320

#: The least part of the watched picture a lit place covers: less is a flicker or an edge, not a thing that lit.
LEAST_LIT_SHARE = 0.002

#: How long the screen must stand unchanged, once something has lit, for the showing to be over; and how long, where
#: nothing has changed at all, before the screen is taken to show nothing in turn.
SHOWING_OVER_S = 1.0
NOTHING_LIT_S = 2.0

#: The least time between two pictures watched: a showing is followed at thirty pictures a second, however fast pictures
#: can be had, so a still screen looked at in a tight loop is not thousands of pictures in two seconds.
WATCH_EVERY_S = 1 / 30


@dataclass
class _Lit:
    """One lighting of a place: where it is (its middle, and its box, as shares of the picture), the box in the watched
    picture's pixels, and the shape of what changed there."""

    place: tuple[float, float]
    box: tuple[int, int, int, int]
    mark: np.ndarray
    share: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def _small(picture: Any) -> np.ndarray:
    pixels = np.asarray(picture)
    pixels = pixels if pixels.ndim == 3 else pixels[..., None]
    step = max(1, int(np.ceil(max(pixels.shape[:2]) / WATCHED_SIDE)))
    return np.ascontiguousarray(pixels[::step, ::step, :3]).astype(np.uint8, copy=False)


async def _watched(look: Callable[[], Awaitable[Any]], seconds: float) -> list[np.ndarray]:
    """Pictures of a showing, from now until it is over: nothing has changed for a while after something did, nothing
    has changed at all for longer, or ``seconds`` have gone."""
    watched: list[np.ndarray] = []
    began = changed_at = None
    while True:
        seen = await look()
        if seen is None:
            break
        picture, at = seen
        if began is not None and watched and at - began < WATCH_EVERY_S * len(watched):
            await asyncio.sleep(WATCH_EVERY_S)
            continue
        began = at if began is None else began
        small = _small(picture)
        if watched and small.shape == watched[-1].shape and np.abs(small.astype(np.int16) - watched[-1]).max() > LIT_BY:
            changed_at = at
        watched.append(small)
        if at - began >= seconds:
            break
        if changed_at is None and at - began >= NOTHING_LIT_S:
            break
        if changed_at is not None and at - changed_at >= SHOWING_OVER_S:
            break
    return watched


def _lit_in(watched: list[np.ndarray], *, lit_at_first: bool = True) -> list[_Lit]:
    """The places that lit one after another in a run of pictures, in their order.

    A place is lit where it looks unlike its usual self, brighter or darker or another colour: a beige arrow going red
    is darker, and a lit place found by brightness alone was missed. Its usual self is how it looked most of the time
    it was watched, since a place is lit only now and then and the watching may begin while one already is. Lit is
    then unlit: a change that is still there at the end (the screen's words going from WATCH to YOUR TURN) is a new
    state of the screen, not a place in the order. Without ``lit_at_first`` a place already lit in the first picture is
    left out: watched just after she acted, it is the screen answering her last press, not the next showing.
    """
    from core.perception.picture_arithmetic import pieces

    watched = [w for w in watched if w.shape == watched[-1].shape] if watched else []
    if len(watched) < 3:
        return []
    usual = np.median(np.stack(watched), axis=0).astype(np.int16)
    tall, wide = usual.shape[:2]
    least = max(4, int(LEAST_LIT_SHARE * tall * wide))
    lit: list[_Lit] = []
    before: list[tuple[int, int, int, int]] = []
    for at, picture in enumerate(watched):
        changed = np.abs(picture.astype(np.int16) - usual).max(axis=2) > LIT_BY
        count, labels, stats, centres = pieces(changed)
        now: list[tuple[int, int, int, int]] = []
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            if area < least:
                continue
            box = (x, y, x + w, y + h)
            now.append(box)
            # A place that lights where nothing had changed a picture ago is the next in the order: every place that
            # changes is followed, not only the largest, since the screen's words changing beside a lit arrow made a
            # larger change than the arrow, and one lighting was read as two.
            if (lit_at_first or at) and not any(_overlaps(box, other) for other in before):
                cx, cy = centres[label]
                lit.append(_Lit(((cx + 0.5) / wide, (cy + 0.5) / tall), box, labels[y:y + h, x:x + w] == label,
                                (x / wide, y / tall, (x + w) / wide, (y + h) / tall)))
        before = now
    last = np.abs(watched[-1].astype(np.int16) - usual).max(axis=2) > LIT_BY
    return [one for one in lit if last[one.box[1]:one.box[3], one.box[0]:one.box[2]].mean() < 0.5]


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


async def lit_in_turn(look: Callable[[], Awaitable[Any]], seconds: float, *, places: int = 16) -> list[tuple[float, float]]:
    """Watch until the showing is over, ``seconds`` at most: the places that lit one after another, in the order they
    lit, as shares of the picture. ``places`` is kept for callers that said how many there were; a lit place is found
    where it is, not in a grid."""
    return [one.place for one in _lit_in(await _watched(look, seconds))]


async def shown_in_turn(look: Callable[[], Awaitable[Any]], seconds: float) -> Shown:
    """What a screen shows one thing at a time, watched until the showing is over, as what to do again: where every
    place that lit is an arrow, the keys they point to (a book of arrows lit one after another asks for the arrow keys,
    not for clicks on the book); else the places, to be clicked in their order.

    What is already lit as the watching begins is left out. Watched straight after she acted, it is the screen answering
    her last press: offline 2026-10-11 a pressed arrow lit for a quarter second, and a spell was read with her own last
    key in front of it; and the screen's words turning from YOUR TURN to WATCH were clicked as a place. Begun in the
    middle of a showing, the first of it is missed, the copy is wrong, and the showing comes again to be watched whole.
    """
    lit = _lit_in(await _watched(look, seconds), lit_at_first=False)
    ways = [_the_way_it_points(one.mark) for one in lit]
    if lit and all(ways):
        return Shown(keys=[str(way) for way in ways], lit=lit)
    return Shown(places=[one.place for one in lit], lit=lit)


async def follow(look: Callable[[], Awaitable[Any]], hands: Any, lights: Any, *, keys: list[str], clicks: bool,
                 rounds_at_most: int, watch_s: float, read_words: Callable[[Any], Any] | None = None,
                 say: Callable[[str], Any] | None = None, each_round: Callable[[dict[str, Any]], Any] | None = None) -> int:
    """Watch for what the screen shows, do it again, and watch for the next, ``rounds_at_most`` times at most: how many
    rounds were done. ``lights`` is what her presses have lit in this place (core/agency/what_a_press_lights.py),
    ``keys`` the keys she may press, ``clicks`` whether the place is worked by the pointer; ``each_round`` is told of
    each round done."""
    from core.agency.what_a_press_lights import CLICK, pressed_and_watched, watched_for

    rounds, guessed = 0, False
    while rounds < rounds_at_most:
        shown = await shown_in_turn(look, watch_s)
        for one in shown.lit:
            lights.saw_place(one.share)
        if not shown:
            # Nothing lit: a row of arrows that stands still is the order, done once, and what the screen does about it
            # is read on the next look.
            seen = await look()
            row = arrows_in(seen[0], read_words(seen[0]) if read_words else None) if seen is not None else []
            if len(row) >= 2:
                done = await do_again(Shown(keys=row), hands, say=say)
                _a_round(each_round, {"keys": row, "places": []}, done)
                return rounds + 1
            # Missed: the showing was over before she looked. A press to have it shown again (a wrong press usually
            # is), watched for what it lights, and then the showing watched from its start.
            guess = (lights.untried(keys) or keys)[:1]
            if guessed or not guess:
                break
            guessed = True
            logger.info("missed what was shown; pressing %s to have it shown again", guess[0])
            await pressed_and_watched(guess[0], hands, look, lights.places, lights)
            continue
        if say is not None:
            say(f"Doing again the {len(shown.lit)} thing{'s' if len(shown.lit) != 1 else ''} it showed.")
        acts: list[tuple[str, Any]] = []
        for one in shown.lit:
            # Chosen at the moment it is done, with what the presses before it have just lit: two places no key is
            # known for are not both given the same untried key.
            act = lights.act_for(one.share, _the_way_it_points(one.mark) or "", keys=keys, clicks=clicks)
            if act is None:
                logger.info("no way known to do again the thing shown at %s", one.share)
                break
            kind, what = act
            acts.append(act)
            if kind == CLICK:
                await hands.click(*what)
                await watched_for(look, BETWEEN_S)
                continue
            lit = await pressed_and_watched(what, hands, look, lights.places, lights, watched_for_s=BETWEEN_S)
            if lit is not None and lights.control_for(one.share) != what:
                # The press lit another place than the one shown: this round is lost, and the screen will show it again.
                logger.info("pressing %s lit another place than the one shown", what)
                break
        done = len(acts)
        _a_round(each_round, {"keys": [what for kind, what in acts if kind != CLICK],
                              "places": [what for kind, what in acts if kind == CLICK]}, done)
        rounds += 1
    return rounds


def _a_round(each_round: Callable[[dict[str, Any]], Any] | None, followed: dict[str, Any], done: int) -> None:
    logger.info("did again what was shown: %s (%d acts)", followed, done)
    if each_round is not None:
        each_round({"pictures": done, "gains": 0, "losses": 0, "ended": "did again what was shown", "followed": followed})


async def do_again(shown: Shown, hands: Any, *, say: Callable[[str], Any] | None = None) -> int:
    """Do what was shown, in its order, at a person's pace: each key pressed, or each place clicked. How many acts."""
    done = 0
    if say is not None and shown:
        what = ", ".join(shown.keys) if shown.keys else f"{len(shown.places)} places in the order they lit"
        say(f"Doing it again as it was shown: {what}.")
    for key in shown.keys:
        await hands.tap(key)
        done += 1
        await asyncio.sleep(BETWEEN_S)
    for place in shown.places:
        await hands.click(*place)
        done += 1
        await asyncio.sleep(BETWEEN_S)
    return done
