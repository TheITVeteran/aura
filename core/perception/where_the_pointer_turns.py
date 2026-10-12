"""Where the pointer turns into a hand: the places on a screen that answer to a click, found as a person finds them,
by moving the mouse over it.

Much of what can be clicked on a game's screen looks like nothing in particular: the doorway to the next room, the
drawer with something in it, the edge of a scene that leads on, a word drawn into the picture. A person playing a
point-and-click game moves the mouse about and watches the pointer. Where it turns into a hand, something answers.
LIVE-like 2026-10-11 a search through a house's rooms was clicked at shapes picked out by colour for two minutes,
and the way to the next room was found only at the end.

So, once a screen, the pointer is swept over the part of the page that draws, on a grid, and where it is shown as a
hand is noted. Neighbouring places where it is a hand are one place, and the middle of each is offered as a thing to
click that stands out. A screen that is a hand everywhere (a screen clicked anywhere to go on) or nowhere says
nothing about where.

What the pointer is shown as is read from the page, as a person reads it off the screen. Nothing here knows a game.
"""
from __future__ import annotations

import logging
from typing import Any

__all__ = ["places_the_pointer_turns", "shown_as_a_hand", "where_the_hand_is_now", "where_the_pointer_turns"]

logger = logging.getLogger("Aura.WhereThePointerTurns")

#: The grid the pointer is swept over, across and down.
ACROSS, DOWN = 12, 9
#: The most places kept for one screen.
MOST = 8
#: How the pointer is shown over a thing that answers to a click.
_A_HAND = frozenset({"pointer", "hand"})

#: What the pointer is shown as at a place of the page, looking into the player the game is drawn in.
_SHOWN_AS = """(([x, y]) => {
  let e = document.elementFromPoint(x, y), shown = '';
  for (let i = 0; e && i < 6; i++) {
    shown = getComputedStyle(e).cursor;
    const inner = e.shadowRoot && e.shadowRoot.elementFromPoint(x, y);
    if (!inner || inner === e) break;
    e = inner;
  }
  return shown;
})"""


def places_the_pointer_turns(hands: set[tuple[int, int]], across: int = ACROSS, down: int = DOWN) -> list[tuple[float, float]]:
    """The places of a swept grid where the pointer was a hand, neighbours taken together, as the middle of each in
    shares of what was swept; none where it was a hand nearly everywhere or nowhere."""
    if not hands or len(hands) >= 0.8 * across * down:
        return []
    left, places = set(hands), []
    while left:
        start = left.pop()
        group, frontier = [start], [start]
        while frontier:
            column, row = frontier.pop()
            for near in ((column + 1, row), (column - 1, row), (column, row + 1), (column, row - 1)):
                if near in left:
                    left.discard(near)
                    group.append(near)
                    frontier.append(near)
        places.append(((sum(c for c, _ in group) / len(group) + 0.5) / across,
                       (sum(r for _, r in group) / len(group) + 0.5) / down, len(group)))
    places.sort(key=lambda place: -place[2])
    return [(x, y) for x, y, _size in places[:MOST]]


async def where_the_pointer_turns(page: Any, over: tuple[float, float, float, float] | None = None) -> list[tuple[float, float]]:
    """The places in ``over`` (shares of the page's view, left, top, right, bottom) where the pointer turns into a
    hand, as shares of ``over``; none where the page cannot be asked."""
    size = getattr(page, "viewport_size", None) or {}
    width, height = float(size.get("width") or 0), float(size.get("height") or 0)
    if not width or not height:
        return []
    left, top, right, bottom = over or (0.0, 0.0, 1.0, 1.0)
    hands: set[tuple[int, int]] = set()
    try:
        for row in range(DOWN):
            for column in range(ACROSS):
                x = (left + (right - left) * (column + 0.5) / ACROSS) * width
                y = (top + (bottom - top) * (row + 0.5) / DOWN) * height
                await page.mouse.move(x, y)
                if str(await page.evaluate(_SHOWN_AS, [x, y]) or "") in _A_HAND:
                    hands.add((column, row))
    except Exception as why:  # noqa: BLE001 - a page that cannot be asked is a screen with no such places found
        logger.info("the pointer could not be swept over the page: %s", str(why)[:120])
        return []
    places = places_the_pointer_turns(hands)
    logger.info("where the pointer turns into a hand: %s", [(round(x, 2), round(y, 2)) for x, y in places])
    return places


async def shown_as_a_hand(page: Any, x: float, y: float) -> bool:
    """Whether the pointer, moved to ``x``, ``y`` in the page's pixels, is shown as a hand there now."""
    await page.mouse.move(x, y)
    return str(await page.evaluate(_SHOWN_AS, [x, y]) or "") in _A_HAND


async def where_the_hand_is_now(page: Any, x: float, y: float, left: float, top: float, wide: float,
                                tall: float) -> tuple[float, float]:
    """Where a place found to answer the pointer answers now, as shares of the part swept: where it was, where the pointer
    is still a hand there; else the nearest place within a grid step or so where it is; else where it was. A button
    that moves (a top spinning about a title, a door that slides) is not where it was when the screen was swept: offline
    2026-10-11 a title's start was a spinning top, and every click went to where it had been."""
    try:
        if await shown_as_a_hand(page, left + x * wide, top + y * tall):
            return x, y
        found = []
        for row in range(-NEAR, NEAR + 1):
            for column in range(-NEAR, NEAR + 1):
                there = (x + column / (2 * ACROSS), y + row / (2 * DOWN))
                if not (0.0 <= there[0] <= 1.0 and 0.0 <= there[1] <= 1.0) or (row, column) == (0, 0):
                    continue
                if await shown_as_a_hand(page, left + there[0] * wide, top + there[1] * tall):
                    found.append(there)
    except Exception as why:  # noqa: BLE001 - a page that cannot be asked is clicked where the place was
        logger.info("where the hand is now could not be asked: %s", str(why)[:120])
        return x, y
    if not found:
        return x, y
    near = min(found, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
    return near


#: How many half grid steps either way a place found to answer the pointer is looked for where it has gone.
NEAR = 3

