"""A screen that shows a sequence and waits for it to be done again, drawn as pixels on a clock of its own.

One family, many looks: places set as a pad of arrows, a row or a grid; lit brighter, darker or in another colour;
done again by the arrow keys, by number keys in an order nothing on the screen says, or by clicks on the places; a
pressed place lit back as an echo, or nothing lit back; the screen's words changing between WATCH and YOUR TURN as
bars of writing do. A wrong act costs a life and the showing comes again; the whole sequence done is a point and the
next showing is one longer. The clock moves a thirtieth of a second each look, so a run of many rounds takes no time.

Nothing that plays it reads its state: it is given pictures and its hands, as a game on a screen gives them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

WIDE, TALL = 240, 160
BACK = (43, 29, 58)
PLACE = (150, 140, 120)
LIT = {"brighter": (250, 245, 230), "darker": (40, 40, 40), "recolour": (230, 50, 50)}
ARROWS = ("up", "right", "down", "left")


def _arrow_mask(size: int, way: str) -> np.ndarray:
    tile = np.zeros((size, size), bool)
    third = size // 3
    tile[size // 2 - 3:size // 2 + 3, 0:size - third] = True
    for i in range(third):
        tile[size // 2 - (third - i) - 2:size // 2 + (third - i) + 2, size - third + i] = True
    return {"right": tile, "left": tile[:, ::-1], "down": tile.T, "up": tile.T[::-1, :]}[way]


@dataclass
class AShowing:
    layout: str = "pad"            # "pad" of four arrows round a middle, "row" of four squares, "grid" of nine squares
    lit: str = "recolour"          # how a place lights: "brighter", "darker", "recolour"
    answered_by: str = "arrows"    # "arrows": the arrow keys; "numbers": keys 1-n in a hidden order; "clicks"
    echo: bool = True              # a pressed place lit back for a moment
    words: bool = True             # bars of writing at the foot that change between showing and waiting
    seed: int = 1
    shown_s: float = 0.6
    gap_s: float = 0.25
    starts_after_s: float = 0.6
    lives: int = 3
    t: float = 0.0
    score: int = 0
    spell: list[int] = field(default_factory=list)
    done: list[int] = field(default_factory=list)
    showing_since: float | None = 0.0
    echo_until: float = -1.0
    echo_place: int = -1
    presses: int = 0

    def __post_init__(self) -> None:
        self.rand = random.Random(self.seed)
        if self.layout == "pad":
            middles = [(120, 30), (160, 70), (120, 110), (80, 70)]
        elif self.layout == "row":
            middles = [(48 + 48 * i, 70) for i in range(4)]
        else:
            middles = [(70 + 50 * (i % 3), 30 + 40 * (i // 3)) for i in range(9)]
        self.boxes = [(x - 14, y - 14, x + 14, y + 14) for x, y in middles]
        self.keys = list(ARROWS) if self.answered_by == "arrows" else [str(i + 1) for i in range(len(self.boxes))]
        if self.answered_by == "numbers":
            self.rand.shuffle(self.keys)
        self.grow()

    def grow(self) -> None:
        self.spell.append(self.rand.randrange(len(self.boxes)))
        self.show()

    def show(self) -> None:
        self.showing_since, self.done = self.t + self.starts_after_s, []

    def _showing_now(self) -> int:
        """The place lit by the showing now; -1 where none is; the showing ended once its last place has gone dark."""
        if self.showing_since is None or self.t < self.showing_since:
            return -1
        slot = self.shown_s + self.gap_s
        at = int((self.t - self.showing_since) / slot)
        if at >= len(self.spell):
            self.showing_since = None
            return -1
        return self.spell[at] if (self.t - self.showing_since) - at * slot < self.shown_s else -1

    def picture(self) -> np.ndarray:
        frame = np.full((TALL, WIDE, 3), BACK, np.uint8)
        lit = self._showing_now()
        if lit < 0 and self.t < self.echo_until:
            lit = self.echo_place
        for i, (left, top, right, bottom) in enumerate(self.boxes):
            colour = LIT[self.lit] if i == lit else PLACE
            if self.layout == "pad":
                mask = _arrow_mask(right - left, ARROWS[i])
                frame[top:bottom, left:right][mask] = colour
            else:
                frame[top:bottom, left:right] = colour
        if self.words:
            # WATCH while showing, YOUR TURN while waiting: bars of different lengths where the words stand.
            waiting = self.showing_since is None
            for n, length in enumerate((14, 10, 18, 12) if waiting else (22, 8)):
                x = 70 + sum((14, 10, 18, 12)[:n]) + 4 * n if waiting else 90 + 30 * n
                frame[146:154, x:x + length] = (255, 216, 74)
        return frame

    async def look(self):
        self.t += 1 / 30
        return self.picture(), self.t

    def _acted_on(self, place: int) -> None:
        self.presses += 1
        self.t += 0.06
        if place < 0 or self.showing_since is not None:
            return
        if self.echo:
            self.echo_place, self.echo_until = place, self.t + 0.25
        if place == self.spell[len(self.done)]:
            self.done.append(place)
            if len(self.done) == len(self.spell):
                self.score += 1
                self.grow()
        else:
            self.lives -= 1
            self.show()

    async def tap(self, key: str) -> None:
        if self.answered_by == "clicks":
            self._acted_on(-1)
            return
        self._acted_on(self.keys.index(key) if key in self.keys else -1)

    async def click(self, x: float, y: float) -> None:
        if self.answered_by != "clicks":
            self._acted_on(-1)
            return
        px, py = x * WIDE, y * TALL
        hit = [i for i, (left, top, right, bottom) in enumerate(self.boxes) if left <= px <= right and top <= py <= bottom]
        self._acted_on(hit[0] if hit else -1)

    def keys_a_player_is_told(self) -> list[str]:
        return list(ARROWS) if self.answered_by == "arrows" else sorted(self.keys)
