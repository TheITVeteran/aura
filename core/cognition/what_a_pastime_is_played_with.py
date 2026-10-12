"""What a pastime is played with, and where that goes: what anyone knows of golf before they see a course.

A person told "mini golf" knows before the first screen that there is a ball,
that it is theirs to hit, and that it goes into a hole; told "basketball", that
the ball goes through a hoop; told "bowling", that it rolls at pins. Nobody
learns that from the course in front of them: it is what the word means. A
place's title and instructions name its pastime, and that names the thing she
works with and the thing it is sent toward, each with how such a thing looks:
a ball is small, round and solid; a hole is a dark round mouth in the ground.
Where her model reads the place it says the same and more; this is what holds
where it says nothing, and what tells her eyes what to look for.

Nothing here knows a game: it knows pastimes, as anyone does.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["DARK_MOUTH", "Pastime", "pastime_named_in", "where_it_goes"]

#: How a thing looks: small, round and solid (a ball, a top, a puck); a dark round mouth (a hole, a cup, a pocket).
SMALL_ROUND = "small round"
DARK_MOUTH = "dark mouth"


@dataclass(frozen=True)
class Pastime:
    """A pastime: the words that name it, what it is played with and how that looks, and what that is sent toward."""

    name: str
    named_by: re.Pattern[str]
    played_with: str
    looks: str
    toward: str
    toward_looks: str = ""


PASTIMES: tuple[Pastime, ...] = (
    Pastime("golf", re.compile(r"\b(?:mini[- ]?)?golf\b|\bputt(?:s|ing|er)?\b|\bfore!", re.I), "ball", SMALL_ROUND,
            "hole", DARK_MOUTH),
    Pastime("pool", re.compile(r"\bbilliards?\b|\bsnooker\b|\bcue ball\b|\b8[- ]ball\b", re.I), "cue ball", SMALL_ROUND,
            "pocket", DARK_MOUTH),
    Pastime("basketball", re.compile(r"\bbasketball\b|\bshoot(?:ing)? (?:hoops|baskets)\b|\bslam dunk\b", re.I), "ball",
            SMALL_ROUND, "hoop"),
    Pastime("bowling", re.compile(r"\bbowling\b|\bbowl(?:s|ers?)?\b|\bstrike\b.*\bpins?\b", re.I), "ball", SMALL_ROUND,
            "pins"),
    Pastime("darts", re.compile(r"\bdarts?\b|\bbull'?s[- ]?eye\b", re.I), "dart", "", "board"),
    Pastime("football", re.compile(r"\bsoccer\b|\bpenalt(?:y|ies)\b|\bscore a goal\b", re.I), "ball", SMALL_ROUND,
            "goal"),
    Pastime("spinning tops", re.compile(r"\bspinning tops?\b|\byour top\b|\btops? out of\b", re.I), "top", SMALL_ROUND,
            ""),
    Pastime("slingshot", re.compile(r"\bsling ?shot\b|\bcatapult\b|\blaunch(?:er|ing)?\b", re.I), "what is launched", "",
            ""),
)


def pastime_named_in(words: str) -> Pastime | None:
    """The pastime ``words`` name, or None: the first one named, by the order of what is most particular."""
    said = " ".join(str(words or "").split())
    return next((pastime for pastime in PASTIMES if pastime.named_by.search(said)), None)


#: A dark mouth is darker than most of the picture by this much, covers this share of it, and is this much wider than
#: tall at most (a hole seen from above is round; from aslant, an ellipse lying down).
MOUTH_DARKER_BY = 0.5
MOUTH_SIZES = (0.0003, 0.02)
MOUTH_WIDER = (0.8, 3.5)
#: The width it is looked for at.
MOUTH_LOOKED_AT = 320
#: How near what is to be sent a dark thing may be and not be where it goes; how nearly a mouth fills the ellipse its
#: box holds; and by how much the best of several must beat the next to be taken for the mouth.
NOT_WHERE_IT_IS = 0.06
A_MOUTH_FILLS = 0.85
PLAINLY_MORE_A_MOUTH = 0.08


def where_it_goes(pastime: Pastime | None, picture: Any,
                  apart_from: tuple[float, float] | None = None) -> tuple[float, float] | None:
    """Where on ``picture`` what the pastime is played with is to be sent, as shares, where its look can be found: the
    dark mouth on a course. None where the pastime names nothing to find, or the picture shows none, or shows several
    that cannot be told apart: guessing between two holes is no aim.

    Of several dark things, the mouth is the one drawn as a mouth: a filled ellipse, wider than tall as a round hole is
    from aslant. And it is not where the thing to be sent already is (``apart_from``): offline 2026-10-12 a second
    hole's tee, a dark mat with the ball on it, was as dark as the hole, and with two dark things she aimed at neither.
    """
    if pastime is None or pastime.toward_looks != DARK_MOUTH:
        return None
    from core.perception.picture_arithmetic import pieces, shrink

    pixels = np.asarray(picture)[:, :, :3].astype(np.uint8)
    tall, wide = pixels.shape[:2]
    if wide > MOUTH_LOOKED_AT:
        pixels = shrink(pixels, MOUTH_LOOKED_AT, max(1, round(tall * MOUTH_LOOKED_AT / wide)))
    high, across = pixels.shape[:2]
    bright = pixels.astype(np.float32).mean(axis=2)
    dark = bright < (1.0 - MOUTH_DARKER_BY) * float(np.median(bright))
    count, _labels, stats, centres = pieces(dark.astype(np.uint8))
    found = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        share = area / (high * across)
        if not MOUTH_SIZES[0] <= share <= MOUTH_SIZES[1] or not MOUTH_WIDER[0] <= w / max(1, h) <= MOUTH_WIDER[1]:
            continue
        if area < 0.6 * w * h or x == 0 or y == 0 or x + w >= across or y + h >= high:
            continue  # not solid, or running off the picture: a shadow, a frame, the dark of the sky
        place = (round(float(centres[label][0]) / across, 4), round(float(centres[label][1]) / high, 4))
        if apart_from is not None and math.dist(place, apart_from) <= NOT_WHERE_IT_IS:
            continue
        # How nearly it fills the ellipse its box holds: 1 for an ellipse, less for a mat, a sign, a patch of shade.
        found.append((1.0 - abs(1.0 - area / (math.pi / 4.0 * w * h)), place))
    found.sort(reverse=True)
    if len(found) == 1 or (len(found) > 1 and found[0][0] >= A_MOUTH_FILLS and found[0][0] - found[1][0] >= PLAINLY_MORE_A_MOUTH):
        return found[0][1]
    return None
