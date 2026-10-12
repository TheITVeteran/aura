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


def where_it_goes(pastime: Pastime | None, picture: Any) -> tuple[float, float] | None:
    """Where on ``picture`` what the pastime is played with is to be sent, as shares, where its look can be found: the
    one dark mouth on a course. None where the pastime names nothing to find, or the picture shows none, or more than
    one: guessing between two holes is no aim."""
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
        found.append((round(float(centres[label][0]) / across, 4), round(float(centres[label][1]) / high, 4)))
    return found[0] if len(found) == 1 else None
