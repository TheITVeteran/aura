"""What a place's code says of the things she sees: which is her, what each costs her or is worth, and how it goes.

Her eyes give her kinds of things by their look (a colour, a size); a reading of the place's code
(core/perception/reading_a_world_s_code.py) gives her its things by what it keeps them as and what it fills each in. A
kind she sees in the colour the code fills a thing of its own with is that thing: the code draws her body in #f2f2f2
and its bugs in #2a9a2a, and its bugs cost a life when they touch her. So before a bug has touched her she keeps clear
of it; before her keys have been tried, the white square is taken for her, until her trials say otherwise; and what the
code says of how fast a kind wanders is how far it could swerve (core/agency/how_far_a_thing_could_swerve.py).

A colour two of the code's things share tells neither apart (two paddles drawn alike), and is left to play to say.
Offline 2026-10-13, put to the paint world in its own code with her whole live play, she pressed keys for half a
minute before taking the white square for hers and went to meet a bug before learning it cost her.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.WhatItsCodeSaysSheSees")

__all__ = ["GROUND_GONE_OVER", "CodeKind", "a_strip_of_ground", "ground_gone_over", "kinds_its_code_draws", "nothing_inside_stops_her",
           "paces_its_code_says", "swerves_its_code_says", "what_its_code_says_she_sees"]

#: How far a colour seen may be from a colour the code fills with and be that colour, on its most different channel, out
#: of 255: what anti-aliasing and the picture's packing leave of a flat fill.
SEEN_AS_DRAWN = 40
#: How much nearer one of the code's colours must be than any other for a kind to be taken for it.
CLEARER_BY = 12


@dataclass(frozen=True)
class CodeKind:
    """A kind she sees, as the code has it: the name the code keeps it under ("hers" for her body), what touching her
    does when it looks so, how it goes, and how fast it wanders in her picture's pixels a second (0 where it does not)."""

    name: str
    touching: str = ""
    moves: str = ""
    wanders: float = 0.0


def _rgb(colour: str) -> tuple[int, int, int] | None:
    hex_ = str(colour or "").lstrip("#")
    if len(hex_) == 3:
        hex_ = "".join(c * 2 for c in hex_)
    if len(hex_) != 6:
        return None
    try:
        return int(hex_[0:2], 16), int(hex_[2:4], 16), int(hex_[4:6], 16)
    except ValueError:
        return None


def _apart(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return max(abs(int(x) - int(y)) for x, y in zip(a, b, strict=True))


#: What a kind is where it is the marks of ground gone over (painted squares): ground, not a thing.
GROUND_GONE_OVER = "ground gone over"


def _drawn(world: Any, scale: float) -> list[tuple[tuple[int, int, int], CodeKind]]:
    out = []
    hers = getattr(world, "hers", None)
    if hers is not None and _rgb(hers.drawn_in):
        out.append((_rgb(hers.drawn_in), CodeKind("hers")))
    ground = getattr(world, "ground", None)
    if ground is not None and _rgb(getattr(ground, "drawn_in", "")):
        out.append((_rgb(ground.drawn_in), CodeKind(GROUND_GONE_OVER, moves="stands")))
    for thing in getattr(world, "things", []) or []:
        for colour, does in thing.looks:
            rgb = _rgb(colour)
            if rgb is None:
                continue
            touching = does or (thing.touching if " or " not in thing.touching else "")
            wanders = (thing.speed or 0.0) * scale if thing.moves == "wanders" else 0.0
            out.append((rgb, CodeKind(thing.name, touching, thing.moves, wanders)))
    return out


def kinds_its_code_draws(world: Any, kinds: Any, scale: float = 1.0) -> dict[int, CodeKind]:
    """Each kind she sees (``kinds``, with a number and a colour) that is drawn as one of the code's things, by its
    colour: the nearest of the code's colours, near enough and clearly nearer than any other."""
    drawn = _drawn(world, scale)
    out: dict[int, CodeKind] = {}
    if not drawn:
        return out
    for kind in kinds or []:
        colour = tuple(getattr(kind, "colour", ()) or ())
        if len(colour) != 3:
            continue
        ranked = sorted(((_apart(colour, rgb), code) for rgb, code in drawn), key=lambda one: one[0])
        near, code = ranked[0]
        if near > SEEN_AS_DRAWN:
            continue
        rivals = [other for apart, other in ranked[1:] if apart - near < CLEARER_BY and other != code]
        if rivals:
            continue
        out[int(kind.number)] = code
    return out


def _laid(world: Any, moves: Any, guide: Any) -> float:
    from core.agency.ground_her_code_names import stage_in_her_picture

    stage = getattr(world, "stage", None)
    laid = stage_in_her_picture(stage, moves.shape, getattr(guide, "stage_in_her_picture", None)) if stage else None
    return laid[0] if laid else 1.0


def _the_world(guide: Any = None) -> tuple[Any, Any]:
    if guide is None:
        from core.cognition.a_guide_to_a_place import THE_GUIDE

        guide = THE_GUIDE.get()
    return getattr(guide, "world_its_code_makes", None), guide


def swerves_its_code_says(moves: Any, guide: Any = None) -> dict[int, float]:
    """How fast each kind she sees wanders, in her picture's pixels a second, where the place's code says it does."""
    world, guide = _the_world(guide)
    if not world:
        return {}
    matched = kinds_its_code_draws(world, getattr(moves, "kinds", []), _laid(world, moves, guide))
    return {kind: code.wanders for kind, code in matched.items() if code.wanders}


def nothing_inside_stops_her(moves: Any, mine: Any, guide: Any = None) -> bool:
    """Whether the place's code says nothing inside where she is kept stops her body (no wall, no box she walks into),
    and the thing she takes for hers is the body its code draws: then what play took for something in her way was not.
    Offline 2026-10-13 she took the painted squares of a floor for something that stopped her, and kept off them."""
    world, guide = _the_world(guide)
    hers = getattr(world, "hers", None) if world else None
    if hers is None or hers.stopped_inside or not hers.paces or mine is None:
        return False
    matched = kinds_its_code_draws(world, getattr(moves, "kinds", []), _laid(world, moves, guide))
    return getattr(matched.get(int(getattr(mine, "kind", -1))), "name", "") == "hers"


def a_strip_of_ground(bar: Any, shape: tuple[int, int], guide: Any = None) -> bool:
    """Whether a strip she follows as a bar (core/perception/how_full_a_bar_is.py), in a picture of ``shape`` (rows,
    columns), is the ground the place's code says is to be gone over: drawn in the colour its code marks ground gone
    over in, or lying on that ground. A row of floor shortens as it is painted, and it is the floor, not a bar of what
    she has left. Offline 2026-10-13 rows of a painted floor were taken for bars, one of them named for the PAINTED
    beside it, and what she learned from them falling was that holding down cost her."""
    world, guide = _the_world(guide)
    ground = getattr(world, "ground", None) if world else None
    if ground is None:
        return False
    marks, seen = _rgb(getattr(ground, "drawn_in", "")), tuple(getattr(bar, "colour", ()) or ())
    if marks is not None and len(seen) == 3 and _apart(seen, marks) <= SEEN_AS_DRAWN:
        return True
    from core.agency.ground_her_code_names import ground_in_her_picture

    laid = ground_in_her_picture(world, shape, getattr(guide, "stage_in_her_picture", None))
    if laid is None:
        return False
    (left, top), (wide, tall), (columns, rows) = laid.origin, laid.cell, laid.cells
    return (top <= float(bar.row) <= top + rows * tall
            and left - wide / 2 <= float(bar.start) and float(bar.end) <= left + (columns + 0.5) * wide)


def ground_gone_over(moves: Any, guide: Any = None) -> set[int]:
    """The kinds she sees that are the marks of ground gone over, by the colour the place's code draws them in."""
    world, guide = _the_world(guide)
    if not world:
        return set()
    matched = kinds_its_code_draws(world, getattr(moves, "kinds", []), _laid(world, moves, guide))
    return {kind for kind, code in matched.items() if code.name == GROUND_GONE_OVER}


def paces_its_code_says(moves: Any, mine: Any, keys: Any, guide: Any = None) -> dict[str, tuple[float, float]]:
    """How fast each of ``keys`` moves her, in her picture's pixels a second, where the place's code says it sets her
    pace and the thing she takes for hers is the body its code draws: what her trials have not measured yet, they need
    not find before she can use it. Offline 2026-10-13 her trials of left and right went unmeasured on the paint world
    and she played ninety-seven seconds of it moving only up and down."""
    world, guide = _the_world(guide)
    hers = getattr(world, "hers", None) if world else None
    if hers is None or not hers.paces or hers.pushes or mine is None:
        return {}
    scale = _laid(world, moves, guide)
    matched = kinds_its_code_draws(world, getattr(moves, "kinds", []), scale)
    if getattr(matched.get(int(getattr(mine, "kind", -1))), "name", "") != "hers":
        return {}
    return {key: (pace[0] * scale, pace[1] * scale) for key, pace in hers.paces.items() if key in set(keys or [])}


def what_its_code_says_she_sees(run: Any, moves: Any, hers: Any, meeting: Any, at: float, tell: Any = None,
                                guide: Any = None) -> dict[int, CodeKind]:
    """One picture: each kind she sees matched to the code's things, and what that says put where she keeps it: a stance
    for each kind not yet met that the code says costs her or is worth getting; and her body, taken for hers where no
    trial of her keys has found it yet and one thing on the screen is drawn as it. The matches."""
    from core.agency.what_meeting_things_does import AVOID, IGNORE, MEET

    world, guide = _the_world(guide)
    if not world:
        return {}
    matched = kinds_its_code_draws(world, getattr(moves, "kinds", []), _laid(world, moves, guide))
    for kind, code in matched.items():
        if code.name == GROUND_GONE_OVER:
            meeting.told[kind] = IGNORE                          # ground gone over: nothing to meet or keep clear of
            continue
        if code.name == "hers" or meeting.evidence.get(kind) is not None:
            continue
        stance = {"costs a life": AVOID, "scores": MEET}.get(code.touching)
        if stance and meeting.told.get(kind) != stance:
            meeting.told[kind] = stance
            if tell is not None:
                many = code.name.endswith("s")
                what = ("cost" if many else "costs") + " me" if stance == AVOID else ("are" if many else "is") + " worth getting"
                tell(f"Its code says the {code.name} {what}.", f"code says {code.name}")
    mine = [kind for kind, code in matched.items() if code.name == "hers"]
    if len(mine) == 1 and getattr(hers, "number", None) is None:
        hers_seen = [t for t in getattr(moves, "things", {}).values() if int(t.kind) == mine[0]]
        if len(hers_seen) == 1:
            hers.seen_to_be_her(hers_seen[0], at, f"its code draws me in {world.hers.drawn_in}")
            if tell is not None:
                tell("Its code says which one is me; I'll check it answers my keys.", "code says me")
    return matched
