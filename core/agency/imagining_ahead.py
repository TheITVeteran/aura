"""Imagining the next moment of a world under each thing she could do, and doing the one that comes out best.

A finite game she can solve outright: she holds its rules exactly and searches
them (2048, a board in turns). A world that moves has no rules written down,
but she learns its parts as she plays: how each key moves her
(core/agency/which_one_answers_to_her.py), the path a press takes her on
(core/agency/what_a_press_does.py), how each kind of thing falls, bounces and
leaves (core/perception/how_things_move_here.py), and what meeting each kind
does to her (core/agency/what_meeting_things_does.py). Put together, those are
a simulator of the world as she knows it. This runs it.

For each plan she could follow over the next second or so (hold a key, hold
one then another, hold one and make a press partway), her thing is run
forward under it and every thing that matters is run forward under the world's
own going. The plan is scored by what it meets: what is worth getting, the
sooner the better; what costs her, heavily, and wider of it the lower she is;
and how near it ends to where she means to be. She does the first part of the
best plan, and plans again at the next picture: model-predictive control, as
the general video game players and robots do it, on a model she learned
herself.

What this replaces is a set of one-step rules, one per kind of play (dodge,
catch, intercept, jump over). A plan found by imagining needs no rule for its
kind of play: going left and then up to get under a falling thing past a wall,
or jumping while running so as to land beyond what is coming, comes out of the
imagining. Nothing here knows a game.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["HORIZON_S", "Plan", "best_plan", "plans_to_imagine"]

#: How far ahead she imagines, and in what steps, in seconds.
HORIZON_S = 1.2
STEP_S = 1 / 30
#: When in a plan she may change from one key to another, or make a press, in seconds from now.
CHANGES_AT = (0.15, 0.3, 0.5, 0.75)
PRESSES_AT = (0.0, 0.2, 0.4)
#: What a thing worth getting is worth, what a thing that costs her costs, and how much each step of waiting takes off.
GAIN = 1.0
LOSS = 4.0
SOONER = 0.97
#: How much ending far from where she means to be costs, for the whole width of what she can cross in the horizon.
FAR_FROM_THE_GOAL = 0.5
#: How much a plan that changes what she is holding now costs: her hands are not free.
CHANGING = 0.02
#: How much waiting before a plan starts doing what it does costs, a second of it: her model of the world is never
#: exact, and of two plans that come out alike, the one that moves sooner leaves room for what it gets wrong.
WAITING = 0.2
#: The most things imagined: the nearest that matter.
MOST_THINGS = 16
#: How much faster and slower than she has measured the world is imagined going, plan by plan: her measure of it is
#: never exact, and a plan that comes out well only at exactly the pace measured is no plan to stake a life on. A plan
#: worth following is one that comes out well however the world goes within that; this is what makes her get under a
#: falling thing and wait, as people do, rather than pass under it at the moment it arrives.
PACES = (0.85, 1.0, 1.15)


@dataclass(frozen=True)
class Plan:
    """What she does over the horizon: keys held in turn, each from its start, and presses made at times."""

    held: tuple[tuple[float, str], ...]
    presses: tuple[tuple[float, str], ...] = ()

    @property
    def first(self) -> str:
        return self.held[0][1]

    def press_now(self) -> str | None:
        return next((key for when, key in self.presses if when <= 0.0), None)


@dataclass
class Imagined:
    """What a plan came to: its worth, and what it met."""

    plan: Plan
    worth: float
    gets: int = 0
    costs: int = 0
    ends_at: tuple[float, float] = (0.0, 0.0)
    #: Whether the plans imagined differ in what they meet: where none does, imagining has nothing to choose between.
    decides: bool = False


@dataclass
class _Scene:
    """Her thing and the world, run forward once for every plan to be held against."""

    times: np.ndarray
    start: tuple[float, float]
    size: tuple[float, float]
    bounds: tuple[float, float, float, float]
    ways: dict[str, tuple[float, float]]
    rest: tuple[float, float]
    paths: dict[str, Callable[[float], tuple[float, float]]]
    #: Each thing's place at each step (paces x steps x things x 2), its size, what it is to her, and whether it is there.
    where: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0, 2)))
    sizes: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    worth: np.ndarray = field(default_factory=lambda: np.zeros(0))
    there: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0), dtype=bool))
    barred: Any = None


def plans_to_imagine(choices: Sequence[str], presses: Sequence[str], held: str) -> list[Plan]:
    """Every plan worth imagining: each key (or none) held throughout; each held and then another from a change time;
    and each press made at a time while a key is held. ``choices`` includes "" (nothing held)."""
    plans = [Plan(((0.0, key),)) for key in choices]
    for first in choices:
        for then in choices:
            if then != first:
                plans += [Plan(((0.0, first), (when, then))) for when in CHANGES_AT]
    for press in presses:
        for key in choices:
            plans += [Plan(((0.0, key),), ((when, press),)) for when in PRESSES_AT]
    # The key held now first, so that among plans that come out alike she goes on as she is.
    return sorted(plans, key=lambda plan: plan.first != held)


def best_plan(choosing: Any, held: str, goal: tuple[float | None, float | None], *,
              horizon: float = HORIZON_S) -> Imagined | None:
    """The plan that comes out best for ``choosing`` (the real-time player's arithmetic of one decision), holding
    ``held`` now and meaning to be at ``goal``; None where there is nothing to imagine with: no thing of hers, or no way
    she knows of moving it."""
    scene = _the_scene(choosing, horizon)
    if scene is None:
        return None
    plans = plans_to_imagine(["", *scene.ways], list(scene.paths), held)
    imagined = [_run(scene, plan, goal, held, choosing.caution) for plan in plans]
    best = max(imagined, key=lambda one: one.worth)
    best.decides = len({(one.gets, one.costs) for one in imagined}) > 1
    return best


def _the_scene(choosing: Any, horizon: float) -> _Scene | None:
    mine = getattr(choosing, "mine", None)
    ways = dict(getattr(choosing, "ways", {}) or {})
    # Nothing to imagine with: no thing of hers, no way she knows of moving it, or no world seen around it.
    if mine is None or not ways or getattr(choosing, "moves", None) is None or not hasattr(choosing, "others"):
        return None
    tall, wide = choosing.moves.shape
    times = np.arange(1, int(round(horizon / STEP_S)) + 1) * STEP_S
    moving = ways.get(getattr(choosing, "held_now", ""), (0.0, 0.0))
    response = float(getattr(choosing, "response_s", 0.0) or 0.0)
    # What she holds now goes on moving her while the next command is on its way.
    start = (mine.x + moving[0] * response, mine.y + moving[1] * response)
    paths: dict[str, Callable[[float], tuple[float, float]]] = {}
    presses = getattr(choosing, "press_paths", None)
    for key in getattr(choosing, "lifts", []) or []:
        path = presses(key) if callable(presses) else None
        if path is not None:
            paths[key] = path
    scene = _Scene(times=times, start=start, size=(float(mine.w), float(mine.h)),
                   bounds=(mine.w / 2, mine.h / 2, wide - mine.w / 2, tall - mine.h / 2), ways=ways,
                   rest=tuple(getattr(choosing, "rest", (0.0, 0.0))), paths=paths, barred=getattr(choosing, "barred", None))
    _the_world_run_forward(choosing, scene)
    return scene


def _the_world_run_forward(choosing: Any, scene: _Scene) -> None:
    """Every thing that matters to her, nearest first, run forward under the world's own going."""
    from core.agency.what_meeting_things_does import AVOID, CLICK, MEET, SHOOT

    mine = choosing.mine
    worth_of = {MEET: GAIN, CLICK: GAIN, AVOID: -LOSS, SHOOT: -LOSS}
    things = [(t, worth_of.get(choosing.stance(t), 0.0)) for t in choosing.others()]
    things = [(t, w) for t, w in things if w != 0.0]
    things.sort(key=lambda pair: math.hypot(pair[0].x - mine.x, pair[0].y - mine.y))
    things = things[:MOST_THINGS]
    steps, count = len(scene.times), len(things)
    where = np.zeros((len(PACES), steps, count, 2))
    there = np.ones((len(PACES), steps, count), dtype=bool)
    physics = getattr(choosing, "physics", None)
    for column, (thing, _worth) in enumerate(things):
        path = _its_path(physics, thing, float(scene.times[-1]) * max(PACES))
        for layer, pace in enumerate(PACES):
            for row, when in enumerate(scene.times):
                spot = path(float(when) * pace)
                if spot is None:
                    there[layer, row:, column] = False
                    break
                where[layer, row, column] = spot
    scene.where, scene.there = where, there
    scene.sizes = np.array([[float(t.w), float(t.h)] for t, _w in things]).reshape(count, 2)
    scene.worth = np.array([w for _t, w in things])


def _its_path(physics: Any, thing: Any, seconds: float) -> Callable[[float], tuple[float, float] | None]:
    """Where a thing will be after so many seconds, under the world's physics as learned where they are known, else
    along its going; None once it has left."""
    imagined = None
    if physics is not None and getattr(thing, "moved", False):
        try:
            imagined = physics.imagine(thing, seconds, step=STEP_S)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            imagined = None
    if imagined is None or not getattr(imagined, "path", None):
        return lambda after: thing.where_at(after)
    points = imagined.path
    leaves = getattr(imagined, "ends", "") == "leaves"
    last = points[-1][0]

    def at(after: float) -> tuple[float, float] | None:
        if after > last:
            return None if leaves else (points[-1][1], points[-1][2])
        index = min(len(points) - 1, max(0, int(round(after / STEP_S))))
        return points[index][1], points[index][2]

    return at


def _run(scene: _Scene, plan: Plan, goal: tuple[float | None, float | None], held: str, caution: float) -> Imagined:
    """Her thing run under ``plan``, held against the world as it goes, and what that comes to."""
    times = scene.times
    xs, ys = np.empty(len(times)), np.empty(len(times))
    x, y = scene.start
    low_x, low_y, high_x, high_y = scene.bounds
    previous = 0.0
    for index, when in enumerate(times):
        key = next(key for start, key in reversed(plan.held) if start <= when - STEP_S / 2 or start == 0.0)
        vx, vy = scene.ways.get(key, scene.rest) if key else scene.rest
        step = float(when) - previous
        previous = float(when)
        nx, ny = x + vx * step, y + vy * step
        if not _barred(scene.barred, nx, ny):
            x, y = nx, ny
        x = min(max(x, low_x), high_x) if high_x >= low_x else x
        y = min(max(y, low_y), high_y) if high_y >= low_y else y
        xs[index], ys[index] = x, y
    for at, press in plan.presses:
        path = scene.paths.get(press)
        if path is None:
            continue
        for index, when in enumerate(times):
            if when >= at:
                dx, dy = path(float(when) - at)
                xs[index] += dx
                ys[index] += dy
    worth, gets, costs = 0.0, 0, 0
    if scene.worth.size:
        # The plan held against the world at each pace, and what it comes to taken over all of them.
        for layer in range(len(PACES)):
            met, got, lost = _what_it_meets(scene, layer, xs, ys, caution)
            worth += met / len(PACES)
            gets, costs = (got, lost) if PACES[layer] == 1.0 else (gets, costs)
    gx, gy = goal
    if gx is not None or gy is not None:
        reach = max(1.0, max((math.hypot(*way) for way in scene.ways.values()), default=1.0) * float(times[-1]))
        far = math.hypot((xs[-1] - gx) if gx is not None else 0.0, (ys[-1] - gy) if gy is not None else 0.0)
        worth -= FAR_FROM_THE_GOAL * min(2.0, far / reach)
    if plan.first != held:
        worth -= CHANGING
    if not plan.first and len(plan.held) > 1:
        worth -= WAITING * plan.held[1][0]
    return Imagined(plan, worth, gets, costs, (float(xs[-1]), float(ys[-1])))


def _what_it_meets(scene: _Scene, layer: int, xs: np.ndarray, ys: np.ndarray, caution: float) -> tuple[float, int, int]:
    """What her path meets in the world run at one pace: its worth, how many worth getting it gets, how many cost her."""
    where, there = scene.where[layer], scene.there[layer]
    apart_x = np.abs(where[:, :, 0] - xs[:, None])
    apart_y = np.abs(where[:, :, 1] - ys[:, None])
    reach_x = (scene.sizes[:, 0][None, :] + scene.size[0]) / 2
    reach_y = (scene.sizes[:, 1][None, :] + scene.size[1]) / 2
    costly = scene.worth < 0
    # What costs her is given a berth, the wider the lower she is.
    berth = np.where(costly, 2.0 * caution, 0.0)[None, :]
    meets = (apart_x <= reach_x + berth) & (apart_y <= reach_y + berth) & there
    worth, gets, costs, lost_at = 0.0, 0, 0, len(xs)
    firsts = [int(hits[0]) if hits.size else -1 for hits in (np.flatnonzero(meets[:, c]) for c in range(scene.worth.size))]
    for column, first in enumerate(firsts):
        if first >= 0 and costly[column]:
            costs += 1
            worth += scene.worth[column] * caution * (SOONER ** first)
            lost_at = min(lost_at, first)
    for column, first in enumerate(firsts):
        # What is got after she has been hit is not got.
        if first >= 0 and not costly[column] and first < lost_at:
            gets += 1
            worth += scene.worth[column] * (SOONER ** first)
    return worth, gets, costs


def _barred(barred: Any, x: float, y: float) -> bool:
    if not barred:
        return False
    cells, cell = barred
    return bool(cell) and (int(x // cell), int(y // cell)) in cells
