"""What she has found she cannot get to: the cells beyond what stops her, and places she gets no nearer to.

Kept apart from the play it serves (core/agency/playing_as_it_happens.py), which asks it every picture.
"""
from __future__ import annotations

import math
from typing import Any

__all__ = ["NO_NEARER_S", "OUT_OF_REACH_S", "out_of_her_reach"]


#: How long her thing stays put under a way held before what lies that way is taken to be out of her reach.
OUT_OF_REACH_S = 0.6

#: How long she makes for a place and gets no nearer before it is taken to be out of her reach.
NO_NEARER_S = 1.5


def out_of_her_reach(run: Any, choosing: Any, at: float) -> None:
    """What she has found she cannot get to: the cells beyond what stops her, and places she gets no nearer to.

    A room has walls, and the ground she means to go over is only the ground
    she can get to: a person walking into a wall learns there is a wall that way
    and looks elsewhere. Where she holds a way and her thing does not go, the
    cells that way to the edge of the picture, across the band of her thing's
    own size. Where she makes for a place, by whatever ways, and for a while
    gets no nearer to it, that place: LIVE 2026-10-07 her hero went left and
    right under a spot on the wall above it a hundred and forty times.

    Where her keys take her every way over a floor seen from above, what stops
    her is a wall with ends, not the edge of what she can get to: the cells
    just ahead are marked a wall and the way round it is found.
    """
    mine, cell = choosing.mine, run.cell
    if mine is None or not cell:
        run.stuck = run.making_for = None
        return
    tall, wide = choosing.moves.shape
    # On a floor seen from above, a wall met is gone round (core/agency/a_way_through.py): how near she is is how far
    # the way round is, a place the walls found shut off is out of her reach, and a wall newly found is getting
    # somewhere, though it takes her no nearer.
    through = getattr(run, "way_through", None)
    through = through if through is not None and through.a_floor(choosing.ways) else None
    (gx, gy), why, _aim = choosing.target()
    if why in ("meet", "cover") and gx is not None and gy is not None:
        place, far = (int(gx // cell), int(gy // cell)), math.hypot(gx - mine.x, gy - mine.y)
        walls = len(through.walls) if through is not None else 0
        if through is not None and walls:
            far = through.way_length(float(mine.x), float(mine.y), gx, gy, wide, tall)
        if math.isinf(far):
            run.barred.add(place)
            run.making_for = None
        elif (run.making_for is None or run.making_for[0] != place or far < run.making_for[1] - 0.25 * cell
                or walls != run.making_for[3]):
            run.making_for = (place, far, at, walls)
        elif walls:
            # With walls known on a floor, a place is out of her reach when they leave no way to it, and only then:
            # where she gets no nearer along a way that is there, what stops her is a wall not yet found, and being
            # stopped by it finds it.
            pass
        elif at - run.making_for[2] >= NO_NEARER_S:
            run.barred.add(place)
            run.making_for = None
    else:
        run.making_for = None
    way = choosing.ways.get(run.held) if run.held else None
    if way is None:
        run.stuck = None
        return
    if run.stuck is None or run.stuck[0] != run.held or math.dist(run.stuck[1:3], (mine.x, mine.y)) > 0.25 * cell:
        run.stuck = (run.held, mine.x, mine.y, at)
        return
    if at - run.stuck[3] < OUT_OF_REACH_S:
        return
    # What she was stopped by, where it is a thing on the screen, is a thing in her way (core/agency/who_is_who.py).
    who = getattr(run, "who", None)
    if who is not None:
        who.stopped(mine, way, choosing.moves.things.values())
    if through is not None:
        through.stopped(float(mine.x), float(mine.y), way, (float(mine.w), float(mine.h)))
        run.stuck = None
        return
    along = 0 if abs(way[0]) >= abs(way[1]) else 1
    step = 1 if way[along] > 0 else -1
    middle, half = (mine.y, mine.h / 2) if along == 0 else (mine.x, mine.w / 2)
    band = range(int((middle - half) // cell), int((middle + half) // cell) + 1)
    ahead = int((mine.x, mine.y)[along] // cell) + step
    while 0 <= ahead <= int((wide, tall)[along] // cell):
        for across in band:
            run.barred.add((ahead, across) if along == 0 else (across, ahead))
        ahead += step
    run.stuck = None
