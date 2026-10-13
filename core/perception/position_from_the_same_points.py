"""Measure a path from the same visible marks while an outline changes.

Offline 2026-10-12 marks on a falling body went on moving while its box
included exhaust and pixels at an earlier place. The box's middle was not
the body's path. Marks that keep their names can supply a separate path;
new marks cannot silently take over from the ones that started it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.perception.rigid_motion_from_named_points import rigid_motion
from core.verify import invariant


@dataclass
class PointPose:
    born: float
    at: float
    origin: tuple[float, float]
    anchors: dict[int, tuple[float, float]]
    position: tuple[float, float]
    reference_at: float | None = None


def measured_position(thing: Any) -> tuple[float, float]:
    """Control samples use one measured path, rather than a changing box's middle."""
    position = getattr(thing, "motion_position", None)
    if position is not None and getattr(thing, "motion_measured_at", None) == thing.seen:
        return position
    return float(thing.x), float(thing.y)


def position_from_points(thing: Any, blob: dict, motion: Any, at: float) -> tuple | None:
    """Return previous and current positions supported by continuing owned marks.

    An anchor starts at the current box's middle. Its offset need not be a
    physical centre: only displacement is learned from it. At least three
    previously admitted, spread-out marks must remain, move together, and
    still belong to current visible pixels. They can admit a minority of
    new marks into the same origin. An unbridged replacement starts afresh.
    """
    previous = getattr(thing, "point_pose", None)
    current = _owned_marks(thing, blob, motion, at)
    thing.point_pose = None
    if current is None:
        return None
    origin = float(blob["x"]), float(blob["y"])
    seed = PointPose(thing.born, at, origin, current, origin, reference_at=at)
    if previous is None or previous.born != thing.born or previous.at != thing.seen:
        thing.point_pose = seed
        return None
    names = previous.anchors.keys() & current.keys()
    if len(names) < 3 or len(names) < .6 * len(previous.anchors):
        thing.point_pose = seed
        return None
    starts = np.array([previous.anchors[n] for n in names])
    ends = np.array([current[n] for n in names])
    shifts = ends - starts
    middle = np.median(shifts, axis=0)
    coherent = np.linalg.norm(shifts - middle, axis=1) <= 1.
    if coherent.sum() < 3 or coherent.mean() < .7 or not _spread(starts[coherent]):
        return _continue_rotating_marks(thing, previous, current, names, starts, ends, motion, at, seed)
    if getattr(motion, 'rigid', None) is not None:
        return _continue_rotating_marks(thing, previous, current, names, starts, ends, motion, at, seed)
    middle = np.median(shifts[coherent], axis=0)
    position = tuple(float(v) for v in np.asarray(previous.origin) + middle)
    step = np.subtract(position, previous.position)
    if math.dist(step, (motion.dx, motion.dy)) > 1.:
        thing.point_pose = seed
        return None
    before = previous.position
    qualified = [name for name, good in zip(names, coherent, strict=True) if good]
    anchors = {name: previous.anchors[name] for name in qualified}
    # Current owned marks join only after three retained marks independently
    # located the same body. Back-project them into that origin; no new box
    # centre is substituted. Admissions cannot outnumber the retained marks.
    for name in sorted(current.keys() - names)[:len(qualified) // 2]:
        anchors[name] = tuple(float(v) for v in np.asarray(current[name]) - middle)
    previous.anchors = anchors
    previous.position, previous.at = position, at
    thing.point_pose = previous
    return before, position


def _continue_rotating_marks(thing: Any, previous: PointPose, current: dict, names: set,
                            starts: np.ndarray, ends: np.ndarray, motion: Any, at: float, seed: PointPose) -> tuple | None:
    """Rotate the retained origin with its original marks; admit new marks through that same transform."""
    thing.point_pose = seed
    fit = rigid_motion(starts, ends)
    if fit is None:
        return None
    position = tuple(float(v) for v in fit.apply(previous.origin))
    current_fit = getattr(motion, 'rigid', None)
    expected = (current_fit.apply(previous.position) if current_fit is not None
                else np.asarray(previous.position) + (motion.dx, motion.dy))
    if math.dist(position, expected) > 1.:
        return None
    ordered = list(names)
    qualified = [ordered[i] for i in fit.inliers]
    anchors = {name: previous.anchors[name] for name in qualified}
    for name in sorted(current.keys() - names)[:len(qualified) // 2]:
        anchors[name] = tuple(float(v) for v in fit.inverse(current[name]))
    before = previous.position
    previous.anchors, previous.position, previous.at = anchors, position, at
    thing.point_pose = previous
    return before, position


def _owned_marks(thing: Any, blob: dict, motion: Any, at: float) -> dict | None:
    """The current motion must name unique marks inside current observed support."""
    if getattr(motion, 'rigid', None) is not None and not motion.has_rigid_pose():
        return None
    if (motion is None or motion.before != thing.seen or motion.at != at
            or not all(math.isfinite(v) for v in (at, thing.seen, thing.born, motion.dx, motion.dy))
            or not all(math.isfinite(blob[key]) for key in ("x", "y", "w", "h"))
            or not 0 < at - thing.seen <= .35):
        return None
    names, locations = tuple(motion.points), tuple(motion.locations)
    if (len(names) != len(locations) or len(names) < 3 or len(set(names)) != len(names)
            or motion.support != len(names)):
        return None
    support = blob.get("support")
    if (not isinstance(support, np.ndarray) or support.ndim != 2 or min(blob["w"], blob["h"]) <= 0
            or support.shape != (round(blob["h"]), round(blob["w"]))):
        return None
    left, top = blob["x"] - blob["w"] / 2, blob["y"] - blob["h"] / 2
    current = {}
    for name, (x, y) in zip(names, locations, strict=True):
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        col, row = int(round(x - left)), int(round(y - top))
        if 0 <= col < support.shape[1] and 0 <= row < support.shape[0] and support[row, col]:
            current[name] = (x, y)
    if len(current) < 3 or len(current) < .6 * len(names) or not _spread(np.asarray(list(current.values()))):
        return None
    return current


def _spread(places: np.ndarray) -> bool:
    """Marks along one line cannot locate a patch in both directions."""
    centered = places - places.mean(axis=0)
    return float(np.linalg.svd(centered, compute_uv=False)[-1]) >= 1.


@invariant("perception.a_point_path_keeps_its_original_marks", scope="perception",
           owner="core/perception/position_from_the_same_points.py", observational=False)
def _point_path_invariant() -> tuple:
    from types import SimpleNamespace

    from core.perception.points_that_keep_their_motion import PointMotion

    thing = SimpleNamespace(born=0., seen=1., point_pose=None)
    blob = dict(x=20., y=20., w=20., h=20., support=np.ones((20, 20), bool))
    marks = ((15., 15.), (25., 15.), (15., 25.), (25., 25.))
    def motion(before, at, dx, names):
        return PointMotion(dx, 0., .01, 4, before, at, names, tuple((x + dx, y) for x, y in marks))
    assert position_from_points(thing, blob, motion(1., 1.1, 0., (1, 2, 3, 4)), 1.1) is None
    thing.seen = 1.1
    assert position_from_points(thing, blob, motion(1.1, 1.2, 2., (1, 2, 3, 4)), 1.2) == ((20., 20.), (22., 20.))
    thing.seen = 1.2
    assert position_from_points(thing, blob, motion(1.2, 1.3, 2., (5, 6, 7, 8)), 1.3) is None
    return ()


@invariant("perception.new_marks_need_a_measured_bridge_to_the_same_origin", scope="perception",
           owner="core/perception/position_from_the_same_points.py", observational=False)
def _point_handoff_invariant() -> tuple:
    from types import SimpleNamespace

    from core.perception.points_that_keep_their_motion import PointMotion

    thing = SimpleNamespace(born=0., seen=1., point_pose=None)
    blob = dict(x=20., y=20., w=20., h=20., support=np.ones((20, 20), bool))
    places = {1: (15., 15.), 2: (25., 15.), 3: (15., 25.), 4: (25., 25.)}
    for at, names in ((1.1, (1, 2, 3)), (1.2, (1, 2, 3, 4)), (1.3, (2, 3, 4))):
        motion = PointMotion(0., 0., .01, len(names), thing.seen, at, names, tuple(places[n] for n in names))
        result = position_from_points(thing, blob, motion, at)
        assert result is None if at == 1.1 else result == ((20., 20.), (20., 20.))
        thing.seen = at
    assert thing.point_pose.origin == (20., 20.)
    return ()
