"""An input can change speed or bend motion without choosing one screen direction.

A forward push on a turned body has a different screen vector. Compare
repeatable changes in speed and signed curvature, each measured within an
isolated input interval. These are control witnesses, never a facing label.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any

from core.verify import invariant


def _slope(times: list[float], values: list[float]) -> tuple[float, float]:
    middle_t, middle = statistics.fmean(times), statistics.fmean(values)
    spread = sum((t - middle_t) ** 2 for t in times)
    rate = sum((t - middle_t) * (v - middle) for t, v in zip(times, values, strict=True)) / spread
    residual = max(abs(v - middle - rate * (t - middle_t)) for t, v in zip(times, values, strict=True))
    return rate, residual


def _within_one_input(samples: Any) -> tuple[float, float] | None:
    if len(samples) < 4 or any(len(sample) != 3 for sample in samples):
        return None
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for sample in samples for v in sample):
        return None
    times = [sample[0] for sample in samples]
    if times[-1] - times[0] < .15 or any(b <= a for a, b in zip(times, times[1:], strict=False)):
        return None
    speeds = [math.hypot(vx, vy) for _at, vx, vy in samples]
    change, error = _slope(times, speeds)
    if error > max(2., .15 * (max(speeds) - min(speeds))):
        return None
    lateral = 0.
    if min(speeds) > 4.:
        angles = [math.atan2(samples[0][2], samples[0][1])]
        for _at, vx, vy in samples[1:]:
            difference = (math.atan2(vy, vx) - angles[-1] + math.pi) % (2 * math.pi) - math.pi
            if abs(difference) >= math.pi / 2:
                return None  # direction is aliased across these pictures
            angles.append(angles[-1] + difference)
        turn, error = _slope(times, angles)
        if error > .08:
            return None
        lateral = turn * statistics.median(speeds)
    return change, lateral


def response_in_motion_coordinates(evidence: Any) -> tuple[float, float]:
    """Repeated isolated trials, including neutral, must distinguish their response.

    An autonomous curve shared by every input supplies no effect. An initial
    heading is a nuisance variable, removed by expressing speed change and
    signed lateral acceleration in the trajectory's own frame.
    """
    from core.agency.which_one_answers_to_her import PUSHED_FOR_S, _contingency

    groups: dict[str, list] = defaultdict(list)
    for (key, began), samples in getattr(evidence, 'timed', {}).items():
        if (not isinstance(key, str) or not isinstance(began, (int, float)) or not math.isfinite(began)
                or not samples or samples[0][0] <= began or (key, began) in getattr(evidence, 'interrupted', ())):
            continue
        response = _within_one_input(samples)
        if response is not None:
            groups[key].append(response)
    groups = {key: values for key, values in groups.items() if len(values) >= 2}
    if '' not in groups or len(groups) < 2:
        return 0., 0.
    statistic, gap = _contingency(groups)
    return statistic, gap * PUSHED_FOR_S


@invariant('agency.rotated_control_trials_keep_their_own_motion', scope='agency',
           owner='core/agency/an_input_answers_relative_to_motion.py', observational=False)
def _motion_coordinates_invariant() -> tuple:
    from types import SimpleNamespace

    evidence = SimpleNamespace(timed={})
    for index, (key, heading) in enumerate((('a', 0.), ('', .4), ('a', math.pi), ('', -1.))):
        began = 1. + index
        evidence.timed[key, began] = [(began + elapsed, (40 + (60 if key else 0) * elapsed) * math.cos(heading),
                                     (40 + (60 if key else 0) * elapsed) * math.sin(heading))
                                     for elapsed in (.2, .3, .4, .5)]
    statistic, effect = response_in_motion_coordinates(evidence)
    assert statistic > 20. and effect > 20.
    evidence.timed = {(key, began): samples for (key, began), samples in evidence.timed.items() if key}
    assert response_in_motion_coordinates(evidence) == (0., 0.)
    return ()
