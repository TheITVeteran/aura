"""Integrate a checked direct-response model over acknowledged input history."""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.verify import invariant


@dataclass(frozen=True)
class VectorInput:
    at: float
    velocity: tuple[float, ...] | None
    began: float | None = None


def input_clock_uncertainty(began: float, until: float, inputs: tuple[VectorInput, ...]) -> tuple | None:
    """Bound displacement uncertainty while a requested transition awaited acknowledgement."""
    if (not isinstance(inputs, tuple) or not 1 <= len(inputs) <= 128
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                       for v in (began, until)) or not began <= until <= began + 2.
            or any(not isinstance(i, VectorInput) or not isinstance(i.at, (int, float))
                   or isinstance(i.at, bool) or not math.isfinite(i.at) for i in inputs)
            or any(a.at > b.at for a, b in zip(inputs, inputs[1:], strict=False))):
        return None
    current = next((i for i in reversed(inputs) if i.at <= began), None)
    if (current is None or not isinstance(current.velocity, tuple) or not 1 <= len(current.velocity) <= 16
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                       for v in current.velocity)):
        return None
    error = [0.] * len(current.velocity)
    for following in (i for i in inputs if i.at > began):
        start = following.at if following.began is None else following.began
        if (not isinstance(start, (int, float)) or isinstance(start, bool) or not math.isfinite(start) or start > following.at
                or not isinstance(current.velocity, tuple) or not isinstance(following.velocity, tuple)
                or len(current.velocity) != len(following.velocity)):
            return None
        width = max(0., min(until, following.at) - max(began, start))
        for axis, (before, after) in enumerate(zip(current.velocity, following.velocity, strict=True)):
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (before, after)):
                return None
            error[axis] += max(abs(before), abs(after), abs(after - before)) * width
        current = following
        if following.at >= until:
            break
    return tuple(error)


def position_from_inputs(origin: tuple[float, ...], began: float, until: float,
                         inputs: tuple[VectorInput, ...], bounds: tuple[tuple[float, float], ...]) -> tuple | None:
    """Unknown intervals remain unknown; boundaries saturate each acknowledged span."""
    if (not isinstance(origin, tuple) or not 1 <= len(origin) <= 16
            or not isinstance(bounds, tuple) or len(bounds) != len(origin)
            or any(not isinstance(pair, tuple) or len(pair) != 2 for pair in bounds)
            or not isinstance(inputs, tuple) or not 1 <= len(inputs) <= 128
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                       for v in (*origin, began, until, *(v for pair in bounds for v in pair)))
            or not began <= until <= began + 2. or any(low > high for low, high in bounds)
            or any(not isinstance(i, VectorInput) or not isinstance(i.at, (int, float))
                   or isinstance(i.at, bool) or not math.isfinite(i.at) for i in inputs)
            or any(a.at > b.at for a, b in zip(inputs, inputs[1:], strict=False))):
        return None
    previous = next((i for i in reversed(inputs) if i.at <= began), None)
    if previous is None:
        return None
    position, at = origin, began
    for following in (*[i for i in inputs if began < i.at < until], VectorInput(until, None)):
        velocity = previous.velocity
        if (not isinstance(velocity, tuple) or len(velocity) != len(origin)
                or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in velocity)):
            return None
        position = tuple(min(max(p + v * (following.at - at), low), high)
                         for p, v, (low, high) in zip(position, velocity, bounds, strict=True))
        previous, at = following, following.at
    return position


@invariant('agency.input_history_covers_the_observed_response_interval', scope='agency',
           owner='core/agency/what_acknowledged_inputs_predict.py', observational=False)
def _acknowledged_prediction_invariant() -> tuple:
    history = (VectorInput(1., (100.,)), VectorInput(1.05, (-100.,)), VectorInput(1.17, (0.,)))
    predicted = position_from_inputs((40.,), 1., 1.2, history, ((0., 100.),))
    assert predicted is not None and math.isclose(predicted[0], 33.)
    assert position_from_inputs((40.,), .9, 1.2, history, ((0., 100.),)) is None
    assert position_from_inputs((40.,), 1., 1.2, (VectorInput(1., None),), ((0., 100.),)) is None
    return ()
