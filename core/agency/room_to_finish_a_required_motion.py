"""Reserve time and space for cancelling motion and satisfying arrival state.

A body can be nearer its destination and less able to arrive safely.
Predict its resting place and the time needed for the remaining transfer,
then let the forward model show what else happens during that preparation.
The caller supplies measured or source-verified rates in consistent units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.verify import invariant


@dataclass(frozen=True)
class AfterBraking:
    position: float
    other_values: tuple[float, ...]
    time: float


def after_braking(position: float, rate: float, braking: float, *,
                  other_values: tuple[float, ...] = (), other_changes: tuple[float, ...] = ()) -> AfterBraking | None:
    """Project a constant opposing change through a stop, including coupled posture or supply.

    ``braking`` is the positive magnitude opposing ``rate``. Other changes
    are signed rates under that same braking input. This assumes they remain
    constant until the stop; a saturated or changing law needs its own model.
    """
    if (not all(math.isfinite(v) for v in (position, rate, braking, *other_values, *other_changes))
            or braking <= 0 or len(other_values) != len(other_changes)):
        return None
    duration = abs(rate) / braking
    end = position + .5 * rate * duration
    others = tuple(value + change * duration for value, change in zip(other_values, other_changes, strict=True))
    if not all(math.isfinite(value) for value in (duration, end, *others)):
        return None
    return AfterBraking(end, others, duration)


def transfer_from_rest(distance: float, acceleration: float, speed_limit: float) -> float | None:
    """Time for a transfer that accelerates then brakes, with a finite speed cap.

    The ramp is triangular when it never reaches the cap, trapezoidal when
    it does. Initial and terminal rates are zero, with equal acceleration
    and braking magnitudes. Obstacles and changing forces remain outside
    this one-axis calculation.
    """
    if not all(math.isfinite(v) for v in (distance, acceleration, speed_limit)) or min(acceleration, speed_limit) <= 0:
        return None
    distance = abs(distance)
    threshold = speed_limit * speed_limit / acceleration
    result = (2 * math.sqrt(distance / acceleration) if distance <= threshold else
              distance / speed_limit + speed_limit / acceleration)
    return result if math.isfinite(result) else None


def time_to_prepare(stopped: AfterBraking, destination: float, tolerance: float,
                    acceleration: float, speed_limit: float) -> float | None:
    """A feasible stop then transfer needs this time before another constraint is crossed."""
    if (not all(math.isfinite(v) for v in (stopped.position, stopped.time, destination, tolerance))
            or stopped.time < 0 or tolerance < 0):
        return None
    remaining = max(0., abs(destination - stopped.position) - tolerance)
    transfer = transfer_from_rest(remaining, acceleration, speed_limit)
    result = stopped.time + transfer if transfer is not None else math.inf
    return result if math.isfinite(result) else None


@invariant("agency.arrival_preparation_needs_room_to_finish", scope="agency",
           owner="core/agency/room_to_finish_a_required_motion.py", observational=False)
def _preparation_room_invariant() -> tuple:
    stopping = after_braking(10., 4., 2., other_values=(6.,), other_changes=(-3.,))
    assert stopping == AfterBraking(14., (0.,), 2.)
    assert time_to_prepare(stopping, 14., 0., 2., 4.) == 2.
    assert after_braking(10., 4., 0.) is None
    assert transfer_from_rest(10., 0., 4.) is None
    assert math.isclose(transfer_from_rest(8., 2., 4.), 4.)
    return ()
