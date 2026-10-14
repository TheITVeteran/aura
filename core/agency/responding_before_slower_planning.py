"""Connect fresh play observations to the general consequence reflex circuit."""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

from core.agency.inputs_while_measuring_control import hold_control, release_control_if_owned
from core.agency.reflexes_from_measured_consequences import (
    ReflexCircuit,
    ReflexObservation,
    ReflexResponse,
)
from core.perception.translation_from_visible_support import motion_was_measured
from core.runtime.task_ownership import fire_and_forget


@dataclass
class _ReflexInput:
    context: tuple
    key: str
    since: float
    until: float
    receipt: Any


def _response_context(run: Any, mine: Any, hers: Any, choosing: Any, ways: dict) -> tuple:
    owner = mine.number, mine.born, hers.identification.epoch, tuple(choosing.moves.shape)
    previous = getattr(run, '_reflex_response_model', None)
    changed = (previous is None or previous[0] != owner or set(previous[1]) != set(ways)
               or any(math.dist(way, previous[1][key]) > max(2., .2 * math.hypot(*previous[1][key]))
                      for key, way in ways.items()))
    if changed:
        version = 1 if previous is None else previous[2] + 1
        previous = run._reflex_response_model = owner, dict(ways), version
    return (*owner, previous[2])


def _observation(run: Any, hers: Any, choosing: Any, at: float, now: float) -> ReflexObservation | None:
    mine = choosing.mine
    if (mine is None or not 0 <= now - at <= .15 or not motion_was_measured(mine, at)
            or mine.seen != at or hers.number != mine.number or hers.kind != mine.kind
            or getattr(hers, 'carried', False) or choosing.pointing or choosing.lifts
            or (choosing.faces is not None and choosing.faces.turns_her())
            or len(choosing.moves.things) > 128 or len(run.keys) > 31 or run.held_trying
            or getattr(run, '_prepared_input_set', None) is not None
            or run.letting_go or run.charging.key
            or (run.shown is not None and run.shown.asks(at))
            or (run.coming is not None and run.coming.asks(at))):
        return None
    ways = {}
    for key in run.keys:
        measured = hers.way_of(key)
        if (measured is not None and len(measured) == 2 and all(math.isfinite(v) for v in measured)
                and key in choosing.ways and math.dist(measured, choosing.ways[key]) <= 2.):
            ways[key] = tuple(measured)
    if not ways or (run.held and run.held not in ways):
        return None
    ways[''] = (0., 0.)
    through = choosing.way_through
    if through is not None and through.walls and through.a_floor(ways):
        ways = {key: way for key, way in ways.items() if key == run.held
                or not through.ahead_is_a_wall(float(mine.x), float(mine.y), way, (float(mine.w), float(mine.h)))}
    # Evaluate the same response horizon for the held input and each alternative.
    # Picture age and measured delivery latency consume some of that horizon.
    latency = now - at + choosing.response_s
    if not math.isfinite(latency) or not 0 <= latency <= .2:
        return None
    times = tuple(latency + t for t in (.04, .08, .12, .18))
    prior = float(mine.vx), float(mine.vy)
    if not all(math.isfinite(v) for v in prior):
        return None
    costs = {key: choosing.danger_along(lambda after, v=way:
             (prior[0] * min(after, latency) + v[0] * max(0., after - latency),
              prior[1] * min(after, latency) + v[1] * max(0., after - latency)), times=times)
             for key, way in ways.items()}
    if any(not math.isfinite(v) or v < 0 for v in costs.values()):
        return None
    context = _response_context(run, mine, hers, choosing, ways)
    return ReflexObservation(context, float(at), costs[run.held],
                             tuple(ReflexResponse(key, cost, .12) for key, cost in costs.items() if key != run.held))


async def _release(hands: Any, run: Any, hers: Any, active: _ReflexInput) -> None:
    if getattr(run, '_reflex_input', None) is not active:
        return
    if run.held == active.key and run.held_since == active.since:
        await release_control_if_owned(hands, run, hers, active.key, active.since)
        run._reflex_circuit.ended(active.receipt, time.monotonic())
    else:
        run._reflex_circuit.retire('physical input preempted')
    if getattr(run, '_reflex_input', None) is active:
        run._reflex_input = None


async def _expire(hands: Any, run: Any, hers: Any, active: _ReflexInput) -> None:
    await asyncio.sleep(max(0., active.until - time.monotonic()))
    await _release(hands, run, hers, active)


async def respond_before_planning(hands: Any, run: Any, hers: Any, choosing: Any, at: float) -> bool:
    """Return control only while a short reflex owns the actual physical input."""
    now = time.monotonic()
    circuit = getattr(run, '_reflex_circuit', None)
    if circuit is None:
        circuit = run._reflex_circuit = ReflexCircuit()
    observed = _observation(run, hers, choosing, at, now)
    active = getattr(run, '_reflex_input', None)
    if active is not None:
        owned = run.held == active.key and run.held_since == active.since
        if (observed is None or observed.context != active.context or not owned or now >= active.until):
            await _release(hands, run, hers, active)
            if not owned or observed is None or observed.context != active.context:
                circuit.retire('response ownership or observation unavailable')
        else:
            return True
    if observed is None:
        circuit.retire('current response model unavailable')
        return False
    response = circuit.inspect(observed, now)
    if response is None:
        return False
    began = time.monotonic()
    await hold_control(hands, run, hers, response.action, at)
    delivered = time.monotonic()
    if not circuit.fresh(observed, delivered):
        await release_control_if_owned(hands, run, hers, run.held, run.held_since)
        circuit.retire('delivery exceeded the observation deadline')
        return False
    receipt = circuit.delivered(observed, response, began, delivered)
    active = _ReflexInput(observed.context, response.action, run.held_since,
                         delivered + response.duration_s, receipt)
    run._reflex_input = active
    run._reflex_input_deadline = fire_and_forget(_expire(hands, run, hers, active),
                name='short_reflex_input_deadline', owner='realtime_control', bounded=True)
    return True


def reflex_record(run: Any) -> dict:
    circuit = getattr(run, '_reflex_circuit', None)
    if circuit is None:
        return {'observed_outcomes': 0, 'unknown_outcomes': 0, 'outcomes': []}
    circuit.retire('play observation owner departed')
    return {'observed_outcomes': sum(o.observed_cost is not None for o in circuit.outcomes),
            'unknown_outcomes': sum(o.observed_cost is None for o in circuit.outcomes),
            'outcomes': [{'action': o.receipt.response.action, 'predicted_cost': o.receipt.response.cost,
                          'cost_before': o.receipt.observation.cost, 'cost_after': o.observed_cost,
                          'delivered_at': o.receipt.delivered_at, 'ended_at': o.receipt.ended_at,
                          'captured_at': o.captured_at, 'reason': o.reason} for o in circuit.outcomes]}
