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
    ReflexEffect,
    ReflexObservation,
    ReflexResponse,
)
from core.agency.what_acknowledged_inputs_predict import (
    VectorInput,
    input_clock_uncertainty,
    position_from_inputs,
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
    origin: tuple[float, float]
    ways: dict
    known_origin: bool
    before_key: str
    requested_at: float


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


def _same_response_owner(run: Any, hers: Any, choosing: Any, active: _ReflexInput, at: float) -> bool:
    mine = choosing.mine
    return (hers.number == active.context[0] and hers.identification.epoch == active.context[2]
            and tuple(choosing.moves.shape) == active.context[3]
            and (mine is None or (mine.number == active.context[0] and mine.born == active.context[1]
                                 and mine.kind == hers.kind))
            and not getattr(hers, 'carried', False) and not choosing.pointing
            and not (choosing.faces is not None and choosing.faces.turns_her())
            and not run.held_trying and not run.letting_go and not run.charging.key
            and getattr(run, '_prepared_input_set', None) is None
            and not (run.shown is not None and run.shown.asks(at))
            and not (run.coming is not None and run.coming.asks(at)))


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
    prior = ways[run.held]
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
        if await release_control_if_owned(hands, run, hers, active.key, active.since):
            run._reflex_circuit.ended(active.receipt, time.monotonic())
            run._reflex_feedback_input = active
        else:
            run._reflex_circuit.retire('physical release ownership changed')
    else:
        run._reflex_circuit.retire('physical input preempted')
    if getattr(run, '_reflex_input', None) is active:
        run._reflex_input = None


async def abort_reflex_response(hands: Any, run: Any, hers: Any) -> None:
    """Another input ends this response before its own physical delivery."""
    active = getattr(run, '_reflex_input', None)
    if active is None:
        return
    await _release(hands, run, hers, active)
    run._reflex_circuit.retire('another input took control')


def _effect(run: Any, hers: Any, choosing: Any, observed: ReflexObservation) -> ReflexEffect | None:
    active, receipt = getattr(run, '_reflex_feedback_input', None), run._reflex_circuit.pending
    if (active is None or receipt is None or receipt.ended_at is None
            or active.context != observed.context or receipt.context != observed.context
            or active.receipt.delivered_at != receipt.delivered_at
            or observed.captured_at <= receipt.ended_at or not active.known_origin
            or hers._held_at(choosing.at) is None):
        return None
    duration = receipt.ended_at - receipt.delivered_at
    mine = choosing.mine
    tall, wide = choosing.moves.shape
    held = [h for h in hers._held if h[0] <= observed.captured_at]
    before = next((i for i in reversed(held) if i[0] <= receipt.observation.captured_at), None)
    if before is None:
        return None
    inputs_by_stamp = {float(at): VectorInput(float(at), active.ways.get(key))
                       for at, key, _trying in held if at >= before[0]}
    for began, ended, key in getattr(run, '_input_clock_receipts', []):
        if before[0] <= ended:
            inputs_by_stamp[float(ended)] = VectorInput(float(ended), active.ways.get(key), float(began))
    inputs = tuple(inputs_by_stamp[t] for t in sorted(inputs_by_stamp))
    bounds = (mine.w / 2, wide - mine.w / 2), (mine.h / 2, tall - mine.h / 2)
    expected = position_from_inputs(active.origin, receipt.observation.captured_at, observed.captured_at, inputs,
                                   bounds)
    if expected is None:
        return None
    clock = input_clock_uncertainty(receipt.observation.captured_at, observed.captured_at, inputs)
    if clock is None:
        return None
    alternative_inputs = tuple(i for i in inputs if i.at <= active.requested_at) + (
        VectorInput(active.requested_at, active.ways.get(active.before_key)),)
    alternative = position_from_inputs(active.origin, receipt.observation.captured_at,
                                       observed.captured_at, alternative_inputs, bounds)
    return ReflexEffect(receipt.context, receipt.delivered_at, observed.captured_at,
                        (float(mine.x), float(mine.y)), expected,
                        4. + .2 * math.hypot(*active.ways[active.key]) * duration + math.hypot(*clock), alternative)


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
        same_owner = _same_response_owner(run, hers, choosing, active, at)
        changed_model = observed is not None and observed.context != active.context
        if (not same_owner or changed_model or not owned or now >= active.until):
            await _release(hands, run, hers, active)
            if not owned or not same_owner or changed_model:
                circuit.retire('response owner or model changed')
        else:
            return True
    receipt, feedback = circuit.pending, getattr(run, '_reflex_feedback_input', None)
    if receipt is not None and receipt.ended_at is not None and feedback is not None:
        if not _same_response_owner(run, hers, choosing, feedback, at):
            circuit.retire('response feedback owner changed')
        elif now > receipt.ended_at + .15:
            circuit.retire('response feedback deadline expired')
        elif observed is None or observed.captured_at <= receipt.ended_at:
            return True
    if observed is None:
        circuit.retire('current response model unavailable')
        return False
    response = circuit.inspect(observed, now, _effect(run, hers, choosing, observed))
    if response is None:
        return False
    began = time.monotonic()
    before_key = run.held
    ways = {key: tuple(hers.way_of(key)) for key in ('', *run.keys) if hers.way_of(key) is not None}
    known_origin = hers._held_at(at) is not None
    await hold_control(hands, run, hers, response.action, at)
    delivered = time.monotonic()
    if not circuit.fresh(observed, delivered):
        await release_control_if_owned(hands, run, hers, run.held, run.held_since)
        circuit.retire('delivery exceeded the observation deadline')
        return False
    receipt = circuit.delivered(observed, response, began, delivered)
    mine = choosing.mine
    active = _ReflexInput(observed.context, response.action, run.held_since,
                         delivered + response.duration_s, receipt,
                         (float(mine.x), float(mine.y)), ways, known_origin, before_key, began)
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
                          'response_observed': o.response_observed,
                          'motion': None if o.effect is None else {'observed': o.effect.observed,
                                    'expected': o.effect.expected, 'alternative': o.effect.alternative,
                                    'tolerance': o.effect.tolerance},
                          'delivered_at': o.receipt.delivered_at, 'ended_at': o.receipt.ended_at,
                          'captured_at': o.captured_at, 'reason': o.reason} for o in circuit.outcomes]}
