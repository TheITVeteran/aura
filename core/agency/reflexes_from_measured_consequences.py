"""Select short responses from fresh observations and bounded consequence estimates.

Adapters supply costs in their own units and actions with checked response
models. This circuit knows no objects, games, keys or task vocabulary.
Delivery receipts and later observations are separate from predictions.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass

from core.verify import invariant


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class ReflexResponse:
    action: Hashable
    cost: float
    duration_s: float

    def __post_init__(self) -> None:
        hash(self.action)
        if (self.action is None or not _finite(self.cost) or self.cost < 0
                or not _finite(self.duration_s) or not 0 < self.duration_s <= .25):
            raise ValueError('a reflex response needs a finite cost and short duration')


@dataclass(frozen=True)
class ReflexObservation:
    context: Hashable
    captured_at: float
    cost: float
    responses: tuple[ReflexResponse, ...]

    def __post_init__(self) -> None:
        hash(self.context)
        if (self.context is None or not _finite(self.captured_at) or not _finite(self.cost) or self.cost < 0
                or not isinstance(self.responses, tuple) or len(self.responses) > 32
                or any(not isinstance(r, ReflexResponse) for r in self.responses)
                or len({r.action for r in self.responses}) != len(self.responses)):
            raise ValueError('a reflex observation needs bounded unique responses')


@dataclass(frozen=True)
class ReflexReceipt:
    context: Hashable
    observation: ReflexObservation
    response: ReflexResponse
    delivered_at: float
    ended_at: float | None


@dataclass(frozen=True)
class ReflexOutcome:
    receipt: ReflexReceipt
    captured_at: float | None
    observed_cost: float | None
    reason: str


class ReflexCircuit:
    """Interrupt only for an improving response; retire ineffective model responses.

    Cost after an action is feedback, not a causal proof that the action
    produced it. Two non-improving observations suspend that response until
    the adapter supplies a different model context. Missing data stays unknown.
    """

    def __init__(self, *, max_age_s: float = .15, improvement: float = .5,
                 cooldown_s: float = .06) -> None:
        if (not all(_finite(x) for x in (max_age_s, improvement, cooldown_s))
                or not 0 < max_age_s <= .5 or not 0 < improvement < 1 or not 0 <= cooldown_s <= .5):
            raise ValueError('a reflex circuit needs finite observation and switching bounds')
        self.max_age_s, self.improvement, self.cooldown_s = max_age_s, improvement, cooldown_s
        self.context: Hashable | None = None
        self.pending: ReflexReceipt | None = None
        self.offered: tuple[ReflexObservation, ReflexResponse] | None = None
        self.last_capture = -math.inf
        self.next_at = -math.inf
        self.failures: dict[Hashable, int] = {}
        self.outcomes: deque[ReflexOutcome] = deque(maxlen=64)

    def fresh(self, observation: ReflexObservation, now: float) -> bool:
        return _finite(now) and 0 <= now - observation.captured_at <= self.max_age_s

    def inspect(self, observation: ReflexObservation, now: float) -> ReflexResponse | None:
        self.offered = None
        if not self.fresh(observation, now) or observation.captured_at <= self.last_capture:
            return None
        self.last_capture = observation.captured_at
        if observation.context != self.context:
            self.retire('model context changed')
            self.context, self.failures = observation.context, {}
        if self.pending is not None:
            receipt = self.pending
            if receipt.ended_at is None or observation.captured_at <= receipt.ended_at:
                return None
            self.pending = None
            self.outcomes.append(ReflexOutcome(receipt, observation.captured_at, observation.cost, 'observed'))
            action = receipt.response.action
            if action not in self.failures and len(self.failures) >= 64:
                self.failures.pop(next(iter(self.failures)))
            self.failures[action] = (self.failures.get(action, 0) + 1
                                     if observation.cost >= receipt.observation.cost * self.improvement else 0)
        if now < self.next_at or observation.cost <= 0:
            return None
        eligible = [r for r in observation.responses if self.failures.get(r.action, 0) < 2
                    and r.cost < observation.cost * self.improvement]
        response = min(eligible, key=lambda r: (r.cost, r.duration_s)) if eligible else None
        self.offered = (observation, response) if response is not None else None
        return response

    def delivered(self, observation: ReflexObservation, response: ReflexResponse,
                  began: float, ended: float) -> ReflexReceipt:
        if (not self.fresh(observation, began) or not self.fresh(observation, ended)
                or self.offered != (observation, response)
                or observation.context != self.context or self.pending is not None
                or not _finite(ended) or not began <= ended <= began + .5):
            raise ValueError('a reflex receipt needs current context and an acknowledged delivery')
        self.offered = None
        receipt = ReflexReceipt(observation.context, observation, response, ended, None)
        self.pending = receipt
        self.next_at = ended + self.cooldown_s
        return receipt

    def ended(self, receipt: ReflexReceipt, at: float) -> None:
        if self.pending is not receipt or not _finite(at) or at < receipt.delivered_at:
            return
        self.pending = ReflexReceipt(receipt.context, receipt.observation, receipt.response, receipt.delivered_at, at)
        self.next_at = at + self.cooldown_s

    def retire(self, reason: str) -> None:
        self.offered = None
        if self.pending is not None:
            self.outcomes.append(ReflexOutcome(self.pending, None, None, reason))
            self.pending = None


@invariant('agency.reflexes_need_fresh_improving_observed_consequences', scope='agency',
           owner='core/agency/reflexes_from_measured_consequences.py', observational=False)
def _reflex_consequence_invariant() -> tuple:
    circuit = ReflexCircuit()
    response = ReflexResponse('bounded response', 0., .1)
    sample = ReflexObservation(('owner', 'model'), 1., 4., (response,))
    assert circuit.inspect(sample, 1.01) is response
    receipt = circuit.delivered(sample, response, 1.01, 1.02)
    assert circuit.inspect(sample, 1.02) is None
    circuit.ended(receipt, 1.12)
    assert circuit.inspect(ReflexObservation(sample.context, 1.2, 0., (response,)), 1.21) is None
    assert circuit.outcomes[-1].observed_cost == 0.
    assert circuit.inspect(ReflexObservation(sample.context, 2., 4., (response,)), 3.) is None
    return ()
