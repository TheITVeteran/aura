"""Known equations for controls with stored response and coupled outputs.

Rules are supplied as data: a channel can approach a commanded target by a
fixed amount each tick, or decay and drift when released. Output matrices
turn channel values into movement and observed properties. No rules or
initial states from an archived game are supplied by this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from core.verify import invariant


@dataclass(frozen=True)
class ChannelRule:
    target: float | None = None
    change: float = 0.
    decay: float = 1.
    drift: float = 0.
    clamp: bool = True
    ordered: bool = False

    def next(self, value: float) -> float:
        """Apply one physical tick, including response that continues after release."""
        if self.target is None:
            return self.decay * value + self.drift
        if self.ordered:
            value += self.change * (self.target > value)
            value -= self.change * (self.target < value)
            return value
        error = self.target - value
        amount = min(abs(error), self.change) if self.clamp else self.change if error else 0.
        return value + math.copysign(amount, error)


@dataclass(frozen=True)
class MotionState:
    position: tuple[float, ...]
    channels: tuple[float, ...]


@dataclass(frozen=True)
class GradualControls:
    """A known forward transition, given a rule for each control and channel."""

    rules: tuple[tuple[ChannelRule, ...], ...]
    movement: tuple[tuple[float, ...], ...]
    properties: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self):
        if not self.rules or not self.rules[0]:
            raise ValueError("controls need at least one channel")
        size = len(self.rules[0])
        if any(len(rules) != size for rules in self.rules):
            raise ValueError("all controls need the same channel state")
        if any(len(row) != size for row in (*self.movement, *self.properties)):
            raise ValueError("outputs must name each channel")
        for rules in self.rules:
            for rule in rules:
                values = (rule.change, rule.decay, rule.drift, *((rule.target,) if rule.target is not None else ()))
                if not all(math.isfinite(value) for value in values) or rule.change < 0:
                    raise ValueError("channel rules must be finite")
        if not all(math.isfinite(value) for row in (*self.movement, *self.properties) for value in row):
            raise ValueError("output coefficients must be finite")

    def advance(self, state: MotionState, control: int) -> MotionState:
        self._validate(state, control)
        channels = tuple(rule.next(value) for rule, value in zip(self.rules[control], state.channels, strict=True))
        movement = self.outputs(channels, self.movement)
        position = tuple(p + delta for p, delta in zip(state.position, movement, strict=True))
        if not all(math.isfinite(value) for value in (*position, *channels)):
            raise ValueError("the predicted state is not finite")
        return MotionState(position, channels)

    def _validate(self, state: MotionState, control: int) -> None:
        if type(control) is not int or not 0 <= control < len(self.rules):
            raise ValueError("the input has no supplied transition")
        if len(state.channels) != len(self.rules[0]) or len(state.position) != len(self.movement):
            raise ValueError("observation and forward model have different dimensions")
        if not all(math.isfinite(value) for value in (*state.position, *state.channels)):
            raise ValueError("forward motion needs a finite state")

    def after(self, state: MotionState, control: int, ticks: int) -> MotionState:
        """Project affine channels in constant work, retaining stored response.

        This predicts an endpoint only. A caller must check intermediate
        contact and failure conditions separately. Targeted or near-unit
        decay laws use bounded physical ticks until their sums are available.
        """
        if type(ticks) is not int or not 0 <= ticks <= 10000:
            raise ValueError("the forecast needs a bounded count of physical ticks")
        self._validate(state, control)
        if ticks == 0:
            return state
        rules = self.rules[control]
        if any(rule.target is not None or 0 < abs(rule.decay - 1.) < 1e-6 for rule in rules):
            for _ in range(ticks):
                state = self.advance(state, control)
            return state
        following, totals = [], []
        for value, rule in zip(state.channels, rules, strict=True):
            if rule.decay == 1.:
                following.append(value + ticks * rule.drift)
                totals.append(ticks * value + rule.drift * ticks * (ticks + 1) / 2)
            else:
                try:
                    power = rule.decay ** ticks
                except OverflowError as error:
                    raise ValueError("the predicted state is not finite") from error
                series = rule.decay * (1. - power) / (1. - rule.decay)
                following.append(value * power + rule.drift * (1. - power) / (1. - rule.decay))
                totals.append(value * series + rule.drift * (ticks - series) / (1. - rule.decay))
        displacement = self.outputs(tuple(totals), self.movement)
        position = tuple(p + delta for p, delta in zip(state.position, displacement, strict=True))
        if not all(math.isfinite(value) for value in (*position, *following)):
            raise ValueError("the predicted state is not finite")
        return MotionState(position, tuple(following))

    @staticmethod
    def outputs(channels: tuple[float, ...], matrix: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
        return tuple(sum(coefficient * value for coefficient, value in zip(row, channels, strict=True))
                     for row in matrix)


@invariant("agency.released_control_keeps_its_stored_response", scope="agency",
           owner="core/agency/controls_that_take_time_to_build.py", observational=False)
def _stored_response_invariant() -> tuple:
    model = GradualControls(((ChannelRule(decay=.8),),), ((1.,),))
    state = MotionState((0.,), (2.,))
    predicted = model.after(state, 0, 4)
    for _ in range(4):
        state = model.advance(state, 0)
    assert math.isclose(predicted.position[0], state.position[0])
    assert math.isclose(predicted.channels[0], state.channels[0])
    assert predicted.position[0] > 0. and predicted.channels[0] > 0.
    return ()
