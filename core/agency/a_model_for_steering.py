"""Use a bounded kinematic hypothesis before its controls have been calibrated.

An owned body's current motion anchors the model. Opaque controls acquire turn
priors only when a caller explicitly arms them. Their signs and rates are
assumptions until current input receipts supply observations. This component
plans experiments; it grants neither body identity nor measured control status.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from core.verify import invariant

Owner = tuple[int, float, int]


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _arc(x: float, y: float, heading: float, speed: float, rate: float, seconds: float) -> tuple[float, float]:
    half = rate * seconds / 2
    distance = speed * seconds * (math.sin(half) / half if abs(half) > 1e-9 else 1.)
    return x + distance * math.cos(heading + half), y + distance * math.sin(heading + half)


@dataclass(frozen=True)
class SteeringPrediction:
    model: str
    rate: float
    speed: float
    endpoint: tuple[float, float]
    heading: float
    path: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class SteeringAction:
    key: str
    duration: float
    owner: Owner
    prediction_id: int
    predictions: tuple[SteeringPrediction, ...]
    status: str = 'provisional'


@dataclass(frozen=True)
class RateEstimate:
    rate: float
    uncertainty: float
    observations: int
    began: float


@dataclass(frozen=True)
class _Pose:
    at: float
    x: float
    y: float
    heading: float | None
    vx: float
    vy: float
    error: float


class ProvisionalSteeringBank:
    """Competing finite forecasts, corrected by fresh owned observations.

    ``arm`` supplies eligibility, not a direction or rate. A prior's turn
    radius is one current body extent. Neutral travel competes with controlled
    curvature. All public action records remain provisional, even after a fit.
    """

    def __init__(self, *, longest_gap: float = .35) -> None:
        if not _finite(longest_gap) or longest_gap <= 0:
            raise ValueError('invalid observation gap')
        self.longest_gap = longest_gap
        self.owner: Owner | None = None
        self.turn_keys: set[str] = set()
        self.body_extent = 1.
        self.estimates: dict[str, RateEstimate] = {}
        self.retired_keys: set[str] = set()
        self.pending: SteeringAction | None = None
        self.last_result = 'unobserved'
        self._pose: _Pose | None = None
        self._origin: _Pose | None = None
        self._receipt: tuple[str, float] | None = None
        self._samples: deque[tuple[float, float, float]] = deque(maxlen=32)
        self._drivers: dict[str, tuple[float, float, float]] = {}
        self._prediction_id = 0
        self._delivered_at: float | None = None
        self._delay = 0.
        self._stationary_probe = False

    def arm(self, turn_keys: Iterable[str], *, body_extent: float | None = None) -> None:
        """Only caller-qualified control roles may receive a turning hypothesis."""
        self.turn_keys = {key for key in turn_keys if isinstance(key, str) and key}
        if body_extent is not None:
            if not _finite(body_extent) or body_extent <= 0:
                raise ValueError('invalid body extent')
            self.body_extent = body_extent

    def _reset(self, owner: Owner | None) -> None:
        self.owner = owner
        self.estimates.clear()
        self.retired_keys.clear()
        self.pending = None
        self._pose = self._origin = None
        self._receipt = None
        self._samples.clear()
        self._drivers.clear()
        self._delivered_at = None
        self._stationary_probe = False
        self.last_result = 'unobserved'

    def saw(self, owner: Owner | None, key: str, began: float, at: float, x: float, y: float,
            vx: float, vy: float, *, position_error: float = 1., measured: bool = True) -> None:
        """An absent interval breaks verification; it cannot teach a stationary response."""
        valid_owner = (isinstance(owner, tuple) and len(owner) == 3 and isinstance(owner[0], int)
                       and isinstance(owner[2], int) and _finite(owner[1]) and owner[2] >= 0)
        if not valid_owner:
            self._reset(None)
            return
        if owner != self.owner:
            self._reset(owner)
        if (not measured or not isinstance(key, str) or not _finite(began, at, x, y, vx, vy, position_error)
                or began > at or position_error < 0):
            self.pending = None
            self._pose = None
            self._samples.clear()
            self._receipt = None
            self.last_result = 'unobserved'
            return
        previous = self._pose
        if previous is not None and at <= previous.at:
            return
        gap = previous is not None and at - previous.at > self.longest_gap
        if gap:
            self.pending = None
            self._samples.clear()
            self._receipt = None
            self.last_result = 'observation gap'
        speed = math.hypot(vx, vy)
        interval = at - previous.at if previous is not None and not gap else None
        velocity_error = ((position_error + previous.error) / interval
                          if interval is not None and interval > 0 else math.inf)
        moving = speed > velocity_error if interval is not None else speed > 0
        heading = math.atan2(vy, vx) if moving else (previous.heading if previous and not gap else None)
        if heading is not None and moving and key not in self.turn_keys and key:
            sign = 1.
            if previous is not None and previous.heading is not None:
                sign = 1. if math.cos(heading - previous.heading) >= 0 else -1.
            if sign < 0:
                heading = _wrap(heading + math.pi)
            self._drivers[key] = (sign * speed, heading, at)
        elif heading is not None and moving and previous is not None and previous.heading is not None:
            if math.cos(heading - previous.heading) < 0:
                heading = _wrap(heading + math.pi)
        pose = _Pose(at, x, y, heading, vx, vy, position_error)
        receipt = key, began
        if receipt != self._receipt:
            self._receipt = receipt
            self._samples.clear()
        if not moving:
            self._samples.clear()
        # A velocity interval crossing input delivery does not belong to this input.
        if (moving and heading is not None and interval is not None and previous is not None
                and previous.at >= began and not gap):
            angle_error = math.asin(min(1., velocity_error / speed))
            if self._samples:
                heading = self._samples[-1][1] + _wrap(heading - self._samples[-1][1])
            self._samples.append((at, heading, angle_error))
            estimate = self._fit(began)
            if estimate is not None:
                self.estimates[key] = estimate
                self._retire_unresponsive_keys()
            elif len(self._samples) >= 2 and key in self.estimates and self.estimates[key].began == began:
                del self.estimates[key]
                self.retired_keys.discard(key)
                self.last_result = 'rate contradicted'
        self._pose = pose
        self._verify(key, began, pose, previous, gap)

    def _fit(self, began: float) -> RateEstimate | None:
        """Intersect error-bounded angular slopes; shared position errors are not independent samples."""
        if len(self._samples) < 2:
            return None
        low, high = -math.inf, math.inf
        for i, (at, angle, error) in enumerate(self._samples):
            for after, direction, uncertainty in tuple(self._samples)[i + 1:]:
                span = after - at
                low = max(low, (direction - angle - uncertainty - error) / span)
                high = min(high, (direction - angle + uncertainty + error) / span)
        if not _finite(low, high) or low > high:
            return None
        times, angles, _errors = zip(*self._samples, strict=True)
        middle_t, middle_a = sum(times) / len(times), sum(angles) / len(angles)
        denom = sum((at - middle_t) ** 2 for at in times)
        rate = sum((at - middle_t) * (angle - middle_a) for at, angle in zip(times, angles, strict=True)) / denom
        rate = min(high, max(low, rate))
        return RateEstimate(rate, max(rate - low, high - rate), len(times), began)

    def _retire_unresponsive_keys(self) -> None:
        neutral = self.estimates.get('')
        if neutral is None:
            return
        for key in self.turn_keys:
            fit = self.estimates.get(key)
            if fit is None:
                continue
            error = fit.uncertainty + neutral.uncertainty
            prior = max(math.hypot(self._pose.vx, self._pose.vy) if self._pose else 0., 1.) / self.body_extent
            if abs(fit.rate - neutral.rate) <= error and error < prior / 2:
                self.retired_keys.add(key)
            elif abs(fit.rate - neutral.rate) > error:
                self.retired_keys.discard(key)

    def _rate(self, key: str, speed: float) -> tuple[tuple[float, float, str], ...]:
        neutral = self.estimates.get('')
        base, base_error = (neutral.rate, neutral.uncertainty) if neutral else (0., 0.)
        fit = self.estimates.get(key)
        if fit is not None:
            return ((fit.rate, fit.uncertainty, 'fitted moving turn'),)
        radius = max(self.body_extent, 4 * (self._pose.error if self._pose else 0.), 1.)
        prior = abs(speed) / radius
        return ((base - prior, base_error + prior, 'moving turn prior'),
                (base + prior, base_error + prior, 'moving turn prior'))

    def _prediction(self, pose: _Pose, speed: float, rate: float, error_rate: float, duration: float,
                    delay: float, model: str, *, heading: float | None = None) -> SteeringPrediction:
        heading = pose.heading if heading is None else heading
        assert heading is not None
        start_x, start_y = pose.x + pose.vx * delay, pose.y + pose.vy * delay
        samples = []
        for step in range(1, 5):
            after = duration * step / 4
            x, y = _arc(start_x, start_y, heading, speed, rate, after)
            error = pose.error + abs(speed) * error_rate * after * after / 2
            samples.append((delay + after, x, y, error))
        return SteeringPrediction(model, rate, speed, samples[-1][1:3],
                                  _wrap(heading + rate * duration), tuple(samples))

    def action(self, keys: Iterable[str], driver: str, target: tuple[float, float], *, horizon: float = .2,
               clearance: Callable[[float, float, float], bool] | None = None,
               delay_s: float = 0., now: float | None = None) -> SteeringAction | None:
        """Select one finite target-directed pulse, with alternatives kept for the next observation."""
        pose = self._pose
        if (pose is None or self.owner is None or pose.heading is None or self.pending is not None
                or not _finite(horizon, delay_s, *target) or horizon <= 0 or delay_s < 0
                or delay_s > self.longest_gap or driver not in self._drivers
                or (now is not None and (not _finite(now) or not 0 <= now - pose.at <= self.longest_gap))):
            return None
        controls = set(keys)
        if driver not in controls:
            return None
        driving_speed, _driver_heading, driver_at = self._drivers[driver]
        if pose.at - driver_at > 4 * self.longest_gap and not self.estimates:
            return None
        signed_speed = pose.vx * math.cos(pose.heading) + pose.vy * math.sin(pose.heading)
        duration = min(horizon, self.longest_gap,
                       self.body_extent / max(abs(driving_speed), abs(signed_speed), 1.))
        neutral = self.estimates.get('')
        base, base_error = (neutral.rate, neutral.uncertainty) if neutral else (0., 0.)
        eligible = sorted((self.turn_keys & controls) - self.retired_keys - {driver})
        # One curve asks for neutral travel; it does not establish a control effect.
        need_neutral = neutral is None and any(key in self.estimates for key in eligible) and abs(signed_speed) > 0
        candidates = []
        for key in ([''] if need_neutral else [driver, *eligible]):
            speed = driving_speed if key == driver else signed_speed
            rates = ((base, base_error, 'body drive'),) if key in ('', driver) else self._rate(key, driving_speed)
            if self._stationary_probe and key != driver:
                continue
            for rate, rate_error, model in rates:
                prediction = self._prediction(pose, speed, rate, rate_error, duration, delay_s, model)
                if key != driver and key and abs(speed) < 1e-9:
                    prediction = self._prediction(pose, 0., rate, rate_error, duration, delay_s, 'stationary turn prior')
                if clearance is not None and not all(clearance(x, y, error) for _, x, y, error in prediction.path):
                    continue
                gx, gy = target
                bearing = math.atan2(gy - prediction.endpoint[1], gx - prediction.endpoint[0])
                sign = -1 if driving_speed < 0 else 1
                goes = prediction.heading + (math.pi if sign < 0 else 0.)
                cost = math.dist(prediction.endpoint, target) + .5 * self.body_extent * abs(_wrap(bearing - goes))
                candidates.append((cost, key, prediction, rate_error))
        if not candidates:
            return None
        _cost, key, chosen, _rate_error = min(candidates, key=lambda item: (item[0], item[1]))
        alternatives = [chosen]
        if key in eligible:
            for rate, uncertainty, model in self._rate(key, driving_speed):
                other = self._prediction(pose, signed_speed, rate, uncertainty, duration, delay_s, model)
                if other != chosen:
                    alternatives.append(other)
            alternatives.extend((self._prediction(pose, signed_speed, 0., 0., duration, delay_s, 'no effect'),
                                 self._prediction(pose, signed_speed, base, base_error, duration, delay_s, 'autonomous curve')))
        else:
            _speed, world_heading, _when = self._drivers[driver]
            alternatives.append(self._prediction(pose, driving_speed, 0., 0., duration, delay_s,
                                                 'world-fixed drive', heading=world_heading))
        if clearance is not None and not all(clearance(x, y, error) for pred in alternatives for _, x, y, error in pred.path):
            return None
        self._prediction_id += 1
        action = SteeringAction(key, duration, self.owner, self._prediction_id, tuple(alternatives))
        self.pending, self._origin, self._delivered_at, self._delay = action, pose, None, delay_s
        self._stationary_probe = key in eligible and abs(signed_speed) < 1e-9
        if key == driver:
            self._stationary_probe = False
        return action

    def delivered(self, action: SteeringAction, began: float) -> bool:
        """Bind the forecast to actual input delivery without reading its response."""
        if (self.pending != action or action.owner != self.owner or self._origin is None
                or not _finite(began) or began > self._origin.at + self.longest_gap):
            return False
        self._delivered_at = began
        self._delay = max(0., began - self._origin.at)
        return True

    def _verify(self, key: str, began: float, pose: _Pose, previous: _Pose | None, gap: bool) -> None:
        action, origin = self.pending, self._origin
        if action is None or origin is None:
            return
        if gap or action.owner != self.owner:
            self.pending = None
            self.last_result = 'observation gap'
            return
        delivered = began if self._delivered_at is None else self._delivered_at
        if key != action.key or began != delivered:
            if pose.at >= delivered:
                self.pending = None
                self.last_result = 'input changed'
            return
        if previous is None or previous.at < delivered or pose.at <= delivered:
            return
        if pose.at + 1e-9 < origin.at + self._delay + action.duration:
            return
        seconds = pose.at - origin.at - self._delay
        errors = []
        for prediction in action.predictions:
            initial_heading = _wrap(prediction.heading - prediction.rate * action.duration)
            x, y = _arc(origin.x + origin.vx * self._delay, origin.y + origin.vy * self._delay,
                        initial_heading, prediction.speed, prediction.rate, seconds)
            bound = prediction.path[-1][3] + pose.error
            errors.append(math.dist((x, y), (pose.x, pose.y)) <= bound)
        self.last_result = 'consistent' if any(errors) else 'contradicted'
        if math.hypot(pose.vx, pose.vy) == 0 and action.key in self.turn_keys:
            self.last_result = 'heading unobserved'
        self.pending = None


@invariant('agency.provisional_steering_keeps_its_observed_owner', scope='agency',
           owner='core/agency/a_model_for_steering.py', observational=False)
def _provisional_steering_invariant() -> tuple:
    model = ProvisionalSteeringBank()
    model.arm(['b'], body_extent=10.)
    model.saw((1, 0., 1), 'a', 0., .3, 0., 0., 30., 0., position_error=.01)
    action = model.action(['a', 'b'], 'a', (0., 100.))
    assert action is not None and action.status == 'provisional'
    model.saw((1, 1., 1), 'a', .3, .4, 3., 0., 30., 0., position_error=.01)
    assert model.pending is None and not model.estimates
    assert not model.delivered(action, .3)
    return ()
