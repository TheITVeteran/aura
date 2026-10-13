"""A control can curve a body's travel even when it cannot turn that body at rest.

Offline 2026-10-12 a rolling body kept moving during its steering presses.
Reading every moving press as a fixed direction excluded its steering keys.
Continuing observed travel can instead measure a turn within the press.
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field

from core.verify import invariant


@dataclass(frozen=True)
class TravelTurn:
    key: str
    began: float
    ended: float
    rate: float
    samples: int = 4
    span: float = .15


@dataclass
class TurningWhileTravelling:
    """Contrast repeated input intervals with observed neutral travel.

    Four continuing settled velocity readings must describe one angular
    rate. A missing reading or a gap invalidates that press. An autonomous
    curve, present during neutral travel too, supplies no control effect.
    These rates apply while travelling; none establish turning at rest.
    """

    moving: float
    settle_s: float
    gap_s: float
    turns: float
    trials: deque[TravelTurn] = field(default_factory=lambda: deque(maxlen=64))
    samples: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=64))
    receipt: tuple[str, float] | None = None
    last_at: float | None = None
    valid: bool = True
    travelling: bool = False
    last_velocity: tuple[float, float] | None = None

    def saw(self, key: str, began: float, at: float, vx: float, vy: float, *, measured: bool) -> None:
        if (not all(math.isfinite(v) for v in (began, at)) or began > at
                or (self.last_at is not None and at <= self.last_at)):
            return
        receipt = key, began
        changed = receipt != self.receipt
        if changed:
            self._finish(began)
            self.receipt, self.valid = receipt, True
            self.samples.clear()
        gap = not changed and self.last_at is not None and at - self.last_at > self.gap_s
        self.last_at = at
        self.travelling = bool(measured and all(math.isfinite(v) for v in (vx, vy))
                               and math.hypot(vx, vy) >= self.moving)
        self.last_velocity = (vx, vy) if self.travelling else None
        if gap or not measured or not all(math.isfinite(v) for v in (vx, vy)):
            self.valid = False
            self.samples.clear()
        if not self.valid or at - began < self.settle_s:
            return
        if not self.travelling:
            # Crossing a stop or reversal cannot share a continuous heading.
            if self.samples:
                self._finish(at)
                self.valid = False
            self.samples.clear()
            return
        angle = math.atan2(vy, vx)
        if self.samples:
            change = (angle - self.samples[-1][1] + math.pi) % (2 * math.pi) - math.pi
            if abs(change) > math.pi / 2:
                self.valid = False
                self.samples.clear()
                return
            angle = self.samples[-1][1] + change
        self.samples.append((at, angle))

    def _finish(self, ended: float) -> None:
        if not self.valid or self.receipt is None or len(self.samples) < 2:
            return
        key, began = self.receipt
        span = self.samples[-1][0] - self.samples[0][0]
        if ended <= began or self.samples[-1][0] > ended or span < .06:
            return
        times, angles = zip(*self.samples, strict=True)
        middle_t, middle_a = statistics.fmean(times), statistics.fmean(angles)
        rate = sum((t - middle_t) * (a - middle_a) for t, a in self.samples) / sum((t - middle_t) ** 2 for t in times)
        if max(abs(a - middle_a - rate * (t - middle_t)) for t, a in self.samples) > .08:
            return
        if any(trial.key == key and trial.began == began for trial in self.trials):
            return
        self.trials.append(TravelTurn(key, began, ended, rate, len(self.samples), span))

    def _repeated(self, key: str) -> float | None:
        rates = [trial.rate for trial in self.trials if trial.key == key and trial.samples >= 4 and trial.span >= .15][-2:]
        if len(rates) < 2:
            return None
        rate = statistics.fmean(rates)
        return rate if max(abs(value - rate) for value in rates) <= max(.15, .15 * abs(rate)) else None

    def rates(self) -> dict[str, float]:
        baseline = self._repeated('')
        if baseline is None:
            return {}
        known = {}
        for key in {trial.key for trial in self.trials if trial.key}:
            rate = self._repeated(key)
            if rate is not None and abs(rate - baseline) >= self.turns and abs(rate) >= self.turns:
                known[key] = rate
        return known

    def needs_neutral_travel(self) -> bool:
        """An unobserved neutral rate still needs its own rolling interval."""
        return self._repeated('') is None

    def curved(self, key: str) -> bool:
        """One observed curve can request a repeat, without supplying a control model."""
        return any(trial.key == key and abs(trial.rate) >= self.turns for trial in self.trials)

    def repeats(self, key: str) -> int:
        return 2 if self._repeated(key) is not None else int(self.curved(key))


@invariant("agency.travel_turns_require_repeated_observed_input_contrasts", scope="agency",
           owner="core/agency/turning_while_travelling.py", observational=False)
def _travel_turns_invariant() -> tuple:
    turns = TurningWhileTravelling(moving=20., settle_s=.2, gap_s=.35, turns=.5)
    for index, (key, rate) in enumerate((('', 0.), ('bend', 1.4), ('', 0.), ('bend', 1.4), ('', 0.))):
        began = index * .8
        for offset in (.0, .2, .3, .4, .5, .6):
            angle = rate * offset
            turns.saw(key, began, began + offset, 80 * math.cos(angle), 80 * math.sin(angle), measured=True)
    assert math.isclose(turns.rates()['bend'], 1.4)
    alone = TurningWhileTravelling(moving=20., settle_s=.2, gap_s=.35, turns=.5)
    alone.trials.extend(TravelTurn(t.key, t.began, t.ended, 1.4) for t in turns.trials)
    assert not alone.rates()
    return ()
