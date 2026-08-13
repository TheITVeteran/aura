"""Steady-state dynamics properties for loops, accumulators and competitions.

A contract test asks *does submit() return a bool*. A dynamics test asks
*after five hundred ticks, is anybody starving*. The distinction is not
academic: the global-workspace monopoly fixed in cc1b00c68 was covered by 46
contract tests and survived all of them, because not one ran a competition for
more than a single tick. It took twenty-four ticks to become obvious.

The lesson generalises. Wherever this codebase has a loop, an accumulator or a
competition, the defect will be in the *steady state*, and a single-step
assertion cannot see a steady state by construction.

Two failure modes, and why both checks are mandatory
----------------------------------------------------
Fixing a monopoly is easy if you are allowed to break arbitration. The first
repair attempted on the workspace — hard-inhibiting the winner — passed every
"is it fair" check and was *worse*: an urgent source at 0.90 and an idle one at
0.20 alternated 50/50, and a source bidding alone won half its ticks. It had
stopped being a monopoly by ceasing to be an arbiter.

So a correct competition must satisfy an opposing pair:

* :func:`no_starvation` / :func:`rotation_entropy` — nobody is locked out.
* :func:`order_preserving` — a stronger bid still wins more often.

Either one alone is trivially satisfiable by a broken mechanism. Together they
pin the behaviour. :func:`competition_health` bundles them with
:func:`lone_source_retention` (a source with no rival keeps the workspace),
which is the third degree of freedom the two-source tests miss.

Usage::

    traj = run_trajectory(steps=200, step=lambda i: workspace.tick())
    for finding in competition_health(traj, bids={"a": 0.90, "b": 0.88}):
        raise AssertionError(finding)

Everything here is pure and offline: no clocks, no I/O, no imports from the
subsystems being measured. That is deliberate — see ``core/verify/DEPS``.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

__all__ = [
    "DynamicsFinding",
    "Trajectory",
    "run_trajectory",
    "no_starvation",
    "rotation_entropy",
    "order_preserving",
    "lone_source_retention",
    "competition_health",
    "bounded",
    "no_unbounded_growth",
    "converges",
    "recovers_after",
    "no_limit_cycle",
]

T = TypeVar("T")

#: Below this, a share is treated as "effectively never wins". It is not a
#: tuning knob: it is the reciprocal of the largest trajectory this module is
#: expected to be run with, so a participant winning a single tick in a
#: thousand still registers as non-zero rather than rounding to starvation.
_SHARE_EPSILON = 1e-3


@dataclass(frozen=True)
class DynamicsFinding:
    """One violated steady-state property.

    Carries the measurement, not just a verdict, because the number is what
    tells you whether the mechanism is subtly skewed or completely inverted.
    """

    prop: str
    subject: str
    message: str
    measured: float | None = None
    threshold: float | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"[{self.prop}] {self.subject}: {self.message}"]
        if self.measured is not None:
            got = f"measured={self.measured:.4g}"
            if self.threshold is not None:
                got += f" threshold={self.threshold:.4g}"
            parts.append(got)
        return " — ".join(parts)


@dataclass
class Trajectory(Generic[T]):
    """An ordered recording of what a mechanism did over many steps.

    ``observations`` holds one entry per step *after* warmup. ``warmup`` steps
    are executed and discarded, because a competition's first few ticks are
    transient and a fairness claim about the transient is not a claim about the
    steady state.
    """

    observations: list[T] = field(default_factory=list)
    warmup: int = 0
    aborted: str | None = None

    def __len__(self) -> int:
        return len(self.observations)

    def winners(self, key: Callable[[T], str | None] | None = None) -> list[str]:
        """Winner label per step, skipping steps that produced no winner."""
        extract = key or (lambda obs: None if obs is None else str(obs))
        return [w for w in (extract(o) for o in self.observations) if w is not None]

    def shares(self, key: Callable[[T], str | None] | None = None) -> dict[str, float]:
        """Fraction of decided steps won by each participant."""
        won = self.winners(key)
        if not won:
            return {}
        counts = Counter(won)
        total = float(len(won))
        return {src: n / total for src, n in counts.items()}

    def series(self, key: Callable[[T], float]) -> list[float]:
        """Pull a scalar time series out of the observations."""
        return [float(key(o)) for o in self.observations]


def run_trajectory(
    *,
    steps: int,
    step: Callable[[int], T],
    warmup: int = 0,
    stop_when: Callable[[T], bool] | None = None,
) -> Trajectory[T]:
    """Drive ``step`` for ``steps`` iterations and record every result.

    ``steps`` is a hard bound (NASA/JPL Power-of-Ten rule 2: every loop has a
    statically evident upper bound). ``stop_when`` may end the run early; that
    is recorded in ``aborted`` so a property cannot silently pass on three
    observations when it asked for three hundred.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if warmup < 0:
        raise ValueError("warmup must not be negative")

    traj: Trajectory[T] = Trajectory(warmup=warmup)
    for i in range(warmup + steps):
        obs = step(i)
        if i >= warmup:
            traj.observations.append(obs)
        if stop_when is not None and stop_when(obs):
            traj.aborted = f"stop_when triggered at step {i}"
            break
    return traj


# --------------------------------------------------------------------------
# Competition properties
# --------------------------------------------------------------------------


def no_starvation(
    shares: Mapping[str, float],
    *,
    expected: Iterable[str],
    min_share: float,
    subject: str = "competition",
) -> list[DynamicsFinding]:
    """Every expected participant wins at least ``min_share`` of decided steps.

    This is the check that would have caught the workspace monopoly on its
    first run: three of four sources measured 0.000.
    """
    findings: list[DynamicsFinding] = []
    for src in expected:
        got = shares.get(src, 0.0)
        if got < min_share:
            findings.append(
                DynamicsFinding(
                    prop="no_starvation",
                    subject=f"{subject}/{src}",
                    message=(
                        "never won the competition"
                        if got <= _SHARE_EPSILON
                        else "won far less often than the floor allows"
                    ),
                    measured=got,
                    threshold=min_share,
                    detail={"shares": dict(shares)},
                )
            )
    return findings


def rotation_entropy(
    shares: Mapping[str, float],
    *,
    min_normalised: float,
    subject: str = "competition",
) -> list[DynamicsFinding]:
    """Winner distribution is spread out, measured as normalised entropy.

    1.0 is a perfectly even rotation, 0.0 is one source taking every step.
    Normalising by ``log(n)`` keeps the threshold meaningful as the number of
    participants changes.
    """
    live = {k: v for k, v in shares.items() if v > 0.0}
    if len(live) <= 1:
        return [
            DynamicsFinding(
                prop="rotation_entropy",
                subject=subject,
                message="a single source took every decided step",
                measured=0.0,
                threshold=min_normalised,
                detail={"shares": dict(shares)},
            )
        ]
    entropy = -sum(p * math.log(p) for p in live.values())
    normalised = entropy / math.log(len(shares) or len(live))
    if normalised < min_normalised:
        return [
            DynamicsFinding(
                prop="rotation_entropy",
                subject=subject,
                message="broadcast is concentrated on too few sources",
                measured=normalised,
                threshold=min_normalised,
                detail={"shares": dict(shares)},
            )
        ]
    return []


def order_preserving(
    shares: Mapping[str, float],
    bids: Mapping[str, float],
    *,
    subject: str = "competition",
    tolerance: float = 0.0,
) -> list[DynamicsFinding]:
    """A source that bids higher must not win less often than one that bids lower.

    The guard against over-correcting a monopoly into indifference. Compares
    every strictly-ordered pair rather than a rank correlation, so the finding
    names the specific inversion instead of reporting a coefficient nobody can
    act on. ``tolerance`` allows a share gap that is within sampling noise.
    """
    findings: list[DynamicsFinding] = []
    ranked = sorted(bids.items(), key=lambda kv: kv[1], reverse=True)
    for i, (hi_src, hi_bid) in enumerate(ranked):
        for lo_src, lo_bid in ranked[i + 1 :]:
            if hi_bid <= lo_bid:
                continue  # tied bids may resolve in either direction
            hi_share = shares.get(hi_src, 0.0)
            lo_share = shares.get(lo_src, 0.0)
            if hi_share + tolerance < lo_share:
                findings.append(
                    DynamicsFinding(
                        prop="order_preserving",
                        subject=f"{subject}/{hi_src}<{lo_src}",
                        message=(
                            f"{hi_src} bids {hi_bid:.3g} but wins less than "
                            f"{lo_src} bidding {lo_bid:.3g} — arbitration "
                            "ignores bid strength"
                        ),
                        measured=hi_share - lo_share,
                        threshold=-tolerance,
                        detail={"shares": dict(shares), "bids": dict(bids)},
                    )
                )
    return findings


def lone_source_retention(
    shares: Mapping[str, float],
    *,
    source: str,
    min_share: float,
    subject: str = "competition",
) -> list[DynamicsFinding]:
    """A source with no rival keeps winning.

    Fatigue, refractory periods and inhibition all risk making an unopposed
    source lose to nothing at all. That is silence with nothing else to attend
    to, and it is a real regression that fairness checks applaud.
    """
    got = shares.get(source, 0.0)
    if got < min_share:
        return [
            DynamicsFinding(
                prop="lone_source_retention",
                subject=f"{subject}/{source}",
                message="an unopposed source failed to hold the workspace",
                measured=got,
                threshold=min_share,
                detail={"shares": dict(shares)},
            )
        ]
    return []


def competition_health(
    shares: Mapping[str, float],
    bids: Mapping[str, float],
    *,
    subject: str = "competition",
    min_share: float,
    min_normalised_entropy: float,
    order_tolerance: float = 0.0,
) -> list[DynamicsFinding]:
    """The full opposing pair: nobody starves *and* bid strength still decides.

    Checking only one half is how both of the workspace's failure modes got
    written. Prefer this over calling the individual properties.
    """
    findings: list[DynamicsFinding] = []
    findings += no_starvation(
        shares, expected=bids.keys(), min_share=min_share, subject=subject
    )
    findings += rotation_entropy(
        shares, min_normalised=min_normalised_entropy, subject=subject
    )
    findings += order_preserving(shares, bids, subject=subject, tolerance=order_tolerance)
    return findings


# --------------------------------------------------------------------------
# Accumulator / time-series properties
# --------------------------------------------------------------------------


def bounded(
    series: Sequence[float],
    *,
    lo: float,
    hi: float,
    subject: str = "series",
) -> list[DynamicsFinding]:
    """An accumulator stays inside its declared range for the whole run.

    Reported on the first excursion, with the step index, because the step
    number is what makes a drift bug reproducible.
    """
    for i, value in enumerate(series):
        if value < lo or value > hi:
            return [
                DynamicsFinding(
                    prop="bounded",
                    subject=subject,
                    message=f"left [{lo:g}, {hi:g}] at step {i}",
                    measured=value,
                    threshold=hi if value > hi else lo,
                    detail={"step": i},
                )
            ]
    return []


def no_unbounded_growth(
    series: Sequence[float],
    *,
    max_slope_per_step: float,
    subject: str = "series",
    min_points: int = 8,
) -> list[DynamicsFinding]:
    """The series does not trend upward without limit.

    Least-squares slope over the run. This is the leak check: a buffer that
    grows a little every tick looks perfectly healthy in any single-step
    assertion and exhausts memory in a four-hour soak.
    """
    n = len(series)
    if n < min_points:
        return [
            DynamicsFinding(
                prop="no_unbounded_growth",
                subject=subject,
                message=f"too few points to estimate a trend ({n} < {min_points})",
                measured=float(n),
                threshold=float(min_points),
            )
        ]
    mean_x = (n - 1) / 2.0
    mean_y = statistics.fmean(series)
    denom = sum((i - mean_x) ** 2 for i in range(n))
    if denom == 0.0:
        return []
    slope = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(series)) / denom
    if slope > max_slope_per_step:
        return [
            DynamicsFinding(
                prop="no_unbounded_growth",
                subject=subject,
                message="series trends upward faster than the allowed slope",
                measured=slope,
                threshold=max_slope_per_step,
                detail={"first": series[0], "last": series[-1], "steps": n},
            )
        ]
    return []


def converges(
    series: Sequence[float],
    *,
    tolerance: float,
    window: int,
    subject: str = "series",
) -> list[DynamicsFinding]:
    """The last ``window`` samples sit within ``tolerance`` of their mean."""
    if len(series) < window:
        return [
            DynamicsFinding(
                prop="converges",
                subject=subject,
                message=f"run shorter than the settling window ({len(series)} < {window})",
                measured=float(len(series)),
                threshold=float(window),
            )
        ]
    tail = series[-window:]
    centre = statistics.fmean(tail)
    spread = max(abs(v - centre) for v in tail)
    if spread > tolerance:
        return [
            DynamicsFinding(
                prop="converges",
                subject=subject,
                message="still oscillating at the end of the run",
                measured=spread,
                threshold=tolerance,
                detail={"centre": centre},
            )
        ]
    return []


def recovers_after(
    series: Sequence[float],
    *,
    perturbation_step: int,
    baseline: float,
    tolerance: float,
    within_steps: int,
    subject: str = "series",
) -> list[DynamicsFinding]:
    """After a shock the value returns to baseline within a deadline.

    Homeostasis is the claim; this is the measurement. A system that is
    knocked off baseline and settles somewhere *else* has not recovered, it has
    relocated, and only a post-perturbation deadline distinguishes the two.
    """
    tail = series[perturbation_step : perturbation_step + within_steps + 1]
    if not tail:
        return [
            DynamicsFinding(
                prop="recovers_after",
                subject=subject,
                message="no samples after the perturbation step",
                measured=float(len(series)),
                threshold=float(perturbation_step),
            )
        ]
    for value in tail:
        if abs(value - baseline) <= tolerance:
            return []
    closest = min(tail, key=lambda v: abs(v - baseline))
    return [
        DynamicsFinding(
            prop="recovers_after",
            subject=subject,
            message=f"did not return to baseline within {within_steps} steps",
            measured=closest,
            threshold=baseline,
            detail={"tolerance": tolerance, "samples": len(tail)},
        )
    ]


def no_limit_cycle(
    labels: Sequence[str],
    *,
    max_period: int,
    min_repeats: int,
    subject: str = "sequence",
) -> list[DynamicsFinding]:
    """The mechanism is not stuck repeating a short fixed pattern.

    A strict A-B-A-B alternation is the signature of arbitration that has
    stopped reading its inputs — it looks fair to an entropy check and is
    exactly the failure the winner-inhibition repair produced.
    """
    n = len(labels)
    for period in range(1, max_period + 1):
        needed = period * min_repeats
        if n < needed:
            continue
        tail = labels[-needed:]
        if all(tail[i] == tail[i % period] for i in range(needed)):
            return [
                DynamicsFinding(
                    prop="no_limit_cycle",
                    subject=subject,
                    message=(
                        f"locked into a period-{period} cycle repeated "
                        f"{min_repeats} times: {list(tail[:period])}"
                    ),
                    measured=float(period),
                    threshold=float(max_period),
                    detail={"cycle": list(tail[:period])},
                )
            ]
    return []
