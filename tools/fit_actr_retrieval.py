#!/usr/bin/env python3
"""Fit ACT-R's retrieval parameters to Aura's own measured recall, and report what does not fit.

``latency_model.unfitted_parameters`` was registered as an undischarged
assumption: the module shipped published defaults for the latency factor F and
the threshold tau, so only relative orderings stood. This is the measurement
that settles it — including, if that is what the data says, settling it as a
negative.

Two models are fitted separately, because they transfer differently:

  P(recall) = 1 / (1 + exp(-(A - tau)/s))     the retrieval curve
  T         = F * exp(-A)                     the latency equation

The retrieval curve is a claim about *which* memories come back, and Aura's
ranking is genuinely activation-driven, so it should fit. The latency equation
is a claim about *how long* retrieval takes, and it earns that shape in ACT-R
because retrieval there is a race between activations. Aura's recall is a
ranked scan: its cost is a function of how many candidates there are and what
the store does, not of how strong the winning trace is. If that is true, the
correlation will be absent and the honest result is to say so rather than to
fit F to noise — F is a pure scale factor and will happily absorb any timing.

Fitting a parameter to a relationship that is not there is the specific failure
RETRACTION.json is about. So this tool reports r^2 for the latency model and
refuses to emit an F when the relationship is not present.

Run: ``python tools/fit_actr_retrieval.py [--trials N] [--json PATH]``
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.cognition.actr_activation import (  # noqa: E402
    DEFAULT_PARAMETERS,
    base_level_activation,
    latency_sensitivity,
    retrieval_probability,
)

DAY = 86400.0

#: Below this r^2 the latency model has no relationship to fit. Not a
#: significance threshold: 0.10 means the activation term explains under a
#: tenth of the variance, at which point any F is a scale factor bolted onto
#: noise.
_MIN_LATENCY_R2 = 0.10


def _build_population(n: int, now: float, rng: random.Random) -> list[dict[str, float]]:
    """Traces spanning the age and rehearsal range the live system sees."""
    population = []
    for _ in range(n):
        age = math.exp(rng.uniform(math.log(60.0), math.log(365 * DAY)))
        accesses = rng.choice([0, 0, 0, 1, 2, 5, 12, 30])
        presentations = [now - age]
        if accesses:
            last = now - age * rng.uniform(0.0, 0.9)
            step = (last - (now - age)) / accesses
            presentations += [now - age + step * (i + 1) for i in range(accesses)]
        population.append(
            {
                "activation": base_level_activation(presentations, now),
                "importance": rng.uniform(0.0, 1.0),
            }
        )
    return population


def _measure(trials: int, batch: int, seed: int) -> dict[str, object]:
    """Rank real batches through the live ranking blend and time each one."""
    from core.memory.episodic_memory import Episode, EpisodicMemory

    rng = random.Random(seed)
    now = time.time()
    ranker = EpisodicMemory.__new__(EpisodicMemory)

    samples: list[tuple[float, float, int]] = []  # activation, seconds, recalled
    top_k = max(1, batch // 5)

    for _ in range(trials):
        specs = _build_population(batch, now, rng)
        episodes = []
        for i, spec in enumerate(specs):
            # Reconstruct a trace with the intended activation by placing a
            # single presentation at the age that produces it: B = -d*ln(t).
            age = math.exp(-spec["activation"] / DEFAULT_PARAMETERS.decay)
            episodes.append(
                Episode(
                    id=f"e{i}",
                    timestamp=now - age,
                    importance=spec["importance"],
                )
            )
        started = time.perf_counter()
        ranked = EpisodicMemory._static_rank(ranker, episodes, now)
        elapsed = time.perf_counter() - started

        recalled = {ep.episode_id for ep in ranked[:top_k]}
        for i, spec in enumerate(specs):
            samples.append((spec["activation"], elapsed, int(f"e{i}" in recalled)))

    return {"samples": samples, "top_k": top_k, "batch": batch, "trials": trials}


def _fit_latency(samples: list[tuple[float, float, int]]) -> dict[str, object]:
    """Least squares of ln(T) on -A. Slope should be 1 and intercept ln(F)."""
    xs = [-a for a, _t, _r in samples]
    ys = [math.log(t) for _a, t, _r in samples if t > 0.0]
    if len(xs) != len(ys) or len(xs) < 8:
        return {"fitted": False, "reason": "not enough timing samples"}

    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return {"fitted": False, "reason": "no variation in activation"}
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    intercept = my - slope * mx

    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    result: dict[str, object] = {
        "r2": round(r2, 6),
        "slope": round(slope, 6),
        "n": len(xs),
        "mean_latency_s": round(statistics.fmean([t for _a, t, _r in samples]), 9),
    }
    if r2 < _MIN_LATENCY_R2:
        result["fitted"] = False
        result["reason"] = (
            f"activation explains r2={r2:.4f} of latency variance, below "
            f"{_MIN_LATENCY_R2}. Aura's recall is a ranked scan whose cost tracks "
            "candidate count and store behaviour, not trace strength, so the "
            "ACT-R latency equation has no mechanism here. F is a pure scale "
            "factor and would absorb any timing; emitting one would be fitting "
            "a parameter to a relationship that is not present."
        )
        return result
    result["fitted"] = True
    result["latency_factor_F"] = round(math.exp(intercept), 9)
    return result


def _fit_retrieval_curve(samples: list[tuple[float, float, int]]) -> dict[str, object]:
    """Grid-search tau and s by log-likelihood on whether the trace was recalled."""
    activations = [a for a, _t, _r in samples]
    outcomes = [r for _a, _t, r in samples]
    if len(set(outcomes)) < 2:
        return {"fitted": False, "reason": "recall outcome never varied"}

    lo, hi = min(activations), max(activations)
    best = None
    steps = 60
    # The noise grid must extend past any plausible optimum: a maximum found at
    # the edge of the search is a railed parameter, not a fit, and the first
    # run of this tool railed at s=1.5 against a 1.5 ceiling.
    s_values = [0.05 * i for i in range(1, 121)]
    for ti in range(steps + 1):
        tau = lo + (hi - lo) * ti / steps
        for s in s_values:
            ll = 0.0
            for a, r in zip(activations, outcomes, strict=True):
                p = retrieval_probability(a, threshold=tau, noise_s=s)
                p = min(max(p, 1e-9), 1 - 1e-9)
                ll += math.log(p) if r else math.log(1.0 - p)
            if best is None or ll > best[0]:
                best = (ll, tau, s)

    assert best is not None
    ll, tau, s = best
    predicted = [
        retrieval_probability(a, threshold=tau, noise_s=s) for a in activations
    ]
    brier = statistics.fmean(
        (p - r) ** 2 for p, r in zip(predicted, outcomes, strict=True)
    )
    base_rate = statistics.fmean(outcomes)
    brier_base = statistics.fmean((base_rate - r) ** 2 for r in outcomes)
    skill = 1.0 - brier / brier_base if brier_base > 0.0 else 0.0

    railed = s <= s_values[0] + 1e-9 or s >= s_values[-1] - 1e-9
    return {
        "fitted": not railed,
        "railed": railed,
        "reason": (
            f"noise_s optimum sits at the edge of the search grid (s={s:.3f}); "
            "a maximum at the boundary is not a located optimum"
            if railed
            else ""
        ),
        "threshold_tau": round(tau, 4),
        "noise_s": round(s, 4),
        "log_likelihood": round(ll, 3),
        "brier": round(brier, 6),
        "brier_skill_vs_base_rate": round(skill, 4),
        "base_rate": round(base_rate, 4),
        "n": len(activations),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--batch", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    measured = _measure(args.trials, args.batch, args.seed)
    samples = measured["samples"]  # type: ignore[index]

    latency = _fit_latency(samples)  # type: ignore[arg-type]
    curve = _fit_retrieval_curve(samples)  # type: ignore[arg-type]

    report = {
        "measured": {
            "trials": measured["trials"],
            "batch": measured["batch"],
            "top_k": measured["top_k"],
            "samples": len(samples),  # type: ignore[arg-type]
        },
        "retrieval_curve": curve,
        "latency_model": latency,
        "sensitivity_at_A1": [
            {
                "parameter": b.parameter,
                "low": round(b.low, 6),
                "high": round(b.high, 6),
                "nominal": round(b.nominal, 6),
                "spread_ratio": round(b.spread_ratio, 4),
            }
            for b in latency_sensitivity(1.0)
        ],
    }

    print(json.dumps(report, indent=2))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}", file=sys.stderr)

    print("\n--- reading ---", file=sys.stderr)
    if curve.get("fitted"):
        print(
            f"retrieval curve FITTED: tau={curve['threshold_tau']} s={curve['noise_s']}, "
            f"Brier skill {curve['brier_skill_vs_base_rate']} over base rate",
            file=sys.stderr,
        )
    if latency.get("fitted"):
        print(f"latency FITTED: F={latency['latency_factor_F']}s", file=sys.stderr)
    else:
        print(f"latency NOT FITTED: {latency.get('reason')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
