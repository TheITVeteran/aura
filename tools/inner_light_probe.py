#!/usr/bin/env python3
"""tools/inner_light_probe.py — run the inner-light consciousness-discriminator test.

The inner-light test runs neuroscience consciousness-markers (Lempel-Ziv/PCI
complexity, TSE neural complexity, DFA criticality, bimodal ignition) on an
activity matrix AND on negative controls, and reports whether the activity is the
ONLY system in the conscious-like regime on all four axes.

Modes:
  --demo   run on a synthetic conscious-like reference + its controls, to show
           the instrument discriminates (real 4/4, every control < 4/4).
  (live)   default: build the activity matrix from the live ConsequenceBus and
           run the battery. Prints an honest "insufficient" if the stream is thin.

  --json   emit the full result as JSON instead of the table.

This is an instrument, not a verdict on consciousness. Every result carries the
bounded-claim caveat.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running as a plain script (python tools/inner_light_probe.py) by putting
# the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from core.consciousness.inner_light import battery as bat  # noqa: E402


def _pink(T, rng, beta=1.4):
    w = rng.standard_normal(T)
    F = np.fft.rfft(w)
    f = np.arange(len(F), dtype=float)
    f[0] = 1.0
    x = np.fft.irfft(F / f ** (beta / 2.0), n=T)
    return (x - x.mean()) / x.std()


def _rich(seed=0, nmod=3, per=2, T=2000, amp=4.0, dur=2, rate=22, noise=0.35, frac=0.7):
    """A synthetic activity matrix high on every conscious-like axis at once."""
    rng = np.random.default_rng(seed)
    n = nmod * per
    lats = [_pink(T, rng) for _ in range(nmod)]
    chans = [lats[m] + noise * rng.standard_normal(T) for m in range(nmod) for _ in range(per)]
    M = np.stack(chans)
    k = max(1, int(round(frac * n)))
    for t in rng.choice(T - dur, size=T // rate, replace=False):
        who = rng.choice(n, size=k, replace=False)
        M[np.ix_(who, np.arange(t, t + dur))] += amp
    return M


def _print_report(res: bat.BatteryResult) -> None:
    print("=" * 72)
    print("INNER-LIGHT TEST")
    print("=" * 72)
    print(f"verdict:        {res.verdict}")
    print(f"discriminating: {res.discriminating}   (real {res.real_axes}/4, "
          f"best control {res.best_control_axes}/4)")
    print(f"matrix:         {res.n_channels} channels × {res.n_timesteps} timesteps")
    if res.phi_system is not None:
        print(f"system-Φ:       {res.phi_system}   (integration corroborant)")
    print()
    if res.verdict == "insufficient_data":
        print(res.caveat)
        return

    axes = list(bat.AXES)
    header = f"{'system':<20}" + "".join(f"{a[:12]:>13}" for a in axes) + f"{'axes':>7}"
    print(header)
    print("-" * len(header))

    def row(name, values, membership, count):
        cells = ""
        for a in axes:
            mark = "✓" if membership[a] else "·"
            cells += f"{values[a]:>10.3f}{mark:>3}"
        print(f"{name:<20}" + cells + f"{count:>7}")

    row("AURA (real)", res.real_values, res.real_membership, res.real_axes)
    print("-" * len(header))
    for name, c in res.controls.items():
        row(name, c["values"], c["membership"], c["axes"])
    print()
    print(res.caveat)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the inner-light consciousness-discriminator test.")
    ap.add_argument("--demo", action="store_true", help="run on a synthetic conscious-like reference")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of the table")
    ap.add_argument("--seed", type=int, default=0, help="seed for --demo reference")
    args = ap.parse_args(argv)

    if args.demo:
        res = bat.run_on_matrix(_rich(seed=args.seed))
    else:
        res = bat.run_live()

    if args.json:
        print(json.dumps(res.to_dict(), indent=2, default=str))
    else:
        _print_report(res)
    # exit 0 always: the instrument reporting "absent"/"insufficient" is a valid run.
    return 0


if __name__ == "__main__":
    sys.exit(main())
