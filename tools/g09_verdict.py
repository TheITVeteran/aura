#!/usr/bin/env python3
"""G09's three criteria, read off tools/g09_analysis.py's output exactly as a preregistration states them.

* Primary: pooled over the domains, kept answers gain more requests than they
  lose, with an exact two-sided McNemar p below ``--alpha``.
* Breadth: at least ``--breadth-need`` of the ``--breadth-among`` domains, in
  the analysis of held-out kinds only, gain more than they lose at p below
  ``--alpha``.
* No harm: no domain, counting every request a kept procedure answered, loses
  more than it gains at p below ``--harm-alpha``.

G09 closes only if all three hold. Attempt 2: ``--alpha 0.05 --breadth-need 2
--breadth-among code,planning,math``; attempt 3: ``--alpha 0.0167
--breadth-need 2 --breadth-among math,transfer``.

Usage:
    g09_verdict.py --all FILE --held-out FILE --alpha A --breadth-need N --breadth-among D,D [--harm-alpha 0.05]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def gains(line: dict[str, Any], alpha: float) -> bool:
    return int(line["gained"]) > int(line["lost"]) and float(line["two_sided_p"]) < alpha


def harms(line: dict[str, Any], alpha: float) -> bool:
    return int(line["lost"]) > int(line["gained"]) and float(line["two_sided_p"]) < alpha


def verdict(every: dict[str, Any], held_out: dict[str, Any], *, alpha: float, breadth_need: int,
            breadth_among: list[str], harm_alpha: float) -> dict[str, Any]:
    pooled = every["pooled"]
    broad = [d for d in breadth_among if d in held_out["domains"] and gains(held_out["domains"][d], alpha)]
    harmed = [d for d, line in every["domains"].items() if harms(line, harm_alpha)]
    out = {
        "primary": {"holds": gains(pooled, alpha), "gained": pooled["gained"], "lost": pooled["lost"],
                    "two_sided_p": pooled["two_sided_p"], "alpha": alpha},
        "breadth": {"holds": len(broad) >= breadth_need, "domains_with_gain": broad, "needed": breadth_need,
                    "among": breadth_among,
                    "by_domain": {d: {k: held_out["domains"][d][k] for k in ("gained", "lost", "two_sided_p")}
                                  for d in breadth_among if d in held_out["domains"]}},
        "no_harm": {"holds": not harmed, "domains_harmed": harmed, "alpha": harm_alpha},
    }
    out["closes"] = all(out[name]["holds"] for name in ("primary", "breadth", "no_harm"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", type=Path, required=True, help="g09_analysis.py output over every domain")
    parser.add_argument("--held-out", type=Path, required=True, help="g09_analysis.py output over held-out kinds")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--breadth-need", type=int, required=True)
    parser.add_argument("--breadth-among", required=True)
    parser.add_argument("--harm-alpha", type=float, default=0.05)
    args = parser.parse_args()
    result = verdict(json.loads(args.all.read_text()), json.loads(args.held_out.read_text()), alpha=args.alpha,
                     breadth_need=args.breadth_need, breadth_among=args.breadth_among.split(","),
                     harm_alpha=args.harm_alpha)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
