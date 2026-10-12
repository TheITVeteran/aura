"""A pass of play over many games, as the scorecard and signals her self-improvement weighs.

    python tools/a_pass_of_play_as_evidence.py PASS_DIR [--before PASS_DIR] [--held-out FILE]

Prints the scorecard (score over the games she may learn from, the held-out mean on its own, regressions against the
pass before) and the signals, as JSON (core/learning/what_a_pass_of_play_came_to.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.learning.what_a_pass_of_play_came_to import scorecard, signals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pass_dir")
    parser.add_argument("--before", default=None)
    parser.add_argument("--held-out", default=str(Path("~/.aura/flash-campaign/held_out.json").expanduser()))
    args = parser.parse_args()
    held_out = set(json.loads(Path(args.held_out).read_text()).get("held_out") or []) if Path(args.held_out).exists() else set()
    print(json.dumps({"scorecard": scorecard(args.pass_dir, held_out, before=args.before),
                      "signals": signals(args.pass_dir, held_out)}, indent=1))


if __name__ == "__main__":
    main()
