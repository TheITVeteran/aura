#!/usr/bin/env python3
"""The dimensions of a place to play, counted over the 56 games: what each game needs, and what she has for it.

Each game's needs are read from what is already kept of it: the mechanics, ways and situations its own file showed
(docs/design-docs/what_it_takes_to_beat.json) and the ways the design map gives it
(docs/design-docs/THE_56_GAMES_AND_WHAT_THEY_ASK.md). A game needs a value of a dimension where any of that value's
signs is among them (core/agency/dimensions_of_play.py). Then the counts: how many games need each value, how many
distinct combinations the 56 make, how many have an answer for every value they need, and which values she has no
answer for, by how many games each stops.

    python tools/the_dimensions_of_play.py
    python tools/the_dimensions_of_play.py --json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def signs_of_each_game() -> dict[int, set[str]]:
    """Every game's signs: its mechanics, ways and situations as read, and the ways the map gives it."""
    signs: dict[int, set[str]] = {}
    table = (ROOT / "docs/design-docs/THE_56_GAMES_AND_WHAT_THEY_ASK.md").read_text()
    for line in table.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 3 and cells[1].isdigit():
            ways = {part.split("(")[0].strip() for part in cells[3].split(",")}
            signs.setdefault(int(cells[1]), set()).update(w for w in ways if w)
    for row in json.loads((ROOT / "docs/design-docs/what_it_takes_to_beat.json").read_text()):
        if "error" in row:
            continue
        game = signs.setdefault(int(row["number"]), set())
        game.update(row.get("mechanics") or [], row.get("in_its_code") or [], row.get("ways") or [],
                    row.get("situations") or [], row.get("kinds") or [])
    return signs


def counted() -> dict:
    from core.agency.dimensions_of_play import DIMENSIONS

    games = signs_of_each_game()
    needs: dict[int, list[tuple[str, str]]] = {}
    per_value: Counter = Counter()
    for number, signs in sorted(games.items()):
        wanted = [(value.dimension, value.name) for values in DIMENSIONS.values() for value in values
                  if set(value.signs) & signs]
        needs[number] = wanted
        per_value.update(wanted)
    answered = {(v.dimension, v.name) for values in DIMENSIONS.values() for v in values if v.done_by}
    unanswered = Counter()
    fully = []
    for number, wanted in needs.items():
        missing = [w for w in wanted if w not in answered]
        unanswered.update(missing)
        if not missing:
            fully.append(number)
    sizes = {dimension: len(values) for dimension, values in DIMENSIONS.items()}
    return {
        "dimensions": len(DIMENSIONS),
        "values": sum(sizes.values()),
        "values_answered": len(answered),
        "kinds_of_place": math.prod(sizes.values()),
        "games": len(games),
        "distinct_combinations": len({tuple(sorted(w)) for w in needs.values()}),
        "values_needed_by_a_game": {"least": min(map(len, needs.values())), "most": max(map(len, needs.values())),
                                    "mean": round(sum(map(len, needs.values())) / len(needs), 1)},
        "games_with_every_value_answered": len(fully),
        "needed_most": [[f"{d}: {n}", c] for (d, n), c in per_value.most_common()],
        "unanswered_by_games_stopped": [[f"{d}: {n}", c] for (d, n), c in unanswered.most_common()],
        "per_game": {str(k): [f"{d}: {n}" for d, n in v] for k, v in needs.items()},
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    found = counted()
    if args.json:
        print(json.dumps(found, indent=1))
        return 0
    for key in ("dimensions", "values", "values_answered", "kinds_of_place", "games", "distinct_combinations",
                "values_needed_by_a_game", "games_with_every_value_answered"):
        print(f"{key}: {found[key]}")
    print("unanswered, by the games each stops:")
    for name, count in found["unanswered_by_games_stopped"]:
        print(f"  {count:3}  {name}")
    print("needed most:")
    for name, count in found["needed_most"][:25]:
        print(f"  {count:3}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
