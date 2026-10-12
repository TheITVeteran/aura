#!/usr/bin/env python3
"""Hand her a recording of someone playing a game, and keep what it showed with that game.

She reads the recording (core/perception/watching_someone_play.py): which thing was the player's, what they made it
do, as her own keys where she has played the game before, which kinds they went for and kept clear of, and the places
along their way. What it showed is kept under the game's address, as what her own play of it is kept, and used the next
time she plays it: a kind she has not met yet is played as the player played it.

    python tools/watch_someone_play.py recording.mp4 --game https://archive.org/details/food_bash
    python tools/watch_someone_play.py recording.mp4 --game https://... --seconds 120 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv: list[str]) -> dict:
    from core.agency.what_she_keeps_of_a_game import INDEXED_TABLES
    from core.perception.watching_someone_play import read_a_recording
    from core.runtime.what_she_learned import recall, remember

    parser = argparse.ArgumentParser()
    parser.add_argument("recording")
    parser.add_argument("--game", required=True, help="the address she plays the game at")
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    world = f"played as it happens at {args.game}"
    held = recall(world, indexed_tables=INDEXED_TABLES) or {}
    ways, hers_look = {}, None
    kinds = held.get("kinds") or []
    hers = held.get("hers") or {}
    if hers.get("kind") is not None and hers["kind"] < len(kinds):
        kind = kinds[hers["kind"]]
        hers_look = (tuple(kind["colour"]), float(kind["size"]))
        ways = {key: (sum(v[0] for v in values) / len(values), sum(v[1] for v in values) / len(values))
                for key, values in (hers.get("ways") or {}).items() if values and len(values[0]) >= 2}
    footage = read_a_recording(args.recording, most_s=args.seconds, ways=ways, hers_look=hers_look)
    said = {"seconds": round(footage.seconds, 1), "player found": footage.hers is not None,
            "stances": footage.as_memory()["stances"], "moments": len(footage.moments),
            "checkpoints": len(footage.checkpoints), "what it showed": footage.said()}
    print(json.dumps(said, indent=1))
    if not args.dry_run and footage.hers is not None:
        held["watched"] = footage.as_memory()
        remember(world, held, indexed_tables=INDEXED_TABLES)
    return said


if __name__ == "__main__":
    main(sys.argv[1:])
