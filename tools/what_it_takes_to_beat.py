#!/usr/bin/env python3
"""What it takes to beat each game of a list: read offline from each game's own program, said as general ways.

This is a design map, read once and offline, never by her while she plays: each game's Flash file is fetched from its
Internet Archive copy and read by her own program reader (core/perception/reading_a_program.py), and what it says is
put in the general terms her play is built in: the situations its words put a player in
(core/agency/situations_she_knows.py), the mechanics its words and code show (core/agency/mechanics_she_knows.py),
its controls, and what its own words say wins and loses. From those, the kinds of problem that beating it needs:
which general capability does it, so that each is built for every game of its kind and none for one game.

    python tools/what_it_takes_to_beat.py --out docs/design-docs/what_it_takes_to_beat.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

#: What its words say wins and loses, as games say them.
_WINS = re.compile(r"\b(you win|you won|winner|victory|congratulations|you did it|level complete|stage clear|"
                   r"you beat|well done|you saved|mission complete|you escaped|you made it)\b", re.I)
_LOSES = re.compile(r"\b(game over|you lose|you lost|try again|better luck|out of (?:time|lives|fuel)|time'?s up|"
                    r"you crashed|caught you|you failed)\b", re.I)

#: The kinds of problem beating a game needs, each a general capability, and the signs in what a game is that it needs
#: it: ways of playing (core/agency/ways_of_playing.py) and situations (core/agency/situations_she_knows.py).
KINDS = {
    "real-time control by imagining ahead": {"steer", "jump", "shoot", "strike", "carried", "unseen", "a view going by",
                                             "catching what falls", "keeping clear of what comes", "shooting what comes",
                                             "getting to the way out", "a fight at close quarters", "holding on until the end"},
    "a shot aimed and sent": {"send", "launching for distance", "tossing to whoever takes it"},
    "a press timed to a moving thing": {"time a press", "pressing at the right moment"},
    "clicking things as they come": {"click things"},
    "a board searched in turns": {"a board in turns"},
    "a rule found and searched": {"a grid's rule"},
    "a sequence remembered and done again": {"copy a sequence", "keys shown", "doing again what was shown"},
    "pairs remembered": {"remember what was shown", "matching what goes together"},
    "a build planned from parts": {"carry", "a chain", "turn what was placed", "setting off a chain", "stack",
                                   "building up"},
    "an errand of things found and used": {"use things", "serve"},
    "words typed as asked": {"type"},
    "something made, nothing won": {"make", "making something"},
}


def _get(url: str, *, timeout: float = 60.0, tries: int = 4) -> bytes:
    import time
    import urllib.error

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (offline design map)"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as answer:
                return answer.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    return b""


def the_program(address: str) -> tuple[str, bytes]:
    """The Flash file of an Internet Archive item: its largest .swf."""
    identifier = address.rstrip("/").split("/details/")[-1]
    meta = json.loads(_get(f"https://archive.org/metadata/{identifier}"))
    files = [f for f in meta.get("files", []) if str(f.get("name", "")).lower().endswith(".swf")]
    if not files:
        return "", b""
    chosen = max(files, key=lambda f: int(f.get("size") or 0))
    name = chosen["name"]
    return name, _get(f"https://archive.org/download/{identifier}/{urllib.request.quote(name)}")


def what_it_takes(number: int, name: str, address: str) -> dict:
    from core.agency.mechanics_she_knows import mechanics_in
    from core.agency.situations_she_knows import situations_in
    from core.perception.reading_a_program import read_program

    file, data = the_program(address)
    read = read_program(data, file) if data else None
    words = " ".join(read.words) if read else ""
    mechanics = mechanics_in(words, read.text if read else "") if read else {}
    from core.agency.mechanics_she_knows import known

    # What its words say outranks what its code merely can do: a program that handles drags may drag nothing in play.
    worded = sorted(m for m, evidence in mechanics.items() if any(str(e).startswith("says") for e in evidence))
    coded = sorted(m for m in mechanics if m not in worded)
    ways = sorted({way for m in worded for way in (known(m).ways if known(m) else ())})
    situations = [s.name for s in situations_in(words)]
    signs = set(ways) | set(situations)
    kinds = [kind for kind, wants in KINDS.items() if signs & wants]
    return {
        "number": number, "game": name, "address": address, "file": file, "bytes": len(data),
        "said": [w for w in (read.words if read else [])][:24],
        "keys": dict(read.keys) if read else {}, "pointer": list(read.pointer) if read else [],
        "wins": sorted({m.group(0).lower() for m in _WINS.finditer(words)}),
        "loses": sorted({m.group(0).lower() for m in _LOSES.finditer(words)}),
        "situations": situations, "mechanics": worded, "in_its_code": coded, "ways": ways, "kinds": kinds,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="*", type=int, default=[])
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    from tools.measure_on_archived_games import ARCHIVED

    names = _names()
    found = []
    for number in args.games or sorted(ARCHIVED):
        try:
            row = what_it_takes(number, names.get(number, str(number)), ARCHIVED[number])
        except Exception as why:  # noqa: BLE001 - one game not read is a row that says so
            row = {"number": number, "game": names.get(number, str(number)), "error": f"{type(why).__name__}: {why}"}
        found.append(row)
        print(json.dumps({k: row.get(k) for k in ("number", "game", "kinds", "error")}), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(found, indent=1))
    return 0


def _names() -> dict[int, str]:
    """The games' names, as the design map lists them."""
    table = (ROOT / "docs/design-docs/THE_56_GAMES_AND_WHAT_THEY_ASK.md").read_text()
    names = {}
    for line in table.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) > 3 and cells[1].isdigit():
            names[int(cells[1])] = cells[2]
    return names


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
