#!/usr/bin/env python3
"""How she does on the test worlds when the whole of her play is handed them, as a game on a page is.

tools/measure_playing_as_it_happens.py hands a world to her reflexes alone, the loop that plays what does not wait.
A game on a page is played by more than that: the guide she keeps to the place, the rules read off its title screen,
the slower way that reads a screen that waits (a board, a book of arrows, cards face down, a box to type in) and the
plans she makes toward an end. This hands each world in tests/fixtures/worlds_that_move to the same entry a live game
is handed to (core/skills/sovereign_browser_drawing.py), from its title screen, in her own browser, asked to win.

The world's own tally (``window.__world``) is read only afterwards, to say how it went.

    python tools/measure_whole_play_on_worlds.py simon pairs --minutes 3
    python tools/measure_whole_play_on_worlds.py --all --minutes 2 --out report.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WORLDS = ROOT / "tests" / "fixtures" / "worlds_that_move"

#: What a world's tally holds besides score and lives, whichever of them a world keeps.
_TALLY = ("state", "score", "lives", "losses", "games", "ended", "caught", "opened", "mine", "theirs", "returns",
          "strokes", "sunk", "fused", "blown", "copied", "turns", "painted", "switched", "bumps", "parked", "distances",
          "knocked", "shots", "flags", "solved", "tests", "answered", "asked", "cleared")


def every_world() -> list[str]:
    return sorted(path.stem for path in WORLDS.glob("*.html"))


async def _one(world: str, seed: int, minutes: float, goal: str) -> dict:
    from core.capabilities.phantom_browser import PhantomBrowser
    from core.skills import sovereign_browser_drawing as drawing

    said: list[str] = []
    import core.skills.screen_pursuit as pursuit

    original_tell = pursuit._tell

    def tell(line: str) -> None:
        said.append(" ".join(str(line).split())[:200])
        original_tell(line)

    pursuit._tell = tell
    drawing.PLAY_UNTIL_WON_S = minutes * 60.0
    name, _, extra = world.partition("&")
    address = f"file://{WORLDS / (name + '.html')}?seed={seed}" + (f"&{extra}" if extra else "")
    began = time.monotonic()
    browser = PhantomBrowser(visible=False, browser_type="chromium", principal="owner")
    tally: dict = {}
    try:
        if not await browser.ensure_ready():
            return {"world": world, "error": "browser did not start"}
        await browser.page.set_viewport_size({"width": 800, "height": 600})
        await browser.page.goto(address, wait_until="domcontentloaded", timeout=30000)
        await asyncio.sleep(0.5)
        played = await drawing.played_on_the_drawing(browser, goal, {"url": address})
        tally = await browser.page.evaluate(
            "(names) => Object.fromEntries(names.filter((n) => __world[n] !== undefined).map((n) => [n, __world[n]]))",
            list(_TALLY))
    except Exception as why:  # noqa: BLE001 - one world failing is a result, not a stop
        played = {"error": f"{type(why).__name__}: {why}"}
    finally:
        pursuit._tell = original_tell
        await browser.close()
    return {
        "world": world, "seed": seed, "minutes": round((time.monotonic() - began) / 60.0, 1),
        "runs": played.get("runs"), "won": played.get("won"), "best_score": played.get("best_score"),
        "stopped_because": played.get("stopped_because"), "error": played.get("error"),
        "tally": tally, "said": said[:40],
    }


async def main(argv: list[str]) -> list[dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument("worlds", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[1])
    parser.add_argument("--minutes", type=float, default=3.0)
    parser.add_argument("--goal", default="Play this game and win.")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    os.environ.setdefault("AURA_BROWSER_STEALTH", "0")
    results = []
    for world in every_world() if args.all else args.worlds:
        for seed in args.seeds:
            result = await _one(world, seed, args.minutes, args.goal)
            results.append(result)
            print(json.dumps({k: result.get(k) for k in ("world", "seed", "minutes", "runs", "won", "best_score",
                                                         "stopped_because", "error", "tally")}), flush=True)
    return results


if __name__ == "__main__":
    _results = asyncio.run(main(sys.argv[1:]))
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps(_results, indent=1))
