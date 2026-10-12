"""Open a bundled game and picture it through a list of clicks and keys, to see its screens before a runner is written.

    python -m tools.reference_runners.look_around 2 --out /tmp/g2 click:0.5,0.9 wait:3 key:Space wait:2
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from tools.reference_runners.a_game_at_hand import GameAtHand


async def _main(case: int, out: Path, steps: list[str], wait_first: float) -> None:
    async with GameAtHand(case) as game:
        print("TITLE", game.title, "MOVIE", game.movie, flush=True)
        await asyncio.sleep(wait_first)
        picture, _ = await game.look()
        game.keep(picture, out / "00.png")
        print("00", game.said(picture)[:300], flush=True)
        for number, step in enumerate(steps, 1):
            kind, _colon, value = step.partition(":")
            if kind == "click":
                await game.click(*map(float, value.split(",")))
            elif kind == "word":
                picture, _ = await game.look()
                if not await game.click_word(picture, value):
                    print(f"{number:02d}", "no", repr(value), "on the screen", flush=True)
            elif kind == "key":
                await game.tap(value)
            elif kind == "hold":
                key, _comma, seconds = value.partition(",")
                await game.key_down(key)
                await asyncio.sleep(float(seconds or 0.5))
                await game.key_up(key)
            elif kind == "type":
                await game.type_text(value)
            elif kind == "wait":
                await asyncio.sleep(float(value))
                continue
            await asyncio.sleep(0.6)
            picture, _ = await game.look()
            game.keep(picture, out / f"{number:02d}.png")
            print(f"{number:02d}", step, "|", game.said(picture)[:300], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=int)
    parser.add_argument("steps", nargs="*")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wait", type=float, default=8.0)
    args = parser.parse_args()
    asyncio.run(_main(args.case, args.out, args.steps, args.wait))


if __name__ == "__main__":
    main()
