"""What her reading of a world's code says each thing is drawn in is the colour on the world's canvas where it is.

core/perception/reading_a_world_s_code.py reads a world's drawing functions for the colour each thing is filled with,
and where a thing has more than one look (a fruit and a bomb in one falling list), which look goes with what touching
her does. That is what lets the code's kinds be matched to what she sees. Here each world is drawn by a browser, its
clock driven, and the canvas is read at the middle of each thing the world's state places: her body's colour, and each
thing's, is among the looks read for its kind, and where its look is tied to what it does, the right one.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

WORLDS = Path(__file__).parent / "fixtures" / "worlds_that_move"
_A_DRIVEN_CLOCK = """
(() => {
  let now = 1000; const pending = [];
  Object.defineProperty(performance, "now", { value: () => now, configurable: true });
  window.requestAnimationFrame = (fn) => { pending.push(fn); return pending.length; };
  window.__drive = (dt, n) => {
    for (let i = 0; i < n; i++) { now += dt * 1000; const due = pending.splice(0, pending.length); for (const fn of due) fn(now); }
  };
})();
"""


def _read(world: str):
    from core.perception.reading_a_world_s_code import read_a_world_s_code

    page = (WORLDS / f"{world}.html").read_text()
    return read_a_world_s_code([(WORLDS / "world.js").read_text(), *re.findall(r"<script>([\s\S]*?)</script>", page)])


def _rgb(colour: str) -> tuple[int, int, int]:
    hex_ = colour.lstrip("#")
    hex_ = "".join(c * 2 for c in hex_) if len(hex_) == 3 else hex_
    return tuple(int(hex_[i:i + 2], 16) for i in (0, 2, 4))


_SEEN = """
([body, kinds]) => {
  const c = document.querySelector('canvas'), g = c.getContext('2d'), game = __world.game;
  const at = (t, w, h) => { const d = g.getImageData(Math.round(t.x + (t.w || w) / 2), Math.round(t.y + (t.h || h) / 2), 1, 1).data;
                            return [d[0], d[1], d[2]]; };
  const boxes = [];
  const her = body.endsWith('[]') ? null : game[body];
  const out = {her: her ? at(her, 16, 16) : null, things: {}};
  for (const [kind, w, h] of kinds) {
    // Only what is wholly on the canvas: one coming in from an edge is not drawn there yet.
    out.things[kind] = [].concat(game[kind] || []).filter((t) => t && t.x !== undefined && t.y !== undefined
      && t.x >= 0 && t.y >= 0 && t.x + (t.w || w) <= c.width && t.y + (t.h || h) <= c.height)
      .map((t) => ({colour: at(t, w, h), good: t.good, x: t.x, y: t.y}));
  }
  return out;
}
"""


def _near(a, b, within=6) -> bool:
    return max(abs(x - y) for x, y in zip(a, b, strict=True)) <= within


@pytest.mark.parametrize("world", ["paint", "catch", "collect", "dodge", "jump", "lander", "shooter", "brawl", "guard",
                                   "rooms", "storm", "held", "steps", "bounce", "charge"])
def test_each_thing_is_on_the_canvas_in_the_colour_read_for_it(world):
    playwright_api = pytest.importorskip("playwright.async_api")
    reading = _read(world)
    kinds = [[t.name, *(t.size or (12, 12))] for t in reading.things if t.looks]

    async def seen():
        async with playwright_api.async_playwright() as playwright:
            try:
                browser = await playwright.chromium.launch(headless=True)
            except Exception as why:  # noqa: BLE001 - no engine installed here
                pytest.skip(f"no browser engine: {why}")
            page = await browser.new_page(viewport={"width": 480, "height": 400})
            await page.add_init_script(_A_DRIVEN_CLOCK)
            await page.goto(f"file://{WORLDS / (world + '.html')}?seed=4&start=1")
            # Kept in play throughout, so that what is read is the scene and not a screen that says it is over.
            await page.evaluate("(__drive(1 / 60, 1), __world.lives = 1e9)")
            out = []
            for _ in range(12):
                await page.evaluate("__drive(1 / 60, 30)")
                out.append(await page.evaluate(_SEEN, [reading.hers.name, kinds]))
            await browser.close()
            return out

    pictures = asyncio.run(seen())
    if reading.hers.drawn_in:
        her = [p["her"] for p in pictures if p["her"]]
        assert her and sum(_near(c, _rgb(reading.hers.drawn_in)) for c in her) >= len(her) // 2, (world, her[:3])
    for thing in reading.things:
        if not thing.looks:
            continue
        colours = [_rgb(colour) for colour, _does in thing.looks]
        tied = {does: _rgb(colour) for colour, does in thing.looks if does}
        seen_here = [t for p in pictures for t in p["things"].get(thing.name, [])]
        if not seen_here:
            continue
        on_it = [t for t in seen_here if any(_near(t["colour"], c) for c in colours)]
        assert len(on_it) >= 0.8 * len(seen_here), (world, thing.name, seen_here[:4], thing.looks)
        for t in on_it:
            if "good" in t and t["good"] is not None and tied:
                assert _near(t["colour"], tied["scores" if t["good"] else "costs a life"]), (world, thing.name, t)
