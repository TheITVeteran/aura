"""A world run from its code outside a browser is the world a browser runs: the same state after the same keys, clicks
and time, value for value.

tools/oracles/a_world_run_from_its_code.js stands in for the browser (a canvas that draws nowhere, a driven clock) so
that what is said of a world can be checked against its code quickly and exactly. That is only worth anything if the
stand-ins change nothing, so each world is run both ways, the browser's clock driven the same way, and every value of
its tally and its game compared.
"""
from __future__ import annotations

import asyncio
import math
import re
from pathlib import Path

import pytest

from tools.oracles.a_world_run_from_its_code import FRAME_S, HERE, WorldRunFromItsCode, node_is_here

pytestmark = [pytest.mark.unit]

WORLDS = Path(__file__).parent / "fixtures" / "worlds_that_move"
EVERY_WORLD = sorted(path.stem for path in WORLDS.glob("*.html"))

#: What is done to each world: keys held in turn, and the pointer moved and clicked about the screen.
KEYS = ("ArrowLeft", "ArrowUp", "ArrowRight", "ArrowDown", " ", "x", "z", "")
CLICKS = ((120, 90), (300, 200), (240, 300), (400, 60), (60, 250))

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


def _plain_js() -> str:
    source = (HERE / "a_world_run_from_its_code.js").read_text()
    return re.search(r"function plain\(value, depth\) \{.*?\n\}\n", source, re.S).group(0)


def _same(a, b, where="") -> list[str]:
    if isinstance(a, float | int) and isinstance(b, float | int) and not isinstance(a, bool):
        return [] if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) else [f"{where}: {a} != {b}"]
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{where}: keys {sorted(set(a) ^ set(b))}"] if set(a) != set(b) else []
        return out + [d for k in set(a) & set(b) for d in _same(a[k], b[k], f"{where}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{where}: {len(a)} items != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b, strict=True)) for d in _same(x, y, f"{where}[{i}]")]
    return [] if a == b else [f"{where}: {a!r} != {b!r}"]


def _script():
    """The same doings for both: each key held a while, then the pointer moved and pressed and let go at places."""
    for key in KEYS:
        yield ("hold", key, 0.4)
    for x, y in CLICKS:
        yield ("click", (x, y), 0.3)


def _in_the_oracle(world: str, seed: int) -> list[dict]:
    states = []
    with WorldRunFromItsCode(world, seed=seed) as here:
        for what, arg, seconds in _script():
            if what == "hold":
                here.hold(*([arg] if arg else []))
            else:
                here.click(*arg)
            here.run(seconds)
            states.append(here.state())
    return states


async def _in_a_browser(page, world: str, seed: int) -> list[dict]:
    await page.goto(f"file://{WORLDS / (world + '.html')}?seed={seed}&start=1")
    await page.evaluate(f"window.__plain = (() => {{ {_plain_js()} return plain; }})()")
    await page.evaluate(f"__drive({FRAME_S}, 1)")
    held: set[str] = set()
    states = []
    for what, arg, seconds in _script():
        if what == "hold":
            for key in sorted(held - {arg}):
                await page.keyboard.up("Space" if key == " " else key)
            held = {arg} if arg else set()
            if arg:
                await page.keyboard.down("Space" if arg == " " else arg)
        else:
            for key in sorted(held):
                await page.keyboard.up("Space" if key == " " else key)
            held = set()
            await page.evaluate(
                f"""(() => {{ const c = document.querySelector('canvas'), r = c.getBoundingClientRect();
                const at = {{clientX: r.left + {arg[0]} * r.width / 480, clientY: r.top + {arg[1]} * r.height / 320}};
                c.dispatchEvent(new MouseEvent('mousemove', at)); c.dispatchEvent(new MouseEvent('mousedown', at));
                window.dispatchEvent(new MouseEvent('mouseup', at)); }})()""")
        await page.evaluate(f"__drive({FRAME_S}, {max(1, round(seconds / FRAME_S))})")
        states.append(await page.evaluate(
            "({world: __plain(__world, 0), game: __plain(__world.game, 0), held: [...__world.held],"
            " now: performance.now()})"))
    return states


@pytest.mark.parametrize("seed", [3])
def test_every_world_run_from_its_code_is_the_world_the_browser_runs(seed):
    if not node_is_here():
        pytest.skip("node is not installed")
    playwright_api = pytest.importorskip("playwright.async_api")

    async def every_world():
        async with playwright_api.async_playwright() as playwright:
            try:
                browser = await playwright.chromium.launch(headless=True)
            except Exception as why:  # noqa: BLE001 - no engine installed here
                pytest.skip(f"no browser engine: {why}")
            page = await browser.new_page(viewport={"width": 480, "height": 400})
            await page.add_init_script(_A_DRIVEN_CLOCK)
            out = {}
            for world in EVERY_WORLD:
                out[world] = await _in_a_browser(page, world, seed)
            await browser.close()
            return out

    browsed = asyncio.run(every_world())
    differ = {}
    for world in EVERY_WORLD:
        ran = _in_the_oracle(world, seed)
        found = [d for a, b in zip(ran, browsed[world], strict=True) for d in _same(a, b)]
        if found:
            differ[world] = found[:5]
    assert not differ, differ
    # And the doings did something in each world: the comparison is not of worlds standing still.
    assert all(_in_the_oracle(world, seed)[0] != _in_the_oracle(world, seed)[-1] for world in EVERY_WORLD)
