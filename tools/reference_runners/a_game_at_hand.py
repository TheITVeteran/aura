"""One game of the offline bundle, at hand for a reference runner: its picture as drawn, and a hand and keys on it.

A reference runner is a program written to beat one kind of game, knowing what a person knows who has read its rules
and watched it played. It is not Aura and she never runs it: it is the proof of what wins, so that what made it win
can be taken into her as a general way of playing, keyed to what a place shows and says rather than to the game
(docs/design-docs/WHAT_THE_RUNNERS_WON_BY.md says what each runner showed and where it went in her).

The games are the offline bundle's (tools/prepare_offline_flash_bundle.py), played by the pinned emulator from a
server on this machine, in a headless browser with nothing hidden about it being automated.

    async with GameAtHand(2) as game:
        picture, at = await game.look()
        await game.click(0.5, 0.8)
"""
from __future__ import annotations

import asyncio
import base64
import functools
import io
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

BUNDLE = Path("~/.aura/flash-campaign/offline-bundle").expanduser()

#: The page around the player: the emulator fills the page under a title line.
VIEWPORT = {"width": 1280, "height": 860}

#: The emulator's own canvas, in its shadow root, read from inside the page; None when it reads back blank.
_THE_CANVAS = """
(() => {
  let canvas = window.__runnerCanvas;
  if (!canvas || !canvas.isConnected) {
    const found = [];
    const walk = (root) => { for (const el of root.querySelectorAll('*')) {
      if (el.tagName === 'CANVAS') found.push(el); if (el.shadowRoot) walk(el.shadowRoot); } };
    walk(document);
    canvas = found.sort((a, b) => b.width * b.height - a.width * a.height)[0] || null;
    window.__runnerCanvas = canvas;
  }
  if (!canvas) return null;
  const r = canvas.getBoundingClientRect();
  try { return {pixels: canvas.toDataURL('image/jpeg', 0.85), box: [r.left, r.top, r.width, r.height]}; }
  catch (e) { return null; }
})()
"""

#: Where the movie itself sits inside the player: the player letterboxes a movie of another shape.
_THE_PLAYER = """
(() => { const p = document.querySelector('ruffle-player'); if (!p) return null;
  const r = p.getBoundingClientRect(); const m = p.ruffle ? p.ruffle().metadata : p.metadata;
  return {box: [r.left, r.top, r.width, r.height], wide: m ? m.width : null, tall: m ? m.height : null}; })()
"""


#: Whether the movie is playing, as the emulator itself says.
_IS_PLAYING = "(() => { const p = document.querySelector('ruffle-player'); return !!(p && p.isPlaying); })()"


def _served(bundle: Path) -> tuple[ThreadingHTTPServer, str, dict[int, dict]]:
    from tools.serve_offline_flash_bundle import BundleHandler, check_bundle

    class _Quietly(BundleHandler):
        def log_message(self, *_args: Any) -> None:
            pass

    directory = bundle.expanduser().resolve()
    manifest = check_bundle(directory)
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quietly, directory=str(directory)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    cases = {case["id"]: case for case in manifest["cases"] if case.get("status") == "asset_verified"}
    return server, f"http://127.0.0.1:{server.server_port}", cases


def _decoded(data_url: str) -> np.ndarray | None:
    from PIL import Image

    _head, _comma, body = str(data_url or "").partition(",")
    if not body:
        return None
    with Image.open(io.BytesIO(base64.b64decode(body))) as image:
        return np.asarray(image.convert("RGB"))


@dataclass
class GameAtHand:
    """The game numbered ``case`` in the museum's list, open and running, with its movie's place on the page."""

    case: int
    bundle: Path = BUNDLE
    headless: bool = True
    title: str = ""
    #: The movie's place on the page, (left, top, width, height) in page pixels.
    movie: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    pictures_seen: int = 0
    _page: Any = None
    _browser: Any = None
    _playwright: Any = None
    _server: Any = None
    _keys_down: set[str] = field(default_factory=set)
    _blank_canvas: bool = False
    _stream: Any = None

    async def __aenter__(self) -> GameAtHand:
        from playwright.async_api import async_playwright

        self._server, origin, cases = await asyncio.to_thread(_served, self.bundle)
        if self.case not in cases:
            raise LookupError(f"the bundle has no playable program numbered {self.case}")
        self.title = str(cases[self.case].get("title") or "")
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._page = await self._browser.new_page(viewport=VIEWPORT)
        await self._page.goto(f"{origin}/index.html?case={self.case}", wait_until="load", timeout=60000)
        await self._started()
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        for key in list(self._keys_down):
            await self.key_up(key)
        if self._stream is not None:
            await self._stream.close()
        if self._browser is not None:
            await self._browser.close()
        if self._playwright is not None:
            await self._playwright.stop()
        if self._server is not None:
            self._server.shutdown()

    async def _started(self) -> None:
        """The emulator loaded and the movie playing: its splash pressed through if it waits on a press."""
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            player = await self._page.evaluate(_THE_PLAYER)
            if player and player.get("wide"):
                break
            await asyncio.sleep(0.25)
        else:
            player = await self._page.evaluate(_THE_PLAYER)
        if not player:
            raise RuntimeError("the emulator never showed a player")
        left, top, wide, tall = player["box"]
        # The emulator's own splash is pressed until the movie is playing: one press made as the splash was still
        # coming up took nothing, and the runner watched a play button for a minute.
        for _press in range(12):
            await self._page.mouse.move(left + wide / 2 - 9, top + tall / 2 - 7)
            await self._page.mouse.move(left + wide / 2, top + tall / 2, steps=4)
            await self._page.mouse.click(left + wide / 2, top + tall / 2)
            await asyncio.sleep(0.6)
            if await self._page.evaluate(_IS_PLAYING):
                break
        player = await self._page.evaluate(_THE_PLAYER) or player
        left, top, wide, tall = player["box"]
        movie_wide, movie_tall = player.get("wide"), player.get("tall")
        if movie_wide and movie_tall:
            scale = min(wide / movie_wide, tall / movie_tall)
            shown_wide, shown_tall = movie_wide * scale, movie_tall * scale
            self.movie = (left + (wide - shown_wide) / 2, top + (tall - shown_tall) / 2, shown_wide, shown_tall)
        else:
            self.movie = (left, top, wide, tall)

    # -- eyes ----------------------------------------------------------------------------------------------------------

    async def look(self) -> tuple[np.ndarray, float]:
        """The movie as drawn now (RGB, its own place on the page only), and when."""
        picture = None
        if not self._blank_canvas:
            read = await self._page.evaluate(_THE_CANVAS)
            if read:
                whole = await asyncio.to_thread(_decoded, read["pixels"])
                if whole is not None and float(whole.std()) > 2.0:
                    picture = self._cropped_from_canvas(whole, read["box"])
                elif self.pictures_seen == 0:
                    self._blank_canvas = True
        if picture is None:
            picture = await self._streamed()
        self.pictures_seen += 1
        return picture, time.monotonic()

    def _cropped_from_canvas(self, whole: np.ndarray, box: list[float]) -> np.ndarray:
        left, top, wide, tall = box
        across, down = whole.shape[1] / max(1.0, wide), whole.shape[0] / max(1.0, tall)
        x0, y0 = (self.movie[0] - left) * across, (self.movie[1] - top) * down
        x1, y1 = x0 + self.movie[2] * across, y0 + self.movie[3] * down
        part = whole[max(0, int(y0)):int(round(y1)), max(0, int(x0)):int(round(x1))]
        return part if part.size else whole

    async def _streamed(self) -> np.ndarray:
        from core.perception.frames_as_they_are_drawn import PageFrames

        clip = {"x": self.movie[0], "y": self.movie[1], "width": self.movie[2], "height": self.movie[3]}
        if self._stream is None:
            self._stream = PageFrames(self._page)
        got = await self._stream.look(clip)
        if got is not None:
            return got[0]
        shot = await self._page.screenshot(clip=clip, type="png")
        from core.perception.picture_arithmetic import decode

        return decode(shot)

    def words(self, picture: np.ndarray) -> list[dict[str, Any]]:
        """The runs of text on a picture, with where each sits (fractions of the movie, top-left)."""
        from core.perception.what_the_pixels_show import recognize_text

        return recognize_text(np.ascontiguousarray(picture[:, :, ::-1]))

    def where_it_says(self, picture: np.ndarray, word: str) -> tuple[float, float] | None:
        """The middle of the run of text that says ``word``, the shortest such run first, or None."""
        wanted = word.lower()
        runs = [run for run in self.words(picture) if wanted in str(run.get("text") or "").lower()]
        if not runs:
            return None
        run = min(runs, key=lambda r: len(str(r.get("text") or "")))
        return float(run["center_x"]), float(run["center_y"])

    async def click_word(self, picture: np.ndarray, word: str) -> bool:
        where = self.where_it_says(picture, word)
        if where is not None:
            await self.click(*where, arrive_s=0.3, held_s=0.1)
        return where is not None

    def said(self, picture: np.ndarray) -> str:
        return " ".join(str(run.get("text") or "") for run in self.words(picture))

    def keep(self, picture: np.ndarray, path: Path | str) -> None:
        from PIL import Image

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(picture, dtype=np.uint8)).save(path)

    # -- hands ---------------------------------------------------------------------------------------------------------

    def at(self, x: float, y: float) -> tuple[float, float]:
        """A place in the movie, in fractions of it, as a place on the page."""
        left, top, wide, tall = self.movie
        return left + min(1.0, max(0.0, x)) * wide, top + min(1.0, max(0.0, y)) * tall

    async def point(self, x: float, y: float, steps: int = 1) -> None:
        await self._page.mouse.move(*self.at(x, y), steps=steps)

    async def click(self, x: float, y: float, held_s: float = 0.04, arrive_s: float = 0.0) -> None:
        """A click made as a hand makes one: the pointer comes over the place from beside it and, where ``arrive_s``
        is given, rests there before the press. A Flash button answers only a pointer that came over it first."""
        px, py = self.at(x, y)
        if arrive_s:
            await self._page.mouse.move(px - 8, py - 6)
            await self._page.mouse.move(px, py, steps=6)
            await asyncio.sleep(arrive_s)
        else:
            await self._page.mouse.move(px, py)
        await self._page.mouse.down()
        await asyncio.sleep(held_s)
        await self._page.mouse.up()

    async def press(self, x: float, y: float) -> None:
        await self.point(x, y)
        await self._page.mouse.down()

    async def release(self) -> None:
        await self._page.mouse.up()

    async def drag(self, path: list[tuple[float, float]], over_s: float = 0.3) -> None:
        """The button pressed at the path's start, carried along it, and let go at its end."""
        await self.press(*path[0])
        for x, y in path[1:]:
            await asyncio.sleep(over_s / max(1, len(path) - 1))
            await self.point(x, y)
        await self.release()

    async def key_down(self, key: str) -> None:
        self._keys_down.add(key)
        await self._page.keyboard.down(key)

    async def key_up(self, key: str) -> None:
        self._keys_down.discard(key)
        await self._page.keyboard.up(key)

    async def tap(self, key: str, held_s: float = 0.05) -> None:
        await self.key_down(key)
        await asyncio.sleep(held_s)
        await self.key_up(key)

    async def type_text(self, text: str) -> None:
        await self._page.keyboard.type(text, delay=40)


def colour_mask(picture: np.ndarray, colour: tuple[int, int, int], within: float = 30.0) -> np.ndarray:
    """Where a picture is about one colour."""
    difference = picture[:, :, :3].astype(np.int16) - np.asarray(colour, dtype=np.int16)
    return np.abs(difference).max(axis=2) <= within


def report(out: Path | None, what: dict[str, Any]) -> None:
    print(json.dumps(what, default=str), flush=True)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps(what, indent=2, default=str))
