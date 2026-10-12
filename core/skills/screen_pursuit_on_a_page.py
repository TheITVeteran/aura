"""The screen pursuit's eyes and hands on a page in her own browser.

The screen pursuit reaches the world through nine things: read the screen,
press a key, press several, click a point, and five questions about windows —
is hers in front, what page is it, what is in front, what is on top of it, and
bring it back. Every test of it replaces exactly those nine with a world of its
own. A page in her own browser is one more world, and this is those nine
answered by the page: a picture the browser takes of what it draws, keys and
clicks the browser delivers to the page itself, and the page's own address.

Nothing reaches the desktop while a page is the surface. The keys go to the
page whatever the person has in front of them, and no window is ever guessed
at: LIVE 2 October, a hand-over that took the front window to be hers would
have played into the person's own Chrome.

The surface is set for the length of one hand-over (``HER_OWN_PAGE``) and each
of the nine checks it first, so a pursuit over the desktop is exactly what it
was.
"""

from __future__ import annotations

import contextvars
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.runtime.what_stops_it import current

__all__ = ["HER_OWN_PAGE", "OnAPage", "it_was_answered", "on_her_page"]

#: Where keys go when a page is the surface: the pursuit's names to the
#: browser's.
_KEY_NAMES = {
    "up": "ArrowUp",
    "down": "ArrowDown",
    "left": "ArrowLeft",
    "right": "ArrowRight",
    "return": "Enter",
    "enter": "Enter",
    "tab": "Tab",
    "space": "Space",
    "escape": "Escape",
    "esc": "Escape",
    "backspace": "Backspace",
    "delete": "Delete",
    "shift": "Shift",
    "control": "Control",
    "ctrl": "Control",
    "alt": "Alt",
    "home": "Home",
    "end": "End",
    "pageup": "PageUp",
    "pagedown": "PageDown",
}

#: Focus the thing the page draws, across shadow roots, so keys reach it. A
#: player in a shadow root takes the keyboard only once it is focused (measured
#: in the pane 2 October: after "Play Game" the focus stayed on the body).
_FOCUS_THE_DRAWING = """
(function () {
  var best = null, area = 0;
  var walk = function (root, host) {
    var all = root.querySelectorAll('*');
    for (var i = 0; i < all.length; i++) {
      var el = all[i];
      if (el.tagName === 'CANVAS' || el.tagName === 'EMBED' || el.tagName === 'OBJECT') {
        var r = el.getBoundingClientRect();
        if (r.width * r.height > area) { area = r.width * r.height; best = host || el; }
      }
      if (el.shadowRoot) walk(el.shadowRoot, host || el);
    }
  };
  walk(document, null);
  if (!best) return false;
  if (!best.hasAttribute('tabindex')) best.setAttribute('tabindex', '0');
  best.focus();
  return document.activeElement === best;
})()
"""


@dataclass
class OnAPage:
    """One page of her own browser, as the surface a screen pursuit acts on."""

    page: Any
    name: str
    _focused: bool = field(default=False, init=False)
    #: The page watched as one stream while she uses it (core/perception/watching_it_happen.py).
    watching: Any = field(default=None, init=False)
    _watching_over: Any = field(default=None, init=False)
    _pictures: Any = field(default=None, init=False)
    #: The words of the screen the pointer was last swept over, and where it turned into a hand there.
    _swept: Any = field(default=None, init=False)

    #: The longest she waits on what plays by itself inside one look, in seconds.
    WATCH_IN_A_LOOK_S = 5.0

    async def read(self, over: tuple[float, float, float, float] | None = None) -> dict[str, Any]:
        from core.perception.what_her_page_shows import look_at_a_page

        await self._watch(over)
        seen = await look_at_a_page(self.page, over, name=self.name)
        if seen is None:
            return {"ok": False, "text": "", "layout": [], "error": "her page could not be photographed"}
        seen.setdefault("ok", True)
        _what_the_stream_shows(seen, self.watching)
        # Whether what is read is a screen of play in a place played in time, for whoever waits on it to come to rest.
        from .screen_pursuit_as_it_happens import AS_IT_HAPPENS, a_screen_of_play_in_time

        reflexes = AS_IT_HAPPENS.get()
        labels = [str(r.get("text") or "") for r in seen.get("layout") or () if isinstance(r, dict)]
        if reflexes is not None:
            reflexes.play_in_sight = a_screen_of_play_in_time(reflexes, labels)
        if not getattr(reflexes, "play_in_sight", False):
            await self._where_the_pointer_turns(seen, over)
        return seen

    async def _where_the_pointer_turns(self, seen: dict[str, Any], over: tuple[float, float, float, float] | None) -> None:
        """Once a screen, the places on it where the pointer turns into a hand (core/perception/where_the_pointer_turns.py),
        among the shapes she may click: a doorway or a hidden thing looks like nothing in particular."""
        from core.perception.where_the_pointer_turns import where_the_pointer_turns

        words = frozenset(w for w in str(seen.get("text") or "").lower().split() if len(w) >= 3)
        known = self._swept
        if known is not None and (known[0] == words or len(words & known[0]) >= 0.5 * max(1, min(len(words), len(known[0])))):
            places = known[1]
        else:
            places = await where_the_pointer_turns(self.page, over)
            self._swept = (words, places)
        regions = [r for r in [*(seen.get("shapes") or []), *(seen.get("layout") or [])] if isinstance(r, dict)]
        for x, y in places:
            # Where a thing already read stands there, it is that thing which answers the pointer.
            standing = [r for r in regions if abs(x - float(r.get("center_x", -1))) < 0.05
                        and abs(y - float(r.get("center_y", -1))) < 0.05]
            for region in standing:
                region["answers_the_pointer"] = True
            if standing:
                continue
            seen.setdefault("shapes", []).append({"text": f"the one that stands out at {round(x * 100)}% across, "
                                                          f"{round(y * 100)}% down", "center_x": x, "center_y": y,
                                                  "shape": True, "answers_the_pointer": True})

    async def _watch(self, over: tuple[float, float, float, float] | None) -> None:
        """Watch the part being read as a stream from the first look; and while something plays by itself, wait for it."""
        from core.perception.watching_it_happen import Watching

        if self.watching is not None and over is None:
            # A look at the whole page does not move what she watches: the stream stays on the part she acts on, and her
            # acts on it stay hers to be answered. Measured offline 2026-10-11: each act was followed by a whole-page
            # look that began the stream again, every act's answer was lost, and every click was judged by whether the
            # words changed; a choice that only lit up ("1 Player") was taken to do nothing, and she went back and on
            # between two screens for four minutes.
            return
        if self.watching is None or over != self._watching_over:
            await self.stop_watching()
            self._pictures = await _pictures_of(self.page, over)
            self.watching, self._watching_over = Watching(take=self._pictures), over
            self.watching.start()
            return
        from .screen_pursuit import _tell
        from .screen_pursuit_as_it_happens import AS_IT_HAPPENS, a_screen_of_play_in_time

        # Play that goes on without her is not waited out, it is played: offline 2026-10-11 a game's eggs fell while she
        # watched them "play out on its own" twice, and it was over before she played.
        reflexes = AS_IT_HAPPENS.get()
        if getattr(reflexes, "play_in_sight", False) or a_screen_of_play_in_time(reflexes):
            return
        # Inside one look, which has its own few seconds: a wait longer than the look is a look that never comes back.
        await self.watching.watch(at_most_s=self.WATCH_IN_A_LOOK_S, tell=_tell)

    async def stop_watching(self) -> None:
        if self.watching is not None:
            await self.watching.stop()
            self.watching = None
        pictures, self._pictures = self._pictures, None
        if pictures is not None:
            await pictures.close()

    async def _focus(self) -> None:
        if self._focused:
            return
        try:
            self._focused = bool(await self.page.evaluate(_FOCUS_THE_DRAWING))
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            self._focused = False

    async def press(self, key: str) -> bool:
        context = current(whose="page_input.press")
        context.check()
        from core.skills.screen_pursuit_surface import PRESSABLE_KEYS

        name = str(key or "").strip().lower()
        if name not in PRESSABLE_KEYS:
            return False
        await self._focus()
        context.check()
        from .screen_pursuit_as_it_happens import AS_IT_HAPPENS

        reflexes = AS_IT_HAPPENS.get()
        watching = self.watching
        window = watching.begin_act(name) if watching is not None else None
        delivered = False
        try:
            if reflexes is not None:
                # In a game, a key is held a moment and watched, the way a person tries one.
                await reflexes.pressed(name)
            else:
                await self.page.keyboard.press(_KEY_NAMES.get(name, name))
            delivered = True
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            return False
        finally:
            if window is not None:
                watching.finish_act(window, delivered)
        return True

    async def press_many(self, keys: Sequence[str]) -> int:
        landed = 0
        for key in keys:
            if not await self.press(key):
                break
            landed += 1
        return landed

    async def click(self, x: float, y: float, bounds: Sequence[int] | None) -> bool:
        """A click at shares of ``bounds``, in the page's own pixels; never outside them."""
        context = current(whose="page_input.click")
        context.check()
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False
        if bounds and len(bounds) >= 4:
            left, top, wide, tall = (float(value) for value in bounds[:4])
        else:
            from core.perception.what_her_page_shows import the_page_size

            wide, tall = await the_page_size(self.page)
            left = top = 0.0
        if self._swept is not None and any(abs(x - a) < 0.02 and abs(y - b) < 0.02 for a, b in self._swept[1]):
            # A place found by the pointer turning into a hand is clicked where it answers now: it may move.
            from core.perception.where_the_pointer_turns import where_the_hand_is_now

            x, y = await where_the_hand_is_now(self.page, x, y, left, top, wide, tall)
        watching = self.watching
        window = None
        delivered = False
        try:
            context.check()
            window = watching.begin_act(f"click at {x:.2f}, {y:.2f}") if watching is not None else None
            await self.page.mouse.click(left + x * wide, top + y * tall)
            delivered = True
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            return False
        finally:
            if window is not None:
                watching.finish_act(window, delivered)
        # A click on the drawing is also what gives it the keyboard.
        self._focused = True
        return True

    #: The steps a carry is made in, and the pause between them: a page sees the pointer travel with the button down,
    #: as a hand carries it, and not a jump from the one place to the other.
    CARRY_STEPS = 12
    CARRY_STEP_S = 0.02

    async def carry(self, start: tuple[float, float], end: tuple[float, float], bounds: Sequence[int] | None) -> bool:
        """A press at ``start`` carried with the button held to ``end`` and let go, at shares of ``bounds``."""
        import asyncio

        context = current(whose="page_input.carry")
        context.check()

        if not all(0.0 <= v <= 1.0 for v in (*start, *end)):
            return False
        if bounds and len(bounds) >= 4:
            left, top, wide, tall = (float(value) for value in bounds[:4])
        else:
            from core.perception.what_her_page_shows import the_page_size

            wide, tall = await the_page_size(self.page)
            left = top = 0.0
        (x0, y0), (x1, y1) = ((left + x * wide, top + y * tall) for x, y in (start, end))
        watching = self.watching
        window = None
        delivered = False
        try:
            context.check()
            await self.page.mouse.move(x0, y0)
            context.check()
            window = watching.begin_act(
                f"carry from {start[0]:.2f}, {start[1]:.2f} to {end[0]:.2f}, {end[1]:.2f}"
            ) if watching is not None else None
            try:
                await self.page.mouse.down()
                for step in range(1, self.CARRY_STEPS + 1):
                    context.check()
                    share = step / self.CARRY_STEPS
                    await self.page.mouse.move(x0 + (x1 - x0) * share, y0 + (y1 - y0) * share)
                    await asyncio.sleep(self.CARRY_STEP_S)
            finally:
                await self.page.mouse.up()
            delivered = True
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            return False
        finally:
            if window is not None:
                watching.finish_act(window, delivered)
        self._focused = True
        return True

    async def identity(self) -> dict[str, str]:
        try:
            title = str(await self.page.title() or "")
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            title = ""
        return {"url": str(getattr(self.page, "url", "") or ""), "title": title, "error": ""}


async def _pictures_of(page: Any, over: tuple[float, float, float, float] | None) -> Any:
    """Where the stream's pictures come from: the page's own frames as it draws them, else screenshots of the part."""
    from core.perception.frames_as_they_are_drawn import CanvasFrames, PageFrames
    from core.perception.what_her_page_shows import the_page_size

    wide, tall = await the_page_size(page)
    left, top, right, bottom = over if over is not None else (0.0, 0.0, 1.0, 1.0)
    clip = {"x": left * wide, "y": top * tall, "width": max(1.0, (right - left) * wide), "height": max(1.0, (bottom - top) * tall)}
    frames = PageFrames(page, every_nth_frame=4)

    # Reading a drawing directly avoids streaming an entire window at the
    # display's refresh rate just to sample its changes eight times a second.
    canvas = CanvasFrames(page) if over is not None else None
    return _PagePictures(page, clip, frames, canvas)


@dataclass
class _PagePictures:
    """The pictures and the stream that produces them have the same owner."""

    page: Any
    clip: dict[str, float]
    frames: Any
    canvas: Any = None

    async def __call__(self) -> Any:

        if self.canvas is not None and not self.canvas.unavailable:
            drawn = await self.canvas.look(self.clip)
            if drawn is not None:
                return drawn
        if not self.frames.unavailable:
            streamed = await self.frames.look(self.clip)
            if streamed is not None:
                return streamed
        from core.perception.a_picture_of_her_page import picture_of

        # The whole window, cut down here: a picture of part of the page made a visible window flicker (LIVE 2026-10-09).
        picture = await picture_of(self.page, self.clip, css=False, kind="png")
        return None if picture is None else picture[:, :, ::-1].copy()

    async def close(self) -> None:
        if self.canvas is not None:
            await self.canvas.close()
        await self.frames.close()


def _what_the_stream_shows(seen: dict[str, Any], watching: Any) -> None:
    """Mark the writing that keeps changing by itself, a scene's lines, which are read and not pressed; and whether her last act was answered."""
    if watching is None:
        return
    for region in seen.get("layout") or []:
        try:
            box = (float(region["x"]), float(region["y"]), float(region["width"]), float(region["height"]))
        except (KeyError, TypeError, ValueError):
            continue
        if watching.keeps_changing_on_its_own(box):
            region["of_its_own"] = True
    if watching.acts:
        when, _act = watching.acts[-1]
        seen["answered"] = watching.answered(when)
        seen["answer_window"] = watching.answer_window()
        # Where what her last act set going came to rest: where a machine she built stopped when tested.
        seen["came_to_rest_at"] = watching.where_it_came_to_rest(when)


def it_was_answered(changed: bool | None, observation: dict[str, Any]) -> bool | None:
    """Use the owned pixel response, or still comparisons when no stream owns it.

    Identical words can accompany a visual response. An incomplete stream
    supplies None; a measured autonomous change keeps its False veto.
    """
    if changed is None:
        return None
    window = observation.get("answer_window")
    if isinstance(window, dict):
        if window.get("delivered") is not True:
            return None
        measured = observation.get("answered")
        return measured if type(measured) is bool else None
    return bool(changed) and observation.get("answered") is not False


#: The page a hand-over is playing on, for as long as it plays.
HER_OWN_PAGE: contextvars.ContextVar[OnAPage | None] = contextvars.ContextVar(
    "aura_her_own_page", default=None
)


def on_her_page() -> OnAPage | None:
    """The page the pursuit is acting on, when it is acting on one."""
    return HER_OWN_PAGE.get()
