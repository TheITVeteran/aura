"""Read the current screen before old menu words hold back a running world.

Offline 2026-10-12 a falling body reached the ground while its old PLAY menu
was still what the reflex went by. Reading a frame's words takes less time
than interpreting its whole scene. Empty or old readings leave the state
unknown; they cannot say that a menu has gone.
"""
from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import numpy as np

from core.verify import invariant


def remember_menu_pixels(reflexes: Any, observation: Any, *, menu: bool) -> None:
    """Bind a menu reading to its captured word patches and this view's owner."""
    from core.perception.words_kept_with_their_pixels import words_with_their_pixels

    if menu:
        words = words_with_their_pixels(observation)
        if words is not None:
            reflexes.menu_pixels = (id(reflexes.page), getattr(reflexes.page, "url", ""), reflexes.band, words)
    elif str(observation.get("text") or "").strip():
        reflexes.menu_pixels = None


async def _old_menu_changed(reflexes: Any, pixels: Any, at: float) -> float | None:
    kept = getattr(reflexes, "menu_pixels", None)
    if kept is None or kept[:3] != (id(reflexes.page), getattr(reflexes.page, "url", ""), reflexes.band):
        return None
    if kept[3].still_in(pixels) is not False:
        return None
    await asyncio.sleep(.05)
    again = await reflexes.look()
    if again is None or not math.isfinite(again[1]) or not at < again[1] <= time.monotonic():
        return None
    second = np.asarray(again[0])
    if second.ndim != 3 or second.shape[2] < 3 or time.monotonic() - again[1] > .25:
        return None
    return again[1] if kept[3].still_in(second[:, :, 2::-1]) is False else None


def words_of_this_frame(regions: list[dict], *, at: float, now: float, band: tuple) -> dict | None:
    """A recent frame's measured words, placed in the whole view; None without a current reading."""
    if not all(math.isfinite(v) for v in (at, now)) or not 0 <= now - at <= .25:
        return None
    measured = [dict(r) for r in regions if isinstance(r, dict) and str(r.get("text") or "").strip()]
    if not measured:
        return None
    left, top, right, bottom = band
    for region in measured:
        for axis, origin, span in (("x", left, right - left), ("y", top, bottom - top)):
            for key in (axis, "center_" + axis):
                if key in region:
                    region[key] = origin + float(region[key]) * span
        for key, span in (("width", right - left), ("height", bottom - top)):
            if key in region:
                region[key] = float(region[key]) * span
    return {"ok": True, "text": " ".join(str(r["text"]).strip() for r in measured),
            "layout": measured, "captured_at": at}


async def read_current_words(reflexes: Any, *, goes_on: bool) -> dict | None:
    """Refresh a known running world's words before choosing between its menu and its controls."""
    if not goes_on:
        return None
    from core.perception.observed_transfer import ScreenReading, make_snapshot
    from core.perception.what_the_pixels_show import recognize_text
    from core.perception.words_kept_with_their_pixels import words_with_their_pixels

    looked = await reflexes.look()
    if looked is None:
        return None
    picture, at = looked
    pixels = np.asarray(picture)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        return None
    # The word reader takes BGR; the control stream supplies RGB.
    bgr = pixels[:, :, 2::-1].copy()
    regions = await asyncio.to_thread(recognize_text, bgr)
    observed = words_of_this_frame(regions, at=at, now=time.monotonic(), band=reflexes.band)
    if observed is not None:
        observed = ScreenReading(observed)
        observed.word_pixels = words_with_their_pixels(ScreenReading({"layout": regions}, picture_snapshot=make_snapshot(bgr)))
        reflexes.read(observed)
    elif 0 <= time.monotonic() - at <= .25:
        confirmed_at = await _old_menu_changed(reflexes, bgr, at)
        if confirmed_at is None:
            return None
        # This retires a contradicted reading. It says nothing about what the
        # new view means; her known controls are tried on what its pixels show.
        reflexes.said_now, reflexes.way_on_shown = "", False
        reflexes.words.append("")
        del reflexes.words[:-12]
        observed = {"ok": True, "text": "", "layout": [], "captured_at": confirmed_at, "prior_words_changed": True}
    return observed


@invariant("skills.current_play_words_require_a_measured_frame", scope="skills",
           owner="core/skills/current_words_before_play.py", observational=False)
def _current_play_words_invariant() -> tuple:
    band = (0, 0, 1, 1)
    assert words_of_this_frame([], at=1, now=1.1, band=band) is None
    assert words_of_this_frame([{"text": "PLAY"}], at=0, now=1.1, band=band) is None
    assert words_of_this_frame([{"text": "PLAY"}], at=2, now=1.1, band=band) is None
    assert words_of_this_frame([{"text": "FUEL 73"}], at=1, now=1.1, band=band)["text"] == "FUEL 73"
    return ()
