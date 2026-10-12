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
    from core.perception.what_the_pixels_show import recognize_text

    looked = await reflexes.look()
    if looked is None:
        return None
    picture, at = looked
    pixels = np.asarray(picture)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        return None
    # The word reader takes BGR; the control stream supplies RGB.
    regions = await asyncio.to_thread(recognize_text, pixels[:, :, 2::-1].copy())
    observed = words_of_this_frame(regions, at=at, now=time.monotonic(), band=reflexes.band)
    if observed is not None:
        reflexes.read(observed)
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
