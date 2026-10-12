"""Keep the pixels a screen's words belonged to, so an old reading can be retired.

Offline 2026-10-12 a text-free moving world was still held back by the menu
read before it. Empty OCR could not say that the menu had gone. Its measured
word patches could: most no longer looked like what was read there.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.perception.observed_transfer import _distinctive, _patch, freeze_screen, make_snapshot
from core.perception.visual_anchor import _appearance_error
from core.verify import invariant


@dataclass(frozen=True)
class WordsWithPixels:
    """Bounded word patches in one captured view, with the view's shape."""

    width: int
    height: int
    patches: tuple[tuple[tuple, bytes], ...]

    def still_in(self, picture: Any) -> bool | None:
        """Whether most word patches still fit, or measurably changed; None without enough measured patches."""
        current = make_snapshot(picture)
        if current is None or abs(current.width / current.height - self.width / self.height) > .02:
            return None
        same = changed = 0
        for box, before in self.patches:
            after = _patch(current, box)
            if len(after) != len(before):
                continue
            if _appearance_error(before, after) is not None:
                same += 1
            elif float(np.abs(np.frombuffer(before, np.uint8).astype(float)
                              - np.frombuffer(after, np.uint8).astype(float)).mean()) / 255 > .10:
                changed += 1
        required = .75 * len(self.patches)
        if self.patches and same >= required:
            return True
        if self.patches and changed >= required:
            return False
        return None


def words_with_their_pixels(observation: Any) -> WordsWithPixels | None:
    """Remember only labelled word extents with distinctive pixels from the captured reading."""
    attached = getattr(observation, "word_pixels", None)
    if isinstance(attached, WordsWithPixels):
        return attached
    screen = freeze_screen(observation)
    if screen.picture is None:
        return None
    labels = {str(r.get("text") or "").strip().casefold() for r in observation.get("layout") or () if isinstance(r, dict)}
    patches = []
    for region in screen.regions:
        if not region.label or region.label not in labels:
            continue
        appearance = _patch(screen.picture, region.bbox)
        if _distinctive(appearance):
            patches.append((region.bbox, appearance))
    return WordsWithPixels(screen.picture.width, screen.picture.height, tuple(patches[:24])) if patches else None


@invariant("perception.kept_words_need_their_captured_pixels", scope="perception",
           owner="core/perception/words_kept_with_their_pixels.py", observational=False)
def _kept_word_pixels_invariant() -> tuple:
    from core.perception.observed_transfer import ScreenReading

    picture = np.zeros((32, 32, 3), np.uint8)
    picture[8:24:2, 8:24] = 240
    region = {"text": "PLAY", "x": .25, "y": .25, "width": .5, "height": .5}
    seen = ScreenReading({"layout": [region]}, picture_snapshot=make_snapshot(picture))
    words = words_with_their_pixels(seen)
    assert words is not None and words.still_in(picture) is True
    assert words.still_in(np.full_like(picture, 60)) is False
    assert words.still_in(None) is None
    assert words_with_their_pixels({"layout": [region]}) is None
    return ()
